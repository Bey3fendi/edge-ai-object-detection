"""Kesintiden devamda birebir aynı veri akışı: tohumlu örnekleyici + örnek başına tohumlu artırma.

YOLOX'un varsayılanı devam için uygun değil (fork 3ec6263):
- InfiniteSampler her süreç başlangıcında aynı tohumla baştan başlar → devam edilen koşu epoch 0'ın veri
  sırasını tekrar görür.
- Artırmalar (mosaic, random_affine, HSV, flip, mixup) worker'daki global `random`/`np.random` ile yapılır;
  worker tohumu `uuid4` → tekrar üretilemez.
- Mosaic `close_mosaic()` ile örnekleyici özniteliği değiştirilerek kapanır; önceden üretilmiş (prefetch)
  batch'ler eski bayrakla kalır → kapanış noktası koşudan koşuya kayabilir.

Burada:
- Epoch e'nin sırası = torch.randperm(N, Generator(seed_e)), seed_e = H(koşu tohumu, "perm", e). Epoch başına
  N // B iterasyon (drop_last); her batch öğesi (mosaic, görsel_idx, epoch, mantıksal_konum) taşır.
- Mosaic bayrağı epoch'un fonksiyonudur: e < max_epoch − no_aug_epochs − 1 (YOLOX before_epoch kapanışıyla
  aynı epoch); close_mosaic çağrısı yok sayılır.
- Örnek başına artırma tohumu = H(koşu tohumu, epoch, mantıksal konum, rastgelelik kaynağı); python `random`
  ve `numpy.random` ayrı türetilmiş tohumlarla bu örnek için yeniden tohumlanır. Böylece aynı örnek, hangi
  worker'a düşerse düşsün ve devam noktası neresi olursa olsun aynı artırmayı alır (mosaic'in yardımcı
  görselleri ve mixup seçimi dahil: hepsi bu akışlardan çekilir).
- Çok ölçekli eğitimde giriş boyutu (YOLOX random_resize, 10 iterasyonda bir) = H(koşu tohumu, "rs",
  iterasyon); devam noktasındaki boyut `size_at(progress)` ile yeniden kurulur.
"""

import hashlib
import json
import random

import numpy as np
import torch
from torch.utils.data.sampler import Sampler


def derive_seed(*parts, bits=32):
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "little") % (1 << bits)


def mosaic_on(epoch, max_epoch, no_aug_epochs):
    """YOLOX before_epoch: epoch + 1 == max_epoch − no_aug_epochs olduğunda mosaic kapanır."""
    return epoch < max_epoch - no_aug_epochs - 1


class ResumableBatchSampler(Sampler):
    """Epoch tabanlı, tohumlu, (start_epoch, start_iter)'den devam eden sonsuz batch örnekleyici."""

    def __init__(self, n, batch_size, seed, max_epoch, no_aug_epochs, start_epoch=0, start_iter=0):
        self.n, self.bs, self.seed = int(n), int(batch_size), int(seed)
        self.max_epoch, self.no_aug_epochs = int(max_epoch), int(no_aug_epochs)
        self.start_epoch, self.start_iter = int(start_epoch), int(start_iter)
        self.iters_per_epoch = self.n // self.bs
        assert self.iters_per_epoch > 0
        self.mosaic = True          # YOLOX DataLoader.close_mosaic uyumluluğu; kullanılmaz

    def perm(self, epoch):
        g = torch.Generator()
        g.manual_seed(derive_seed(self.seed, "perm", epoch))
        return torch.randperm(self.n, generator=g).tolist()

    def batch(self, epoch, it, perm=None):
        perm = perm if perm is not None else self.perm(epoch)
        m = mosaic_on(epoch, self.max_epoch, self.no_aug_epochs)
        return [(m, perm[pos], epoch, pos) for pos in range(it * self.bs, (it + 1) * self.bs)]

    def __iter__(self):
        epoch, it = self.start_epoch, self.start_iter
        while True:
            perm = self.perm(epoch)
            for i in range(it, self.iters_per_epoch):
                yield self.batch(epoch, i, perm)
            epoch, it = epoch + 1, 0

    def __len__(self):
        return self.iters_per_epoch


def seed_sample(run_seed, epoch, pos):
    random.seed(derive_seed(run_seed, "aug-py", epoch, pos))
    np.random.seed(derive_seed(run_seed, "aug-np", epoch, pos))


def _seeded_mosaic_cls():
    from yolox.data import MosaicDetection

    class SeededMosaic(MosaicDetection):
        """MosaicDetection; (mosaic, idx, epoch, pos) indeksinde önce örneğe özel tohumlama."""
        run_seed = 0

        def __getitem__(self, index):
            if isinstance(index, tuple) and len(index) == 4:
                mosaic, idx, epoch, pos = index
                seed_sample(self.run_seed, epoch, pos)
                return super().__getitem__((mosaic, idx))
            return super().__getitem__(index)

    SeededMosaic.__module__ = __name__
    SeededMosaic.__qualname__ = "SeededMosaic"
    return SeededMosaic


def __getattr__(name):   # modül düzeyinde (pickle edilebilir) sınıf; yolox ilk kullanımda içe aktarılır
    if name == "SeededMosaic":
        cls = _seeded_mosaic_cls()
        globals()["SeededMosaic"] = cls
        return cls
    raise AttributeError(name)


def make_seeded_mosaic(base_cls=None):
    """Geriye uyumluluk: modül düzeyindeki SeededMosaic (spawn ile başlayan worker'lar da pickle edebilir)."""
    import sys
    return getattr(sys.modules[__name__], "SeededMosaic")

def resize_size(run_seed, progress, random_size, size_factor=1.0):
    """YOLOX random_resize ile aynı aralık; iterasyon numarasından tohumlu."""
    r = random.Random(derive_seed(run_seed, "rs", progress))
    s = r.randint(*random_size)
    return (int(32 * s), 32 * int(s * size_factor))


def size_at(run_seed, progress, random_size, initial):
    """`progress` iterasyonunun başındaki giriş boyutu (YOLOX: (p + 1) % 10 == 0 olan iterasyondan sonra değişir)."""
    if progress < 10 or random_size is None:
        return tuple(initial)
    last = (progress // 10) * 10 - 1
    return resize_size(run_seed, last, random_size)


def read_trace(path):
    out = {}
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:   # kill -9 anında yarım kalmış son satır
                continue
            out.setdefault(r["progress"], []).append(r)
    return out


def compare_traces(ref_path, test_path, loss_rtol=1e-3):
    """Kesintisiz (ref) ve kesilip devam eden (test) koşunun iterasyon izlerini karşılaştırır.

    Birebir aynı olması gerekenler: alt ağ dizisi, sonraki lr, giriş boyutu, batch (görüntü + etiket) sha256'sı.
    Tekrar koşulan iterasyonlar (son doğrulanmış kayıttan sonra kesilenler) ref ile ayrı ayrı karşılaştırılır.
    Kayıplar GPU belirlenimsizliği yüzünden toleransla raporlanır (karar insanın).
    """
    ref, test = read_trace(ref_path), read_trace(test_path)
    exact_fail, loss_dev, repeated = [], 0.0, 0
    for p, recs in test.items():
        if p not in ref:
            exact_fail.append((p, "ref'te yok"))
            continue
        r0 = ref[p][0]
        repeated += len(recs) - 1
        for r in recs:
            for k in ("cfgs", "lr_next", "input_size", "batch_sha256", "epoch", "iter"):
                if r[k] != r0[k]:
                    exact_fail.append((p, k))
            for a, b in zip(r["losses"], r0["losses"]):
                loss_dev = max(loss_dev, abs(a - b) / max(1e-8, abs(b)))
    missing = sorted(set(ref) - set(test))
    return {"iterations_ref": len(ref), "iterations_test": len(test), "repeated_after_resume": repeated,
            "missing_in_test": missing[:10], "exact_mismatches": exact_fail[:20], "n_exact_mismatches": len(exact_fail),
            "max_loss_rel_dev": loss_dev, "loss_within_rtol": loss_dev <= loss_rtol}


def batch_digest(inps, targets):
    """Modele giren (artırılmış) görüntü + etiket tensörlerinin sha256'sı (kesinti testi)."""
    h = hashlib.sha256()
    for t in (inps, targets):
        h.update(t.detach().float().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
