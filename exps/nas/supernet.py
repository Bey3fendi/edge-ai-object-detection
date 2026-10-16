"""Gerçek NAS (Karar 6, Furkan 2026-10-16): YOLOX-Nano üzerinde one-shot ağırlık paylaşımlı süper ağ.

Arama uzayı (genişlik yok; kanal genişliği budamanın işi, Karar 5/6):
    kernel     her depthwise konvolüsyonda ayrı ayrı {3, 5, 7}   (DW_LAYERS: 30 katman)
    depth      backbone dark3 ve dark4 CSP blok sayısı {1, 2, 3}  (önceden eğitilmiş Nano: 3)
    res        giriş çözünürlüğü {320, 352, 384, 416, 448, 480, 512} (yalnız değerlendirmede;
               süper ağ YOLOX'un çok ölçekli tarifiyle eğitilir)
    → 3^30 × 3^2 × 7 ≈ 1,3 × 10^16 alt ağ.

Ağırlık paylaşımı:
    kernel   her DW katmanı tek 7×7 ağırlık taşır; k×k alt kernel bunun merkez kesitidir (BigNAS tarzı,
             OFA'nın dönüşüm matrisi yok: YOLOX optimizer'ı yalnız `weight`/`bias` adlı parametreleri
             gruplar, matrise ağırlık sönümü de birim matristen değil sıfırdan çekerdi).
             Başlangıç: önceden eğitilmiş 3×3 merkezde, kenarlar sıfır → başlangıçta her kernel seçimi
             önceden eğitilmiş Nano ile aynı çıktıyı verir (nano_macro.widen_depthwise ile aynı ilke).
    depth    CSPLayer'ın ilk d Bottleneck'i koşar (OFA elastik derinlik).
    BN       katman başına paylaşımlı; alt ağ değerlendirilmeden önce BN istatistikleri yeniden
             hesaplanır (scripts/supernet_search.py::recalibrate_bn).

Eğitim (sandwich kuralı, BigNAS): her iterasyonda en büyük, en küçük ve `nas_sandwich_random` (=2, BigNAS)
rastgele alt ağ aynı mini-batch üzerinde koşar; kayıplar ortalanıp tek optimizer adımı atılır (ortalama:
lr 0,001/64 tek ağ için doğrulandı, toplam gradyan adımı büyütürdü). BigNAS'tan bilinçli sapma: inplace
distillation yok (tespit başlığı için damıtma kaybı ayrıca gerekçelendirilmesi gereken bir tasarım; Karar 6).

Kesintiye dayanıklılık (Karar 6): tam durum (ham + EMA ağırlık, optimizer, AMP ölçekleyici, RNG, epoch/iter)
epoch sonunda ve `nas_ckpt_every_iters`'te bir doğrulanmış depoya (scripts/ckpt_store.py) yazılır; veri
sırası, artırmalar ve çok ölçekli boyut tohumludur (scripts/nas_data.py) → devam noktasından birebir aynı
akış. Süreç her başlatıldığında depodaki en yeni doğrulanmış kayıttan kendiliğinden devam eder.

YOLOX submodule'üne dokunulmaz; model cerrahisi ve trainer kancası bu dosyadadır.
"""

import json
import os
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolox.exp.default.yolox_nano import Exp as _Base

_SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

KERNELS = (3, 5, 7)
DEPTH_STAGES = ("dark3", "dark4")
DEPTHS = (1, 2, 3)
RESOLUTIONS = (320, 352, 384, 416, 448, 480, 512)
MAX_K = max(KERNELS)


# ---- dinamik modüller ---------------------------------------------------------------------------

class DynamicDWConv(nn.Conv2d):
    """Depthwise Conv2d; 7×7 ağırlık taşır, `active_k` × `active_k` merkez kesitiyle koşar."""

    def __init__(self, channels, stride, max_k=MAX_K):
        super().__init__(channels, channels, max_k, stride=stride, padding=max_k // 2,
                         groups=channels, bias=False)
        self.active_k = 3

    @classmethod
    def from_conv3(cls, conv):
        assert conv.kernel_size == (3, 3) and conv.groups == conv.in_channels == conv.out_channels
        assert conv.bias is None and conv.padding == (1, 1)
        new = cls(conv.in_channels, conv.stride[0]).to(conv.weight.device, conv.weight.dtype)
        p = (MAX_K - 3) // 2
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:, :, p:p + 3, p:p + 3].copy_(conv.weight)
        return new

    def active_weight(self, k=None):
        k = self.active_k if k is None else k
        s = (MAX_K - k) // 2
        return self.weight[:, :, s:s + k, s:s + k]

    def forward(self, x):
        k = self.active_k
        return F.conv2d(x, self.active_weight(k), None, self.stride, k // 2, 1, self.groups)


class ElasticSequential(nn.Sequential):
    """İlk `active_n` modülü koşar (CSPLayer.m için elastik derinlik)."""

    def __init__(self, *mods):
        super().__init__(*mods)
        self.active_n = len(mods)

    def forward(self, x):
        for i, m in enumerate(self):
            if i >= self.active_n:
                break
            x = m(x)
        return x


def _set_child(root, dotted_name, new_module):
    parent_name, _, child = dotted_name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child, new_module)


def dw_layer_names(model):
    """Standart Nano'daki tüm 3×3 depthwise Conv2d'lerin adları (model sırasıyla)."""
    return [n for n, m in model.named_modules()
            if isinstance(m, nn.Conv2d) and m.groups > 1 and m.groups == m.in_channels
            and m.kernel_size in ((3, 3), (MAX_K, MAX_K))]


def csp_of(model, stage):
    return getattr(model.backbone.backbone, stage)[1]


def to_supernet(model):
    """Standart (önceden eğitilmiş) Nano'yu yerinde süper ağa çevirir; DW katman adlarını döndürür."""
    names = dw_layer_names(model)
    for n in names:
        _set_child(model, n, DynamicDWConv.from_conv3(model.get_submodule(n)))
    for stage in DEPTH_STAGES:
        csp = csp_of(model, stage)
        csp.m = ElasticSequential(*list(csp.m))
        assert len(csp.m) == max(DEPTHS), (stage, len(csp.m))
    return names


# ---- alt ağ yapılandırması -----------------------------------------------------------------------
# Yapılandırma = {"res": int, "depth": {"dark3": d, "dark4": d}, "k": [30 kernel]}; "k" sırası
# dw_layer_names() sırasıdır (DW_LAYERS'a eşit olduğu to_supernet sonrası doğrulanır).

def max_config(n_layers):
    return {"res": max(RESOLUTIONS), "depth": {s: max(DEPTHS) for s in DEPTH_STAGES},
            "k": [max(KERNELS)] * n_layers}


def min_config(n_layers):
    return {"res": min(RESOLUTIONS), "depth": {s: min(DEPTHS) for s in DEPTH_STAGES},
            "k": [min(KERNELS)] * n_layers}


def nano_config(n_layers, res=416):
    """Dokunulmamış YOLOX-Nano mimarisi (kontrol)."""
    return {"res": res, "depth": {s: 3 for s in DEPTH_STAGES}, "k": [3] * n_layers}


def random_config(n_layers, rng=random):
    return {"res": rng.choice(RESOLUTIONS), "depth": {s: rng.choice(DEPTHS) for s in DEPTH_STAGES},
            "k": [rng.choice(KERNELS) for _ in range(n_layers)]}


def config_id(cfg):
    """Kısa, okunur kimlik: r416_d33_k + 30 hane (3/5/7)."""
    d = "".join(str(cfg["depth"][s]) for s in DEPTH_STAGES)
    return f"r{cfg['res']}_d{d}_k{''.join(str(k) for k in cfg['k'])}"


def parse_config_id(cid):
    r, d, k = cid.split("_")
    return {"res": int(r[1:]), "depth": dict(zip(DEPTH_STAGES, (int(c) for c in d[1:]))),
            "k": [int(c) for c in k[1:]]}


def inactive_layers(names, cfg):
    """Seçilen derinlikte koşmayan Bottleneck'lerin içindeki DW katmanları (kernel'leri anlamsız)."""
    out = set()
    for stage in DEPTH_STAGES:
        prefix = f"backbone.backbone.{stage}.1.m."
        for n in names:
            if n.startswith(prefix) and int(n[len(prefix):].split(".")[0]) >= cfg["depth"][stage]:
                out.add(n)
    return out


def canonical(cfg, names):
    """Koşmayan katmanların kernel'ini 3'e sabitler: aynı ağı veren yapılandırmalar tek kimlik alır."""
    off = inactive_layers(names, cfg)
    return {**cfg, "k": [3 if n in off else k for n, k in zip(names, cfg["k"])]}


def set_active(model, names, cfg):
    assert len(cfg["k"]) == len(names), (len(cfg["k"]), len(names))
    for n, k in zip(names, cfg["k"]):
        assert k in KERNELS, k
        model.get_submodule(n).active_k = int(k)
    for stage in DEPTH_STAGES:
        d = int(cfg["depth"][stage])
        assert d in DEPTHS, d
        csp_of(model, stage).m.active_n = d


# ---- alt ağı süper ağdan çıkarma (statik model: ince ayar, ONNX) ---------------------------------

def extract_subnet(supernet, names, cfg):
    """Süper ağdan seçili alt ağın statik kopyası: DW katmanları k×k nn.Conv2d, koşmayan bloklar silinir.

    Çıkan model standart YOLOX nesnesidir (DynamicDWConv/ElasticSequential içermez); budama ve ONNX
    dışa aktarımı için doğrudan kullanılabilir. Çıktı eşdeğerliği tests ile doğrulanır.
    """
    import copy
    sub = copy.deepcopy(supernet)
    set_active(sub, names, cfg)
    for n in names:
        m = sub.get_submodule(n)
        k = m.active_k
        conv = nn.Conv2d(m.in_channels, m.out_channels, k, stride=m.stride, padding=k // 2,
                         groups=m.groups, bias=False).to(m.weight.device, m.weight.dtype)
        with torch.no_grad():
            conv.weight.copy_(m.active_weight(k))
        _set_child(sub, n, conv)
    for stage in DEPTH_STAGES:
        csp = csp_of(sub, stage)
        csp.m = nn.Sequential(*list(csp.m)[:csp.m.active_n])
    return sub


def subnet_skeleton(base_model, cfg_k_by_name, depth):
    """Standart Nano iskeletine statik alt ağ yapısını uygular (ağırlık yüklemeden önce; bkz. SubnetExp)."""
    for stage in DEPTH_STAGES:
        csp = csp_of(base_model, stage)
        csp.m = nn.Sequential(*list(csp.m)[:depth[stage]])
    for n, k in cfg_k_by_name.items():
        try:
            old = base_model.get_submodule(n)
        except AttributeError:
            continue   # silinen bloğun katmanı
        if k != 3:
            _set_child(base_model, n, nn.Conv2d(old.in_channels, old.out_channels, k,
                                                stride=old.stride, padding=k // 2,
                                                groups=old.groups, bias=False))
    return base_model


# ---- exp ---------------------------------------------------------------------------------------

FORMAT = "nas_supernet_full_v1"
# Devamda değişmemesi gereken ayarlar (max_epoch hariç: uzatma aşaması onu bilerek büyütür).
FINGERPRINT_ATTRS = ("nas_seed", "nas_seed_data", "nas_sandwich_random", "nas_phase1_epochs",
                     "nas_ext_peak_ratio", "basic_lr_per_img", "warmup_epochs", "warmup_lr", "min_lr_ratio",
                     "no_aug_epochs", "scheduler", "weight_decay", "momentum", "train_ann", "input_size",
                     "random_size", "mosaic_prob", "mosaic_scale", "enable_mixup", "hsv_prob", "flip_prob",
                     "degrees", "translate", "shear")


class Exp(_Base):
    def __init__(self):
        super().__init__()
        self.nas_pretrained = ""          # önceden eğitilmiş yolox_nano.pth (strict yüklenir)
        self.nas_ckpt_dir = ""            # doğrulanmış checkpoint deposu (Drive); boşsa devam yok
        self.nas_ckpt_every_iters = 300   # epoch ortası kayıt aralığı (iterasyon)
        self.nas_ckpt_keep = 3            # sabitlenmemiş son N doğrulanmış kayıt
        self.nas_snapshot_interval = 10   # her N epoch'un sonundaki kayıt sabitlenir (panel değerlendirmesi)
        self.nas_log_dir = ""             # train_log.txt'nin Drive aynası
        self.nas_sandwich_random = 2      # en büyük + en küçük + N rastgele alt ağ / iterasyon (BigNAS: N=2)
        self.nas_seed = 0                 # alt ağ örnekleme tohumu
        self.nas_seed_data = 0            # veri sırası + artırma + çok ölçek tohumu
        self.nas_pause_epoch = 0          # >0: bu epoch'un doğrulanmış kaydından sonra süreç durur (erken kontrol)
        self.nas_trace_path = ""          # kesinti testi: iterasyon başına jsonl iz (batch hash'i dahil)
        self.nas_trace_iters = 0          # izlenecek iterasyon sayısı (0: hepsi)
        self.nas_deterministic = False    # kesinti testi: cudnn deterministic (benchmark kapalı)
        # Uzatma aşaması (Karar 6): ilk aşama nas_phase1_epochs; max_epoch > nas_phase1_epochs ise kalan epoch'lar
        # optimizer korunarak yeni cosine ile koşar: tepe lr = nas_ext_peak_ratio × ilk aşama tepe lr, ısınma yok,
        # son no_aug_epochs + 1 epoch mosaic'siz ve lr min'de (yoloxwarmcos ile aynı yapı).
        self.nas_phase1_epochs = 60
        self.nas_ext_peak_ratio = 0.5
        # Veri: tam train2017 (önceden eğitilmiş Nano ile aynı veri); arama uygunluğu val2017-A'da, BN
        # kalibrasyonu train2017 bn2k'da (scripts/make_splits.py).
        self.data_dir = "/content/datasets/COCO"
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"
        self.max_epoch = 60
        self.no_aug_epochs = 1
        self.basic_lr_per_img = 0.001 / 64.0
        self.warmup_epochs = 1
        self.eval_interval = 10_000
        self.data_num_workers = 8
        self.print_interval = 10
        self.save_history_ckpt = False
        self.exp_name = "nano_supernet"
        self.dw_names = []
        self.nas_summary = {}
        self._trainer_progress = 0

    def fingerprint(self):
        import hashlib
        vals = {k: getattr(self, k, None) for k in FINGERPRINT_ATTRS}
        return hashlib.sha256(json.dumps(vals, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def get_model(self, sublinear=False):
        if "model" in self.__dict__:
            return self.model
        model = super().get_model()
        if self.nas_pretrained:
            state = torch.load(self.nas_pretrained, map_location="cpu")["model"]
            model.load_state_dict(state, strict=True)
            self.nas_summary["pretrained_loaded"] = True
        self.dw_names = to_supernet(model)
        self.nas_summary["dw_layers"] = len(self.dw_names)
        self.nas_summary["space"] = {"kernels": KERNELS, "depth_stages": DEPTH_STAGES,
                                     "depths": DEPTHS, "resolutions": RESOLUTIONS}
        set_active(model, self.dw_names, max_config(len(self.dw_names)))
        self.model = model
        return model

    def sandwich(self, step):
        """step'e bağlı, tekrar üretilebilir alt ağ listesi (kesintiden devamda aynı dizi)."""
        n = len(self.dw_names)
        rng = random.Random(self.nas_seed * 1_000_003 + step)
        cfgs = [max_config(n), min_config(n)]
        cfgs += [random_config(n, rng) for _ in range(int(self.nas_sandwich_random))]
        return cfgs

    # -- veri: kesintiden birebir devam eden akış (scripts/nas_data.py) --
    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=None):
        from yolox.data import DataLoader, MosaicDetection, TrainTransform
        import nas_data as nd
        assert not is_distributed, "süper ağ tek GPU için kuruldu"
        if self.dataset is None:
            self.dataset = self.get_dataset(cache=False, cache_type=cache_img)
        ds = nd.make_seeded_mosaic(MosaicDetection)(
            dataset=self.dataset, mosaic=True, img_size=self.input_size,
            preproc=TrainTransform(max_labels=120, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob),
            degrees=self.degrees, translate=self.translate, mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale, shear=self.shear, enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob, mixup_prob=self.mixup_prob)
        ds.run_seed = int(self.nas_seed_data)
        se, si = getattr(self, "_resume_pos", (0, 0))
        bs = nd.ResumableBatchSampler(len(ds), batch_size, self.nas_seed_data, self.max_epoch,
                                      self.no_aug_epochs, start_epoch=se, start_iter=si)
        return DataLoader(ds, batch_sampler=bs, num_workers=self.data_num_workers, pin_memory=True)

    def random_resize(self, data_loader, epoch, rank, is_distributed):
        import nas_data as nd
        return nd.resize_size(self.nas_seed_data, self._trainer_progress, self.random_size,
                              self.input_size[1] / self.input_size[0])

    def get_trainer(self, args):
        import math
        import time

        import numpy as np
        from loguru import logger
        from yolox.core import Trainer

        import nas_data as nd
        from ckpt_store import CheckpointStore

        exp = self

        class SupernetTrainer(Trainer):
            def __init__(self, exp_, args_):
                super().__init__(exp_, args_)
                self.store = (CheckpointStore(exp.nas_ckpt_dir, os.path.join(self.file_name, "ck_local"),
                                              keep=exp.nas_ckpt_keep, log=logger.info)
                              if exp.nas_ckpt_dir else None)
                self._state, self._rec = self.store.latest() if self.store else (None, None)
                self.start_iter = 0

            # -- devam --
            def resume_train(self, model):
                st = self._state
                self.start_epoch = 0
                if st is None:
                    return model
                assert st["format"] == FORMAT, st.get("format")
                assert st["fingerprint"] == exp.fingerprint(), "tarif değişmiş: kayıt bu koşuya ait değil"
                assert st["batch_size"] == self.args.batch_size
                model.load_state_dict(st["model_raw"], strict=True)
                self.optimizer.load_state_dict(st["optimizer"])
                self.start_epoch, self.start_iter = int(st["epoch"]), int(st["iter"])
                exp._resume_pos = (self.start_epoch, self.start_iter)
                logger.info(f"devam: {self._rec['file']} (epoch {self.start_epoch}, iter {self.start_iter}, "
                            f"ilerleme {st['progress']})")
                return model

            def before_train(self):
                if exp.nas_deterministic:
                    torch.backends.cudnn.benchmark = False
                    torch.backends.cudnn.deterministic = True
                super().before_train()
                # ilk aşamanın lr planı (uzatma koşusunda max_epoch büyüse de değişmez)
                saved = exp.max_epoch
                exp.max_epoch = exp.nas_phase1_epochs
                self.lr_p1 = exp.get_lr_scheduler(exp.basic_lr_per_img * self.args.batch_size, self.max_iter)
                exp.max_epoch = saved
                st = self._state
                if st is not None:
                    self.ema_model.ema.load_state_dict(st["model"], strict=True)
                    self.ema_model.updates = int(st["ema_updates"])
                    self.scaler.load_state_dict(st["scaler"])
                    random.setstate(st["rng"]["python"])
                    np.random.set_state(st["rng"]["numpy"])
                    torch.set_rng_state(st["rng"]["torch"])
                    torch.cuda.set_rng_state_all(st["rng"]["cuda"])
                    p = self.start_epoch * self.max_iter + self.start_iter
                    self.input_size = nd.size_at(exp.nas_seed_data, p, exp.random_size, exp.input_size)
                    assert st["iters_per_epoch"] == self.max_iter
                    self._state = None   # bellek
                logger.info(f"Süper ağ: {exp.nas_summary}")
                logger.info(f"DW katmanları: {json.dumps(exp.dw_names)}")
                logger.info(f"parmak izi {exp.fingerprint()} · başlangıç epoch {self.start_epoch} iter "
                            f"{self.start_iter}")

            def lr_at(self, p):
                p1 = exp.nas_phase1_epochs * self.max_iter
                if p < p1:
                    return self.lr_p1.update_lr(p)
                base = exp.basic_lr_per_img * self.args.batch_size
                peak, low = exp.nas_ext_peak_ratio * base, base * exp.min_lr_ratio
                ext = (self.max_epoch - exp.nas_phase1_epochs - exp.no_aug_epochs) * self.max_iter
                t = min(1.0, (p - p1) / max(1, ext))
                return low + 0.5 * (peak - low) * (1 + math.cos(math.pi * t))

            def train_in_iter(self):
                s = self.start_iter if self.epoch == self.start_epoch else 0
                for self.iter in range(s, self.max_iter):
                    self.before_iter()
                    self.train_one_iter()
                    self.after_iter()

            def train_one_iter(self):
                t0 = time.time()
                p = self.progress_in_iter
                exp._trainer_progress = p
                inps, targets = self.prefetcher.next()
                inps = inps.to(self.data_type)
                targets = targets.to(self.data_type)
                targets.requires_grad = False
                digest = (nd.batch_digest(inps, targets)
                          if exp.nas_trace_path and (not exp.nas_trace_iters or p < exp.nas_trace_iters) else None)
                inps, targets = self.exp.preprocess(inps, targets, self.input_size)
                t_data = time.time()

                cfgs = exp.sandwich(p)
                self.optimizer.zero_grad()
                logs, losses = {}, []
                for i, cfg in enumerate(cfgs):
                    set_active(self.model, exp.dw_names, cfg)
                    with torch.cuda.amp.autocast(enabled=self.amp_training):
                        outputs = self.model(inps, targets)
                    self.scaler.scale(outputs["total_loss"] / len(cfgs)).backward()
                    tag = ("max", "min")[i] if i < 2 else f"rnd{i - 1}"
                    logs[f"loss_{tag}"] = outputs["total_loss"].detach()
                    losses.append(float(outputs["total_loss"].detach()))
                    if i == 0:
                        logs.update({k: v.detach() if torch.is_tensor(v) else v for k, v in outputs.items()})
                self.scaler.step(self.optimizer)
                self.scaler.update()
                set_active(self.model, exp.dw_names, cfgs[0])
                if self.use_model_ema:
                    self.ema_model.update(self.model)
                lr = self.lr_at(p + 1)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr
                self.meter.update(iter_time=time.time() - t0, data_time=t_data - t0, lr=lr, **logs)
                if digest is not None:
                    rec = {"progress": p, "epoch": self.epoch, "iter": self.iter, "input_size": list(self.input_size),
                           "lr_next": lr, "cfgs": [config_id(c) for c in cfgs], "batch_sha256": digest,
                           "losses": losses, "scale": float(self.scaler.get_scale())}
                    with open(exp.nas_trace_path, "a") as f:
                        f.write(json.dumps(rec) + "\n")

            # -- kayıt --
            def save_full(self, epoch, it, pinned=False, meta=None):
                if self.store is None or self.rank != 0:
                    return
                progress = epoch * self.max_iter + it
                st = {"format": FORMAT, "fingerprint": exp.fingerprint(), "batch_size": self.args.batch_size,
                      "epoch": epoch, "iter": it, "progress": progress, "iters_per_epoch": self.max_iter,
                      "start_epoch": epoch, "max_epoch": self.max_epoch,
                      "model_raw": self.model.state_dict(), "model": self.ema_model.ema.state_dict(),
                      "ema_updates": self.ema_model.updates, "optimizer": self.optimizer.state_dict(),
                      "scaler": self.scaler.state_dict(),
                      "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                              "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}}
                t = time.time()
                path = self.store.save(st, progress, meta={**(meta or {}), "epoch": epoch, "iter": it},
                                       pinned=pinned)
                logger.info(f"doğrulanmış kayıt: {os.path.basename(path)} ({time.time() - t:.1f} sn)")
                if exp.nas_log_dir:
                    from nano_macro import mirror_atomic
                    mirror_atomic(os.path.join(self.file_name, "train_log.txt"), exp.nas_log_dir)

            def after_iter(self):
                super().after_iter()
                every = int(exp.nas_ckpt_every_iters or 0)
                if every and (self.iter + 1) % every == 0 and self.iter + 1 < self.max_iter:
                    self.save_full(self.epoch, self.iter + 1)

            def evaluate_and_save_model(self):
                pass   # eğitim içi değerlendirme yok; panel ayrı ölçülür

            def save_ckpt(self, ckpt_name, update_best_ckpt=False, ap=None):
                pass   # YOLOX'un EMA-only kayıtları yerine save_full (tam durum, doğrulanmış)

            def after_epoch(self):
                ep = self.epoch + 1
                interval = int(exp.nas_snapshot_interval or 0)
                pinned = (interval and ep % interval == 0) or ep == self.max_epoch \
                    or ep == exp.nas_phase1_epochs or ep == int(exp.nas_pause_epoch or 0)
                self.save_full(ep, 0, pinned=bool(pinned), meta={"epoch_end": ep})
                if int(exp.nas_pause_epoch or 0) == ep and ep < self.max_epoch:
                    logger.info(f"Süper ağ duraklatıldı: epoch {ep} (erken kontrol; yeniden çalıştırınca devam)")
                    raise SystemExit(0)

        return SupernetTrainer(self, args)
