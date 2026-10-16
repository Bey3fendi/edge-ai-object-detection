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

Eğitim (sandwich kuralı, BigNAS/AttentiveNAS): her iterasyonda en büyük, en küçük ve `nas_sandwich_random`
rastgele alt ağ aynı mini-batch üzerinde koşar; kayıplar ortalanıp tek optimizer adımı atılır (ortalama:
lr 0,001/64 tek ağ için doğrulandı, toplam gradyan adımı büyütürdü). Bilgi damıtma yok (tespit kaybında
iki ağ çıktısını eşleştirmek ayrı bir tasarım; bilinçli olarak dışarıda).

YOLOX submodule'üne dokunulmaz; model cerrahisi ve trainer kancası bu dosyadadır.
"""

import json
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolox.exp.default.yolox_nano import Exp as _Base

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

def copy_atomic(src, dst):
    tmp = dst + ".tmp"
    with open(src, "rb") as fi, open(tmp, "wb") as fo:
        while True:
            chunk = fi.read(1 << 20)
            if not chunk:
                break
            fo.write(chunk)
        fo.flush()
        os.fsync(fo.fileno())
    os.replace(tmp, dst)


class Exp(_Base):
    def __init__(self):
        super().__init__()
        self.nas_pretrained = ""          # önceden eğitilmiş yolox_nano.pth (strict yüklenir)
        self.nas_mirror_dir = ""          # Drive aynası (her epoch atomik kopya)
        self.nas_snapshot_interval = 4    # her N epoch'ta epoch<N>_ckpt.pth anlık görüntüsü
        self.nas_sandwich_random = 1      # en büyük + en küçük + N rastgele alt ağ / iterasyon
        self.nas_seed = 0                 # alt ağ örnekleme tohumu (iterasyon numarasıyla türetilir)
        self.nas_pause_epoch = 0          # >0: bu epoch bitince checkpoint'ler aynalanıp süreç durur
                                          # (erken kontrol için; sonra --resume ile devam, lr planı aynı)
        # Veri: train2017'den arama için ayrılan 5000 görsel çıkarılmış eğitim kümesi
        # (scripts/make_holdout.py). Arama uygunluk değeri val2017'de ölçülmez (sızıntı olurdu).
        self.data_dir = "/content/datasets/COCO"
        self.train_ann = "instances_train2017_minus_nas5k.json"
        self.val_ann = "instances_val2017.json"
        # Tarif: nas_r2 (tam COCO, lr 0,001/64, warmup 1, no_aug son 1 epoch; kontrolü bozmadı) +
        # daha uzun süre. Epoch sayısı pilot ölçümünden sonra Furkan'la sabitlenir (manifestte).
        self.max_epoch = 20
        self.no_aug_epochs = 1
        self.basic_lr_per_img = 0.001 / 64.0
        self.warmup_epochs = 1
        self.eval_interval = 10_000       # eğitim içi değerlendirme yok; alt ağlar ayrı ölçülür
        self.data_num_workers = 8
        self.print_interval = 10
        self.save_history_ckpt = False
        self.exp_name = "nano_supernet"
        self.dw_names = []
        self.nas_summary = {}

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

    def get_trainer(self, args):
        import time

        from loguru import logger
        from yolox.core import Trainer

        exp = self

        class SupernetTrainer(Trainer):
            def before_train(self):
                super().before_train()
                logger.info(f"Süper ağ: {exp.nas_summary}")
                logger.info(f"DW katmanları: {json.dumps(exp.dw_names)}")

            def train_one_iter(self):
                # YOLOX Trainer.train_one_iter + sandwich: aynı batch, birden çok alt ağ, tek adım.
                t0 = time.time()
                inps, targets = self.prefetcher.next()
                inps = inps.to(self.data_type)
                targets = targets.to(self.data_type)
                targets.requires_grad = False
                inps, targets = self.exp.preprocess(inps, targets, self.input_size)
                t_data = time.time()

                cfgs = exp.sandwich(self.progress_in_iter)
                self.optimizer.zero_grad()
                logs = {}
                for i, cfg in enumerate(cfgs):
                    set_active(self.model, exp.dw_names, cfg)
                    with torch.cuda.amp.autocast(enabled=self.amp_training):
                        outputs = self.model(inps, targets)
                    self.scaler.scale(outputs["total_loss"] / len(cfgs)).backward()
                    tag = ("max", "min")[i] if i < 2 else f"rnd{i - 1}"
                    logs[f"loss_{tag}"] = outputs["total_loss"].detach()
                    if i == 0:   # meter'de YOLOX'un alanları en büyük alt ağdan
                        logs.update({k: v.detach() if torch.is_tensor(v) else v
                                     for k, v in outputs.items()})
                self.scaler.step(self.optimizer)
                self.scaler.update()
                set_active(self.model, exp.dw_names, cfgs[0])

                if self.use_model_ema:
                    self.ema_model.update(self.model)

                lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr
                self.meter.update(iter_time=time.time() - t0, data_time=t_data - t0, lr=lr, **logs)

            def evaluate_and_save_model(self):
                self.save_ckpt("last_epoch")

            def after_epoch(self):
                super().after_epoch()
                if not exp.nas_mirror_dir or self.rank != 0:
                    return
                from nano_macro import mirror_atomic
                ep = self.epoch + 1
                interval = int(exp.nas_snapshot_interval or 0)
                os.makedirs(exp.nas_mirror_dir, exist_ok=True)
                if interval and ep % interval == 0:
                    copy_atomic(os.path.join(self.file_name, "latest_ckpt.pth"),
                                os.path.join(exp.nas_mirror_dir, f"epoch{ep}_ckpt.pth"))
                    logger.info(f"Süper ağ anlık görüntü: epoch{ep}_ckpt.pth")
                for name in ("latest_ckpt.pth", "last_epoch_ckpt.pth", "train_log.txt"):
                    mirror_atomic(os.path.join(self.file_name, name), exp.nas_mirror_dir)
                if int(exp.nas_pause_epoch or 0) == ep and ep < self.max_epoch:
                    logger.info(f"Süper ağ duraklatıldı: epoch {ep} (erken kontrol; --resume ile devam)")
                    raise SystemExit(0)   # Exception değil: Trainer.train bunu hata olarak loglamaz

        return SupernetTrainer(self, args)
