"""Makro-mimari araması (N1/N2) için parametrik YOLOX-Nano exp'i.

Aday ayarları komut satırından `exp.merge` ile verilir (train.py / eval.py sonundaki opts):
    nas_act    silu | relu | hardswish   aktivasyon (tüm modeldeki SiLU modülleri değiştirilir)
    nas_depth  full | reduced            reduced: backbone dark3 ve dark4 CSP blokları 3 → 2
                                          (son Bottleneck çıkarılır; n=1 olan bloklara dokunulmaz)
    nas_dw_k   3 | 5                     tüm depthwise konvolüsyonların kernel boyutu; 5'te önceden
                                          eğitilmiş 3×3 ağırlık merkeze konur, kenarlar sıfır
    nas_pretrained  önceden eğitilmiş Nano .pth yolu; verilirse ağırlıklar dönüşümden ÖNCE
                    standart Nano'ya yüklenir (train.py'ye -c verilmez: YOLOX load_ckpt şekli
                    tutmayan ağırlığı sessizce atlar, kernel 5'te 3×3 ağırlıklar kaybolurdu)
    nas_mirror_dir  verilirse her epoch sonunda checkpoint'ler ve train_log buraya atomik
                    olarak kopyalanır (Drive; Colab kesintisine karşı)

Çözünürlük (320/416/512) eğitimi değiştirmez (Seçenek A, Furkan 2026-10-05): tüm adaylar Nano'nun
çok ölçekli tarifiyle (320–640) eğitilir, çözünürlük değerlendirmede `eval.py --tsize` ile seçilir.

YOLOX submodule'üne dokunulmaz; model cerrahisi ve trainer kancası bu dosyadadır.
"""

import os
import shutil

import torch
import torch.nn as nn

from yolox.exp.default.yolox_nano import Exp as _Base

ACTS = ("silu", "relu", "hardswish")
DEPTHS = ("full", "reduced")
DW_KS = (3, 5)
REDUCED_STAGES = ("dark3", "dark4")


def _set_child(root, dotted_name, new_module):
    parent_name, _, child = dotted_name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child, new_module)


def swap_activation(model, act):
    """nn.SiLU modüllerini değiştirir; döndürülen değer değiştirilen modül sayısıdır."""
    if act == "silu":
        return 0
    make = {"relu": lambda: nn.ReLU(inplace=True), "hardswish": lambda: nn.Hardswish(inplace=True)}[act]
    names = [n for n, m in model.named_modules() if isinstance(m, nn.SiLU)]
    for n in names:
        _set_child(model, n, make())
    return len(names)


def reduce_depth(model):
    """dark3/dark4 CSPLayer'larından son Bottleneck'i çıkarır; döndürülen değer yeni blok sayılarıdır."""
    out = {}
    for stage in REDUCED_STAGES:
        csp = getattr(model.backbone.backbone, stage)[1]
        blocks = list(csp.m)
        assert len(blocks) > 1, f"{stage} zaten tek bloklu"
        csp.m = nn.Sequential(*blocks[:-1])
        out[stage] = len(csp.m)
    return out


def widen_depthwise(model, k):
    """3×3 depthwise Conv2d'leri k×k yapar; eski ağırlık merkeze, kenarlar sıfır (çıktı aynı kalır)."""
    if k == 3:
        return 0
    pad = (k - 3) // 2
    names = [
        n for n, m in model.named_modules()
        if isinstance(m, nn.Conv2d) and m.groups > 1 and m.groups == m.in_channels
        and m.kernel_size == (3, 3)
    ]
    for n in names:
        old = model.get_submodule(n)
        new = nn.Conv2d(old.in_channels, old.out_channels, k, stride=old.stride,
                        padding=old.padding[0] + pad, groups=old.groups, bias=old.bias is not None)
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:, :, pad:pad + 3, pad:pad + 3].copy_(old.weight)
            if old.bias is not None:
                new.bias.copy_(old.bias)
        _set_child(model, n, new.to(old.weight.device))
    return len(names)


def mirror_atomic(src, dst_dir):
    """src'yi dst_dir'e önce geçici adla kopyalar, sonra tek hamlede yerine koyar."""
    if not os.path.exists(src):
        return
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, os.path.basename(src))
    tmp = dst + ".tmp"
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


class Exp(_Base):
    def __init__(self):
        super().__init__()
        # Aday (opts ile değişir)
        self.nas_act = "silu"
        self.nas_depth = "full"
        self.nas_dw_k = 3
        self.nas_pretrained = ""
        self.nas_mirror_dir = ""
        # Sabit arama tarifi (Karar 5, 2026-10-04): 10 epoch, son 2 epoch mosaic'siz
        # (YOLOX no_aug_epochs + 1 epoch'u mosaic'siz koşar). Batch 64 ve fp16 komut satırından.
        self.data_dir = "/content/datasets/COCO"
        self.train_ann = "instances_train2017_10k.json"
        self.val_ann = "instances_val2017.json"
        self.max_epoch = 10
        self.no_aug_epochs = 1
        # Öğrenme hızı: YOLOX varsayılanı (0.01/64 görsel başına, min_lr_ratio 0.05, cos). Isınma 5 yerine
        # 1 epoch: önceden eğitilmiş ağırlıktan başlanıyor, YOLOX'un COCO→VOC fine-tune örneği de 1 kullanır
        # (exps/example/yolox_voc/yolox_voc_s.py); 5 epoch 10 epoch'luk koşunun yarısı olurdu.
        self.warmup_epochs = 1
        self.eval_interval = 10_000   # eğitim içi değerlendirme yok; ayrı fp32 eval.py (bkz. trainer)
        self.data_num_workers = 8
        self.print_interval = 10
        self.save_history_ckpt = False
        self.exp_name = "nano_macro"
        self.nas_summary = {}

    @property
    def arch_id(self):
        return f"{self.nas_act}_d{self.nas_depth}_k{self.nas_dw_k}"

    def get_model(self, sublinear=False):
        if "model" in self.__dict__:
            return self.model
        assert self.nas_act in ACTS and self.nas_depth in DEPTHS and int(self.nas_dw_k) in DW_KS, \
            (self.nas_act, self.nas_depth, self.nas_dw_k)
        model = super().get_model()
        if self.nas_pretrained:
            state = torch.load(self.nas_pretrained, map_location="cpu")["model"]
            model.load_state_dict(state, strict=True)   # standart Nano: tüm anahtarlar birebir
            self.nas_summary["pretrained_loaded"] = True
        self.nas_summary["act_swapped"] = swap_activation(model, self.nas_act)
        self.nas_summary["csp_blocks"] = reduce_depth(model) if self.nas_depth == "reduced" else {}
        self.nas_summary["dw_widened"] = widen_depthwise(model, int(self.nas_dw_k))
        self.model = model
        return model

    def get_trainer(self, args):
        from yolox.core import Trainer
        from loguru import logger

        exp = self

        class NasTrainer(Trainer):
            def before_train(self):
                super().before_train()
                logger.info(f"NAS adayı {exp.arch_id}: {exp.nas_summary}")

            def evaluate_and_save_model(self):
                # Eğitim içi (fp16) değerlendirme atlanır; aday ayrı fp32 eval.py ile ölçülür.
                self.save_ckpt("last_epoch")

            def after_epoch(self):
                super().after_epoch()
                if exp.nas_mirror_dir and self.rank == 0:
                    for name in ("latest_ckpt.pth", "last_epoch_ckpt.pth", "train_log.txt"):
                        mirror_atomic(os.path.join(self.file_name, name), exp.nas_mirror_dir)

        return NasTrainer(self, args)
