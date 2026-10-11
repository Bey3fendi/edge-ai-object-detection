"""NAS 2. tur (nas_r2): tam COCO train2017'de uzun ince ayar için nano_macro türevi.

2. turun sorusu (Furkan, 2026-10-12): ReLU'nun 1. turdaki kaybı (416'da −5,5 puan, 10k alt küme, 10 epoch)
kalıcı mı, yoksa önceden eğitilmiş SiLU ağırlıklarının yeni aktivasyona uyum süresi mi? İki kol birebir aynı
tarifle eğitilir: `relu_dfull_k3` ve kontrol `silu_dfull_k3`. Karar ölçütü önceden sabit: son epoch'ta
416'da mAP(ReLU) − mAP(SiLU) ≥ −1,0 (Karar 2/5 NAS bütçesi) → ReLU geçer.

nano_macro'dan farklar (mimari cerrahisi, nas_* seçenekleri ve ağırlık yükleme aynı):
    train_ann            instances_train2017.json (tam train2017, 118 287 görsel)
    max_epoch            12 (son 2 epoch mosaic'siz: no_aug_epochs = 1, 1. turla aynı kural)
    nas_snapshot_interval  her N epoch sonunda latest_ckpt.pth Drive'a epoch<N>_ckpt.pth olarak kopyalanır
                           (öğrenme eğrisi: ara epoch'lar ayrı fp32 eval.py ile değerlendirilir)

lr 0,001/64, ısınma 1 epoch, batch 64, fp16, çok ölçekli eğitim 1. turla aynıdır (nas_r1b'de kontrolü
bozmadığı gösterildi). nano_macro.py değiştirilmez: hash'i nas_r1b manifestlerinde kayıtlıdır.
"""

import os

from nano_macro import Exp as _Macro
from nano_macro import mirror_atomic


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


class Exp(_Macro):
    def __init__(self):
        super().__init__()
        self.train_ann = "instances_train2017.json"
        self.max_epoch = 12
        self.no_aug_epochs = 1
        self.nas_snapshot_interval = 2
        self.exp_name = "nano_macro_long"

    def get_trainer(self, args):
        # nano_macro'daki NasTrainer'ın aynısı + ara epoch anlık görüntüsü. Trainer iki kez kurulmaz
        # (YOLOX Trainer.__init__ logger kurar; ikinci kurulum train_log.txt'ye satırları çift yazar).
        from yolox.core import Trainer
        from loguru import logger

        exp = self

        class NasLongTrainer(Trainer):
            def before_train(self):
                super().before_train()
                logger.info(f"NAS adayı {exp.arch_id}: {exp.nas_summary}")

            def evaluate_and_save_model(self):
                # Eğitim içi (fp16) değerlendirme atlanır; ayrı fp32 eval.py ile ölçülür.
                self.save_ckpt("last_epoch")

            def after_epoch(self):
                super().after_epoch()
                if not exp.nas_mirror_dir or self.rank != 0:
                    return
                ep = self.epoch + 1
                interval = int(exp.nas_snapshot_interval or 0)
                if interval and ep % interval == 0:
                    os.makedirs(exp.nas_mirror_dir, exist_ok=True)
                    copy_atomic(os.path.join(self.file_name, "latest_ckpt.pth"),
                                os.path.join(exp.nas_mirror_dir, f"epoch{ep}_ckpt.pth"))
                    logger.info(f"NAS anlık görüntü: epoch{ep}_ckpt.pth")
                for name in ("latest_ckpt.pth", "last_epoch_ckpt.pth", "train_log.txt"):
                    mirror_atomic(os.path.join(self.file_name, name), exp.nas_mirror_dir)

        return NasLongTrainer(self, args)
