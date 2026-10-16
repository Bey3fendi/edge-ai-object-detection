"""Statik alt ağın SIFIRDAN eğitimi (Karar 6): 03c sıralama referansı (60 ep) ve 03d final + Nano (300 ep × 3 tohum).

Seçenekler (`exp.merge`):
    nas_arch        alt ağ kimliği (supernet.config_id: r416_d33_k<30 hane>); Nano = r416_d33_k3…3
    nas_seed_init   ağırlık başlatma tohumu (torch.manual_seed; 03d'de tekrar tohumu)
    nas_seed_data   veri sırası + artırma + çok ölçek tohumu (scripts/nas_data.py)
    nas_ckpt_dir    doğrulanmış checkpoint deposu (Drive); her başlatmada en yeni doğrulanmış kayıttan devam
    nas_ckpt_every_iters, nas_ckpt_keep, nas_log_dir, nas_trace_path, nas_trace_iters, nas_deterministic
    nas_snapshot_epochs  virgüllü epoch listesi; bu epoch sonlarının kaydı sabitlenir (öğrenme eğrisi)

Tarif YOLOX-Nano'nun sıfırdan tarifidir (yolox_base: lr 0,01/64 görsel, yoloxwarmcos, min_lr_ratio 0,05,
SGD momentum 0,9 nesterov, weight decay 5e-4, EMA 0,9998, mosaic 0,5 (nano), mixup kapalı (nano), mosaic
ölçeği 0,5–1,5, çok ölçek 320–640); epoch / ısınma / no_aug notebook'ta manifestle sabitlenir (03c: 60 ep).
Ön eğitimli ağırlık YOK: YOLOX'un kendi başlatması (init_yolo + initialize_biases) `nas_seed_init` ile.

Kesintiye dayanıklılık supernet.py ile aynı ilkeler (tam durum, doğrulanmış depo, tohumlu veri, epoch ortası
devam); supernet.py eğitim sürerken değiştirilmesin diye mantık burada ayrı tutulur (parmak izi o dosyanın
sha256'sına bağlı). Kesinti testi: scripts/check_trainer_cpu.py --exp scratch.
"""

import json
import os
import random
import sys

import torch

from yolox.exp.default.yolox_nano import Exp as _Nano

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, "..", "..", "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from supernet import DEPTH_STAGES, dw_layer_names, parse_config_id, subnet_skeleton  # noqa: E402

FORMAT = "nas_scratch_full_v1"
FINGERPRINT_ATTRS = ("nas_arch", "nas_seed_init", "nas_seed_data", "max_epoch", "basic_lr_per_img",
                     "warmup_epochs", "warmup_lr", "min_lr_ratio", "no_aug_epochs", "scheduler", "weight_decay",
                     "momentum", "train_ann", "input_size", "random_size", "mosaic_prob", "mosaic_scale",
                     "enable_mixup", "mixup_prob", "hsv_prob", "flip_prob", "degrees", "translate", "shear")


class Exp(_Nano):
    def __init__(self):
        super().__init__()
        self.nas_arch = ""
        self.nas_seed_init = 0
        self.nas_seed_data = 0
        self.nas_ckpt_dir = ""
        self.nas_ckpt_every_iters = 300
        self.nas_ckpt_keep = 3
        self.nas_log_dir = ""
        self.nas_trace_path = ""
        self.nas_trace_iters = 0
        self.nas_deterministic = False
        self.nas_snapshot_epochs = ""
        self.data_dir = "/content/datasets/COCO"
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017_A.json"      # tüm yöntem kararları val2017-A'da (Karar 6)
        self.eval_interval = 10_000                    # eğitim içi değerlendirme yok; ayrı eval
        self.data_num_workers = 8
        self.print_interval = 10
        self.save_history_ckpt = False
        self.exp_name = "nano_scratch_subnet"
        self.nas_summary = {}
        self._trainer_progress = 0

    def fingerprint(self):
        import hashlib
        vals = {k: getattr(self, k, None) for k in FINGERPRINT_ATTRS}
        return hashlib.sha256(json.dumps(vals, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def get_model(self, sublinear=False):
        if "model" in self.__dict__:
            return self.model
        assert self.nas_arch, "nas_arch gerekli"
        cfg = parse_config_id(self.nas_arch)
        torch.manual_seed(int(self.nas_seed_init))   # başlatma tekrar üretilebilir (tohum = tekrar kimliği)
        model = super().get_model()                  # YOLOX init_yolo + initialize_biases
        names = dw_layer_names(model)
        assert len(names) == len(cfg["k"]), (len(names), len(cfg["k"]))
        subnet_skeleton(model, dict(zip(names, cfg["k"])), cfg["depth"])
        # iskelet yeni k×k konvolüsyonları PyTorch varsayılanıyla başlatır; YOLOX başlatması BN/bias içindir,
        # konvolüsyon ağırlıkları YOLOX'ta da PyTorch varsayılanıdır → tutarlı.
        self.model = model
        self.test_size = (cfg["res"], cfg["res"])
        self.nas_summary = {"arch": self.nas_arch, "depth": {s: cfg["depth"][s] for s in DEPTH_STAGES},
                            "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 4),
                            "seed_init": int(self.nas_seed_init)}
        return model

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=None):
        from yolox.data import DataLoader, TrainTransform
        import nas_data as nd
        assert not is_distributed
        if self.dataset is None:
            self.dataset = self.get_dataset(cache=False, cache_type=cache_img)
        ds = nd.make_seeded_mosaic()(
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
        import time

        import numpy as np
        from loguru import logger
        from yolox.core import Trainer

        import nas_data as nd
        from ckpt_store import CheckpointStore

        exp = self
        snaps = {int(x) for x in str(exp.nas_snapshot_epochs).split(",") if str(x).strip()}

        class ScratchTrainer(Trainer):
            def __init__(self, exp_, args_):
                super().__init__(exp_, args_)
                self.store = (CheckpointStore(exp.nas_ckpt_dir, os.path.join(self.file_name, "ck_local"),
                                              keep=exp.nas_ckpt_keep, log=logger.info)
                              if exp.nas_ckpt_dir else None)
                self._state, self._rec = self.store.latest() if self.store else (None, None)
                self.start_iter = 0

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
                    self._state = None
                logger.info(f"Sıfırdan alt ağ: {exp.nas_summary}")
                logger.info(f"parmak izi {exp.fingerprint()} · başlangıç epoch {self.start_epoch} iter "
                            f"{self.start_iter}")

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
                with torch.cuda.amp.autocast(enabled=self.amp_training):
                    outputs = self.model(inps, targets)
                loss = outputs["total_loss"]
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.use_model_ema:
                    self.ema_model.update(self.model)
                lr = self.lr_scheduler.update_lr(p + 1)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr
                self.meter.update(iter_time=time.time() - t0, data_time=t_data - t0, lr=lr,
                                  **{k: v.detach() if torch.is_tensor(v) else v for k, v in outputs.items()})
                if digest is not None:
                    rec = {"progress": p, "epoch": self.epoch, "iter": self.iter, "input_size": list(self.input_size),
                           "lr_next": lr, "cfgs": [exp.nas_arch], "batch_sha256": digest,
                           "losses": [float(loss.detach())], "scale": float(self.scaler.get_scale())}
                    with open(exp.nas_trace_path, "a") as f:
                        f.write(json.dumps(rec) + "\n")

            def save_full(self, epoch, it, pinned=False, meta=None):
                if self.store is None or self.rank != 0:
                    return
                progress = epoch * self.max_iter + it
                st = {"format": FORMAT, "fingerprint": exp.fingerprint(), "batch_size": self.args.batch_size,
                      "arch": exp.nas_arch, "epoch": epoch, "iter": it, "progress": progress,
                      "iters_per_epoch": self.max_iter, "start_epoch": epoch, "max_epoch": self.max_epoch,
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
                pass

            def save_ckpt(self, ckpt_name, update_best_ckpt=False, ap=None):
                pass

            def after_epoch(self):
                ep = self.epoch + 1
                self.save_full(ep, 0, pinned=(ep in snaps or ep == self.max_epoch), meta={"epoch_end": ep})

        return ScratchTrainer(self, args)
