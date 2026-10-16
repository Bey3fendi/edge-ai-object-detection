"""SupernetTrainer'ın CPU'da uçtan uca kesinti testi (sentetik COCO, ~3–5 dk). GPU sürümü notebook 03 hücre 7.

YOLOX Trainer CUDA varsayar; burada yalnız aygıt bağlamaları CPU'ya yönlendirilir (set_device, prefetcher,
aygıt adı). Eğitim mantığı (sandwich, kayıt deposu, devam, tohumlu veri, lr) değiştirilmeden koşar.

Akış: REF kesintisiz (2 epoch) · TEST: 1. deneme kayıt YAZILIRKEN öldürülür (NAS_TEST_KILL_AT_SAVE) →
2. deneme son doğrulanmış kayıttan devam → bitiş. İzler birebir; CPU belirlenimli olduğu için son ağırlıklar da
birebir beklenir.

Kullanım: python scripts/check_trainer_cpu.py [--exp supernet|scratch]   (alt süreçleri kendisi başlatır)
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def child(work, tag, max_epoch, kind="supernet"):
    sys.path[:0] = [HERE, os.path.join(ROOT, "exps", "nas"), os.path.join(ROOT, "YOLOX")]
    import torch
    import yolox.core.trainer as tr

    class CpuPrefetcher:
        def __init__(self, loader):
            self.it = iter(loader)

        def next(self):
            x, t, _, _ = next(self.it)
            return x, t

    tr.DataPrefetcher = CpuPrefetcher
    torch.cuda.set_device = lambda *a, **k: None
    _init = tr.Trainer.__init__

    def init(self, exp, args):
        _init(self, exp, args)
        self.device = "cpu"
    tr.Trainer.__init__ = init

    if kind == "scratch":
        import scratch_subnet as ss_
        exp = ss_.Exp()
        exp.nas_arch = "r320_d23_k" + "357" * 10      # karışık kernel + kesilmiş derinlik
        exp.nas_seed_init = 7
    else:
        import supernet as sn
        exp = sn.Exp()
    exp.data_dir = os.path.join(work, "coco")
    exp.train_ann = exp.val_ann = "train.json"
    exp.num_classes = 80
    exp.input_size = exp.test_size = (256, 256)
    exp.random_size = (6, 10)
    exp.data_num_workers = 2
    exp.max_epoch = max_epoch
    if kind != "scratch":
        exp.nas_phase1_epochs = max_epoch
    exp.no_aug_epochs = 0
    exp.warmup_epochs = 1
    exp.enable_mixup = True
    exp.nas_ckpt_every_iters = 2
    exp.nas_ckpt_dir = os.path.join(work, tag, "ckpt")
    exp.nas_trace_path = os.path.join(work, tag, "trace.jsonl")
    exp.output_dir = os.path.join(work, tag, "out")
    os.makedirs(os.path.join(work, tag), exist_ok=True)
    args = argparse.Namespace(experiment_name="t", batch_size=4, fp16=False, resume=False, ckpt=None,
                              start_epoch=None, logger="tensorboard", cache=None, occupy=False, num_machines=1,
                              machine_rank=0, devices=1, dist_backend="nccl", dist_url=None)
    torch.manual_seed(0)   # rastgele başlatma iki koşuda aynı olsun (Colab testinde ağırlık önceden eğitilmişten yüklenir)
    exp.get_trainer(args).train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", nargs=4)
    ap.add_argument("--exp", choices=("supernet", "scratch"), default="supernet")
    a = ap.parse_args()
    if a.child:
        child(a.child[0], a.child[1], int(a.child[2]), a.child[3])
        return
    sys.path[:0] = [HERE]
    import check_resume as cr
    import nas_data as nd
    from ckpt_store import CheckpointStore
    import torch
    with tempfile.TemporaryDirectory() as work:
        coco = os.path.join(work, "coco")
        cr.synthetic_coco(coco, n=24)
        import shutil
        shutil.copytree(os.path.join(coco, "train2017"), os.path.join(coco, "val2017"))

        def run(tag, env=None):
            r = subprocess.run([sys.executable, __file__, "--child", work, tag, "2", a.exp], capture_output=True, text=True,
                               encoding="utf-8", errors="replace",
                               env={**os.environ, "PYTHONIOENCODING": "utf-8", **(env or {})}, cwd=ROOT)
            return r.returncode, r.stdout + r.stderr

        rc, out = run("ref")
        assert rc == 0, out[-4000:]
        # ipe = 24 // 4 = 6; kayıtlar: 2, 4, 6(epoch), 8, 10, 12(epoch)
        rc1, out1 = run("test", {"NAS_TEST_KILL_AT_SAVE": "8"})
        assert rc1 == 137, (rc1, out1[-3000:])
        rc2, out2 = run("test")
        assert rc2 == 0 and "ilerleme 6)" in out2, (rc2, out2[-3000:])
        cmp_ = nd.compare_traces(os.path.join(work, "ref", "trace.jsonl"), os.path.join(work, "test", "trace.jsonl"))
        print(json.dumps(cmp_, indent=1, default=str))
        A = CheckpointStore(os.path.join(work, "ref", "ckpt"), os.path.join(work, "la")).latest()[0]
        B = CheckpointStore(os.path.join(work, "test", "ckpt"), os.path.join(work, "lb")).latest()[0]
        assert A["progress"] == B["progress"] == 12
        d = max((A["model_raw"][k].float() - B["model_raw"][k].float()).abs().max().item() for k in A["model_raw"])
        de = max((A["model"][k].float() - B["model"][k].float()).abs().max().item() for k in A["model"])
        print(f"son ağırlık farkı: ham {d:.3e}, EMA {de:.3e}; lr dizisi ve batch hash'leri birebir")
        assert d < 1e-5 and de < 1e-5, "CPU'da devam eden koşu kesintisizden farklı ağırlığa ulaştı"
        print(f"CPU TRAINER KESİNTİ TESTİ GEÇTİ ({a.exp})")


if __name__ == "__main__":
    main()
