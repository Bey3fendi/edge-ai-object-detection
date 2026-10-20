"""Uzatma aşamasının (Karar 6) CPU'da uçtan uca testi (sentetik COCO, ~5–10 dk). Notebook 03 hücre 9c'den önce.

Gerçek koşunun küçük kopyası: ilk aşama 3 epoch (son 2'si mosaic'siz) → AYRI süreçte max_epoch 6 ile devam
(uzatma: +3 epoch, optimizer korunur, tepe lr = 0,5 × ilk aşama tepe lr, ısınma yok, son 2 epoch mosaic'siz).
REF: uzatma kesintisiz · TEST: uzatma kayıt YAZILIRKEN öldürülür → son doğrulanmış kayıttan devam → bitiş.

Doğrulanan:
- izler (alt ağ, lr, giriş boyutu, batch sha256) REF ile birebir, son ağırlıklar (ham + EMA) birebir;
- lr: ilk aşamanın son iterasyonundan sonra tepe (0,5 × taban), uzatmada artmayan cosine, son epoch min;
- mosaic bayrağı: uzatmanın ilk epoch'u açık, son 2 epoch kapalı;
- nas_search.check_extension_log: REF ve TEST loglarında geçer (TEST logu ilk aşama bölümünü de içerir),
  yalnız ilk aşama logunda düşer.

Kullanım: python scripts/check_extension_cpu.py
"""

import json
import math
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
P1, MAX, NO_AUG, IPE = 3, 6, 1, 6       # 24 sentetik görüntü / batch 4 = 6 iter/epoch


def main():
    sys.path[:0] = [HERE, os.path.join(ROOT, "exps", "nas"), os.path.join(ROOT, "YOLOX")]
    import check_resume as cr
    import nas_data as nd
    import nas_search as ns
    import supernet as sn
    from ckpt_store import CheckpointStore
    exp = sn.Exp()
    base = exp.basic_lr_per_img * 4
    peak, low = exp.nas_ext_peak_ratio * base, base * exp.min_lr_ratio
    with tempfile.TemporaryDirectory() as work:
        coco = os.path.join(work, "coco")
        cr.synthetic_coco(coco, n=24)
        import shutil
        shutil.copytree(os.path.join(coco, "train2017"), os.path.join(coco, "val2017"))

        def run(tag, max_epoch, env=None):
            r = subprocess.run([sys.executable, os.path.join(HERE, "check_trainer_cpu.py"), "--child", work, tag,
                                str(max_epoch), "supernet", str(P1), str(NO_AUG)], capture_output=True, text=True,
                               encoding="utf-8", errors="replace",
                               env={**os.environ, "PYTHONIOENCODING": "utf-8", **(env or {})}, cwd=ROOT)
            return r.returncode, r.stdout + r.stderr

        def log(tag):
            with open(os.path.join(work, tag, "out", "t", "train_log.txt"), encoding="utf-8") as f:
                return f.read()

        for tag in ("ref", "test"):
            rc, out = run(tag, P1)
            assert rc == 0, out[-4000:]
        p1_log = log("ref")
        rc, out = run("ref", MAX)
        assert rc == 0 and f"ilerleme {P1 * IPE})" in out, out[-4000:]
        # uzatma kayıtları: 20, 22, 24 (epoch sonu), 26 ← burada kayıt yazılırken öldürülür
        kill = (P1 + 1) * IPE + 2
        rc1, out1 = run("test", MAX, {"NAS_TEST_KILL_AT_SAVE": str(kill)})
        assert rc1 == 137, (rc1, out1[-3000:])
        rc2, out2 = run("test", MAX)
        assert rc2 == 0 and f"ilerleme {kill - 2})" in out2, (rc2, out2[-3000:])

        cmp_ = nd.compare_traces(os.path.join(work, "ref", "trace.jsonl"), os.path.join(work, "test", "trace.jsonl"))
        print(json.dumps(cmp_, indent=1, default=str))
        assert cmp_["n_exact_mismatches"] == 0 and not cmp_["missing_in_test"], "iz farkı"
        A = CheckpointStore(os.path.join(work, "ref", "ckpt"), os.path.join(work, "la")).latest()[0]
        B = CheckpointStore(os.path.join(work, "test", "ckpt"), os.path.join(work, "lb")).latest()[0]
        assert A["progress"] == B["progress"] == MAX * IPE and A["max_epoch"] == MAX
        d = max((A["model_raw"][k].float() - B["model_raw"][k].float()).abs().max().item() for k in A["model_raw"])
        de = max((A["model"][k].float() - B["model"][k].float()).abs().max().item() for k in A["model"])
        print(f"son ağırlık farkı: ham {d:.3e}, EMA {de:.3e}")
        assert d < 1e-5 and de < 1e-5, "kesilip devam eden uzatma kesintisizden farklı ağırlığa ulaştı"

        lr = {p: recs[0]["lr_next"] for p, recs in nd.read_trace(os.path.join(work, "ref", "trace.jsonl")).items()}
        close = lambda a, b: math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-12)
        p1, end = P1 * IPE, (MAX - NO_AUG) * IPE
        assert close(lr[p1 - 2], low), ("ilk aşama sonu min lr değil", lr[p1 - 2], low)
        assert close(lr[p1 - 1], peak), ("uzatma tepe lr ile başlamıyor", lr[p1 - 1], peak)
        ext = [lr[p] for p in range(p1 - 1, MAX * IPE)]
        assert all(b <= a + 1e-15 for a, b in zip(ext, ext[1:])), "uzatmada lr artıyor"
        assert all(close(lr[p], low) for p in range(end - 1, MAX * IPE)), "son epoch min lr değil"
        print(f"lr: ilk aşama sonu {lr[p1 - 2]:.3e} → uzatma tepe {lr[p1 - 1]:.3e} → son {lr[MAX * IPE - 1]:.3e}")

        flags = [nd.mosaic_on(e, MAX, NO_AUG) for e in range(P1, MAX)]
        assert flags == [True] * (MAX - NO_AUG - 1 - P1) + [False] * (NO_AUG + 1), flags
        print("mosaic (uzatma epoch'ları):", flags)

        ns.check_extension_log([log("ref")], P1, MAX, NO_AUG)
        ns.check_extension_log([log("test")], P1, MAX, NO_AUG)
        try:
            ns.check_extension_log([p1_log], P1, MAX, NO_AUG)
        except AssertionError:
            pass
        else:
            raise AssertionError("check_extension_log yalnız ilk aşama logunu kabul etti")
        print("UZATMA CPU TESTİ GEÇTİ")


if __name__ == "__main__":
    main()
