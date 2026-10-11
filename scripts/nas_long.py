"""NAS 2. tur (nas_r2): ReLU k3 ile kontrolün tam COCO'da eşit uzun ince ayarı — koşu, değerlendirme, karar.

notebooks/02d_nas_tur2_relu_tam_coco.ipynb tarafından kullanılır; exp `exps/nas/nano_macro_long.py`.
Kesintiye dayanıklılık ilkeleri nas_search.py ile aynıdır (atomik yazım, runs/<koşu>/status.json,
CSV her seferinde durum dosyalarından yeniden üretilir).

Değerlendirme anahtarı `e<epoch>_s<çözünürlük>`: ara epoch'lar (öğrenme eğrisi) yalnız KARAR_RES'te,
son epoch üç çözünürlükte.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nas_search as ns  # noqa: E402

ROUND = "nas_r2"
CONTROL = ns.CONTROL_ARCH            # silu_dfull_k3 — dokunulmamış YOLOX-Nano
TREATMENT = "relu_dfull_k3"
RUNS = (CONTROL, TREATMENT)          # kontrol önce: erken kontrol tarifi doğrular
DECISION_RES = 416
DECISION_MIN_DELTA = -1.0            # Karar 2/5: NAS aşaması bütçesi; Furkan 2026-10-12 (önceden sabit)
EARLY_CHECK_MIN_MAP = 23.0           # 1. turla aynı eşik: kontrol bunun altındaysa tarif bilgiyi bozuyor
FINGERPRINT_KEYS = ("round", "runs", "recipe", "eval_protocol", "decision_rule", "exp_file_sha256",
                    "base_exp_file_sha256", "weights_sha256", "train_ann_sha256", "yolox_commit")


def eval_key(epoch, res):
    return f"e{epoch}_s{res}"


def eval_plan(max_epoch, interval, resolutions=ns.RESOLUTIONS, decision_res=DECISION_RES):
    """(epoch, çözünürlük) listesi: ara anlık görüntüler karar çözünürlüğünde, son epoch hepsinde."""
    plan = [(e, decision_res) for e in range(interval, max_epoch, interval)]
    return plan + [(max_epoch, r) for r in resolutions]


def ckpt_name(epoch, max_epoch):
    return "last_epoch_ckpt.pth" if epoch == max_epoch else f"epoch{epoch}_ckpt.pth"


CURVE_FIELDS = ["run_id", "act", "epoch", "resolution", "map5095", "ap50", "ap_small", "ap_medium",
                "ap_large", "params_m", "gflops", "gpu_forward_ms", "eval_s", "evaluated_utc"]


def rebuild_curve(results_dir, runs=RUNS):
    """Değerlendirilmiş her (koşu, epoch, çözünürlük) için bir satır → curve.csv."""
    rows = []
    for run_id in runs:
        st = ns.get_status(results_dir, run_id)
        for key, ev in st.get("evals", {}).items():
            ep, res = key[1:].split("_s")
            rows.append({"run_id": run_id, "act": ns.arch_params(run_id)["nas_act"], "epoch": int(ep),
                         "resolution": int(res), **{k: ev.get(k) for k in CURVE_FIELDS[4:]}})
    rows.sort(key=lambda r: (r["run_id"] != CONTROL, r["epoch"], r["resolution"]))
    path = os.path.join(results_dir, "curve.csv")
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CURVE_FIELDS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)
    return rows


def gaps(rows):
    """Aynı (epoch, çözünürlük) için mAP(ReLU) − mAP(kontrol)."""
    by = {(r["run_id"], r["epoch"], r["resolution"]): r["map5095"] for r in rows}
    out = []
    for (run, ep, res), m in sorted(by.items(), key=lambda kv: (kv[0][1], kv[0][2])):
        c = by.get((CONTROL, ep, res))
        if run == TREATMENT and c is not None and m is not None:
            out.append({"epoch": ep, "resolution": res, "relu": m, "control": c, "delta": round(m - c, 2)})
    return out


def decision(rows, max_epoch):
    g = {(d["epoch"], d["resolution"]): d for d in gaps(rows)}
    d = g.get((max_epoch, DECISION_RES))
    if d is None:
        return {"status": "eksik", "rule": f"Δ{DECISION_RES} (epoch {max_epoch}) ≥ {DECISION_MIN_DELTA}"}
    return {"status": "ReLU geçer" if d["delta"] >= DECISION_MIN_DELTA else "ReLU geçmez",
            "rule": f"Δ{DECISION_RES} (epoch {max_epoch}) ≥ {DECISION_MIN_DELTA}", **d}
