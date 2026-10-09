"""Makro-mimari araması (N2) koşu yönetimi: ızgara, manifest, durum dosyaları, özet CSV.

notebooks/02_makro_mimari_arama.ipynb tarafından kullanılır. Kesintiye dayanıklılık ilkeleri:
- Her dosya atomik yazılır (geçici ad → os.replace); yarım dosya oluşmaz.
- Her koşunun durumu runs/<run_id>/status.json'dadır; bitmiş adım tekrar yapılmaz.
- candidates.csv her seferinde koşu klasörlerinden YENİDEN üretilir (çift satır, yarım satır yok).
"""

import csv
import datetime as dt
import hashlib
import itertools
import json
import os
import re

ACTS = ("silu", "relu", "hardswish")
DEPTHS = ("full", "reduced")
DW_KS = (3, 5)
RESOLUTIONS = (320, 416, 512)
CONTROL_ARCH = "silu_dfull_k3"          # dokunulmamış YOLOX-Nano = kontrol koşusu
CONTROL_REPEAT = CONTROL_ARCH + "_rep2"  # gürültü ölçümü için ikinci kontrol koşusu

# Manifestte değişmemesi gereken alanlar: tur ortasında tarif değişirse adaylar karşılaştırılamaz.
FINGERPRINT_KEYS = ("round", "grid", "resolutions", "runs", "recipe", "eval_protocol",
                    "exp_file_sha256", "weights_sha256", "train_ann_sha256", "yolox_commit")

STATUSES = ("pending", "training", "trained", "evaluated")


def now_utc():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---- ızgara ---------------------------------------------------------------------------------

def arch_id(act, depth, k):
    return f"{act}_d{depth}_k{k}"


def arch_params(run_id):
    m = re.fullmatch(r"(silu|relu|hardswish)_d(full|reduced)_k(3|5)(?:_rep\d+)?", run_id)
    assert m, run_id
    return {"nas_act": m.group(1), "nas_depth": m.group(2), "nas_dw_k": int(m.group(3))}


def runs():
    """12 mimari (eğitim) + kontrolün ikinci koşusu. Sıra: kontrol önce (referans erken gelir)."""
    archs = [arch_id(a, d, k) for a, d, k in itertools.product(ACTS, DEPTHS, DW_KS)]
    archs.remove(CONTROL_ARCH)
    return [CONTROL_ARCH] + archs + [CONTROL_REPEAT]


def candidate_id(run_id, res):
    return f"{run_id}_s{res}"


def exp_opts(run_id, **extra):
    """train.py / eval.py sonuna eklenen `exp.merge` argümanları."""
    opts = []
    for k, v in {**arch_params(run_id), **extra}.items():
        opts += [k, str(v)]
    return opts


# ---- manifest ---------------------------------------------------------------------------------

def ensure_manifest(path, manifest, keys=FINGERPRINT_KEYS):
    """Yoksa yazar; varsa parmak izi alanlarının (keys) aynı olduğunu doğrular ve mevcut olanı döndürür."""
    old = read_json(path)
    if old is None:
        manifest = {**manifest, "created_utc": now_utc()}
        write_json(path, manifest)
        return manifest
    diff = [k for k in keys if json.dumps(old.get(k), sort_keys=True, default=str)
            != json.dumps(manifest.get(k), sort_keys=True, default=str)]
    if diff:
        raise RuntimeError(f"Manifest farklı ({diff}): tur ortasında tarif/sürüm değişmiş. "
                           f"Bilerek yeni tur açılacaksa ROUND'u değiştir.")
    return old


# ---- koşu durumu ------------------------------------------------------------------------------

def status_path(results_dir, run_id):
    return os.path.join(results_dir, "runs", run_id, "status.json")


def get_status(results_dir, run_id):
    return read_json(status_path(results_dir, run_id), {"run_id": run_id, "status": "pending",
                                                        "attempts": [], "evals": {}})


def set_status(results_dir, run_id, st, **fields):
    assert fields.get("status", st["status"]) in STATUSES
    st = {**st, **fields, "updated_utc": now_utc()}
    write_json(status_path(results_dir, run_id), st)
    return st


# ---- log ayrıştırma (P0 notebook'u ile aynı kalıplar) ------------------------------------------

AP_RE = re.compile(r"Average Precision  \(AP\) @\[ IoU=0.50:0.95 \| area=   all \| maxDets=100 \] = ([\d.]+)")
AP50_RE = re.compile(r"Average Precision  \(AP\) @\[ IoU=0.50      \| area=   all \| maxDets=100 \] = ([\d.]+)")
AP_S_RE = re.compile(r"Average Precision  \(AP\) @\[ IoU=0.50:0.95 \| area= small \| maxDets=100 \] = ([\d.]+)")
AP_M_RE = re.compile(r"Average Precision  \(AP\) @\[ IoU=0.50:0.95 \| area=medium \| maxDets=100 \] = ([\d.]+)")
AP_L_RE = re.compile(r"Average Precision  \(AP\) @\[ IoU=0.50:0.95 \| area= large \| maxDets=100 \] = ([\d.]+)")
FWD_RE = re.compile(r"Average forward time: ([\d.]+) ms, Average NMS time: ([\d.]+) ms")
INFO_RE = re.compile(r"Params: ([\d.]+)M, Gflops: ([\d.]+)")


SHOW_TRAIN = re.compile(r"iter: \d+/|start train|No mosaic|Save weights|NAS adayı|resume training"
                        r"|WARNING|ERROR|Error|Traceback")
SHOW_EVAL = re.compile(r"Average forward|IoU=0.50:0.95 \| area=   all \| maxDets=100|ERROR|Error|Traceback")


def stream(cmd, cwd, show, tag):
    """Alt süreci çalıştırır, `show` ile eşleşen satırları canlı basar; (tam çıktı, süre sn) döndürür."""
    import subprocess
    import time
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    out = []
    for line in proc.stdout:
        out.append(line)
        if show.search(line):
            m = re.search(r"epoch: .*?(?=, lr)", line)
            print(f"[{tag} {time.time() - t0:5.0f}s] {m.group(0) if m else line.strip()[-150:]}",
                  flush=True)
    proc.wait()
    text = "".join(out)
    if proc.returncode != 0:
        print(text[-3000:])
        raise RuntimeError(f"{tag}: çıkış kodu {proc.returncode}")
    return text, time.time() - t0


def parse_eval(text):
    def pct(rx):
        m = rx.search(text)
        return round(float(m.group(1)) * 100, 2) if m else None
    fwd = FWD_RE.search(text)
    info = INFO_RE.search(text)
    return {"map5095": pct(AP_RE), "ap50": pct(AP50_RE), "ap_small": pct(AP_S_RE),
            "ap_medium": pct(AP_M_RE), "ap_large": pct(AP_L_RE),
            "gpu_forward_ms": float(fwd.group(1)) if fwd else None,
            "gpu_nms_ms": float(fwd.group(2)) if fwd else None,
            "params_m": float(info.group(1)) if info else None,
            "gflops": float(info.group(2)) if info else None}


def check_train_log(text, max_epoch, no_aug_epochs):
    """Mosaic'in tam olarak son (no_aug_epochs + 1) epoch'ta kapandığını doğrular.

    Log devam ettirilen koşularda eklenerek büyür; hata kontrolü yalnız son denemeye bakar
    (kesilen bir önceki denemenin hatası, başarıyla biten devamı geçersiz kılmaz).
    """
    last_attempt = text[text.rfind("args: Namespace"):]
    assert "Exception in training" not in last_attempt, "eğitim hatayla bitmiş"
    first_no_aug = max_epoch - no_aug_epochs
    pos_mosaic_off = text.find("No mosaic aug now")
    pos_epoch = text.find(f"start train epoch{first_no_aug}\n")
    if pos_epoch < 0:
        pos_epoch = text.find(f"start train epoch{first_no_aug}")
    assert pos_mosaic_off > pos_epoch >= 0, (
        f"mosaic epoch {first_no_aug}'dan önce kapanmış ya da hiç kapanmamış")
    assert f"start train epoch{max_epoch + 1}" not in text


# ---- özet CSV ---------------------------------------------------------------------------------

CSV_FIELDS = ["candidate_id", "run_id", "arch_id", "is_control", "repeat", "act", "depth", "dw_k",
              "resolution", "params_m", "gflops", "map5095", "ap50", "ap_small", "ap_medium",
              "ap_large", "delta_vs_control", "gpu_forward_ms", "gpu_nms_ms", "train_s",
              "train_attempts", "eval_s", "cu_train", "cu_eval", "repo_commit", "trained_utc",
              "evaluated_utc"]


def rebuild_csv(results_dir, run_ids, cu_per_hour):
    """Değerlendirilmiş her (koşu, çözünürlük) için bir satır; kontrolle fark aynı çözünürlükte."""
    rows = []
    for run_id in run_ids:
        st = get_status(results_dir, run_id)
        p = arch_params(run_id)
        for res, ev in sorted(st.get("evals", {}).items(), key=lambda kv: int(kv[0])):
            rows.append({
                "candidate_id": candidate_id(run_id, res), "run_id": run_id,
                "arch_id": arch_id(p["nas_act"], p["nas_depth"], p["nas_dw_k"]),
                "is_control": run_id.startswith(CONTROL_ARCH), "repeat": 2 if "_rep" in run_id else 1,
                "act": p["nas_act"], "depth": p["nas_depth"], "dw_k": p["nas_dw_k"],
                "resolution": int(res), **{k: ev.get(k) for k in (
                    "params_m", "gflops", "map5095", "ap50", "ap_small", "ap_medium", "ap_large",
                    "gpu_forward_ms", "gpu_nms_ms", "eval_s")},
                "train_s": st.get("train_s"), "train_attempts": len(st.get("attempts", [])),
                "cu_train": round(st["train_s"] / 3600 * cu_per_hour / len(RESOLUTIONS), 4)
                if st.get("train_s") else None,   # eğitim maliyeti 3 çözünürlüğe bölünür
                "cu_eval": round(ev["eval_s"] / 3600 * cu_per_hour, 4) if ev.get("eval_s") else None,
                "repo_commit": st.get("repo_commit"), "trained_utc": st.get("trained_utc"),
                "evaluated_utc": ev.get("evaluated_utc")})
    ctrl = {r["resolution"]: r["map5095"] for r in rows if r["run_id"] == CONTROL_ARCH}
    for r in rows:
        c = ctrl.get(r["resolution"])
        r["delta_vs_control"] = round(r["map5095"] - c, 2) if c is not None and r["map5095"] is not None else None
    path = os.path.join(results_dir, "candidates.csv")
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)
    return rows
