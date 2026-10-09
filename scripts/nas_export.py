"""Makro-mimari araması N3–N4 hazırlığı: N4 aday listesi ve adayların ONNX'e dışa aktarımı.

N3 = aday mimarilerin ONNX'e dışa aktarımı, N4 = cihazda (RPi 5, ön ölçüm Jetson CPU) gecikme ölçümü.
notebooks/02b_aday_onnx_disa_aktarma.ipynb tarafından kullanılır; aday listesi yerelde de üretilebilir:

    python scripts/nas_export.py candidates results/nas_r1b

Dışa aktarma kuralları (AGENTS.md, workflow 6. aşama): opset 13, statik 1×3×H×W, giriş `images`,
çıkış `output`, `decode_in_inference=False` (YOLOX export_onnx.py varsayılanları), onnxsim yok.
PyTorch ve ONNX Runtime çıktılarının eşdeğerliği her aday için kontrol edilir.

Kesintiye dayanıklılık nas_search.py ile aynı: her aday için ayrı JSON kaydı, atomik yazım, özet CSV
her seferinde bu kayıtlardan yeniden üretilir. torch/onnx yalnız dışa aktarma fonksiyonlarında yüklenir.
"""

import csv
import inspect
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nas_search as ns  # noqa: E402

OPSET = 13
INPUT_NAME, OUTPUT_NAME = "images", "output"
ORT_THREADS = 4               # AGENTS.md: ORT CPU ölçümlerinde intra_op_num_threads = 4
STRIDES = (8, 16, 32)
NUM_OUTPUTS = 85              # 4 kutu + 1 nesnellik + 80 sınıf
EQ_RTOL, EQ_ATOL = 1e-3, 1e-3
# Dışa aktarma manifestinde değişmemesi gereken alanlar (aynı turun ONNX'leri aynı yolla üretilmeli)
EXPORT_FINGERPRINT_KEYS = ("round", "candidates", "export_protocol", "exp_file_sha256", "yolox_commit",
                           "versions")

# ---- N4 aday listesi ---------------------------------------------------------------------------

BUDGET = -1.0                 # Karar 5: aynı çözünürlükteki kontrole göre ≥ −1,0 puan
NOISE_MARGIN = 0.3            # nas_r1b kontrol tekrar farkının en büyüğü (|rep2 − rep1|, 320'de)
# Seçime girmeyen bilgi adayları: ReLU'nun ORT CPU'da hız kazancı var mı? 2. turdaki uzun ince ayarın
# (Furkan, 2026-10-07) değip değmeyeceğini gösterir. Gecikme ağırlıktan bağımsızdır.
REFERENCE_ARCHS = ("relu_dfull_k5",)

TIERS = {
    "budget": "bütçe içinde (≥ −1,0); seçilebilir",
    "noise_margin": f"bütçenin {NOISE_MARGIN} puan (ölçülen gürültü) altına kadar; seçim için 2. tur gerekir",
    "reference": "bütçe dışı bilgi adayı (2. tur kararı); seçilemez",
}

CANDIDATE_FIELDS = ["candidate_id", "tier", "selectable", "run_id", "arch_id", "act", "depth", "dw_k",
                    "resolution", "params_m", "gflops", "map5095", "ap50", "delta_vs_control"]


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def select_candidates(rows, budget=BUDGET, margin=NOISE_MARGIN, reference_archs=REFERENCE_ARCHS):
    """candidates.csv satırlarından N4 listesi; ikinci kontrol koşusu (_rep2) aynı mimari olduğu için dışarıda."""
    out = []
    for r in rows:
        if r["repeat"] != "1":
            continue
        d = float(r["delta_vs_control"])
        if d >= budget:
            tier = "budget"
        elif d >= round(budget - margin, 2):
            tier = "noise_margin"
        elif r["arch_id"] in reference_archs:
            tier = "reference"
        else:
            continue
        out.append({**{k: r[k] for k in CANDIDATE_FIELDS if k in r}, "tier": tier,
                    "selectable": tier == "budget"})
    order = {t: i for i, t in enumerate(TIERS)}
    out.sort(key=lambda r: (order[r["tier"]], int(r["resolution"]), r["arch_id"]))
    return out


def write_candidates(results_dir):
    rows = select_candidates(read_csv(os.path.join(results_dir, "candidates.csv")))
    path = os.path.join(results_dir, "n4_candidates.csv")
    write_csv(path, rows, CANDIDATE_FIELDS)
    return path, rows


# ---- dışa aktarma ---------------------------------------------------------------------------------

def n_anchors(res):
    return sum((res // s) ** 2 for s in STRIDES)


def build_trained(exp_file, run_id, ckpt_path):
    """Adayın mimarisini kurar (rastgele başlatma + cerrahi) ve eğitilmiş EMA ağırlığını birebir yükler."""
    import torch
    from yolox.exp import get_exp
    sys.modules.pop("nano_macro", None)   # get_exp import önbelleği (bkz. notebook 02 hücre 6)
    e = get_exp(exp_file, None)
    e.merge(ns.exp_opts(run_id))          # nas_pretrained boş: ağırlık cerrahiden SONRA yüklenir
    model = e.get_model()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)   # trainer EMA modelini "model" olarak kaydeder
    model.eval()
    model.head.decode_in_inference = False
    return e, model, ckpt.get("start_epoch")


def export_onnx(model, res, path):
    import torch
    dummy = torch.zeros(1, 3, res, res)
    kw = dict(input_names=[INPUT_NAME], output_names=[OUTPUT_NAME], opset_version=OPSET,
              do_constant_folding=True, dynamic_axes=None)
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kw["dynamo"] = False   # TorchScript dışa aktarıcı; dynamo yolu opset 13'ü desteklemez
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(model, dummy, tmp, **kw)
    os.replace(tmp, path)


def graph_info(path):
    import onnx
    m = onnx.load(path)
    onnx.checker.check_model(m)
    ops = {}
    for n in m.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    shape = lambda v: [d.dim_value if d.HasField("dim_value") else d.dim_param
                       for d in v.type.tensor_type.shape.dim]
    return {"opset": max(o.version for o in m.opset_import if o.domain in ("", "ai.onnx")),
            "inputs": {v.name: shape(v) for v in m.graph.input},
            "outputs": {v.name: shape(v) for v in m.graph.output},
            "n_nodes": len(m.graph.node), "op_counts": dict(sorted(ops.items()))}


def check_equivalence(model, path, inputs):
    """inputs: (ad, 1×3×H×W float32 numpy) listesi. Her giriş için PyTorch ve ORT çıktısını karşılaştırır."""
    import numpy as np
    import onnxruntime as ort
    import torch
    so = ort.SessionOptions()
    so.intra_op_num_threads = ORT_THREADS
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    res = []
    for name, x in inputs:
        with torch.no_grad():
            ref = model(torch.from_numpy(x)).numpy()
        out = sess.run([OUTPUT_NAME], {INPUT_NAME: x})[0]
        assert out.shape == ref.shape, (out.shape, ref.shape)
        res.append({"input": name, "max_abs_diff": float(np.abs(ref - out).max()),
                    "allclose": bool(np.allclose(ref, out, rtol=EQ_RTOL, atol=EQ_ATOL))})
    return res


def preprocess(img_bgr, res):
    """YOLOX ValTransform (legacy=False) ile aynı: oran korunarak küçült, sağ/alt 114 dolgu, BGR, 0–255."""
    import cv2
    import numpy as np
    padded = np.full((res, res, 3), 114, dtype=np.uint8)
    r = min(res / img_bgr.shape[0], res / img_bgr.shape[1])
    h, w = int(img_bgr.shape[0] * r), int(img_bgr.shape[1] * r)
    padded[:h, :w] = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32)


def equivalence_inputs(res, image_paths=(), seed=0):
    import cv2
    import numpy as np
    rng = np.random.default_rng(seed)
    xs = [("random_u0_255", (rng.random((1, 3, res, res)) * 255).astype(np.float32))]
    for p in image_paths:
        xs.append((os.path.basename(p), preprocess(cv2.imread(p), res)))
    return xs


def record_path(out_dir, cid):
    return os.path.join(out_dir, "candidates", f"{cid}.json")


def export_candidate(cand, exp_file, ckpt_path, onnx_dir, out_dir, image_paths=()):
    """Tek adayın dışa aktarımı + kontroller; kayıt candidates/<aday>.json. Bitmişse tekrar yapılmaz."""
    cid, run_id, res = cand["candidate_id"], cand["run_id"], int(cand["resolution"])
    onnx_path = os.path.join(onnx_dir, f"{cid}.onnx")
    rec = ns.read_json(record_path(out_dir, cid))
    if rec and os.path.exists(onnx_path) and ns.sha256(onnx_path) == rec["onnx_sha256"]:
        return rec
    _, model, epoch = build_trained(exp_file, run_id, ckpt_path)
    export_onnx(model, res, onnx_path)
    info = graph_info(onnx_path)
    assert info["opset"] == OPSET, info["opset"]
    assert info["inputs"] == {INPUT_NAME: [1, 3, res, res]}, info["inputs"]
    assert info["outputs"] == {OUTPUT_NAME: [1, n_anchors(res), NUM_OUTPUTS]}, info["outputs"]
    eq = check_equivalence(model, onnx_path, equivalence_inputs(res, image_paths))
    assert all(e["allclose"] for e in eq), f"{cid}: PyTorch/ORT çıktıları farklı: {eq}"
    rec = {"candidate_id": cid, "tier": cand["tier"], "selectable": cand["selectable"], "run_id": run_id,
           "resolution": res, "ckpt": os.path.basename(ckpt_path), "ckpt_sha256": ns.sha256(ckpt_path),
           "ckpt_epoch": epoch, "onnx_file": os.path.basename(onnx_path),
           "onnx_sha256": ns.sha256(onnx_path), "onnx_bytes": os.path.getsize(onnx_path),
           **info, "equivalence": eq, "exported_utc": ns.now_utc()}
    ns.write_json(record_path(out_dir, cid), rec)
    return rec


EXPORT_FIELDS = ["candidate_id", "tier", "selectable", "run_id", "resolution", "ckpt_epoch", "onnx_file",
                 "onnx_bytes", "onnx_sha256", "ckpt_sha256", "opset", "n_nodes", "op_counts",
                 "max_abs_diff", "exported_utc"]


def rebuild_exports_csv(out_dir, candidate_ids):
    rows = []
    for cid in candidate_ids:
        rec = ns.read_json(record_path(out_dir, cid))
        if rec:
            rows.append({**rec, "op_counts": json.dumps(rec["op_counts"], sort_keys=True),
                         "max_abs_diff": max(e["max_abs_diff"] for e in rec["equivalence"])})
    write_csv(os.path.join(out_dir, "exports.csv"), rows, EXPORT_FIELDS)
    return rows


if __name__ == "__main__":
    assert len(sys.argv) == 3 and sys.argv[1] == "candidates", __doc__
    path, rows = write_candidates(sys.argv[2])
    for r in rows:
        print(f"{r['tier']:13s} {r['candidate_id']:26s} {float(r['map5095']):5.1f}  Δ {float(r['delta_vs_control']):+.1f}")
    print(f"{len(rows)} aday → {path}")
