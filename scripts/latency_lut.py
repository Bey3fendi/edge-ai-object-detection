"""Gerçek NAS (Karar 6): cihaz gecikme tablosu (LUT) — paket üretimi, cihazda ölçüm, doğrulama/kalibrasyon.

Gecikme modeli (supernet_search.LatencyModel):
    t(alt ağ) = a × [ fixed(res) + Σ_aktif DW (dw(res, katman, k) − dw(res, katman, 3)) + Σ_ek blok block(res) ] + b
    fixed  = k3 + dark3/dark4 derinlik 1 olan tam modelin uçtan uca gecikmesi (aranmayan her şey dahil)
    dw     = tek DW katmanı (Conv k×k + BN + aktivasyon) mikro modeli, o katmanın gerçek giriş şekliyle
    block  = derinlik 1'in üstündeki her Bottleneck (k3) mikro modeli
    a, b   = rastgele + referans alt ağların uçtan uca ölçümüne en küçük kareler (fit adımı)
Mikro-op toplamı CPU'da tam modeli birebir vermez (çağrı yükü, önbellek); bu yüzden doğrulama zorunlu:
Spearman ρ ve ortalama mutlak yüzde hata raporlanır, arama yalnız ρ ≥ 0,9 ise LUT'a güvenir.

Adımlar:
    build   (torch; PC/Colab)  python scripts/latency_lut.py build --out pkg_dir [--n-val 40]
    measure (cihaz; numpy+ORT) python3 scripts/latency_lut.py measure --pkg pkg_dir --device jetson_orin_nano_cpu
    fit     (torch; PC/Colab)  python scripts/latency_lut.py fit --pkg pkg_dir --device jetson_orin_nano_cpu

Ölçüm protokolü (LUT için kısaltılmış; seçilen modellerin raporlanan gecikmesi Karar 3 protokolüyle ayrıca
ölçülür): ORT CPU, 4 intra-op thread, ORT_ENABLE_ALL, sıralı; mikro-op: 30 ısınma + 1 sn × 3 tekrar, tam
model: 50 ısınma + 3 sn × 3 tekrar; değer = tekrar medyanlarının ortalaması. Sıra tohumlu karışık; kaldığı
yerden devam (ölçülmüş öğe atlanır).
"""

import argparse
import hashlib
import json
import os
import random
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from nas_search import now_utc, read_json, write_json  # noqa: E402

INPUT_NAME, OUTPUT_NAME, OPSET, ORT_THREADS = "images", "output", 13, 4
MICRO = {"warmup": 30, "duration_s": 1.0, "repeats": 3}
FULL = {"warmup": 50, "duration_s": 3.0, "repeats": 3}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---- build (torch) ------------------------------------------------------------------------------

def _export(module, shape, path):
    import inspect
    import torch
    kw = dict(input_names=[INPUT_NAME], output_names=[OUTPUT_NAME], opset_version=OPSET,
              do_constant_folding=True)
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kw["dynamo"] = False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(module.eval(), torch.zeros(*shape), path + ".tmp", **kw)
    os.replace(path + ".tmp", path)


def build(out, n_val=40, seed=0, ckpt=""):
    import copy
    import torch
    import torch.nn as nn
    sys.path.insert(0, os.path.join(HERE, "..", "exps", "nas"))
    sys.path.insert(0, os.path.join(HERE, "..", "YOLOX"))
    import supernet as sn

    exp = sn.Exp()
    sup = exp.get_model().eval()
    if ckpt:
        st = torch.load(ckpt, map_location="cpu")
        sup.load_state_dict(st.get("model", st), strict=True)
    names = exp.dw_names
    n = len(names)
    shapes = {}
    hooks = [sup.get_submodule(nm).register_forward_hook(
        lambda m, i, o, nm=nm: shapes.__setitem__(nm, tuple(i[0].shape))) for nm in names]
    blocks = [f"{s}.{i}" for s in sn.DEPTH_STAGES for i in range(1, max(sn.DEPTHS))]
    for b in blocks:
        s, i = b.split(".")
        mod = sn.csp_of(sup, s).m[int(i)]
        hooks.append(mod.register_forward_hook(
            lambda m, inp, o, b=b: shapes.__setitem__(b, tuple(inp[0].shape))))
    items = []
    for res in sn.RESOLUTIONS:
        shapes.clear()
        sn.set_active(sup, names, sn.max_config(n))
        with torch.no_grad():
            sup(torch.zeros(1, 3, res, res))
        shp = {k: tuple(int(s) for s in v) for k, v in shapes.items()}   # alt ağ kopyaları kancaları da kopyalar
        for nm in names:
            parent = sup.get_submodule(nm.rpartition(".")[0])          # BaseConv (conv, bn, act)
            for k in sn.KERNELS:
                conv = parent.conv
                m = nn.Sequential(nn.Conv2d(conv.in_channels, conv.out_channels, k, conv.stride,
                                            k // 2, groups=conv.groups, bias=False),
                                  copy.deepcopy(parent.bn), copy.deepcopy(parent.act))
                with torch.no_grad():
                    m[0].weight.copy_(conv.active_weight(k))
                rel = f"dw/r{res}/{nm}_k{k}.onnx"
                _export(m, shp[nm], os.path.join(out, rel))
                items.append({"id": f"dw|{res}|{nm}|{k}", "kind": "micro", "path": rel, "shape": shp[nm]})
        nano = sn.extract_subnet(sup, names, sn.nano_config(n, res))
        for b in blocks:
            s, i = b.split(".")
            m = sn.csp_of(nano, s).m[int(i)]
            rel = f"block/r{res}/{b}.onnx"
            _export(copy.deepcopy(m), shp[b], os.path.join(out, rel))
            items.append({"id": f"block|{res}|{b}", "kind": "micro", "path": rel, "shape": shp[b]})
        fixed = sn.min_config(n)
        fixed["res"] = res
        items.append(_full(sn, sup, names, fixed, out, "fixed"))
    for h in hooks:
        h.remove()
    rng = random.Random(seed)
    vals = [sn.nano_config(n, r) for r in sn.RESOLUTIONS] + [sn.max_config(n)]
    vals += [sn.canonical(sn.random_config(n, rng), names) for _ in range(n_val)]
    for c in vals:
        items.append(_full(sn, sup, names, c, out, "val"))
    idx = {"created_utc": now_utc(), "names": names, "items": items, "seed": seed, "ckpt": ckpt or None,
           "space": {"kernels": sn.KERNELS, "depths": sn.DEPTHS, "resolutions": sn.RESOLUTIONS,
                     "depth_stages": sn.DEPTH_STAGES, "blocks": blocks}}
    for it in items:
        it["sha256"] = sha256(os.path.join(out, it["path"]))
    write_json(os.path.join(out, "lut_index.json"), idx)
    print(f"{len(items)} öğe → {out}")


def _full(sn, sup, names, cfg, out, tag):
    sub = sn.extract_subnet(sup, names, cfg).eval()
    sub.head.decode_in_inference = False                 # N3/N4 ile aynı: ham çıktı, NMS yok
    cid = sn.config_id(cfg)
    rel = f"full/{tag}/{cid}.onnx"
    _export(sub, (1, 3, cfg["res"], cfg["res"]), os.path.join(out, rel))
    return {"id": f"{tag}|{cid}", "kind": "full", "path": rel, "shape": [1, 3, cfg["res"], cfg["res"]],
            "config_id": cid}


# ---- measure (cihaz) ----------------------------------------------------------------------------

def _session(path):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = ORT_THREADS
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def time_session(sess, shape, warmup, duration_s, repeats, seed=0):
    import numpy as np
    x = np.random.default_rng(seed).standard_normal(shape).astype(np.float32)
    feed = {INPUT_NAME: x}
    for _ in range(warmup):
        sess.run(None, feed)
    meds = []
    for _ in range(repeats):
        lat, t_end = [], time.perf_counter() + duration_s
        while time.perf_counter() < t_end:
            a = time.perf_counter_ns()
            sess.run(None, feed)
            lat.append((time.perf_counter_ns() - a) / 1e6)
        meds.append(statistics.median(lat))
    return {"ms": round(statistics.mean(meds), 5), "rep_medians": [round(m, 5) for m in meds]}


def measure(pkg, device, smoke=False, order_seed=0):
    import n4_latency as n4
    idx = read_json(os.path.join(pkg, "lut_index.json"))
    # deneme ölçümü ayrı klasöre: aksi halde tam ölçüm kısa protokollü deneme değerlerini "ölçülmüş" sayıp atlardı
    out = os.path.join(pkg, "measured", device + ("_smoke" if smoke else ""))
    os.makedirs(out, exist_ok=True)
    meta_path = os.path.join(out, "session.json")
    meta = read_json(meta_path) or {"device": device, "device_info": n4.device_info(),
                                    "device_settings": n4.device_settings(), "versions": n4.versions(),
                                    "protocol": {"micro": MICRO, "full": FULL, "threads": ORT_THREADS},
                                    "started_utc": now_utc()}
    assert meta["device_settings"] == n4.device_settings(), "cihaz ayarı oturum ortasında değişmiş"
    write_json(meta_path, meta)
    res_path = os.path.join(out, "measurements.json")
    done = read_json(res_path) or {}
    items = list(idx["items"])
    random.Random(order_seed).shuffle(items)
    if smoke:
        items = items[:12]
    t0 = time.time()
    for i, it in enumerate(items, 1):
        if it["id"] in done and done[it["id"]]["sha256"] == it["sha256"]:
            continue
        p = MICRO if it["kind"] == "micro" else FULL
        if smoke:
            p = {"warmup": 5, "duration_s": 0.3, "repeats": 1}
        sess = _session(os.path.join(pkg, it["path"]))
        r = time_session(sess, it["shape"], **p)
        done[it["id"]] = {**r, "sha256": it["sha256"], "kind": it["kind"], "utc": now_utc(),
                          "temp_c": max(n4.temperatures().values() or [None])}
        if i % 20 == 0 or i == len(items):
            write_json(res_path, done)
            print(f"[{i}/{len(items)}] {time.time() - t0:.0f} sn · son {it['id']} {r['ms']} ms", flush=True)
    write_json(res_path, done)
    meta["finished_utc"] = now_utc()
    write_json(meta_path, meta)


# ---- fit (LUT + doğrulama) ----------------------------------------------------------------------

def spearman(a, b):
    def ranks(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        for k, i in enumerate(order):
            r[i] = float(k)
        return r
    ra, rb = ranks(a), ranks(b)
    ma, mb = statistics.mean(ra), statistics.mean(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    return cov / (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5


def fit(pkg, device):
    sys.path.insert(0, os.path.join(HERE, "..", "exps", "nas"))
    sys.path.insert(0, os.path.join(HERE, "..", "YOLOX"))
    idx = read_json(os.path.join(pkg, "lut_index.json"))
    meas = read_json(os.path.join(pkg, "measured", device, "measurements.json"))
    names = idx["names"]
    lut = {"device": device, "fixed": {}, "dw": {}, "block": {}, "calib": {"a": 1.0, "b": 0.0}}
    for key, v in meas.items():
        kind, res, *rest = key.split("|")
        if kind == "dw":
            nm, k = rest
            lut["dw"].setdefault(res, {}).setdefault(nm, {})[k] = v["ms"]
        elif kind == "block":
            lut["block"].setdefault(res, {})[rest[0]] = v["ms"]
    for it in idx["items"]:
        if it["id"].startswith("fixed|") and it["id"] in meas:
            lut["fixed"][str(it["shape"][2])] = meas[it["id"]]["ms"]
    import supernet_search as ss
    import supernet as sn
    model = ss.LatencyModel(lut)
    pred, true, ids = [], [], []
    for it in idx["items"]:
        if it["id"].startswith("val|") and it["id"] in meas:
            cfg = sn.parse_config_id(it["config_id"])
            pred.append(model.raw(cfg, names))
            true.append(meas[it["id"]]["ms"])
            ids.append(it["config_id"])
    n = len(pred)
    mp, mt = statistics.mean(pred), statistics.mean(true)
    a = sum((p - mp) * (t - mt) for p, t in zip(pred, true)) / sum((p - mp) ** 2 for p in pred)
    b = mt - a * mp
    lut["calib"] = {"a": round(a, 6), "b": round(b, 6)}
    cal = [a * p + b for p in pred]
    report = {"n_val": n, "spearman_raw": round(spearman(pred, true), 4),
              "mape_raw_pct": round(100 * statistics.mean(abs(p - t) / t for p, t in zip(pred, true)), 2),
              "mape_calibrated_pct": round(100 * statistics.mean(abs(c - t) / t for c, t in zip(cal, true)), 2),
              "calib": lut["calib"], "fitted_utc": now_utc(),
              "rows": [{"config_id": i, "measured_ms": t, "lut_raw_ms": round(p, 3), "lut_cal_ms": round(c, 3)}
                       for i, t, p, c in zip(ids, true, pred, cal)]}
    lut["validation"] = {k: v for k, v in report.items() if k != "rows"}
    out = os.path.join(pkg, "measured", device)
    write_json(os.path.join(out, "lut.json"), lut)
    write_json(os.path.join(out, "lut_validation.json"), report)
    print(json.dumps(lut["validation"], indent=1))
    return lut


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--n-val", type=int, default=40)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--ckpt", default="")
    m = sp.add_parser("measure")
    m.add_argument("--pkg", required=True)
    m.add_argument("--device", required=True)
    m.add_argument("--smoke", action="store_true")
    f = sp.add_parser("fit")
    f.add_argument("--pkg", required=True)
    f.add_argument("--device", required=True)
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.out, a.n_val, a.seed, a.ckpt)
    elif a.cmd == "measure":
        measure(a.pkg, a.device, a.smoke)
    else:
        fit(a.pkg, a.device)


if __name__ == "__main__":
    main()
