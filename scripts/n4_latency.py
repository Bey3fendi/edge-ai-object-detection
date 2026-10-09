"""N4: makro-mimari adaylarının cihazda ONNX Runtime CPU gecikme ölçümü (RPi 5; ön ölçüm Jetson Orin Nano).

Cihazda çalışır; yalnız numpy + onnxruntime gerekir (torch gerekmez). Girdi, notebook 02b'nin ürettiği
paket klasörüdür (`nas_r1b_n4_onnx/`: ONNX'ler, n4_candidates.csv, exports.csv, SHA256SUMS).

    python3 scripts/n4_latency.py --pkg ~/nas_r1b_n4_onnx --device jetson_orin_nano_cpu --smoke   # ≈2 dk deneme
    python3 scripts/n4_latency.py --pkg ~/nas_r1b_n4_onnx --device jetson_orin_nano_cpu           # ≈65 dk, 19 aday

Protokol (Karar 3 + AGENTS.md): ORT CPUExecutionProvider, intra_op_num_threads = 4, tüm grafik
eniyilemeleri açık; girdi gerçek görüntü (val2017, 02b'de ValTransform ile önceden işlenmiş .npy); 100 çıkarım ısınma atılır; ardından 60 sn kararlı ölçüm × 3
tekrar; rapor = tekrar ortalamalarının ortalaması ± std. Ölçülen pencere yalnız `session.run` (model
gecikmesi; ön işlem ve NMS hariç — seçim ölçütü bu, uçtan uca FPS ayrı aşamada).

Önlemler:
- Her aday ayrı alt süreçte ölçülür: bellek (VmHWM) ve ORT iş parçacığı havuzu adaylar arasında karışmaz.
- Adaylar sabit tohumlu karışık sırayla ölçülür: ısınan cihazın yavaşlaması tek bir aday grubuna yüklenmez.
- Her tekrarda CPU sıcaklığı/frekansı kaydedilir; ölçüm öncesi governor, nvpmodel, yük kaydedilir ve
  manifestteki değerlerle karşılaştırılır (oturum ortasında cihaz ayarı değişirse ölçüm durur).
- Kaldığı yerden devam: runs/<aday>.json varsa ve ONNX SHA256 aynıysa aday atlanır.
- Güç bu betikte ölçülmez (USB-C güç ölçer elle okunur, Karar 3). Jetson'da --tegrastats ile kart içi güç
  ayrı sütun olarak kaydedilebilir; cihazlar arası karşılaştırmaya girmez.
"""

import argparse
import json
import os
import platform
import random
import re
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nas_search as ns   # noqa: E402  (yalnız standart kütüphane)
import nas_export as nx   # noqa: E402  (torch/onnx yalnız dışa aktarma fonksiyonlarında yüklenir)

PROTOCOL = {"warmup": 100, "duration_s": 60.0, "repeats": 3, "intra_op_threads": nx.ORT_THREADS,
            "inter_op_threads": 1, "execution_mode": "sequential", "graph_optimization": "ORT_ENABLE_ALL",
            "provider": "CPUExecutionProvider", "input": f"val2017 {nx.LATENCY_IMAGE}, ValTransform ön işlemi, 1x3xRxR float32 (paket inputs/)",
            "window": "session.run (yalnız model)", "order_seed": 0}
SMOKE = {"warmup": 10, "duration_s": 5.0, "repeats": 1}
FINGERPRINT_KEYS = ("device", "protocol", "candidates", "versions", "device_settings")
SUMMARY_FIELDS = ["candidate_id", "tier", "selectable", "arch_id", "act", "depth", "dw_k", "resolution",
                  "map5095", "delta_vs_control", "gflops", "latency_ms_mean", "latency_ms_std",
                  "latency_ms_median", "latency_ms_p90", "latency_ms_p99", "fps_model", "n_inferences",
                  "rss_peak_mb", "temp_c_max", "tegra_vdd_in_mw", "onnx_sha256", "measured_utc"]


# ---- cihaz bilgisi ---------------------------------------------------------------------------------

def read(path, default=None):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return default


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:   # komut yoksa / zaman aşımı
        return f"ERR {e}"


def temperatures():
    out = {}
    base = "/sys/class/thermal"
    for z in sorted(os.listdir(base)) if os.path.isdir(base) else []:
        t = read(f"{base}/{z}/temp")
        if t and t.lstrip("-").isdigit():
            out[read(f"{base}/{z}/type", z)] = int(t) / 1000
    return out


def cpu_freqs_mhz():
    out = {}
    for c in range(os.cpu_count() or 0):
        f = read(f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_cur_freq")
        if f:
            out[c] = int(f) // 1000
    return out


def device_settings():
    """Oturum boyunca sabit kalması gereken ayarlar (Karar 3)."""
    gov = sorted({read(f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_governor")
                  for c in range(os.cpu_count() or 0)} - {None})
    s = {"cpu_governor": gov, "cpu_count": os.cpu_count()}
    if os.path.exists("/etc/nv_tegra_release"):
        s["nvpmodel"] = sh("nvpmodel -q 2>/dev/null | head -2")
    return s


def device_info():
    info = {"hostname": platform.node(), "platform": platform.platform(), "machine": platform.machine(),
            "python": platform.python_version(), "cpu_model": sh("lscpu | grep -i 'model name' | head -1"),
            "mem_total": sh("grep MemTotal /proc/meminfo"), "board": read("/proc/device-tree/model", "")}
    if os.path.exists("/etc/nv_tegra_release"):
        info["l4t"] = read("/etc/nv_tegra_release")
        info["jetson_clocks"] = sh("jetson_clocks --show 2>/dev/null | head -5")
    return info


def versions():
    import numpy
    import onnxruntime
    return {"onnxruntime": onnxruntime.__version__, "numpy": numpy.__version__,
            "python": platform.python_version()}


# ---- tek aday ölçümü (alt süreç) ----------------------------------------------------------------------

TEGRA_RE = re.compile(r"(\w+) (\d+)mW/(\d+)mW")


def start_tegrastats():
    try:
        return subprocess.Popen(["tegrastats", "--interval", "500"], stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, text=True)
    except FileNotFoundError:
        return None


def stop_tegrastats(proc):
    proc.terminate()
    out, _ = proc.communicate(timeout=10)
    rails = {}
    for line in out.splitlines():
        for name, inst, _avg in TEGRA_RE.findall(line):
            rails.setdefault(name, []).append(int(inst))
    return {k: round(statistics.mean(v), 1) for k, v in rails.items()}, len(out.splitlines())


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def run_one(onnx_path, input_path, res, warmup, duration_s, repeats, tegrastats):
    import numpy as np
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = PROTOCOL["intra_op_threads"]
    so.inter_op_num_threads = PROTOCOL["inter_op_threads"]
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    t0 = time.perf_counter()
    sess = ort.InferenceSession(onnx_path, so, providers=[PROTOCOL["provider"]])
    load_s = time.perf_counter() - t0
    x = np.load(input_path)
    assert x.shape == (1, 3, res, res) and x.dtype == np.float32, (x.shape, x.dtype)
    feed = {nx.INPUT_NAME: x}
    out = sess.run([nx.OUTPUT_NAME], feed)[0]
    assert out.shape == (1, nx.n_anchors(res), nx.NUM_OUTPUTS), out.shape
    for _ in range(warmup):
        sess.run([nx.OUTPUT_NAME], feed)
    reps = []
    for _ in range(repeats):
        temp0 = temperatures()
        tg = start_tegrastats() if tegrastats else None
        lat = []
        t_end = time.perf_counter() + duration_s
        t_start = time.perf_counter()
        while time.perf_counter() < t_end:
            a = time.perf_counter_ns()
            sess.run([nx.OUTPUT_NAME], feed)
            lat.append((time.perf_counter_ns() - a) / 1e6)
        wall = time.perf_counter() - t_start
        rep = {"n": len(lat), "wall_s": round(wall, 3), "fps": round(len(lat) / wall, 3),
               "mean_ms": round(statistics.mean(lat), 4), "std_ms": round(statistics.pstdev(lat), 4),
               "median_ms": round(statistics.median(lat), 4), "p90_ms": round(pct(lat, 0.90), 4),
               "p99_ms": round(pct(lat, 0.99), 4), "min_ms": round(min(lat), 4), "max_ms": round(max(lat), 4),
               "temp_start_c": temp0, "temp_end_c": temperatures(), "cpu_mhz_end": cpu_freqs_mhz()}
        if tg:
            rep["tegrastats_mw"], rep["tegrastats_samples"] = stop_tegrastats(tg)
        reps.append(rep)
    hwm = re.search(r"VmHWM:\s+(\d+) kB", read("/proc/self/status", ""))
    return {"session_load_s": round(load_s, 3), "repeats": reps,
            "rss_peak_mb": round(int(hwm.group(1)) / 1024, 1) if hwm else None,
            "ort_providers": sess.get_providers()}


# ---- oturum -----------------------------------------------------------------------------------------

def summarize(rec):
    reps = rec["repeats"]
    means = [r["mean_ms"] for r in reps]
    temps = [t for r in reps for t in list(r["temp_start_c"].values()) + list(r["temp_end_c"].values())]
    vdd = [r["tegrastats_mw"].get("VDD_IN") for r in reps if r.get("tegrastats_mw")]
    return {"latency_ms_mean": round(statistics.mean(means), 3),
            "latency_ms_std": round(statistics.stdev(means), 3) if len(means) > 1 else 0.0,
            "latency_ms_median": round(statistics.mean(r["median_ms"] for r in reps), 3),
            "latency_ms_p90": round(statistics.mean(r["p90_ms"] for r in reps), 3),
            "latency_ms_p99": round(statistics.mean(r["p99_ms"] for r in reps), 3),
            "fps_model": round(statistics.mean(r["fps"] for r in reps), 2),
            "n_inferences": sum(r["n"] for r in reps), "rss_peak_mb": rec["rss_peak_mb"],
            "temp_c_max": max(temps) if temps else None,
            "tegra_vdd_in_mw": round(statistics.mean(v for v in vdd if v), 1) if any(vdd) else None}


def rebuild_summary(out_dir, cands):
    rows = []
    for c in cands:
        rec = ns.read_json(os.path.join(out_dir, "runs", f"{c['candidate_id']}.json"))
        if rec:
            rows.append({**c, **rec["summary"], "onnx_sha256": rec["onnx_sha256"],
                         "measured_utc": rec["measured_utc"]})
    nx.write_csv(os.path.join(out_dir, "latency.csv"), rows, SUMMARY_FIELDS)
    return rows


def load_package(pkg):
    cands = nx.read_csv(os.path.join(pkg, "n4_candidates.csv"))
    sums = {}
    for line in open(os.path.join(pkg, "SHA256SUMS")):
        h, name = line.split()
        sums[name] = h
    for c in cands:
        name = f"{c['candidate_id']}.onnx"
        assert name in sums, f"{name} pakette yok"
        c["onnx_path"] = os.path.join(pkg, name)
        c["onnx_sha256"] = sums[name]
        inp = f"inputs/{nx.latency_input_name(c['resolution'])}"
        assert inp in sums, f"{inp} pakette yok (02b hücre 9 gerçek görüntü girdisini ekler)"
        c["input_path"], c["input_sha256"] = os.path.join(pkg, inp), sums[inp]
    return cands


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkg", help="02b paket klasörü (zip açılmış)")
    ap.add_argument("--device", help="ör. jetson_orin_nano_cpu, rpi5")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                                  "results", "nas_r1b", "n4_latency"))
    ap.add_argument("--only", default="", help="virgülle aday kimlikleri (varsayılan: hepsi)")
    ap.add_argument("--smoke", action="store_true", help="kısa deneme: 10 ısınma, 5 sn × 1; <device>_smoke/")
    ap.add_argument("--tegrastats", action="store_true", help="Jetson: kart içi güç rayları (ayrı sütun)")
    ap.add_argument("--run-one", nargs=3, metavar=("ONNX", "INPUT", "RES"), help=argparse.SUPPRESS)
    ap.add_argument("--proto", help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a.run_one:   # alt süreç: tek aday, sonucu stdout'a JSON olarak yazar
        p = json.loads(a.proto)
        print(json.dumps(run_one(a.run_one[0], a.run_one[1], int(a.run_one[2]), p["warmup"], p["duration_s"],
                                 p["repeats"], p["tegrastats"])))
        return

    assert a.pkg and a.device, "--pkg ve --device gerekli"
    proto = {**PROTOCOL, **(SMOKE if a.smoke else {}), "tegrastats": a.tegrastats}
    out_dir = os.path.join(a.out, a.device + ("_smoke" if a.smoke else ""))
    cands = load_package(a.pkg)
    settings = device_settings()
    manifest = ns.ensure_manifest(os.path.join(out_dir, "manifest.json"), {
        "device": a.device, "protocol": proto,
        "candidates": [{"candidate_id": c["candidate_id"], "onnx_sha256": c["onnx_sha256"],
                        "input_sha256": c["input_sha256"]} for c in cands],
        "versions": versions(), "device_settings": settings, "device_info": device_info(),
        "package": os.path.abspath(a.pkg), "script": "scripts/n4_latency.py",
        "repo_commit": sh(f"git -C {os.path.dirname(os.path.abspath(__file__))} rev-parse --short HEAD"),
        "decisions": "Karar 3 (100 ısınma, 60 sn × 3), Karar 5 (seçim RPi 5 + ORT CPU), AGENTS.md (4 thread); "
                     "Jetson CPU ön ölçümü (Furkan, 2026-10-07/09)",
    }, keys=FINGERPRINT_KEYS)

    todo = [c for c in cands if not a.only or c["candidate_id"] in a.only.split(",")]
    random.Random(proto["order_seed"]).shuffle(todo)
    est = len(todo) * (proto["duration_s"] * proto["repeats"] + 15) / 60
    print(f"{a.device}: {len(todo)} aday, sıra tohumu {proto['order_seed']}, en çok ≈{est:.0f} dk · "
          f"ayarlar {settings} · yük {os.getloadavg()} · sıcaklık {temperatures()}", flush=True)
    for i, c in enumerate(todo, 1):
        cid = c["candidate_id"]
        path = os.path.join(out_dir, "runs", f"{cid}.json")
        old = ns.read_json(path)
        if old and old["onnx_sha256"] == c["onnx_sha256"]:
            print(f"· {i}/{len(todo)} {cid}: ölçülmüş, atlandı")
            continue
        assert ns.sha256(c["onnx_path"]) == c["onnx_sha256"], f"{cid}: ONNX SHA256 uyuşmuyor"
        assert ns.sha256(c["input_path"]) == c["input_sha256"], f"{cid}: girdi SHA256 uyuşmuyor"
        now = device_settings()
        assert now == manifest["device_settings"], f"cihaz ayarı değişti: {now} ≠ {manifest['device_settings']}"
        load = os.getloadavg()[0]
        if load > 1.0:
            print(f"  uyarı: 1 dk yük ortalaması {load:.2f} (arka planda iş var mı?)")
        cmd = [sys.executable, os.path.abspath(__file__), "--run-one", c["onnx_path"], c["input_path"],
               c["resolution"],
               "--proto", json.dumps(proto)]
        t0 = time.time()
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(res.stderr[-3000:])
            raise RuntimeError(f"{cid}: alt süreç çıkış kodu {res.returncode}")
        rec = {"candidate_id": cid, "tier": c["tier"], "resolution": int(c["resolution"]),
               "onnx_sha256": c["onnx_sha256"], "input_file": os.path.relpath(c["input_path"], a.pkg),
               "input_sha256": c["input_sha256"], "load_avg_before": load, "order_index": i,
               **json.loads(res.stdout.strip().splitlines()[-1]), "measured_utc": ns.now_utc(),
               "wall_s": round(time.time() - t0, 1)}
        rec["summary"] = summarize(rec)
        ns.write_json(path, rec)
        s = rec["summary"]
        print(f"✔ {i}/{len(todo)} {cid:26s} {s['latency_ms_mean']:8.2f} ± {s['latency_ms_std']:.2f} ms · "
              f"{s['fps_model']:6.1f} FPS · p99 {s['latency_ms_p99']:.2f} · {s['temp_c_max']} °C", flush=True)
        rebuild_summary(out_dir, cands)
    rows = rebuild_summary(out_dir, cands)
    print(f"{len(rows)}/{len(cands)} aday ölçüldü → {os.path.join(out_dir, 'latency.csv')}")


if __name__ == "__main__":
    main()
