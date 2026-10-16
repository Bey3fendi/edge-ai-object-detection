"""Gerçek NAS (Karar 6): arama sonrası aday seçimi (03b) ve sıralama doğrulaması (03c). Saf Python (torch yok).

Aday seçim kuralı (03b → 03c; sonuçlar görülmeden manifeste yazılır):
    L_nano = LUT gecikmesi(Nano@416).
    Pareto seçimi: her hedef oran r ∈ PARETO_TARGETS için, LUT gecikmesi ≤ r × L_nano olan değerlendirilmiş adaylar
        arasından en yüksek uygunluk (fit1k mAP); aynı aday ikinci kez seçilmez (sıradaki en iyi).
    Katmanlı seçim: kalan değerlendirilmiş adaylar (Pareto seçimleri ve Nano hariç) LUT gecikmesine göre sıralanır,
        N_STRATIFIED eşit sayılı dilime bölünür, her dilimden tohumlu rastgele bir aday.
    Nano@416 her zaman eklenir (60 ep referansı ve 1,0 AP sınırının dayanağı).

Sıralama analizi (03c):
    Kendall τ (süper ağ skoru vs. sıfırdan 60 ep val-A mAP), aday paneline koşullu; bootstrap birimi mimari,
    seçim yapısı korunur (grup içinde yeniden örnekleme: pareto / stratified / nano). Benzer gecikmeli çiftlerin
    (|Δ gecikme| ≤ PAIR_LAT_TOL × L_nano) doğru sıralanma oranı ayrıca. Tohum belirsizliği kapsanmaz.
"""

import math
import random

PARETO_TARGETS = (1.00, 0.80, 0.65)     # ÖNERİ (Furkan onayı bekliyor)
N_STRATIFIED = 7
PAIR_LAT_TOL = 0.05                     # ÖNERİ: benzer gecikme = Nano gecikmesinin %5'i içinde
BOOTSTRAP_N = 10_000


def select_candidates(evaluated, nano_id, seed=2026, targets=PARETO_TARGETS, n_strat=N_STRATIFIED):
    """evaluated: {config_id: {"map5095": float, "latency_ms": float}} → [{config_id, group, reason, ...}]"""
    l_nano = evaluated[nano_id]["latency_ms"]
    picked, out = set(), []
    for r in targets:
        pool = sorted(((v["map5095"], -v["latency_ms"], k) for k, v in evaluated.items()
                       if v["latency_ms"] <= r * l_nano and k not in picked and k != nano_id), reverse=True)
        if pool:
            k = pool[0][2]
            picked.add(k)
            out.append({"config_id": k, "group": "pareto", "reason": f"LUT ≤ {r:.2f} × Nano, en yüksek fit1k"})
        else:
            out.append({"config_id": None, "group": "pareto", "reason": f"LUT ≤ {r:.2f} × Nano: aday yok"})
    rest = sorted((v["latency_ms"], k) for k, v in evaluated.items() if k not in picked and k != nano_id)
    rng = random.Random(seed)
    for b in range(n_strat):
        chunk = rest[b * len(rest) // n_strat:(b + 1) * len(rest) // n_strat]
        if chunk:
            k = rng.choice(chunk)[1]
            out.append({"config_id": k, "group": "stratified", "reason": f"gecikme dilimi {b + 1}/{n_strat}"})
    out.append({"config_id": nano_id, "group": "nano", "reason": "referans (Nano@416)"})
    for o in out:
        if o["config_id"]:
            o.update(fit1k=evaluated[o["config_id"]]["map5095"], lut_ms=evaluated[o["config_id"]]["latency_ms"],
                     lut_ratio=round(evaluated[o["config_id"]]["latency_ms"] / l_nano, 4))
    return out


def kendall_tau(a, b):
    n, s = len(a), 0
    for i in range(n):
        for j in range(i + 1, n):
            x = (a[i] - a[j]) * (b[i] - b[j])
            s += 1 if x > 0 else -1 if x < 0 else 0
    return 2 * s / (n * (n - 1)) if n > 1 else float("nan")


def rank_analysis(rows, l_nano, seed=0, n_boot=BOOTSTRAP_N, pair_tol=PAIR_LAT_TOL):
    """rows: [{config_id, group, supernet, scratch, latency_ms}] → τ, grup korumalı bootstrap %95 aralığı, çift doğruluğu."""
    sup = [r["supernet"] for r in rows]
    scr = [r["scratch"] for r in rows]
    tau = kendall_tau(sup, scr)
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault(r["group"], []).append(i)
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        idx = [rng.choice(g) for g in groups.values() for _ in g]
        t = kendall_tau([sup[i] for i in idx], [scr[i] for i in idx])
        if not math.isnan(t):
            boots.append(t)
    boots.sort()
    lo, hi = (boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]) if boots else (None, None)
    pairs, correct = 0, 0
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if abs(rows[i]["latency_ms"] - rows[j]["latency_ms"]) <= pair_tol * l_nano and scr[i] != scr[j]:
                pairs += 1
                correct += (sup[i] - sup[j]) * (scr[i] - scr[j]) > 0
    return {"n": len(rows), "kendall_tau": round(tau, 4), "bootstrap_95": [lo and round(lo, 4), hi and round(hi, 4)],
            "bootstrap_unit": "mimari, grup içinde (pareto/stratified/nano) yeniden örnekleme", "n_boot": n_boot,
            "similar_latency_pairs": pairs, "similar_latency_pair_accuracy": round(correct / pairs, 4) if pairs else None,
            "pair_tolerance": f"|Δ gecikme| ≤ {pair_tol} × L_nano", "scope": "seçilmiş aday paneline koşullu; tohum "
            "belirsizliği kapsanmaz; 60 ep kısmi eğitim referansı (300 ep sıralamasını garanti etmez)"}


def finalist(rows, nano_id, margin, ap_budget=1.0):
    """Karar 6 seçim kuralı. rows: [{config_id, ap_a60, lat_median_ms, lat_p95_ms}]; margin: asgari göreli hızlanma.

    AP_A,60(aday) ≥ AP_A,60(Nano) − ap_budget  VE  medyan gecikme ≤ (1 − margin) × Nano medyanı
    → en düşük medyan gecikme. Aday yoksa None (eşik gevşetilmez).
    """
    nano = next(r for r in rows if r["config_id"] == nano_id)
    ok = [r for r in rows if r["config_id"] != nano_id and r["ap_a60"] >= nano["ap_a60"] - ap_budget
          and r["lat_median_ms"] <= (1 - margin) * nano["lat_median_ms"]]
    best = min(ok, key=lambda r: (r["lat_median_ms"], -r["ap_a60"])) if ok else None
    return {"finalist": best and best["config_id"], "eligible": [r["config_id"] for r in ok],
            "nano": nano, "margin": margin, "ap_budget": ap_budget,
            "verdict": "finalist seçildi" if best else "bu protokolde uygun aday bulunamadı"}


def speed_margin(nano_rep_medians, k=3.0, floor=0.01):
    """Asgari hızlanma payı (aday sonuçları görülmeden, yalnız Nano'nun tekrar ölçümlerinden): k × CV, en az floor."""
    m = sum(nano_rep_medians) / len(nano_rep_medians)
    sd = (sum((x - m) ** 2 for x in nano_rep_medians) / (len(nano_rep_medians) - 1)) ** 0.5 if len(nano_rep_medians) > 1 else 0
    return {"margin": round(max(floor, k * sd / m), 5), "cv": round(sd / m, 6), "k": k, "floor": floor,
            "rule": f"max({floor}, {k} × CV(Nano tekrar medyanları))"}


if __name__ == "__main__":
    rng = random.Random(1)
    ev = {f"c{i}": {"map5095": rng.uniform(10, 26), "latency_ms": rng.uniform(15, 45)} for i in range(300)}
    ev["nano"] = {"map5095": 25.0, "latency_ms": 30.0}
    sel = select_candidates(ev, "nano")
    assert len(sel) == 11 and len({s["config_id"] for s in sel}) == 11, sel
    assert sel[0]["lut_ms"] <= 30 and sel[1]["lut_ms"] <= 24 and sel[2]["lut_ms"] <= 19.5
    rows = [{"config_id": s["config_id"], "group": s["group"], "supernet": s["fit1k"],
             "scratch": s["fit1k"] + rng.gauss(0, 1), "latency_ms": s["lut_ms"]} for s in sel]
    ra = rank_analysis(rows, 30.0, n_boot=2000)
    assert -1 <= ra["kendall_tau"] <= 1 and ra["bootstrap_95"][0] <= ra["bootstrap_95"][1]
    fr = finalist([{"config_id": "nano", "ap_a60": 20.0, "lat_median_ms": 30.0},
                   {"config_id": "a", "ap_a60": 19.2, "lat_median_ms": 24.0},
                   {"config_id": "b", "ap_a60": 18.5, "lat_median_ms": 20.0},
                   {"config_id": "c", "ap_a60": 21.0, "lat_median_ms": 29.9}], "nano", margin=0.02)
    assert fr["finalist"] == "a" and fr["eligible"] == ["a"], fr
    assert finalist([{"config_id": "nano", "ap_a60": 20, "lat_median_ms": 30}], "nano", 0.02)["finalist"] is None
    sm = speed_margin([29.70, 29.72, 29.69])
    print("seçim:", [(s["group"], s["lut_ratio"]) for s in sel])
    print("τ:", ra["kendall_tau"], ra["bootstrap_95"], "çift:", ra["similar_latency_pairs"], ra["similar_latency_pair_accuracy"])
    print("finalist:", fr["finalist"], "| pay:", sm)
    print("RANK_SELECT ÖZ-DENETİMİ GEÇTİ")
