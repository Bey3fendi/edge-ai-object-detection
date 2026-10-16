"""Gerçek NAS (Karar 6): süper ağdan alt ağ değerlendirme, gecikme tahmini, evrimsel arama, alt ağ çıkarma.

notebooks/03*_gercek_nas_*.ipynb tarafından kullanılır; model tanımı exps/nas/supernet.py.

Alt ağ uygunluk değeri (fitness):
    1. Süper ağ (EMA ağırlıkları) seçili yapılandırmaya ayarlanır.
    2. BN istatistikleri sıfırlanır ve train2017 ayrımının bn2k kümesinden (2000 görsel, seçili
       çözünürlükte, ValTransform) kümülatif ortalamayla yeniden hesaplanır — paylaşımlı BN farklı
       alt ağların karışık istatistiklerini taşır, kalibrasyonsuz mAP anlamsızdır (OFA/BigNAS).
    3. fit1k kümesinde (1000 görsel) COCO mAP@[.5:.95], fp16, conf 0,001, NMS 0,65.
val2017 aramada hiç kullanılmaz (scripts/make_holdout.py).

Arama: NSGA-II (çok amaçlı evrim; amaçlar: mAP ↑, tahmini cihaz gecikmesi ↓). Durum her nesilde atomik
JSON'a yazılır; kesintiden sonra kaldığı nesilden devam eder, değerlendirilmiş alt ağ tekrar ölçülmez.
"""

import copy
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exps", "nas"))
sys.path.insert(0, HERE)
import supernet as sn  # noqa: E402
from nas_search import now_utc, read_json, write_json  # noqa: E402

FIT_ANN = "instances_train2017_nas_fit1k.json"
BN_ANN = "instances_train2017_nas_bn2k.json"
HOLDOUT_ANN = "instances_train2017_nas5k.json"


# ---- süper ağ yükleme --------------------------------------------------------------------------

def load_supernet(ckpt_path, device="cuda"):
    """Süper ağ exp'i + checkpoint (YOLOX save_ckpt: "model" = EMA ağırlıkları)."""
    exp = sn.Exp()
    model = exp.get_model()
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model"] if "model" in state else state, strict=True)
    return exp, model.to(device).eval()


# ---- veri ---------------------------------------------------------------------------------------

def make_loader(data_dir, ann, res, batch_size=64, workers=8, limit=None):
    from yolox.data import COCODataset, ValTransform
    ds = COCODataset(data_dir=data_dir, json_file=ann, name="train2017", img_size=(res, res),
                     preproc=ValTransform(legacy=False))
    if limit is not None:
        ds = torch.utils.data.Subset(ds, range(min(limit, len(ds))))
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers,
                                       pin_memory=True)


class Evaluator:
    """Çözünürlük başına önbellekli BN ve uygunluk yükleyicileri."""

    def __init__(self, data_dir, fit_ann=FIT_ANN, bn_ann=BN_ANN, batch_size=64, workers=8,
                 bn_images=2000, device="cuda", half=True):
        self.data_dir, self.fit_ann, self.bn_ann = data_dir, fit_ann, bn_ann
        self.bs, self.workers, self.bn_images = batch_size, workers, bn_images
        self.device, self.half = device, half
        self._fit, self._bn = {}, {}

    def fit_evaluator(self, res):
        if res not in self._fit:
            from yolox.evaluators import COCOEvaluator
            dl = make_loader(self.data_dir, self.fit_ann, res, self.bs, self.workers)
            self._fit[res] = COCOEvaluator(dataloader=dl, img_size=(res, res), confthre=0.001,
                                           nmsthre=0.65, num_classes=80)
        return self._fit[res]

    def bn_loader(self, res):
        if res not in self._bn:
            self._bn[res] = make_loader(self.data_dir, self.bn_ann, res, self.bs, self.workers,
                                        limit=self.bn_images)
        return self._bn[res]


@torch.no_grad()
def recalibrate_bn(model, loader, device="cuda"):
    """BN koşu istatistiklerini sıfırlayıp yükleyicideki tüm görsellerle kümülatif ortalama alır."""
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    saved = [(m.momentum,) for m in bns]
    model.eval()             # YOLOX eğitim kipinde hedef ister; yalnız BN'ler istatistik toplar
    for m in bns:
        m.reset_running_stats()
        m.momentum = None    # kümülatif ortalama
        m.train()
    for batch in loader:
        x = batch[0].to(device).float()
        model(x)
    for m, (mom,) in zip(bns, saved):
        m.momentum = mom
    model.eval()


def evaluate_config(supernet, names, cfg, ev, log=print):
    """Süper ağın kopyasında alt ağı BN kalibrasyonuyla ölçer; {map5095, ap50, ...} döndürür."""
    t0 = time.time()
    net = copy.deepcopy(supernet)
    sn.set_active(net, names, cfg)
    res = int(cfg["res"])
    recalibrate_bn(net, ev.bn_loader(res), ev.device)
    t_bn = time.time() - t0
    if ev.half:
        net = net.half()
    ap, ap50, _ = ev.fit_evaluator(res).evaluate(net, False, ev.half)
    out = {"map5095": round(ap * 100, 2), "ap50": round(ap50 * 100, 2),
           "bn_s": round(t_bn, 1), "eval_s": round(time.time() - t0, 1), "evaluated_utc": now_utc()}
    log(f"[eval] {sn.config_id(cfg)} → mAP {out['map5095']} ({out['eval_s']} sn)")
    del net
    return out


# ---- gecikme tahmini ----------------------------------------------------------------------------

class LatencyModel:
    """Cihazda ölçülmüş LUT'tan alt ağ gecikmesi (ms).

    LUT biçimi (scripts/latency_lut.py measure çıktısı):
        {"device": ..., "fixed": {"<res>": ms},             # aranmayan kısımlar (kernel 3, derinlik 1)
         "dw": {"<res>": {"<katman>": {"3": ms, "5": ms, "7": ms}}},
         "block": {"<res>": {"<stage>.<i>": ms}},            # derinlik 1'in üstündeki her Bottleneck
         "calib": {"a": 1.0, "b": 0.0}}                       # uçtan uca doğrulamadan doğrusal düzeltme
    Tahmin = a × (fixed + Σ dw[k] − Σ dw[3] (aktif katmanlar) + Σ aktif ek bloklar) + b.
    """

    def __init__(self, lut):
        self.lut = lut
        self.a = lut.get("calib", {}).get("a", 1.0)
        self.b = lut.get("calib", {}).get("b", 0.0)

    @classmethod
    def load(cls, path):
        return cls(read_json(path))

    def raw(self, cfg, names):
        r = str(cfg["res"])
        off = sn.inactive_layers(names, cfg)
        t = self.lut["fixed"][r]
        for n, k in zip(names, cfg["k"]):
            if n not in off:
                d = self.lut["dw"][r][n]
                t += d[str(k)] - d["3"]
        for stage in sn.DEPTH_STAGES:
            for i in range(1, cfg["depth"][stage]):
                t += self.lut["block"][r][f"{stage}.{i}"]
        return t

    def __call__(self, cfg, names):
        return self.a * self.raw(cfg, names) + self.b


class ProxyLatency:
    """LUT yokken yer tutucu: depthwise MAC'lerin çözünürlükle ölçeklenmiş toplamı (birim yok).

    Yalnız akış testi içindir; gerçek arama LatencyModel ile yapılır (karar 6: cihaz gecikmesi).
    """

    def __init__(self, model, names):
        self.base = {}
        with torch.no_grad():
            hooks, shapes = [], {}
            for n in names:
                m = model.get_submodule(n)
                hooks.append(m.register_forward_hook(
                    lambda mod, i, o, n=n: shapes.__setitem__(n, (o.shape[1], o.shape[2], o.shape[3]))))
            sn.set_active(model, names, sn.max_config(len(names)))
            was = model.training
            model.eval()
            p = next(model.parameters())
            model(torch.zeros(1, 3, 416, 416, device=p.device, dtype=p.dtype))
            model.train(was)
            for h in hooks:
                h.remove()
        self.shapes = shapes

    def __call__(self, cfg, names):
        s = (cfg["res"] / 416) ** 2
        off = sn.inactive_layers(names, cfg)
        mac = 0.0
        for n, k in zip(names, cfg["k"]):
            if n not in off:
                c, h, w = self.shapes[n]
                mac += c * h * w * k * k
        return 1e-6 * s * mac * (1 + 0.1 * sum(cfg["depth"].values()))


# ---- NSGA-II ------------------------------------------------------------------------------------

def dominates(a, b):
    """a, b = (mAP, gecikme); mAP büyük, gecikme küçük iyi."""
    return a[0] >= b[0] and a[1] <= b[1] and (a[0] > b[0] or a[1] < b[1])


def nondominated_sort(objs):
    n = len(objs)
    S = [[] for _ in range(n)]
    cnt = [0] * n
    fronts = [[]]
    for i in range(n):
        for j in range(n):
            if i != j:
                if dominates(objs[i], objs[j]):
                    S[i].append(j)
                elif dominates(objs[j], objs[i]):
                    cnt[i] += 1
        if cnt[i] == 0:
            fronts[0].append(i)
    k = 0
    while fronts[k]:
        nxt = []
        for i in fronts[k]:
            for j in S[i]:
                cnt[j] -= 1
                if cnt[j] == 0:
                    nxt.append(j)
        k += 1
        fronts.append(nxt)
    return fronts[:-1]


def crowding(objs, front):
    dist = {i: 0.0 for i in front}
    for m in range(2):
        order = sorted(front, key=lambda i: objs[i][m])
        lo, hi = objs[order[0]][m], objs[order[-1]][m]
        dist[order[0]] = dist[order[-1]] = math.inf
        if hi == lo:
            continue
        for a, b, c in zip(order, order[1:], order[2:]):
            dist[b] += (objs[c][m] - objs[a][m]) / (hi - lo)
    return dist


def select(pop, objs, k):
    """NSGA-II çevresel seçim: cephe sırası, sonra kalabalık mesafesi."""
    out = []
    for front in nondominated_sort(objs):
        if len(out) + len(front) <= k:
            out += front
        else:
            d = crowding(objs, front)
            out += sorted(front, key=lambda i: -d[i])[:k - len(out)]
            break
    return [pop[i] for i in out]


def mutate(cfg, rng, p_k=0.1, p_d=0.2, p_r=0.2):
    c = copy.deepcopy(cfg)
    c["k"] = [rng.choice(sn.KERNELS) if rng.random() < p_k else k for k in c["k"]]
    for s in sn.DEPTH_STAGES:
        if rng.random() < p_d:
            c["depth"][s] = rng.choice(sn.DEPTHS)
    if rng.random() < p_r:
        c["res"] = rng.choice(sn.RESOLUTIONS)
    return c


def crossover(a, b, rng):
    return {"res": rng.choice((a["res"], b["res"])),
            "depth": {s: rng.choice((a["depth"][s], b["depth"][s])) for s in sn.DEPTH_STAGES},
            "k": [rng.choice(pair) for pair in zip(a["k"], b["k"])]}


def evolve(state_path, names, fitness, latency, pop_size=64, generations=15, seed=0,
           latency_max=None, seeds=(), log=print):
    """NSGA-II; durum `state_path`'te. fitness(cfg) → dict (map5095 içerir), latency(cfg) → ms.

    latency_max verilirse bunu aşan aday değerlendirilmeden atılır (bütçe dışı). `seeds` başlangıç
    nüfusuna konan yapılandırmalar (ör. dokunulmamış Nano, en büyük/en küçük).
    """
    st = read_json(state_path) or {"seed": seed, "pop_size": pop_size, "generation": -1,
                                   "evaluated": {}, "population": [], "history": [],
                                   "created_utc": now_utc()}
    rng = random.Random(f"{st['seed']}-{st['generation']}")
    ev = st["evaluated"]

    def key(cfg):
        return sn.config_id(sn.canonical(cfg, names))

    def ok(cfg):
        return latency_max is None or latency(cfg, names) <= latency_max

    def score(cfgs):
        for c in cfgs:
            cid = key(c)
            if cid not in ev:
                r = fitness(sn.parse_config_id(cid))
                r["latency_ms"] = round(latency(sn.parse_config_id(cid), names), 3)
                ev[cid] = r
                st["evaluated"] = ev
                write_json(state_path, st)   # her değerlendirmeden sonra: kesinti en çok 1 aday kaybettirir

    if st["generation"] < 0:
        init = [sn.canonical(c, names) for c in seeds]
        while len(init) < pop_size:
            c = sn.canonical(sn.random_config(len(names), rng), names)
            if ok(c) and key(c) not in {key(x) for x in init}:
                init.append(c)
        st["population"] = [key(c) for c in init]
        score(init)
        st["generation"] = 0
        write_json(state_path, st)

    while st["generation"] < generations:
        g = st["generation"] + 1
        rng = random.Random(f"{st['seed']}-{g}")
        parents = [sn.parse_config_id(c) for c in st["population"]]
        children, seen = [], set(st["population"])
        tries = 0
        while len(children) < pop_size // 2 and tries < pop_size * 50:
            tries += 1
            a, b = rng.sample(parents, 2)
            c = sn.canonical(mutate(crossover(a, b, rng), rng), names)
            if ok(c) and key(c) not in seen:
                seen.add(key(c))
                children.append(c)
        score(children)
        pool = st["population"] + [key(c) for c in children]
        objs = [(ev[c]["map5095"], ev[c]["latency_ms"]) for c in pool]
        st["population"] = select(pool, objs, pop_size)
        front = [pool[i] for i in nondominated_sort(objs)[0]]
        st["history"].append({"generation": g, "evaluated_total": len(ev), "front_size": len(front),
                              "best_map": max(ev[c]["map5095"] for c in pool), "utc": now_utc()})
        st["generation"] = g
        write_json(state_path, st)
        log(f"[nsga] nesil {g}/{generations}: değerlendirilen {len(ev)}, cephe {len(front)}")
    return st


def pareto(state):
    ids = list(state["evaluated"])
    objs = [(state["evaluated"][c]["map5095"], state["evaluated"][c]["latency_ms"]) for c in ids]
    return sorted((ids[i] for i in nondominated_sort(objs)[0]),
                  key=lambda c: state["evaluated"][c]["latency_ms"])


# ---- alt ağ çıkarma -----------------------------------------------------------------------------

def save_subnet(supernet, names, cfg, path, bn_loader=None, device="cuda"):
    """Statik alt ağı (BN kalibre edilmiş) {"model": state_dict, "arch": id} olarak kaydeder."""
    sub = sn.extract_subnet(supernet, names, cfg)
    if bn_loader is not None:
        recalibrate_bn(sub, bn_loader, device)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save({"model": sub.state_dict(), "arch": sn.config_id(cfg)}, tmp)
    os.replace(tmp, path)
    return sub
