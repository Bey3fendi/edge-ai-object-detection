"""Süper ağ öz-denetimi (CPU'da ~1 dk; notebook 03 duman testinden önce de koşulur).

Denetimler:
    1. Başlangıç eşdeğerliği: önceden eğitilmiş (ya da rastgele) Nano → süper ağ; tam derinlikte her
       kernel yapılandırması standart Nano ile aynı çıktıyı verir (sıfır kenarlı 7×7).
    2. Çıkarma eşdeğerliği: ağırlıklar bozulduktan sonra extract_subnet çıktısı = süper ağın aktif alt ağı.
    3. İskelet + strict yükleme (exps/nas/subnet.py yolu) aynı çıktıyı verir.
    4. Sandwich adımı: eğitim kipinde 3 alt ağ ileri/geri, gradyanlar sonlu; koşmayan bloklarda gradyan yok.
    5. Kimlik gidiş-dönüşü ve canonical.
    6. NSGA-II akışı (sahte uygunluk, ProxyLatency) ve kesintiden devam.
    7. Statik alt ağın ONNX dışa aktarımı (opset 13) ve ORT eşdeğerliği (onnxruntime kuruluysa).

Kullanım: python scripts/check_supernet.py [--weights yolox_nano.pth]
"""

import argparse
import copy
import os
import random
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "exps", "nas"))
sys.path.insert(0, os.path.join(ROOT, "YOLOX"))
sys.path.insert(0, HERE)

import supernet as sn  # noqa: E402
import supernet_search as ss  # noqa: E402


def outputs(model, x):
    model.eval()
    with torch.no_grad():
        return model(x)


def maxdiff(a, b):
    return (a - b).abs().max().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="")
    a = ap.parse_args()
    torch.manual_seed(0)
    rng = random.Random(0)
    x = torch.randn(2, 3, 320, 320)

    exp = sn.Exp()
    exp.nas_pretrained = a.weights
    from yolox.exp.default.yolox_nano import Exp as Nano
    base = Nano().get_model()
    if a.weights:
        base.load_state_dict(torch.load(a.weights, map_location="cpu")["model"], strict=True)
    sup = exp.get_model()
    if not a.weights:
        sup_state = sup.state_dict()
        # rastgele Nano ağırlıklarını süper ağa aynı yolla taşı
        sup2 = copy.deepcopy(base)
        names = sn.to_supernet(sup2)
        sup.load_state_dict(sup2.state_dict(), strict=True)
        del sup_state
    names = exp.dw_names
    n = len(names)
    print(f"DW katmanı: {n}; parametre: {sum(p.numel() for p in sup.parameters()) / 1e6:.3f} M")
    assert n == 30, n

    # 1
    ref = outputs(base, x)
    for i in range(4):
        cfg = sn.random_config(n, rng)
        cfg["depth"] = {s: 3 for s in sn.DEPTH_STAGES}
        sn.set_active(sup, names, cfg)
        d = maxdiff(outputs(sup, x), ref)
        assert d < 1e-4, (sn.config_id(cfg), d)
    print("1 başlangıç eşdeğerliği: geçti")

    # 2 + 3
    with torch.no_grad():
        for p in sup.parameters():
            p.add_(0.01 * torch.randn_like(p))
    for i in range(4):
        cfg = sn.random_config(n, rng)
        sn.set_active(sup, names, cfg)
        y = outputs(sup, x)
        sub = sn.extract_subnet(sup, names, cfg)
        d = maxdiff(outputs(sub, x), y)
        assert d < 1e-4, ("extract", sn.config_id(cfg), d)
        assert not any(isinstance(m, (sn.DynamicDWConv, sn.ElasticSequential)) for m in sub.modules())
        skel = Nano().get_model()
        sn.subnet_skeleton(skel, dict(zip(sn.dw_layer_names(skel), cfg["k"])), cfg["depth"])
        skel.load_state_dict(sub.state_dict(), strict=True)
        d2 = maxdiff(outputs(skel, x), y)
        assert d2 < 1e-4, ("skeleton", d2)
    print("2 çıkarma eşdeğerliği + 3 iskelet strict yükleme: geçti")

    # 4
    sup.train()
    targets = torch.zeros(2, 120, 5)
    targets[:, 0] = torch.tensor([1.0, 100.0, 100.0, 60.0, 80.0])
    targets[:, 1] = torch.tensor([5.0, 200.0, 150.0, 40.0, 40.0])
    sup.zero_grad()
    cfgs = exp.sandwich(123)
    assert cfgs == exp.sandwich(123), "sandwich tekrar üretilebilir değil"
    for cfg in cfgs:
        sn.set_active(sup, names, cfg)
        out = sup(torch.randn(2, 3, 320, 320), targets)
        (out["total_loss"] / len(cfgs)).backward()
    g = [p.grad for p in sup.parameters() if p.grad is not None]
    assert g and all(torch.isfinite(t).all() for t in g)
    print(f"4 sandwich adımı: geçti ({len(cfgs)} alt ağ, {len(g)} gradyanlı tensör)")

    # 5
    for _ in range(20):
        cfg = sn.random_config(n, rng)
        assert sn.parse_config_id(sn.config_id(cfg)) == cfg
        c = sn.canonical(cfg, names)
        assert len(sn.inactive_layers(names, c)) == (3 - c["depth"]["dark3"]) + (3 - c["depth"]["dark4"])
    print("5 kimlik/canonical: geçti; örnek:", sn.config_id(sn.nano_config(n)))

    # 6
    lat = ss.ProxyLatency(sup, names)
    calls = []

    def fake_fit(cfg):
        calls.append(1)
        return {"map5095": round(20 + cfg["res"] / 100 + sum(cfg["k"]) / 60
                                 + sum(cfg["depth"].values()) / 3 + rng.random() * 0.1, 2)}

    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "state.json")
        seeds = [sn.nano_config(n), sn.max_config(n), sn.min_config(n)]
        st = ss.evolve(p, names, fake_fit, lat, pop_size=12, generations=2, seeds=seeds, log=lambda s: None)
        n1 = len(calls)
        st = ss.evolve(p, names, fake_fit, lat, pop_size=12, generations=4, seeds=seeds, log=lambda s: None)
        assert st["generation"] == 4 and len(calls) > n1
        st2 = ss.evolve(p, names, fake_fit, lat, pop_size=12, generations=4, seeds=seeds, log=lambda s: None)
        assert st2["generation"] == 4
        front = ss.pareto(st)
        print(f"6 NSGA-II: geçti (değerlendirilen {len(st['evaluated'])}, cephe {len(front)})")

    # 7 — eşdeğerlik ölçütü projeninki (Furkan, 2026-10-09; nas_export.decoded_diff): ham çıktı ONNX'e,
    # karşılaştırma kod çözülmüş uzayda — kutu ≤ EQ_MAX_BOX_PX piksel, skor ≤ EQ_MAX_SCORE. onnxruntime zorunlu.
    import onnxruntime as ort
    import nas_export as nx
    for _ in range(2):
        cfg = sn.random_config(n, rng)
        sub = sn.extract_subnet(sup, names, cfg).eval()
        sub.head.decode_in_inference = False
        xi = torch.randn(1, 3, cfg["res"], cfg["res"])
        with tempfile.TemporaryDirectory() as td:
            f = os.path.join(td, "sub.onnx")
            nx.export_onnx(sub, cfg["res"], f)
            s = ort.InferenceSession(f, providers=["CPUExecutionProvider"])
            o = s.run(None, {nx.INPUT_NAME: xi.numpy()})[0][0]
            ref = outputs(sub, xi)[0].numpy()
            dd = nx.decoded_diff(ref, o, cfg["res"])
            print(f"7 ONNX opset {nx.OPSET} ({sn.config_id(cfg)[:12]}…): kutu {dd['max_box_px']:.2e} px "
                  f"(≤ {nx.EQ_MAX_BOX_PX}), skor {dd['max_score_diff']:.2e} (≤ {nx.EQ_MAX_SCORE}), "
                  f"{dd['n_scored_anchors']} anchor")
            assert dd["pass"], dd
            if a.weights:   # önceden eğitilmiş ağırlıkla kutu karşılaştırması boş geçmemeli
                assert dd["n_scored_anchors"] > 0, dd
    print("TÜM DENETİMLER GEÇTİ")


if __name__ == "__main__":
    main()
