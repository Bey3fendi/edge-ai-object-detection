"""Gerçek NAS veri ayrımları (Karar 6, 2026-10-16).

Neden: önceden eğitilmiş YOLOX-Nano train2017'nin tamamıyla eğitildi; train2017'den ayrılan görseller bu
ağırlıklar için görülmemiş değildir. Bu yüzden arama/geliştirme val2017'nin yarısında (A), nihai değerlendirme
diğer yarısında (B; ya da erişilebilirse COCO test-dev) yapılır. B, NAS + budama + nicemleme yöntem kararları
bitene kadar açılmaz. Not: B önçalışmada (nas_r1b/nas_r2, tam val2017) dolaylı kullanıldı; tezde yazılır.

Çıktılar (<COCO>/annotations/):
    instances_val2017_A.json        2500 görsel — arama, checkpoint/epoch/mimari/budama/INT8 kararları
    instances_val2017_A_fit1k.json  A'nın ilk 1000'i (A içindeki tohumlu sırayla) — evrimsel arama uygunluğu
    instances_val2017_B.json        2500 görsel — mühürlü nihai değerlendirme
    instances_train2017_bn2k.json   train2017'den 2000 görsel — alt ağ BN yeniden kalibrasyonu
    nas_splits_manifest.json        tohum, kimlik listelerinin sha256'sı, dosya sha256'ları, dağılım denetimi

Dağılım denetimi: sınıf başına örnek payı ve nesne boyutu payı (COCO: küçük < 32², orta < 96², büyük),
A/B ile tam val2017 arasındaki en büyük mutlak fark (yüzde puan). Basit tohumlu rastgele bölme; katmanlama yok
(sonuç raporlanır, eşik yok).

Kullanım: python scripts/make_splits.py --coco /content/datasets/COCO [--seed 2026]
"""

import argparse
import hashlib
import json
import os
import random
from collections import Counter

N_A, N_FIT, N_BN = 2500, 1000, 2000


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ids_sha(ids):
    return hashlib.sha256(",".join(str(i) for i in ids).encode()).hexdigest()


def subset(data, ids):
    keep = set(ids)
    return {**{k: v for k, v in data.items() if k not in ("images", "annotations")},
            "images": [im for im in data["images"] if im["id"] in keep],
            "annotations": [a for a in data["annotations"] if a["image_id"] in keep]}


def write(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def split_val(val_ids, seed):
    order = random.Random(f"{seed}-val").sample(sorted(val_ids), len(val_ids))
    a, b = order[:N_A], order[N_A:]
    return a, a[:N_FIT], b


def pick_bn(train_ids, seed):
    return random.Random(f"{seed}-bn").sample(sorted(train_ids), N_BN)


def distribution(data, ids):
    keep = set(ids)
    anns = [a for a in data["annotations"] if a["image_id"] in keep and not a.get("iscrowd", 0)]
    n = max(1, len(anns))
    cls = Counter(a["category_id"] for a in anns)
    size = Counter("small" if a["area"] < 32 ** 2 else "medium" if a["area"] < 96 ** 2 else "large" for a in anns)
    return {c: 100 * v / n for c, v in cls.items()}, {s: 100 * v / n for s, v in size.items()}, len(anns)


def max_dev(part, full):
    keys = set(part) | set(full)
    return round(max(abs(part.get(k, 0) - full.get(k, 0)) for k in keys), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", required=True)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    ann = os.path.join(a.coco, "annotations")
    vsrc, tsrc = os.path.join(ann, "instances_val2017.json"), os.path.join(ann, "instances_train2017.json")
    val = json.load(open(vsrc, encoding="utf-8"))
    val_ids = [im["id"] for im in val["images"]]
    A, fit, B = split_val(val_ids, a.seed)
    assert len(set(A) & set(B)) == 0 and len(A) + len(B) == len(val_ids) == 5000
    full_c, full_s, full_n = distribution(val, val_ids)
    check = {}
    for tag, ids in (("A", A), ("B", B), ("A_fit1k", fit)):
        c, s, n = distribution(val, ids)
        check[tag] = {"instances": n, "size_pct": {k: round(v, 2) for k, v in s.items()},
                      "max_class_share_dev_pp": max_dev(c, full_c), "max_size_share_dev_pp": max_dev(s, full_s)}
    check["full_val2017"] = {"instances": full_n, "size_pct": {k: round(v, 2) for k, v in full_s.items()}}
    train = json.load(open(tsrc, encoding="utf-8"))
    bn = pick_bn([im["id"] for im in train["images"]], a.seed)
    manifest = {"seed": a.seed, "method": "random.Random(f'{seed}-val').sample(sorted(val_ids)); A=ilk 2500, "
                "A_fit1k=A'nın ilk 1000'i, B=kalan 2500; bn2k=random.Random(f'{seed}-bn').sample(train_ids, 2000)",
                "val_source_sha256": sha256(vsrc), "train_source_sha256": sha256(tsrc),
                "ids_sha256": {"A": ids_sha(A), "A_fit1k": ids_sha(fit), "B": ids_sha(B), "bn2k": ids_sha(bn)},
                "distribution_check": check, "files": {}}
    outs = (("instances_val2017_A.json", val, A), ("instances_val2017_A_fit1k.json", val, fit),
            ("instances_val2017_B.json", val, B), ("instances_train2017_bn2k.json", train, bn))
    for name, src, ids in outs:
        path = os.path.join(ann, name)
        sub = subset(src, ids)
        write(path, sub)
        manifest["files"][name] = {"images": len(sub["images"]), "annotations": len(sub["annotations"]),
                                   "sha256": sha256(path)}
    write(os.path.join(ann, "nas_splits_manifest.json"), manifest)
    print(json.dumps({k: manifest[k] for k in ("seed", "distribution_check", "files")}, indent=1))


if __name__ == "__main__":
    main()
