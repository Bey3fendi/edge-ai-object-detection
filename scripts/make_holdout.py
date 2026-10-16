"""Gerçek NAS (Karar 6) için train2017'den arama ayrımı.

Arama uygunluk değeri val2017'de ölçülürse son raporlanan mAP aramaya sızar; bu yüzden train2017'den
5000 görsel ayrılır, süper ağ geri kalanla eğitilir, val2017 yalnız son ince ayar sonrası raporlanır.

Çıktılar (<COCO>/annotations/ altına; görüntüler train2017/ klasöründe kalır):
    instances_train2017_minus_nas5k.json   süper ağ eğitimi (113 287 görsel)
    instances_train2017_nas5k.json         ayrılan 5000 (en iyi adayların son sıralaması)
    instances_train2017_nas_fit1k.json     5000'in ilk 1000'i: evrimsel aramanın uygunluk kümesi
    instances_train2017_nas_bn2k.json      5000'in sonraki 2000'i: alt ağ BN yeniden kalibrasyonu
    nas_holdout_manifest.json              tohum, sayılar, sha256

Seçim: tohumlu rastgele (random.Random(seed).sample), görüntü kimlikleri sıralı listeden; anotasyonsuz
görüntüler de havuzda (val2017'de olduğu gibi). Aynı tohum aynı ayrımı üretir.

Kullanım: python scripts/make_holdout.py --coco /content/datasets/COCO [--seed 2026]
"""

import argparse
import hashlib
import json
import os
import random

N_HOLDOUT, N_FIT, N_BN = 5000, 1000, 2000


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def split_ids(all_ids, seed):
    rng = random.Random(seed)
    hold = rng.sample(sorted(all_ids), N_HOLDOUT)
    return hold, hold[:N_FIT], hold[N_FIT:N_FIT + N_BN]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", required=True)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    ann = os.path.join(a.coco, "annotations")
    src = os.path.join(ann, "instances_train2017.json")
    with open(src, encoding="utf-8") as f:
        data = json.load(f)
    all_ids = [im["id"] for im in data["images"]]
    hold, fit, bn = split_ids(all_ids, a.seed)
    rest = sorted(set(all_ids) - set(hold))
    outs = {"minus_nas5k": rest, "nas5k": hold, "nas_fit1k": fit, "nas_bn2k": bn}
    manifest = {"source": "instances_train2017.json", "source_sha256": sha256(src), "seed": a.seed,
                "method": "random.Random(seed).sample(sorted(image_ids), 5000); fit=ilk 1000, bn=sonraki 2000",
                "files": {}}
    for tag, ids in outs.items():
        path = os.path.join(ann, f"instances_train2017_{tag}.json")
        sub = subset(data, ids)
        write(path, sub)
        manifest["files"][os.path.basename(path)] = {
            "images": len(sub["images"]), "annotations": len(sub["annotations"]), "sha256": sha256(path)}
        print(tag, len(sub["images"]), "görsel", len(sub["annotations"]), "anotasyon")
    assert manifest["files"]["instances_train2017_minus_nas5k.json"]["images"] + N_HOLDOUT == len(all_ids)
    write(os.path.join(ann, "nas_holdout_manifest.json"), manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
