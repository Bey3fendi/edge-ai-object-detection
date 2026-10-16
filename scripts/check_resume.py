"""Kesintiye dayanıklılık öz-denetimi (CPU, sentetik COCO; ~1 dk). GPU'lu uçtan uca test notebook 03 hücre 7'de.

Denetimler:
    1. ResumableBatchSampler: (epoch, iter)'den devam eden akış kesintisiz akışla birebir aynı; mosaic bayrağı
       epoch'un fonksiyonu; epoch'lar farklı sırada.
    2. Örnek başına tohumlu artırma: aynı (epoch, konum) aynı görüntü/etiketi verir — sıra, worker sayısı ve
       devam noktasından bağımsız (DataLoader 0 worker vs 3 worker, baştan vs ortadan batch hash'leri).
    3. Çok ölçek: size_at(p), sıralı random_resize çağrılarıyla aynı.
    4. CheckpointStore: doğrulanmış kayıt, yazım sırasında öldürme (alt süreç, NAS_TEST_KILL_AT_SAVE) →
       önceki sağlam kayıt seçilir; bozulmuş son kayıt atlanır; budama sabitlenmiş kayıtları korur.
"""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "YOLOX"))
import nas_data as nd  # noqa: E402
from ckpt_store import CheckpointStore  # noqa: E402


def synthetic_coco(root, n=24, seed=0):
    import cv2
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.join(root, "annotations"), exist_ok=True)
    os.makedirs(os.path.join(root, "train2017"), exist_ok=True)
    images, anns, aid = [], [], 1
    for i in range(n):
        h, w = int(rng.integers(300, 500)), int(rng.integers(300, 600))
        img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
        fn = f"{i:012d}.jpg"
        cv2.imwrite(os.path.join(root, "train2017", fn), img)
        images.append({"id": i + 1, "file_name": fn, "height": h, "width": w})
        for _ in range(int(rng.integers(1, 4))):
            bw, bh = float(rng.integers(20, w // 2)), float(rng.integers(20, h // 2))
            x, y = float(rng.integers(0, w - bw)), float(rng.integers(0, h - bh))
            anns.append({"id": aid, "image_id": i + 1, "category_id": int(rng.integers(1, 4)),
                         "bbox": [x, y, bw, bh], "area": bw * bh, "iscrowd": 0})
            aid += 1
    cats = [{"id": c, "name": f"c{c}", "supercategory": "x"} for c in (1, 2, 3)]
    with open(os.path.join(root, "annotations", "train.json"), "w") as f:
        json.dump({"images": images, "annotations": anns, "categories": cats}, f)


def build_ds(root, seed):
    from yolox.data import COCODataset, MosaicDetection, TrainTransform
    base = COCODataset(data_dir=root, json_file="train.json", name="train2017", img_size=(256, 256),
                       preproc=TrainTransform(max_labels=50, flip_prob=0.5, hsv_prob=1.0))
    ds = nd.make_seeded_mosaic(MosaicDetection)(
        dataset=base, mosaic=True, img_size=(256, 256),
        preproc=TrainTransform(max_labels=120, flip_prob=0.5, hsv_prob=1.0), degrees=10.0, translate=0.1,
        mosaic_scale=(0.5, 1.5), mixup_scale=(0.5, 1.5), shear=2.0, enable_mixup=True, mosaic_prob=0.5,
        mixup_prob=1.0)
    ds.run_seed = seed
    return ds


def digests(ds, sampler, n_batches, workers):
    from yolox.data import DataLoader
    dl = DataLoader(ds, batch_sampler=sampler, num_workers=workers, pin_memory=False)
    out, it = [], iter(dl)
    for _ in range(n_batches):
        x, t, _, _ = next(it)
        out.append(nd.batch_digest(x, t))
    return out


KILL_CHILD = r"""
import sys, torch
sys.path.insert(0, {here!r})
from ckpt_store import CheckpointStore
s = CheckpointStore({root!r}, {local!r})
s.save({{"w": torch.arange(200000.0), "p": 30}}, 30)
"""


def main():
    # 1
    S = lambda e=0, i=0: nd.ResumableBatchSampler(103, 8, 7, max_epoch=5, no_aug_epochs=1, start_epoch=e, start_iter=i)
    full = [b for _, b in zip(range(40), iter(S()))]
    res = [b for _, b in zip(range(40 - 17), iter(S(1, 17 - 12)))]
    assert S().iters_per_epoch == 12 and full[17:] == res, "devam akışı farklı"
    assert all(b[0][0] for b in full[:36]) and not any(b[0][0] for b in full[36:48]), "mosaic bayrağı"
    assert [x[1] for x in full[0]] != [x[1] for x in full[12]]
    print("1 örnekleyici devamı: geçti")

    with tempfile.TemporaryDirectory() as td:
        synthetic_coco(td)
        ds = build_ds(td, seed=11)
        # 2
        a = digests(ds, nd.ResumableBatchSampler(len(ds), 4, 11, 5, 1), 10, 0)
        b = digests(ds, nd.ResumableBatchSampler(len(ds), 4, 11, 5, 1), 10, 3)
        assert a == b, "worker sayısı artırmayı değiştirdi"
        c = digests(ds, nd.ResumableBatchSampler(len(ds), 4, 11, 5, 1, start_epoch=1, start_iter=1), 3, 2)
        assert c == a[len(ds) // 4 + 1:len(ds) // 4 + 4], "sonraki epoch ortasından devam farklı"
        c2 = digests(ds, nd.ResumableBatchSampler(len(ds), 4, 11, 5, 1, start_iter=3), 4, 2)
        assert c2 == a[3:7], "ortadan devam eden batch'ler farklı"
        d = digests(build_ds(td, seed=12), nd.ResumableBatchSampler(len(ds), 4, 11, 5, 1), 3, 0)
        assert d[0] != a[0], "artırma tohumu etkisiz"
        print(f"2 tohumlu artırma: geçti (0/3 worker eşit, ortadan devam eşit; {len(ds)} görsel)")

    # 3
    rs = (10, 20)
    sizes, cur = [], (416, 416)
    for p in range(200):
        sizes.append(cur)
        if (p + 1) % 10 == 0:
            cur = nd.resize_size(5, p, rs)
    assert all(nd.size_at(5, p, rs, (416, 416)) == sizes[p] for p in range(200))
    print("3 çok ölçek devamı: geçti")

    # 4
    with tempfile.TemporaryDirectory() as td:
        root, local = os.path.join(td, "drive"), os.path.join(td, "local")
        st = CheckpointStore(root, local, keep=2, log=lambda s: None)
        st.save({"p": 10}, 10, pinned=True)
        st.save({"p": 20}, 20)
        env = {**os.environ, "NAS_TEST_KILL_AT_SAVE": "30"}
        r = subprocess.run([sys.executable, "-c", KILL_CHILD.format(here=HERE, root=root, local=local)],
                           env=env, capture_output=True, text=True)
        assert r.returncode == 137, (r.returncode, r.stderr[-500:])
        assert os.path.exists(os.path.join(root, "ck_0000000030.pth.partial"))
        state, rec = CheckpointStore(root, local, log=lambda s: None).latest()
        assert state["p"] == 20, "yazım sırasında kesintiden sonra önceki sağlam kayıt seçilmedi"
        st.save({"p": 40}, 40)
        with open(os.path.join(root, "ck_0000000040.pth"), "r+b") as f:
            f.seek(10)
            f.write(b"\x00bozuk\x00")
        state, rec = CheckpointStore(root, local, log=lambda s: None).latest()
        assert state["p"] == 20, "bozuk kayıt atlanmadı"
        st.save({"p": 50}, 50)
        st.save({"p": 60}, 60)
        progs = [r["progress"] for r in st.records()]
        assert 10 in progs and 20 not in progs and progs[-2:] == [50, 60], progs
        print(f"4 kayıt deposu: geçti (kesinti → önceki sağlam; bozuk atlandı; kalanlar {progs})")
    print("TÜM KESİNTİ DENETİMLERİ GEÇTİ")


if __name__ == "__main__":
    main()
