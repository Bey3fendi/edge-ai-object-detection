"""Doğrulanmış checkpoint deposu (Drive): sağlam kayıt hiçbir zaman ezilmez.

Yazım:
    1. Durum yerel diske `ck_<progress>.pth.tmp` olarak yazılır, fsync, sha256 + boyut hesaplanır.
    2. Depoya (Drive) benzersiz adla `ck_<progress:010d>.pth.partial` kopyalanır, sonra `ck_<...>.pth`'ye
       yeniden adlandırılır (aynı ad daha önce varsa — yarım kalmış eski deneme — üstüne yazılır; o dosyanın
       .ok.json'u yoksa zaten geçerli sayılmamıştır).
    3. Depodaki dosya geri okunur: boyut, sha256 ve torch.load denetimi.
    4. Ancak bundan sonra `ck_<...>.ok.json` (sha256, boyut, meta) yazılır → kayıt "doğrulanmış" olur.
    5. Budama: yalnız yeni kayıt doğrulandıktan sonra; son `keep` doğrulanmış kayıt + `pinned` (anlık görüntü)
       kayıtlar kalır.
Devam: .ok.json'lar ilerlemeye göre azalan sırayla; dosyası var, boyutu ve sha256'sı tutan ve yüklenebilen
ilk kayıt seçilir. Bozuk/yarım kayıtlar atlanır ve raporlanır.

Test kancası: NAS_TEST_KILL_AT_SAVE=<progress> ortam değişkeni, o ilerlemenin kaydı depoya yarım
kopyalanırken süreci os._exit(137) ile öldürür (checkpoint yazılırken kesinti testi).
"""

import hashlib
import json
import os
import shutil
import time

import torch


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path, obj):
    tmp = path + ".partial"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class CheckpointStore:
    def __init__(self, root, local_dir, keep=3, log=print):
        self.root, self.local_dir, self.keep, self.log = root, local_dir, int(keep), log
        os.makedirs(root, exist_ok=True)
        os.makedirs(local_dir, exist_ok=True)

    @staticmethod
    def name(progress):
        return f"ck_{int(progress):010d}"

    def save(self, state, progress, meta=None, pinned=False):
        base = self.name(progress)
        local = os.path.join(self.local_dir, base + ".pth.tmp")
        with open(local, "wb") as f:
            torch.save(state, f)
            f.flush()
            os.fsync(f.fileno())
        digest, size = sha256_file(local), os.path.getsize(local)
        dst = os.path.join(self.root, base + ".pth")
        partial = dst + ".partial"
        kill = os.environ.get("NAS_TEST_KILL_AT_SAVE")
        with open(local, "rb") as fi, open(partial, "wb") as fo:
            if kill is not None and int(kill) == int(progress):
                fo.write(fi.read(size // 2))
                fo.flush()
                os.fsync(fo.fileno())
                self.log(f"[test] checkpoint yazılırken kesinti: {base}")
                os._exit(137)
            shutil.copyfileobj(fi, fo, 1 << 20)
            fo.flush()
            os.fsync(fo.fileno())
        os.replace(partial, dst)
        ok = self._verify(dst, digest, size)
        if not ok:
            raise RuntimeError(f"checkpoint doğrulanamadı: {dst}")
        _write_json(os.path.join(self.root, base + ".ok.json"),
                    {"file": base + ".pth", "sha256": digest, "bytes": size, "progress": int(progress),
                     "pinned": bool(pinned), "meta": meta or {},
                     "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        os.remove(local)
        self._prune()
        return dst

    def _verify(self, path, digest, size):
        try:
            if os.path.getsize(path) != size or sha256_file(path) != digest:
                return False
            torch.load(path, map_location="cpu", weights_only=False)
            return True
        except Exception as e:   # bozuk dosya, okunamayan Drive önbelleği
            self.log(f"doğrulama hatası {path}: {e}")
            return False

    def records(self):
        out = []
        for fn in os.listdir(self.root):
            if fn.endswith(".ok.json"):
                try:
                    with open(os.path.join(self.root, fn), encoding="utf-8") as f:
                        out.append(json.load(f))
                except Exception as e:
                    self.log(f"okunamayan kayıt {fn}: {e}")
        return sorted(out, key=lambda r: r["progress"])

    def _prune(self):
        recs = [r for r in self.records() if not r.get("pinned")]
        for r in recs[:-self.keep] if len(recs) > self.keep else []:
            for suffix in (".ok.json", ".pth"):
                p = os.path.join(self.root, r["file"].replace(".pth", suffix))
                if os.path.exists(p):
                    os.remove(p)

    def latest(self):
        """(state, record) — en yeni doğrulanmış ve yüklenebilen kayıt; yoksa (None, None)."""
        for r in reversed(self.records()):
            p = os.path.join(self.root, r["file"])
            if not os.path.exists(p):
                self.log(f"kayıt dosyası yok, atlandı: {r['file']}")
                continue
            if os.path.getsize(p) != r["bytes"] or sha256_file(p) != r["sha256"]:
                self.log(f"bozuk kayıt (boyut/sha256), atlandı: {r['file']}")
                continue
            try:
                return torch.load(p, map_location="cpu", weights_only=False), r
            except Exception as e:
                self.log(f"yüklenemeyen kayıt, atlandı: {r['file']}: {e}")
        return None, None

    def find(self, progress):
        for r in self.records():
            if r["progress"] == int(progress):
                p = os.path.join(self.root, r["file"])
                return torch.load(p, map_location="cpu", weights_only=False), r
        return None, None
