#!/usr/bin/env python3
"""
dedupe_rules.py - Aturan pengelompokan duplikat versi v2 (dipakai prepare_dataset.py dan alat bantu analisis).

Komponen:
  * parse_ds1_name / k4_pairs : kunci ID foto K4 = (keluarga, jenis, ID, ekstensi) lintas awalan (hanya ds1).
  * Deskriptor SIFT per gambar (cache per gambar di data/interim/sift/), pencocokan + RANSAC (transformasi
    similaritas) -> jumlah inlier. Dua orientasi dicoba (asli dan flip horizontal).
  * knn_pairs : kandidat pasangan = K tetangga terdekat menurut embedding DINOv2 (maks atas 8 varian dihedral),
    TANPA ambang cosine. Pasangan dengan inlier >= MIN_INLIERS dianggap foto yang sama.

CLI: python scripts/dedupe_rules.py estimate [--config configs/classes_v2.yaml]   (estimasi waktu SIFT, tanpa verifikasi penuh)
"""
import argparse
import hashlib
import math
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

cv2.setNumThreads(1)      # satu thread per proses: menghindari kontensi saat banyak worker

sys.path.insert(0, str(Path(__file__).parent))
import aug_detect as ad  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SIFT_DIR = ROOT / "data/interim/sift2"      # sift2: menyimpan juga ukuran bingkai (h,w) untuk metrik sebaran inlier
MAXPX, NFEAT, RATIO, RANSAC_PX = 512, 400, 0.75, 4.0
MIN_INLIERS, KNN = 21, 50
MIN_FRAC = 0.15            # kotak pembatas inlier harus >= 15% luas gambar (menolak kecocokan watermark)
QUARANTINE_INLIERS = 100   # pasangan lintas kelas dengan inlier > ini -> karantina
TIME_GAP_S = 2.0
RULE_VERSION = "v5"
RULE_DESC = ("v4 = phash16 (8 varian dihedral) + embedding DINOv2-small (8 varian) + nama dasar Roboflow + kunci ID K4 (ds1) "
             "+ SIFT (inlier minimum DAN sebaran kotak inlier minimum) pada K tetangga terdekat embedding (semua kelas dan sumber) "
             "+ jendela waktu GESER per sumber (semua pasangan <= jendela) (ds4 dari Original_Filename, ds1 dari nama berkas; pemisah tanggal-jam opsional) "
             "+ karantina pasangan lintas kelas berinlier tinggi. SEMUA kelompok dibentuk pada himpunan lengkap (termasuk foto karantina dan duplikat bertentangan) SEBELUM foto tersebut dibuang. NILAI PARAMETER: lihat bagian 'parameter' di dataset_meta.json.")

RE_ID = re.compile(r"^(?P<origin>r6k|rls)_(?P<sub>train|test|val|valid)_(?P<fam>img_blb|blight|img_brown|brownspot)_"
                   r"(?:(?P<kind>orig|rotated)_)?(?P<id>\d+)(?:_(?P<ext>[A-Za-z]+))?(?:_jpg)?$")


def parse_ds1_name(path):
    stem = re.sub(r"\.rf\.[0-9a-f]+$", "", Path(path).stem)
    m = RE_ID.match(stem)
    if not m:
        return None
    d = m.groupdict()
    d["id"] = int(d["id"])
    d["kind"] = d["kind"] or "-"
    d["ext"] = d["ext"] or "-"
    d["prefix"] = f"{d['origin']}_{d['sub']}"
    return d


def k4_pairs(cands):
    """Pasangan (i,j) dengan (fam,jenis,id,ekstensi) sama dan awalan berbeda; hanya sumber ds1 (nama ber-ID)."""
    by = {}
    for i, c in enumerate(cands):
        if c["src"] != "ds1":
            continue
        p = parse_ds1_name(c.get("orig_path", c["path"]))
        if p:
            by.setdefault((p["fam"], p["kind"], p["id"], p["ext"]), []).append((i, p["prefix"]))
    out = []
    for v in by.values():
        for a in range(len(v)):
            for b_ in range(a + 1, len(v)):
                if v[a][1] != v[b_][1]:
                    out.append((v[a][0], v[b_][0]))
    return np.array(out, int).reshape(-1, 2)


# ------------------------------------------------------------------ jam pemotretan (ds4 dari metadata, ds1 dari nama berkas)
_TS_RE = re.compile(r"(20\d{2})(\d{2})(\d{2})[_-]?(\d{2})(\d{2})(\d{2})")   # pemisah opsional: IMG20250425181206 dan 20250425_181206


def ts_from_text(s):
    import datetime as _dt
    m = _TS_RE.search(str(s))
    if not m:
        return None
    try:
        t = _dt.datetime(*(int(x) for x in m.groups()))
    except ValueError:
        return None
    return t if 2015 <= t.year <= 2026 else None


def ds4_timestamps(xlsx_path):
    """New_Filename (huruf kecil) -> datetime pemotretan, dari tanggal-jam pada Original_Filename."""
    import pandas as pd
    xl = pd.read_excel(xlsx_path, sheet_name="Image_Metadata")
    out = {}
    for n, o in zip(xl["New_Filename"], xl["Original_Filename"]):
        t = ts_from_text(o)
        if t is not None:
            out[str(n).lower()] = t
    return out


def ds1_timestamp(path):
    """Jam dari nama berkas ds1 (mis. rls_train_20231006_170253_jpg.rf.<hash>.jpg); bagian hash Roboflow dibuang dulu."""
    return ts_from_text(re.sub(r"\.rf\.[0-9a-f]+$", "", Path(path).stem))


def timestamp_of(c, ds4_map):
    p = c.get("orig_path", c["path"])
    if c["src"] == "ds4":
        return ds4_map.get(Path(p).name.lower())
    if c["src"] == "ds1":
        return ds1_timestamp(p)
    return None


def time_pairs(cands, ts_fn, gap_s=TIME_GAP_S, sources=("ds4",)):
    """Pasangan berantai antar foto SATU sumber yang selisih waktunya <= gap_s (jam antar sumber tidak dibandingkan).
    Kembalikan (pasangan, jumlah bertanggal per sumber)."""
    pairs, counts = [], {}
    for src in sources:
        items = sorted((t, i) for i, c in enumerate(cands) if c["src"] == src for t in [ts_fn(c)] if t is not None)
        counts[src] = len(items)
        # jendela geser: SEMUA pasangan dengan selisih <= gap_s (bukan hanya tetangga berurutan)
        j0 = 0
        for k in range(len(items)):
            j = k + 1
            while j < len(items) and (items[j][0] - items[k][0]).total_seconds() <= gap_s:
                pairs.append((items[k][1], items[j][1]))
                j += 1
    return np.array(pairs, int).reshape(-1, 2), counts


# ------------------------------------------------------------------ SIFT
def sift_key(path):
    return hashlib.sha1(ad.sig(Path(path)).encode()).hexdigest()[:20]


def _desc_one(path):
    key = sift_key(path)
    f = SIFT_DIR / f"{key}.npz"
    if f.exists():
        return key
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        from PIL import Image
        im = cv2.cvtColor(np.asarray(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    s = MAXPX / max(g.shape)
    if s < 1:
        g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    g = cv2.medianBlur(g, 3)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    sift = cv2.SIFT_create(nfeatures=NFEAT)
    out = {"hw": np.array(g.shape[:2], np.int32)}
    for t, gg in enumerate((g, cv2.flip(g, 1))):
        k, d = sift.detectAndCompute(gg, None)
        if d is None:
            out[f"p{t}"], out[f"d{t}"] = np.zeros((0, 2), np.float32), np.zeros((0, 128), np.uint8)
        else:
            out[f"p{t}"], out[f"d{t}"] = np.float32([p.pt for p in k]), d.astype(np.uint8)
    SIFT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = SIFT_DIR / f"{key}.tmp.npz"
    np.savez(tmp, **out)
    tmp.replace(f)
    return key


def ensure_descriptors(paths, workers=None):
    """Hitung deskriptor yang belum ada di cache (paralel). Kembalikan jumlah yang dihitung."""
    todo = sorted({str(p) for p in paths if not (SIFT_DIR / f"{sift_key(p)}.npz").exists()})
    if todo:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            list(ex.map(_desc_one, todo, chunksize=8))
    return len(todo)


_CACHE = {}
_BF = None


def _load(key):
    d = _CACHE.get(key)
    if d is None:
        z = np.load(SIFT_DIR / f"{key}.npz")
        d = ((z["p0"], z["d0"]), (z["p1"], z["d1"]), z["hw"])
        if len(_CACHE) > 600:
            _CACHE.clear()
        _CACHE[key] = d
    return d


def _bbox_frac(pts, hw):
    if len(pts) < 3:
        return 0.0
    w = float(pts[:, 0].max() - pts[:, 0].min())
    h = float(pts[:, 1].max() - pts[:, 1].min())
    return (w * h) / float(hw[0] * hw[1])


def match(da, db):
    """
    Inlier RANSAC maksimum atas {A asli, A flip} vs B.
    Kembalikan (inlier, skala, rotasi_derajat, frac): frac = luas kotak pembatas titik inlier / luas gambar,
    diambil nilai terbesar dari kedua gambar (kecocokan watermark = inlier menumpuk di area kecil pada KEDUA gambar).
    """
    global _BF
    if _BF is None:
        _BF = cv2.BFMatcher(cv2.NORM_L2)
    best = (0, 1.0, 0.0, 0.0)
    for fa in (0, 1):
        pa, xa = da[fa]
        pb, xb = db[0]
        if len(xa) < 8 or len(xb) < 8:
            continue
        m = _BF.knnMatch(xa.astype(np.float32), xb.astype(np.float32), k=2)
        good = [x[0] for x in m if len(x) == 2 and x[0].distance < RATIO * x[1].distance]
        if len(good) < 6:
            continue
        src = pa[[g.queryIdx for g in good]]
        dst = pb[[g.trainIdx for g in good]]
        M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_PX, maxIters=3000, confidence=0.995)
        if M is None:
            continue
        mask = inl.ravel().astype(bool)
        n = int(mask.sum())
        sc = math.hypot(M[0, 0], M[1, 0])
        if 0.3 <= sc <= 3.3 and n > best[0]:
            fr = max(_bbox_frac(src[mask], da[2]), _bbox_frac(dst[mask], db[2]))
            best = (n, round(sc, 3), round(math.degrees(math.atan2(M[1, 0], M[0, 0])), 1), round(fr, 4))
    return best


def _verify_chunk(keys):
    return [match(_load(a), _load(b)) for a, b in keys]


def verify_pairs(path_pairs, workers=None, chunk=300):
    """path_pairs: list (pathA, pathB). Deskriptor harus sudah ada (ensure_descriptors). Kembalikan list (inlier, skala, rot)."""
    keys = [(sift_key(a), sift_key(b)) for a, b in path_pairs]
    chunks = [keys[i:i + chunk] for i in range(0, len(keys), chunk)]
    out = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(_verify_chunk, chunks):
            out.extend(r)
    return out


# ------------------------------------------------------------------ kandidat kNN
def knn_pairs(E, k=KNN, chunk=1024):
    """K tetangga terdekat per gambar (maks cosine atas 8 varian dihedral), tanpa ambang. Pasangan unik (i<j)."""
    n = len(E)
    O = E[:, 0]
    pairs = set()
    for s in range(0, n, chunk):
        S = np.max([E[s:s + chunk, v] @ O.T for v in range(E.shape[1])], axis=0)
        for r in range(S.shape[0]):
            S[r, s + r] = -9
        idx = np.argpartition(-S, k, axis=1)[:, :k]
        for r in range(S.shape[0]):
            i = s + r
            for j in idx[r]:
                pairs.add((min(i, int(j)), max(i, int(j))))
    return np.array(sorted(pairs), int).reshape(-1, 2)


def knn_pairs_cross(Eq, Et, k=KNN, chunk=1024):
    """Untuk tiap gambar query (Eq) ambil k tetangga di himpunan target (Et, indeks 0 = asli). Kembalikan (qi, tj) array."""
    O = Et[:, 0]
    out = []
    kk = min(k, len(Et))
    for s in range(0, len(Eq), chunk):
        S = np.max([Eq[s:s + chunk, v] @ O.T for v in range(Eq.shape[1])], axis=0)
        idx = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
        for r in range(S.shape[0]):
            out += [(s + r, int(j)) for j in idx[r]]
    return np.array(out, int).reshape(-1, 2)


def build_pool(cfg_path):
    """Kumpulkan kandidat sumber (tanpa hash/split) persis seperti prepare_dataset. Kembalikan (cfg, classes, cands)."""
    import random
    import yaml
    import prepare_dataset as pd_
    cfg = yaml.safe_load((ROOT / cfg_path).read_text(encoding="utf-8"))
    classes = cfg["disease_classes"]
    excl, meta, rng, cands = pd_.Excl(), {}, random.Random(cfg["seed"]), []
    for s in cfg["sources"]:
        if s.get("role", "pool") != "pool":
            continue
        if s["type"] == "yolo":
            cands += pd_.collect_yolo(s, classes, excl, meta)
        elif s["type"] == "yolo_folders":
            cands += pd_.collect_yolo_folders(s, classes, excl, meta)
        else:
            cands += pd_.collect_classification(s, classes, excl, meta, rng)
    pd_.resize_before_hash(cands)
    return cfg, classes, cands


def estimate(cfg_path):
    """Ukur waktu deskriptor dan pencocokan pada sampel kecil lalu ekstrapolasi (tanpa verifikasi penuh)."""
    import os
    cfg, classes, cands = build_pool(cfg_path)
    paths = [c["path"] for c in cands]
    n = len(paths)
    miss = [p for p in paths if not (SIFT_DIR / f"{sift_key(p)}.npz").exists()]
    E = ad.embeddings(paths)
    t = time.time()
    P = knn_pairs(E, KNN)
    t_knn = time.time() - t
    rng = np.random.default_rng(0)
    # waktu deskriptor per gambar (serial, satu proses): hitung ulang 40 gambar
    probe = [paths[i] for i in rng.choice(n, 40, replace=False)]
    t = time.time()
    for p in probe:
        (SIFT_DIR / f"{sift_key(p)}.npz").unlink(missing_ok=True)
        _desc_one(p)
    t_desc = (time.time() - t) / len(probe)
    # waktu per pasangan (serial): 400 pasangan kNN acak
    pick = P[rng.choice(len(P), min(400, len(P)), replace=False)]
    need = {paths[i] for i, _ in pick} | {paths[j] for _, j in pick}
    ensure_descriptors(need)
    keys = [(sift_key(paths[i]), sift_key(paths[j])) for i, j in pick]
    t = time.time()
    res = _verify_chunk(keys)
    t_pair = (time.time() - t) / len(keys)
    w = max(1, (os.cpu_count() or 8) - 2)
    print(f"gambar pool: {n} | deskriptor belum ada di cache: {len(miss)}")
    print(f"pasangan kandidat kNN (k={KNN}, unik): {len(P)} (hitung kNN {t_knn:.1f} s)")
    print(f"waktu per gambar (deskriptor, serial): {t_desc * 1000:.0f} ms | waktu per pasangan (serial): {t_pair * 1000:.1f} ms | worker: {w}")
    print(f"sampel 400 pasangan kNN: lolos aturan (inlier>=21 dan sebaran>=15%): {sum(r[0] >= MIN_INLIERS and r[3] >= MIN_FRAC for r in res)}")
    print(f"ESTIMASI deskriptor: {len(miss) * t_desc / w / 60:.1f} menit (paralel) / {len(miss) * t_desc / 60:.1f} menit (serial)")
    print(f"ESTIMASI verifikasi SIFT: {len(P) * t_pair / w / 60:.1f} menit (paralel) / {len(P) * t_pair / 60:.1f} menit (serial)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["estimate"])
    ap.add_argument("--config", default="configs/classes_v2.yaml")
    a = ap.parse_args()
    estimate(a.config)
