#!/usr/bin/env python3
"""
analyze_ds4.py - Evaluasi dataset kandidat (RiceLeafDiseaseBD = ds4) TANPA menggabungkannya.

Keluaran (semua dihitung dari file nyata, tidak ada angka karangan):
  1. Kelas, jumlah gambar/box per kelas, ID kelas yang benar-benar dipakai di label,
     dan jumlah gambar dengan >= 2 kelas.
  2. Statistik luas box relatif terhadap gambar per kelas, dibandingkan dengan ds1.
  3. Tumpang tindih dengan ds1 dan ds2 lewat perceptual hash (phash 16x16 = 256 bit),
     termasuk versi flip horizontal. Hasil hash di-cache ke data/interim/.

Pemakaian:
  python scripts/analyze_ds4.py
"""
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from inventory import IMG_EXT, find_yolo_label  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DS4 = ROOT / "data/raw/ds4/images_labels/RiceLeafDiseaseBD"
DS4_IMG = DS4 / "Original images"
DS4_LAB = DS4 / "Annotated images ( visual with labels)"
INTERIM = ROOT / "data/interim"
OUT = ROOT / "reports/ds4_analysis.json"

HASH_SIZE = 16          # sama dengan configs/classes.yaml (phash 16x16 = 256 bit)
DUP_THRESHOLD = 10      # sama dengan dup_threshold di configs/classes.yaml

# ds1: ID -> nama (dari data.yaml ds1). Hanya 3 kelas target + pembanding.
DS1_NAMES = {0: "bacterial_leaf_blight", 1: "brown_spot", 4: "leaf_blast"}


# ------------------------------------------------------------------ label
def read_boxes(label: Path):
    """Kembalikan list (cls, cx, cy, w, h) dari satu file label YOLO."""
    rows = []
    if label is None:
        return rows
    for line in label.read_text(errors="ignore").splitlines():
        t = line.split()
        if len(t) >= 5:
            try:
                rows.append((int(float(t[0])), *map(float, t[1:5])))
            except ValueError:
                pass
    return rows


def area_stats(areas):
    a = np.asarray(areas, dtype=float)
    if a.size == 0:
        return {}
    return {
        "n_boxes": int(a.size),
        "mean_area_pct": round(float(a.mean() * 100), 3),
        "median_area_pct": round(float(np.median(a) * 100), 3),
        "p10_area_pct": round(float(np.percentile(a, 10) * 100), 3),
        "p90_area_pct": round(float(np.percentile(a, 90) * 100), 3),
        "share_below_1pct": round(float((a < 0.01).mean()), 3),
        "share_below_10pct": round(float((a < 0.10).mean()), 3),
    }


def analyze_ds4_labels():
    res = {}
    for cdir in sorted(p for p in DS4_LAB.iterdir() if p.is_dir()):
        cls = cdir.name
        labs = sorted((cdir / "labels").glob("*.txt"))
        ids, areas, ws, hs = Counter(), [], [], []
        n_multi = n_empty = 0
        per_img = []
        for lb in labs:
            rows = read_boxes(lb)
            per_img.append(len(rows))
            if not rows:
                n_empty += 1
            if len({r[0] for r in rows}) >= 2:
                n_multi += 1
            for c, _, _, w, h in rows:
                ids[c] += 1
                areas.append(w * h)
                ws.append(w)
                hs.append(h)
        n_orig = len([p for p in (DS4_IMG / cls).iterdir() if p.suffix.lower() in IMG_EXT]) \
            if (DS4_IMG / cls).exists() else 0
        res[cls] = {
            "images_original": n_orig,
            "label_files": len(labs),
            "empty_label_files": n_empty,
            "images_2plus_classes": n_multi,
            "class_ids_in_labels": dict(ids),
            "boxes_per_image_mean": round(float(np.mean(per_img)), 2) if per_img else None,
            "mean_box_w_pct": round(float(np.mean(ws) * 100), 2) if ws else None,
            "mean_box_h_pct": round(float(np.mean(hs) * 100), 2) if hs else None,
            **area_stats(areas),
        }
    # Healthy: tidak punya folder label
    healthy = DS4_IMG / "Healthy"
    if healthy.exists():
        res["Healthy"] = {"images_original": len([p for p in healthy.iterdir() if p.suffix.lower() in IMG_EXT]),
                          "catatan": "tanpa file label (gambar sehat)"}
    return res


def analyze_ds1_labels():
    areas, per_img = defaultdict(list), defaultdict(list)
    imgs = [p for p in (ROOT / "data/raw/ds1").rglob("*") if p.suffix.lower() in IMG_EXT]
    for im in imgs:
        rows = read_boxes(find_yolo_label(im))
        cnt = Counter(r[0] for r in rows)
        for c, n in cnt.items():
            per_img[c].append(n)
        for c, _, _, w, h in rows:
            areas[c].append(w * h)
    out = {}
    for cid, name in DS1_NAMES.items():
        out[name] = {"images": len(per_img[cid]),
                     "boxes_per_image_mean": round(float(np.mean(per_img[cid])), 2),
                     **area_stats(areas[cid])}
    return out


# ------------------------------------------------------------------ phash
def _hash_one(path_str):
    """phash 256-bit asli + flip horizontal. Kembalikan (orig, flip) sebagai array bool."""
    import imagehash
    try:
        with Image.open(path_str) as im:
            im.draft("RGB", (512, 512))          # percepat decode JPEG besar
            im = im.convert("RGB")
            h0 = imagehash.phash(im, hash_size=HASH_SIZE).hash.flatten()
            h1 = imagehash.phash(im.transpose(Image.FLIP_LEFT_RIGHT), hash_size=HASH_SIZE).hash.flatten()
        return h0, h1
    except Exception:
        return None


def hash_folder(name, paths):
    cache = INTERIM / f"phash_{name}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if list(z["paths"]) == [str(p) for p in paths]:
            return z["orig"], z["flip"]
    print(f"  hashing {name}: {len(paths)} gambar ...", flush=True)
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(_hash_one, map(str, paths), chunksize=32))
    ok = [i for i, r in enumerate(res) if r is not None]
    if len(ok) != len(res):
        print(f"  [!] {len(res) - len(ok)} gambar gagal di-hash di {name}")
    orig = np.zeros((len(res), HASH_SIZE ** 2), dtype=bool)
    flip = np.zeros_like(orig)
    for i in ok:
        orig[i], flip[i] = res[i]
    INTERIM.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, orig=orig, flip=flip, paths=np.array([str(p) for p in paths]))
    return orig, flip


def hamming_min(a, b, same=False, chunk=1024):
    """Jarak Hamming minimum tiap baris a ke seluruh b (dan indeks tetangganya)."""
    A = a.astype(np.float32)
    B = b.astype(np.float32)
    n = A.shape[1]
    dmin = np.empty(len(A), dtype=np.int32)
    imin = np.empty(len(A), dtype=np.int64)
    for s in range(0, len(A), chunk):
        x = A[s:s + chunk]
        d = n - (x @ B.T + (1 - x) @ (1 - B).T)
        if same:                                  # jangan cocokkan gambar dengan dirinya sendiri
            for k in range(len(x)):
                d[k, s + k] = 1e9
        imin[s:s + chunk] = d.argmin(1)
        dmin[s:s + chunk] = d.min(1).round().astype(np.int32)
    return dmin, imin


def summarize_dist(d):
    return {f"<={t}": int((d <= t).sum()) for t in (0, 5, DUP_THRESHOLD)} | {"total": int(len(d))}


def by_class(d, cls_of, thr=DUP_THRESHOLD):
    c = Counter(cls_of[i] for i in np.where(d <= thr)[0])
    return dict(c)


def analyze_overlap():
    p4 = sorted(p for p in DS4_IMG.rglob("*") if p.suffix.lower() in IMG_EXT)
    p1 = sorted(p for p in (ROOT / "data/raw/ds1").rglob("*") if p.suffix.lower() in IMG_EXT)
    p2 = sorted(p for p in (ROOT / "data/raw/ds2").rglob("*") if p.suffix.lower() in IMG_EXT)
    cls4 = [p.parent.name for p in p4]
    h4o, h4f = hash_folder("ds4", p4)
    h1o, _ = hash_folder("ds1", p1)
    h2o, _ = hash_folder("ds2", p2)

    out = {}
    for name, ref in (("ds1", h1o), ("ds2", h2o)):
        d_o, _ = hamming_min(h4o, ref)
        d_f, _ = hamming_min(h4f, ref)       # ds4 di-flip horizontal vs referensi asli
        d_any = np.minimum(d_o, d_f)
        out[f"ds4_vs_{name}"] = {
            "asli": summarize_dist(d_o),
            "flip_horizontal": summarize_dist(d_f),
            "asli_atau_flip": summarize_dist(d_any),
            f"per_kelas_ds4_dekat(<={DUP_THRESHOLD})": by_class(d_any, cls4),
            "median_jarak_terdekat": float(np.median(d_any)),
        }
    # duplikat di dalam ds4 sendiri (asli vs asli, dan asli vs flip)
    d_in, _ = hamming_min(h4o, h4o, same=True)
    d_in_f, _ = hamming_min(h4f, h4o, same=True)
    d_in_f = np.minimum(d_in_f, 1e9)
    out["ds4_internal"] = {
        "asli": summarize_dist(d_in),
        "asli_atau_flip": summarize_dist(np.minimum(d_in, d_in_f)),
        f"per_kelas(<={DUP_THRESHOLD})": by_class(np.minimum(d_in, d_in_f), cls4),
    }
    # baseline pembanding: duplikat internal ds1 (agar angka ds4-vs-ds1 bisa ditafsirkan)
    d1, _ = hamming_min(h1o, h1o, same=True)
    out["ds1_internal_baseline"] = {"asli": summarize_dist(d1)}
    return out


def main():
    print("[1/3] label ds4 ...", flush=True)
    labels = analyze_ds4_labels()
    print("[2/3] label ds1 ...", flush=True)
    ds1 = analyze_ds1_labels()
    print("[3/3] phash ...", flush=True)
    overlap = analyze_overlap()
    rep = {"ds4_labels": labels, "ds1_labels_target_classes": ds1, "overlap_phash": overlap,
           "phash": {"hash_size": HASH_SIZE, "dup_threshold": DUP_THRESHOLD}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    print(f"\nLaporan disimpan ke {OUT}")


if __name__ == "__main__":
    main()
