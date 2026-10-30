#!/usr/bin/env python3
"""
blb_pair_analysis4.py - Kontrol sintetis: transformasi kecil (rotasi ~1 derajat, geser beberapa piksel, kenaikan
kecerahan) pada gambar BLB train/val; recall detektor (phash16 dihedral<=14, dengan/tanpa median blur,
embedding>=0.97, SIFT>=21 inlier). Menguji penyebab lolosnya pasangan audit #01/#09. HANYA MEMBACA; test tidak dibuka.
Keluaran: reports/prov/blb_pair_analysis4.json
"""
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import aug_detect as ad  # noqa: E402
import blb_pair_analysis as b  # noqa: E402
import blb_pair_analysis2 as g2  # noqa: E402
from prepare_dataset import ROOT  # noqa: E402

T = 21


def affine(im, rot=0.0, dx=0, dy=0):
    a = np.asarray(im)
    h, w = a.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rot, 1.0)
    M[0, 2] += dx
    M[1, 2] += dy
    return Image.fromarray(cv2.warpAffine(a, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT))


def bright(im, delta):
    return Image.fromarray(np.clip(np.asarray(im, np.float32) + delta, 0, 255).astype(np.uint8))


def main():
    blb = [r for r in b.MAN if r["class"] == "bacterial_leaf_blb" or r["class"] == "bacterial_leaf_blight"]
    tv = [r for r in blb if r["split"] in ("train", "val")]
    samp = random.Random(3).sample(tv, 100)
    O = [b.load(ROOT / r["orig_path"]) for r in samp]
    tfs = {
        "rotasi_1deg": lambda im: affine(im, rot=1.0),
        "geser_5px": lambda im: affine(im, dx=4, dy=-5),
        "rotasi_1deg+geser_5px": lambda im: affine(im, rot=1.0, dx=4, dy=-5),
        "kecerahan_+46": lambda im: bright(im, 46),
        "rotasi_1deg+geser_5px+kecerahan_+46 (meniru #01/#09)": lambda im: bright(affine(im, rot=1.0, dx=4, dy=-5), 46),
        "rotasi_1deg+geser_5px+kecerahan_+46+noise_1%": lambda im: b.salt_pepper(bright(affine(im, rot=1.0, dx=4, dy=-5), 46), 0.01, np.random.default_rng(1)),
    }
    Eo_raw = ad.embed_items(O, batch=16, workers=0)
    Eo_m3 = ad.embed_items([b.preprocess(x, "median3") for x in O], batch=16, workers=0)
    Ho = {m: [b.dih_hash(b.preprocess(x, m), 16) for x in O] for m in ("none", "median5")}
    Do = []
    out = {}
    for name, fn in tfs.items():
        N = [fn(x) for x in O]
        row = {}
        for m in ("none", "median5"):
            d = [b.dmin(b.dih_hash(b.preprocess(n, m), 16), Ho[m][i])[0] for i, n in enumerate(N)]
            row[f"phash16|{m}|recall<=14"] = round(float(np.mean([x <= 14 for x in d])), 3)
            row[f"phash16|{m}|jarak_median"] = float(np.median(d))
        En_raw = ad.embed_items(N, batch=16, workers=0)
        En_m3 = ad.embed_items([b.preprocess(x, "median3") for x in N], batch=16, workers=0)
        c_raw = [max(En_raw[i, v] @ Eo_raw[i, 0] for v in range(8)) for i in range(len(N))]
        c_m3 = [max(En_m3[i, v] @ Eo_m3[i, 0] for v in range(8)) for i in range(len(N))]
        row["embedding|raw|recall>=0.97"] = round(float(np.mean(np.array(c_raw) >= 0.97)), 3)
        row["embedding|median3|recall>=0.97"] = round(float(np.mean(np.array(c_m3) >= 0.97)), 3)
        # SIFT: simpan sementara ke file agar memakai jalur kode yang sama dengan analisis nyata
        tmp = ROOT / "data/interim/_tmp_synth"
        tmp.mkdir(parents=True, exist_ok=True)
        hits = []
        for i, (x, y) in enumerate(zip(O, N)):
            x.save(tmp / f"o{i}.png")
            y.save(tmp / f"n{i}.png")
            hits.append(g2.match(g2._desc(str(tmp / f"n{i}.png")), g2._desc(str(tmp / f"o{i}.png")))[0] >= T)
        row["SIFT|recall>=21"] = round(float(np.mean(hits)), 3)
        out[name] = row
        print(name, json.dumps(row), flush=True)
    for p in (ROOT / "data/interim/_tmp_synth").glob("*.png"):
        p.unlink()
    (ROOT / "data/interim/_tmp_synth").rmdir()
    (ROOT / "reports/prov/blb_pair_analysis4.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
