#!/usr/bin/env python3
"""
cek_orientasi_ds4.py - Audit (HANYA BACA): apakah foto ds4 "Original images" berorientasi sama dengan gambar
"Annotated images/<kelas>/visuals" yang dipakai penulis ds4 saat membuat label txt.

Untuk tiap foto ds4 di manifest prov_v6, foto asli dibandingkan dengan visual-nya pada 8 varian dihedral
(rotasi 0/90/180/270 x flip), grayscale 64x64; varian dengan selisih terkecil = orientasi label. Varian != asli berarti
label txt TIDAK cocok dengan foto yang dipakai dataset (kotak label bergeser/terputar).
Foto dibaca dengan ImageOps.exif_transpose (= cara dataloader Ultralytics/cv2 IMREAD_COLOR membacanya).
--sumber-dir DIR : periksa berkas DIR/<file> (mis. hasil perbaikan) alih-alih foto mentah ds4.
Keluaran: <--out> (baku reports/prov_v6/audit_orientasi_ds4)/{orientasi_per_gambar.csv, ringkasan.json}
"""
import argparse
import csv
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
ANN = ROOT / "data/raw/ds4/images_labels/RiceLeafDiseaseBD/Annotated images ( visual with labels)"
OUT = ROOT / "reports/prov_v6/audit_orientasi_ds4"
VARIAN = ["asli", "rot90", "rot180", "rot270", "flip", "flip_rot90", "flip_rot180", "flip_rot270"]


def kecil(im):
    return np.asarray(im.convert("L").resize((64, 64), Image.BILINEAR), np.float32) / 255


def ubah(a, varian):
    """Terapkan varian dihedral (urutan sama dengan VARIAN) pada gambar PIL."""
    i = VARIAN.index(varian)
    x = a.transpose(Image.FLIP_LEFT_RIGHT) if i >= 4 else a
    return x.rotate(90 * (i % 4), expand=True)


def satu(arg):
    orig_path, fold, stem, sumber = arg
    v = ANN / fold / "visuals" / (stem + ".jpg")
    if not v.exists():
        cand = list((ANN / fold / "visuals").glob(stem + ".*"))
        if not cand:
            return orig_path, None, None, None
        v = cand[0]
    a = ImageOps.exif_transpose(Image.open(sumber))
    b = kecil(Image.open(v))
    d = []
    for f in (False, True):
        x = a.transpose(Image.FLIP_LEFT_RIGHT) if f else a
        for k in range(4):
            d.append(float(np.abs(kecil(x.rotate(90 * k, expand=True)) - b).mean()))
    i = int(np.argmin(d))
    s = sorted(d)
    return orig_path, VARIAN[i], round(s[0], 4), round(s[1] - s[0], 4)


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--sumber-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dataset-dir", default=None, help="periksa berkas <dir>/images/<split>/<file> (mis. data/processed/prov_v7)")
    a_ = ap.parse_args()
    if a_.out:
        OUT = ROOT / a_.out
    OUT.mkdir(parents=True, exist_ok=True)
    m = [r for r in csv.DictReader(open(ROOT / "data/processed/prov_v6/manifest.csv", encoding="utf-8")) if r["source"] == "ds4"]
    if a_.dataset_dir:
        src = lambda r: ROOT / a_.dataset_dir / "images" / r["split"] / r["file"]
    elif a_.sumber_dir:
        src = lambda r: ROOT / a_.sumber_dir / r["file"]
    else:
        src = lambda r: ROOT / r["orig_path"]
    args = [(r["orig_path"], Path(r["orig_path"]).parent.name, Path(r["orig_path"]).stem, src(r)) for r in m]
    with ProcessPoolExecutor(max_workers=8) as ex:
        res = {o: (v, d, g) for o, v, d, g in ex.map(satu, args, chunksize=32)}
    rows = []
    for r in m:
        v, d, g = res[r["orig_path"]]
        rows.append({"file": r["file"], "split": r["split"], "class": r["class"], "orig_path": r["orig_path"],
                     "orientasi_label": v if v else "tanpa_visual", "selisih_terbaik": d, "jarak_ke_terbaik_kedua": g,
                     "label_tidak_cocok": bool(v and v != "asli")})
    with open(OUT / "orientasi_per_gambar.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summ = {"sumber_foto": a_.dataset_dir or a_.sumber_dir or "foto mentah ds4 (Original images)", "pembacaan": "ImageOps.exif_transpose",
            "definisi": "label_tidak_cocok = foto 'Original images' perlu diputar/dibalik agar sama dengan visual buatan penulis ds4 (tempat label dibuat)",
            "per_split": {}}
    for sp in ("train", "val", "test", "test_new_session"):
        rs = [r for r in rows if r["split"] == sp]
        lab = [r for r in rs if r["class"] != "background"]
        summ["per_split"][sp] = {"ds4_gambar": len(rs), "ds4_berlabel": len(lab),
                                 "label_tidak_cocok": sum(r["label_tidak_cocok"] for r in lab),
                                 "persen_berlabel_tidak_cocok": round(100 * sum(r["label_tidak_cocok"] for r in lab) / len(lab), 1) if lab else None,
                                 "orientasi": dict(Counter(r["orientasi_label"] for r in lab))}
    summ["jarak_terbaik_kedua_min_pada_tidak_cocok"] = min((r["jarak_ke_terbaik_kedua"] for r in rows if r["label_tidak_cocok"]), default=None)
    summ["selisih_terbaik_maks"] = max((r["selisih_terbaik"] or 0) for r in rows)
    (OUT / "ringkasan.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summ, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
