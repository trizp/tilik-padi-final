#!/usr/bin/env python3
"""orientasi_visual_ds4.py - Untuk setiap foto ds4 berlabel di prov_v7: apakah foto bertanda penulis (V3 "visuals", sumber label V3)
berorientasi sama dengan foto asli, atau sama dengan foto asli setelah diputar seperti perbaikan prov_v7?
Ukuran: korelasi piksel abu-abu 64x64 antara foto bertanda dan (a) foto asli, (b) foto asli diputar 90/180/270 derajat. Hanya membaca.
Keluaran: reports/ds4_versi/orientasi_visual.json"""
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "data/raw/ds4/images_labels/RiceLeafDiseaseBD"
PUTAR = {"0": None, "90": Image.Transpose.ROTATE_270, "180": Image.Transpose.ROTATE_180, "270": Image.Transpose.ROTATE_90}   # searah jarum jam
PERBAIKAN = {"-": "0", "jpegtran_rotate_cw_90": "90", "jpegtran_rotate_cw_180": "180", "jpegtran_rotate_cw_270": "270"}


def vek(im):
    a = np.asarray(im.convert("L").resize((64, 64)), float).ravel()
    a -= a.mean()
    return a / (np.linalg.norm(a) + 1e-9)


def main():
    man = [r for r in csv.DictReader(open(ROOT / "data/processed/prov_v7/manifest.csv", encoding="utf-8"))
           if r["source"] == "ds4" and r["class"] != "background"]
    hasil = Counter()
    beda = []
    for r in man:
        op = ROOT / r["orig_path"]
        vis = R / "Annotated images ( visual with labels)" / op.parent.name / "visuals" / op.name
        asli = Image.open(op)
        v = vek(Image.open(vis))
        skor = {k: float(vek(asli.transpose(t) if t is not None else asli) @ v) for k, t in PUTAR.items()}
        terbaik = max(skor, key=skor.get)
        harap = PERBAIKAN[r["perbaikan_orientasi"]]
        hasil[(f"perbaikan prov_v7 {harap}", f"visual cocok dengan putaran {terbaik}")] += 1
        if terbaik != harap:
            beda.append({"file": r["file"], "perbaikan": harap, "visual_cocok": terbaik, "skor": skor})
    out = {"foto_ds4_berlabel": len(man), "hasil": {f"{a} -> {b}": n for (a, b), n in sorted(hasil.items())},
           "tidak_sesuai": beda}
    (ROOT / "reports/ds4_versi/orientasi_visual.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "tidak_sesuai"}, indent=1, ensure_ascii=False), "| tidak sesuai:", len(beda))
    for b in beda[:10]:
        print(b)


if __name__ == "__main__":
    main()
