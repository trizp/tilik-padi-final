#!/usr/bin/env python3
"""
bangun_data_daun_seg.py - Dataset segmentasi satu kelas "daun" untuk distilasi SAM2 -> YOLOv8n-seg (KEPUTUSAN_PROYEK.md 2026-10-20, fase 1).

- Foto: prov_v9 split train dan val yang BENAR-BENAR dipisah SAM2 (latar_dihapus = ya); gambar = foto ASLI prov_v7 (berlatar), hardlink.
  Foto cadangan tidak dipakai. Test, test sesi baru, dan ds3 tidak dipakai.
- Label: masker SAM2 aturan v2 setelah pasca-proses yang sama dengan prov_v9 (umum.pasca_proses) -> kontur luar -> poligon
  (approxPolyDP, epsilon 0,2% keliling), koordinat ternormalisasi; tiap komponen = satu objek kelas 0.

  .venv/Scripts/python scripts/v9_latar/bangun_data_daun_seg.py
Keluaran: data/derived/v9_daun_seg/{images,labels}/{train,val}, data.yaml, ringkasan.json
"""
import csv
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).parent))
import umum  # noqa: E402

ROOT = umum.ROOT
OUT = ROOT / "data/derived/v9_daun_seg"
MASKER = umum.KERJA / "sam2_v2/masker_mentah"


def poligon(m):
    h, w = m.shape
    kont, _ = cv2.findContours(m.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in kont:
        if cv2.contourArea(c) < 0.001 * h * w:
            continue
        a = cv2.approxPolyDP(c, 0.002 * cv2.arcLength(c, True), True).reshape(-1, 2)
        if len(a) < 3:
            continue
        out.append("0 " + " ".join(f"{x / w:.5f} {y / h:.5f}" for x, y in a))
    return out


def main():
    if OUT.exists():
        sys.exit(f"[DITOLAK] {OUT} sudah ada")
    man = [r for r in csv.DictReader(open(ROOT / "data/processed/prov_v9/manifest.csv", encoding="utf-8"))
           if r["split"] in ("train", "val") and r["latar_dihapus"] == "ya"]
    stat = Counter()
    for sp in ("train", "val"):
        (OUT / "images" / sp).mkdir(parents=True)
        (OUT / "labels" / sp).mkdir(parents=True)
    for i, r in enumerate(man):
        sp, f = r["split"], r["file"]
        src = ROOT / "data/processed/prov_v7/images" / sp / f
        try:
            os.link(src, OUT / "images" / sp / f)
        except OSError:
            shutil.copy2(src, OUT / "images" / sp / f)
        mm = cv2.imread(str(MASKER / (Path(f).stem + ".png")), cv2.IMREAD_GRAYSCALE)
        pol = poligon(umum.pasca_proses(mm >= 128))
        (OUT / "labels" / sp / (Path(f).stem + ".txt")).write_text("\n".join(pol) + ("\n" if pol else ""))
        stat[f"{sp}|foto"] += 1
        stat[f"{sp}|objek"] += len(pol)
        stat[f"{sp}|tanpa_poligon"] += int(not pol)
        if i % 1000 == 0:
            print(i, len(man), flush=True)
    (OUT / "data.yaml").write_text(yaml.safe_dump({"path": str(OUT).replace("\\", "/"), "train": "images/train", "val": "images/val",
                                                   "names": {0: "daun"}}), encoding="utf-8")
    ring = {"catatan": "distilasi SAM2 aturan v2 -> YOLOv8n-seg; foto asli prov_v7; hanya foto prov_v9 latar_dihapus=ya (train, val)",
            **dict(stat)}
    (OUT / "ringkasan.json").write_text(json.dumps(ring, indent=1, ensure_ascii=False), encoding="utf-8")
    print(ring)


if __name__ == "__main__":
    main()
