#!/usr/bin/env python3
"""
perbaiki_orientasi_ds4.py - Perbaikan MEKANIS orientasi foto ds4 untuk prov_v7 (label TIDAK diubah).

Untuk tiap foto ds4 di manifest prov_v6 (nama berkas sama):
  * orientasi yang cocok dengan label = varian dihedral yang membuat foto sama dengan visual buatan penulis ds4
    (reports/prov_v6/audit_orientasi_ds4/orientasi_per_gambar.csv, dibaca seperti Ultralytics: exif_transpose);
  * cocok dan tanpa tag EXIF Orientation  -> disalin BYTE-IDENTIK;
  * tidak cocok atau bertag EXIF            -> dibaca dengan exif_transpose, diputar ke varian yang cocok, disimpan JPEG
                                               kualitas 95 TANPA tag orientasi (semua pemuat membaca sama);
  * background (tanpa label/visual)         -> disalin byte-identik (bila bertag: exif_transpose lalu simpan tanpa tag).
Sumber: data/processed/prov_v6/images/<split>/<file> (hardlink ke foto mentah). prov_v6 dan data mentah tidak diubah.
Keluaran: data/interim/ds4_orientasi_v7/<file> + reports/prov_v7_prep/perbaikan_orientasi.csv
"""
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).parent))
from cek_orientasi_ds4 import ubah  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/processed/prov_v6/images"
DST = ROOT / "data/interim/ds4_orientasi_v7"
REP = ROOT / "reports/prov_v7_prep"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    if DST.exists():
        sys.exit(f"[DITOLAK] {DST} sudah ada; hapus manual bila ingin membuat ulang.")
    DST.mkdir(parents=True)
    REP.mkdir(parents=True, exist_ok=True)
    audit = {r["file"]: r for r in csv.DictReader(open(ROOT / "reports/prov_v6/audit_orientasi_ds4/orientasi_per_gambar.csv", encoding="utf-8"))}
    man = [r for r in csv.DictReader(open(ROOT / "data/processed/prov_v6/manifest.csv", encoding="utf-8")) if r["source"] == "ds4"]
    rows = []
    for r in man:
        src = SRC / r["split"] / r["file"]
        dst = DST / r["file"]
        a = audit[r["file"]]
        varian = a["orientasi_label"] if a["orientasi_label"] != "tanpa_visual" else "asli"
        with Image.open(src) as im:
            tag = im.getexif().get(274)
            ukuran_lama = im.size
        if varian == "asli" and tag in (None, 1):
            shutil.copy2(src, dst)
            aksi = "salin_byte_identik"
        else:
            with Image.open(src) as im:
                x = ubah(ImageOps.exif_transpose(im).convert("RGB"), varian)
                x.save(dst, "JPEG", quality=95)          # tanpa argumen exif -> tidak ada tag orientasi
            aksi = "diputar_disimpan_ulang" if varian != "asli" else "tag_exif_diterapkan_disimpan_ulang"
        with Image.open(dst) as im2:
            tag2, ukuran_baru = im2.getexif().get(274), im2.size
        rows.append({"file": r["file"], "split": r["split"], "class": r["class"], "varian_diterapkan": varian, "tag_exif_lama": tag,
                     "tag_exif_baru": tag2, "aksi": aksi, "ukuran_lama": "x".join(map(str, ukuran_lama)),
                     "ukuran_baru": "x".join(map(str, ukuran_baru)), "sha256_lama": sha(src), "sha256_baru": sha(dst)})
    with open(REP / "perbaikan_orientasi.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summ = {"n": len(rows), "aksi": dict(Counter(r["aksi"] for r in rows)),
            "aksi_per_split": {sp: dict(Counter(r["aksi"] for r in rows if r["split"] == sp)) for sp in ("train", "val", "test", "test_new_session")},
            "varian": dict(Counter(r["varian_diterapkan"] for r in rows)),
            "masih_bertag_exif": sum(1 for r in rows if r["tag_exif_baru"] not in (None, 1)),
            "ukuran_berubah": sum(1 for r in rows if r["ukuran_lama"] != r["ukuran_baru"]),
            "salinan_identik_sha_cocok": all(r["sha256_lama"] == r["sha256_baru"] for r in rows if r["aksi"] == "salin_byte_identik")}
    (REP / "perbaikan_orientasi_ringkasan.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summ, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
