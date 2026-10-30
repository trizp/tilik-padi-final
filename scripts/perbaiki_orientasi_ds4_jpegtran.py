#!/usr/bin/env python3
"""
perbaiki_orientasi_ds4_jpegtran.py - Perbaikan orientasi ds4 untuk prov_v7 dengan rotasi JPEG LOSSLESS (jpegtran).
Label TIDAK diubah. Menggantikan versi JPEG q95 (ditolak pemilik skripsi, 2026-10-10).

Alat: tools/libjpeg-turbo/jpegtran.exe (libjpeg-turbo 3.2.0, conda-forge build hfd05255_1; di luar venv proyek).
Per foto ds4 di manifest prov_v6 (sumber data/processed/prov_v6/images/<split>/<file>):
  * cocok & tanpa tag EXIF Orientation -> salin byte-identik
  * perlu diputar (audit: varian rotK = putar K x 90 derajat BERLAWANAN jarum jam agar sama dengan visual penulis)
        -> jpegtran -perfect -copy none -rotate (360 - 90K)   [jpegtran memutar SEARAH jarum jam]
  * bertag EXIF (mis. 6) dan sudah cocok setelah tag diterapkan -> jpegtran -rotate sesuai tag, -copy none (tag hilang)
-copy none membuang semua metadata (EXIF termasuk tag orientasi, komentar, ICC) sehingga semua pemuat membaca sama.
Bukti lossless: piksel hasil (didekode PIL) dibandingkan dengan np.rot90 piksel asli yang didekode; dilaporkan jumlah tidak identik.
Keluaran: data/interim/ds4_orientasi_v7_jpegtran/<file>, reports/prov_v7_prep/jpegtran_{per_foto.csv,ringkasan.json}
"""
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
JPEGTRAN = ROOT / "tools/libjpeg-turbo/jpegtran.exe"
SRC = ROOT / "data/processed/prov_v6/images"
DST = ROOT / "data/interim/ds4_orientasi_v7_jpegtran"
REP = ROOT / "reports/prov_v7_prep"
K_CCW = {"asli": 0, "rot90": 1, "rot180": 2, "rot270": 3}
TAG_CW = {3: 180, 6: 90, 8: 270}            # tag EXIF -> derajat searah jarum jam agar tegak


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def info_meta(p):
    with Image.open(p) as im:
        return {"tag": im.getexif().get(274), "icc": bool(im.info.get("icc_profile")), "komentar": bool(im.info.get("comment")),
                "ukuran": im.size}


def main():
    if DST.exists():
        sys.exit(f"[DITOLAK] {DST} sudah ada")
    ver = subprocess.run([str(JPEGTRAN), "-version"], capture_output=True, text=True)
    versi = (ver.stdout + ver.stderr).strip()
    DST.mkdir(parents=True)
    audit = {r["file"]: r for r in csv.DictReader(open(ROOT / "reports/prov_v6/audit_orientasi_ds4/orientasi_per_gambar.csv", encoding="utf-8"))}
    man = [r for r in csv.DictReader(open(ROOT / "data/processed/prov_v6/manifest.csv", encoding="utf-8")) if r["source"] == "ds4"]
    rows = []
    for r in man:
        src, dst = SRC / r["split"] / r["file"], DST / r["file"]
        a = audit[r["file"]]
        varian = a["orientasi_label"] if a["orientasi_label"] != "tanpa_visual" else "asli"
        if varian not in K_CCW:
            sys.exit(f"[GAGAL] varian {varian} belum didukung ({r['file']})")
        m = info_meta(src)
        tag = m["tag"] if m["tag"] not in (None, 1) else None
        if varian == "asli" and tag is None:
            shutil.copy2(src, dst)
            rows.append({"file": r["file"], "split": r["split"], "class": r["class"], "aksi": "salin_byte_identik", "rotasi_cw_jpegtran": 0,
                         "identik_dengan_np_rot90": "", "selisih_maks": "", "icc_asli": m["icc"], "komentar_asli": m["komentar"],
                         "tag_baru": None, "sha256_lama": sha(src), "sha256_baru": sha(dst)})
            continue
        if tag is not None:
            if varian != "asli":
                sys.exit(f"[GAGAL] foto bertag EXIF sekaligus perlu rotasi tambahan: {r['file']} (tangani manual)")
            cw = TAG_CW[tag]
            aksi = f"tag_exif_{tag}_diterapkan"
        else:
            cw = (360 - 90 * K_CCW[varian]) % 360
            aksi = "diputar_lossless"
        res = subprocess.run([str(JPEGTRAN), "-perfect", "-copy", "none", "-rotate", str(cw), "-outfile", str(dst), str(src)],
                             capture_output=True, text=True)
        if res.returncode != 0:
            sys.exit(f"[GAGAL] jpegtran {r['file']}: {res.stderr}")
        with Image.open(src) as im:
            asli = np.asarray(im.convert("RGB"))                 # dekode MENTAH (tanpa menerapkan tag)
        harap = np.rot90(asli, k=(-cw // 90) % 4)                  # putar searah jarum jam cw derajat
        with Image.open(dst) as im2:
            hasil = np.asarray(im2.convert("RGB"))
            tag2 = im2.getexif().get(274)
        same = harap.shape == hasil.shape and np.array_equal(harap, hasil)
        dmax = int(np.abs(harap.astype(int) - hasil.astype(int)).max()) if harap.shape == hasil.shape else -1
        rows.append({"file": r["file"], "split": r["split"], "class": r["class"], "aksi": aksi, "rotasi_cw_jpegtran": cw,
                     "identik_dengan_np_rot90": same, "selisih_maks": dmax, "icc_asli": m["icc"], "komentar_asli": m["komentar"],
                     "tag_baru": tag2, "sha256_lama": sha(src), "sha256_baru": sha(dst)})
    REP.mkdir(parents=True, exist_ok=True)
    with open(REP / "jpegtran_per_foto.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    diputar = [r for r in rows if r["aksi"] != "salin_byte_identik"]
    summ = {"alat": {"jpegtran": str(JPEGTRAN.relative_to(ROOT)), "versi": versi, "sha256_jpegtran_exe": sha(JPEGTRAN),
                     "sumber_paket": "conda-forge libjpeg-turbo-3.2.0-hfd05255_1.conda (sha256 df78ab4c...9b53)",
                     "perintah": "jpegtran -perfect -copy none -rotate <cw> -outfile <dst> <src>"},
            "n_ds4": len(rows), "aksi": dict(Counter(r["aksi"] for r in rows)),
            "aksi_per_split": {sp: dict(Counter(r["aksi"] for r in rows if r["split"] == sp)) for sp in ("train", "val", "test", "test_new_session")},
            "foto_diubah": len(diputar),
            "tidak_identik_dengan_np_rot90": sum(1 for r in diputar if r["identik_dengan_np_rot90"] is not True),
            "selisih_piksel_maks": max((r["selisih_maks"] for r in diputar), default=None),
            "masih_bertag_exif": sum(1 for r in rows if r["tag_baru"] not in (None, 1)),
            "metadata_ikut_terbuang_pada_foto_diubah": {"icc": sum(1 for r in diputar if r["icc_asli"]), "komentar": sum(1 for r in diputar if r["komentar_asli"])},
            "salinan_identik_sha_cocok": all(r["sha256_lama"] == r["sha256_baru"] for r in rows if r["aksi"] == "salin_byte_identik")}
    (REP / "jpegtran_ringkasan.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summ, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
