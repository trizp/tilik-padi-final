#!/usr/bin/env python3
"""
ds3_tanpa_latar.py - Siapkan ds3 TANPA latar untuk evaluasi v9 dan laporkan kualitas pemisahan daun SEBELUM model dijalankan
(KEPUTUSAN_PROYEK.md 2026-10-17). Hanya gambar; TIDAK menjalankan model deteksi.

- Masker: SAM2 aturan v2 + cadangan (sama dengan prov_v9), data/derived/v9_latar/sam2_v2_ds3/masker_mentah (dibuat tanpa label).
- Gambar: data/derived/v9_latar/ds3_tanpa_latar/<folder>/<berkas> (latar abu-abu 114, JPEG q95; cadangan = salinan byte asli).
- Daftar: data/derived/v9_latar/ds3_tanpa_latar/daftar/{ds3_uji_utama, ds3_tambahan_160, ds3_tungro_luar_cakupan}.txt (group_id sama).
- Laporan: reports/final_eval_v9/ds3_pemisahan/{ringkasan.json, RINGKASAN.md, montase_ds3_20.jpg}

  .venv/Scripts/python scripts/v9_latar/ds3_tanpa_latar.py
"""
import json
import random
import shutil
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import umum  # noqa: E402

ROOT = umum.ROOT
DS3 = ROOT / "data/raw/ds3"
MASKER = umum.KERJA / "sam2_v2_ds3"
OUT_IMG = umum.KERJA / "ds3_tanpa_latar"
UD = ROOT / "data/annotation/verifikasi_ds3/uji_utama_final"
HIMP = {"utama": "ds3_uji_utama.txt", "tambahan": "ds3_tambahan_160.txt", "tungro": "ds3_tungro_luar_cakupan.txt"}
REP = ROOT / "reports/final_eval_v9/ds3_pemisahan"


def ubin(img, s=220):
    h, w = img.shape[:2]
    k = s / max(h, w)
    kecil = cv2.resize(img, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
    kan = np.full((s, s, 3), 255, np.uint8)
    y0, x0 = (s - kecil.shape[0]) // 2, (s - kecil.shape[1]) // 2
    kan[y0:y0 + kecil.shape[0], x0:x0 + kecil.shape[1]] = kecil
    return kan


def main():
    REP.mkdir(parents=True, exist_ok=True)
    (OUT_IMG / "daftar").mkdir(parents=True, exist_ok=True)
    info = json.loads((MASKER / "info.json").read_text(encoding="utf-8"))
    daftar = {n: [b.split("\t") for b in (UD / f).read_text(encoding="utf-8").splitlines() if b.strip()] for n, f in HIMP.items()}
    semua = sorted({b[0].replace("data/raw/ds3/", "") for v in daftar.values() for b in v})
    luas, cad = {}, {}
    for rel in semua:
        bgr = cv2.imread(str(DS3 / rel))
        mm = cv2.imread(str(MASKER / "masker_mentah" / (rel.replace("/", "__").rsplit(".", 1)[0] + ".png")), cv2.IMREAD_GRAYSCALE)
        c = bool(info.get(rel, {}).get("cadangan"))
        tujuan = OUT_IMG / rel
        tujuan.parent.mkdir(parents=True, exist_ok=True)
        if c:
            shutil.copy2(DS3 / rel, tujuan)
            luas[rel] = 1.0
        else:
            m = umum.pasca_proses(mm >= 128)
            out = bgr.copy()
            out[~m] = umum.ABU[::-1]
            cv2.imwrite(str(tujuan), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
            luas[rel] = float(m.mean())
        cad[rel] = c
    for n, rows in daftar.items():
        jalur = [f"{(OUT_IMG / b[0].replace('data/raw/ds3/', '')).relative_to(ROOT).as_posix()}\t{b[1]}" for b in rows]
        (OUT_IMG / "daftar" / HIMP[n]).write_text("\n".join(jalur) + "\n", encoding="utf-8")
    ring = {"LABEL": "kualitas pemisahan daun pada ds3 (gambar saja, sebelum model dijalankan)", "per_himpunan": {}, "per_folder": {}}
    for n, rows in daftar.items():
        fs = [b[0].replace("data/raw/ds3/", "") for b in rows]
        per = defaultdict(list)
        for f in fs:
            per[f.split("/")[0]].append(f)
        ring["per_himpunan"][n] = {"n": len(fs), "cadangan": sum(cad[f] for f in fs),
                                   "persen_cadangan": round(100 * sum(cad[f] for f in fs) / len(fs), 1),
                                   "median_luas_masker_dipisah": round(st.median([luas[f] for f in fs if not cad[f]] or [float("nan")]), 4),
                                   "per_folder": {k: {"n": len(v), "cadangan": sum(cad[f] for f in v), "persen_cadangan": round(100 * sum(cad[f] for f in v) / len(v), 1)}
                                                  for k, v in sorted(per.items())}}
    for k in ("blast", "blight", "tungro"):
        fs = [f for f in semua if f.startswith(k + "/")]
        ring["per_folder"][k] = {"n": len(fs), "persen_cadangan": round(100 * sum(cad[f] for f in fs) / len(fs), 1),
                                 "median_luas_masker_dipisah": round(st.median([luas[f] for f in fs if not cad[f]]), 4) if any(not cad[f] for f in fs) else None}
    # montase 20 contoh (7 blast, 7 blight, 6 tungro; acak seed tetap), asli | masker | hasil
    rng = random.Random(umum.SEED + 3)
    contoh = []
    for k, n in (("blast", 7), ("blight", 7), ("tungro", 6)):
        contoh += rng.sample([f for f in semua if f.startswith(k + "/")], n)
    sel = []
    for f in contoh:
        bgr = cv2.imread(str(DS3 / f))
        mm = cv2.imread(str(MASKER / "masker_mentah" / (f.replace("/", "__").rsplit(".", 1)[0] + ".png")), cv2.IMREAD_GRAYSCALE)
        m = np.ones(bgr.shape[:2], bool) if cad[f] else umum.pasca_proses(mm >= 128)
        hasil = cv2.imread(str(OUT_IMG / f))
        trip = np.hstack([ubin(bgr), ubin(np.repeat((m.astype(np.uint8) * 255)[..., None], 3, 2)), ubin(hasil)])
        teks = f"{f.split('/')[0]}{' (CADANGAN)' if cad[f] else ''}"
        cv2.putText(trip, teks, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(trip, teks, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        sel.append(np.pad(trip, ((4, 4), (4, 4), (0, 0)), constant_values=255))
    baris = [np.hstack(sel[i:i + 2]) for i in range(0, len(sel), 2)]
    judul = np.full((36, baris[0].shape[1], 3), 255, np.uint8)
    cv2.putText(judul, "ds3: asli | masker | hasil (latar abu 114). SAM2 aturan v2 + cadangan. Tanpa model deteksi.", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    cv2.imwrite(str(REP / "montase_ds3_20.jpg"), np.vstack([judul] + baris), [cv2.IMWRITE_JPEG_QUALITY, 88])
    ring["contoh_montase"] = contoh
    (REP / "ringkasan.json").write_text(json.dumps(ring, indent=2, ensure_ascii=False), encoding="utf-8")
    md = ["# Kualitas pemisahan daun pada ds3 (sebelum model dijalankan)", "",
          "Gambar saja; model deteksi belum dijalankan pada ds3. SAM2 aturan v2 + cadangan (sama dengan prov_v9).", "",
          "| Himpunan | Foto | Cadangan (tidak dipisah) | Median luas masker (yang dipisah) |", "|---|---|---|---|"]
    md += [f"| {n} | {v['n']} | {v['cadangan']} ({v['persen_cadangan']}%) | {100 * v['median_luas_masker_dipisah']:.1f}% foto |" for n, v in ring["per_himpunan"].items()]
    md += ["", "| Folder | Foto | % cadangan | Median luas masker |", "|---|---|---|---|"]
    md += [f"| {k} | {v['n']} | {v['persen_cadangan']}% | {('-' if v['median_luas_masker_dipisah'] is None else f'{100 * v['median_luas_masker_dipisah']:.1f}% foto')} |"
           for k, v in ring["per_folder"].items()]
    md += ["", "Montase 20 contoh: `montase_ds3_20.jpg`."]
    (REP / "RINGKASAN.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
