"""Montase pasangan 'kembar' menurut ambang longgar (ANALISIS UNTUK REVISI PAPER): pemeriksaan visual apakah pasangan
DINOv2 0,95, SIFT 15, dan waktu 10 dtk benar-benar foto daun yang sama. Membaca per_foto_kembar.csv dari
sensitivitas_kelompok.py; foto dibaca dari data/processed/prov_v7 (hanya dibaca)."""
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/revisi_paper/sensitivitas_kelompok"
IMG = ROOT / "data/processed/prov_v7/images"
m = pd.read_csv(ROOT / "data/processed/prov_v7/manifest.csv").set_index("file")
p = pd.read_csv(OUT / "per_foto_kembar.csv")
S = 220


def baca(f):
    im = cv2.imread(str(IMG / m.loc[f, "split"] / f))
    h, w = im.shape[:2]
    s = S / max(h, w)
    im = cv2.resize(im, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    kan = np.full((S, S, 3), 255, np.uint8)
    kan[(S - im.shape[0]) // 2:(S - im.shape[0]) // 2 + im.shape[0], (S - im.shape[1]) // 2:(S - im.shape[1]) // 2 + im.shape[1]] = im
    return kan


def label(im, teks):
    im = im.copy()
    cv2.rectangle(im, (0, 0), (S, 18), (0, 0, 0), -1)
    cv2.putText(im, teks[:34], (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    return im


blok = [("DINOv2 0,95: test", "DINOv2 0,95", "test", "dino_0.95"),
        ("DINOv2 0,95: test_new_session", "DINOv2 0,95", "test_new_session", "dino_0.95"),
        ("SIFT 15: test", "SIFT 15 inlier", "test", "sift_15"),
        ("waktu 10 dtk: test", "waktu 10 dtk", "test", "waktu_10")]
rng = random.Random(0)
baris_img, daftar = [], []
for judul, peng, split, krit in blok:
    q = p[(p.pengaturan == peng) & (p.split == split) & (p.contoh_kriteria == krit)]
    pilih = q.sample(min(4, len(q)), random_state=rng.randint(0, 10 ** 6)) if len(q) else q
    sel = []
    for r in pilih.itertuples():
        nilai = f"cos {r.nilai:.3f}" if krit.startswith("dino") else (f"{int(r.nilai)} inlier" if krit.startswith("sift") else f"{r.nilai:.0f} dtk")
        sel.append(np.hstack([label(baca(r.foto), f"{split[:8]} {r.kelas[:10]}"), label(baca(r.contoh_kembaran_train), f"train {nilai}")]))
        daftar.append(f"{judul} | {r.foto} ~ {r.contoh_kembaran_train} | {nilai}")
    while len(sel) < 4:
        sel.append(np.full((S, 2 * S, 3), 255, np.uint8))
    head = np.full((24, 8 * S + 3 * 6, 3), 235, np.uint8)
    cv2.putText(head, judul, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    sep = np.full((S, 6, 3), 160, np.uint8)
    baris_img += [head, np.hstack([sel[0], sep, sel[1], sep, sel[2], sep, sel[3]])]
cv2.imwrite(str(OUT / "montase_kembar_longgar.jpg"), np.vstack(baris_img), [cv2.IMWRITE_JPEG_QUALITY, 88])
(OUT / "montase_daftar.txt").write_text("\n".join(daftar) + "\n", encoding="utf-8")
print("\n".join(daftar))
