"""umum.py - Bagian bersama eksperimen v9 (latar dihapus): jalur, pemilihan sampel tahap 0, pasca-proses masker, pengukuran.

Pemisahan daun TIDAK memakai label: kotak bercak hanya dibaca oleh fungsi pengukuran (retensi_kotak), tidak oleh pembuat masker.
Hanya VAL prov_v7; test, test_new_session, dan ds3 tidak disentuh.
"""
import csv
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
VAL_IMG = ROOT / "data/processed/prov_v7/images/val"
VAL_LBL = ROOT / "data/processed/prov_v7/labels/val"
MANIFEST = ROOT / "data/processed/prov_v7/manifest.csv"
KERJA = ROOT / "data/derived/v9_latar"           # masker dan citra hasil (bukan dataset latih)
LAPORAN = ROOT / "reports/v9_latar"
ABU = (114, 114, 114)
SEED = 20261016
STRATA = {("ds4", "leaf_blast"): 30, ("ds4", "brown_spot"): 30, ("ds4", "background"): 20, ("ds1_provisional", "bacterial_leaf_blight"): 20}


def manifest_val():
    return [r for r in csv.DictReader(open(MANIFEST, encoding="utf-8")) if r["split"] == "val"]


def sampel_tahap0():
    """±100 foto VAL, acak berstrata (seed tetap), tidak dipilih berdasarkan hasil."""
    rows = manifest_val()
    rng = random.Random(SEED)
    out = []
    for (src, kls), n in STRATA.items():
        calon = sorted(r["file"] for r in rows if r["source"] == src and r["class"] == kls)
        out += [(f, src, kls) for f in rng.sample(calon, n)]
    return out


def baca_kotak(berkas, w, h):
    p = VAL_LBL / (Path(berkas).stem + ".txt")
    kotak = []
    if p.exists():
        for baris in p.read_text().split("\n"):
            if baris.strip():
                c, x, y, bw, bh = map(float, baris.split()[:5])
                kotak.append((int(c), (x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h))
    return kotak


def isi_lubang(m):
    """Isi lubang di dalam masker (bercak di tengah daun sering terpisah dari daun), lewat flood fill dari tepi."""
    import cv2
    m = m.astype(np.uint8)
    h, w = m.shape
    pad = np.zeros((h + 2, w + 2), np.uint8)
    pad[1:-1, 1:-1] = m
    luar = pad.copy()
    cv2.floodFill(luar, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 1)
    lubang = (luar == 0)[1:-1, 1:-1]
    return (m > 0) | lubang


def pasca_proses(m, min_relatif=0.10):
    """Masker biner -> buang komponen kecil (< 10% komponen terbesar) -> isi lubang -> tutup celah kecil."""
    import cv2
    m = (m > 0).astype(np.uint8)
    if m.sum() == 0:
        return m.astype(bool)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 2:
        luas = stats[1:, cv2.CC_STAT_AREA]
        simpan = 1 + np.where(luas >= min_relatif * luas.max())[0]
        m = np.isin(lab, simpan).astype(np.uint8)
    k = max(3, round(min(m.shape) / 150)) | 1
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return isi_lubang(m)


def terapkan(im_rgb, m):
    out = im_rgb.copy()
    out[~m] = ABU
    return out


def retensi_kotak(m, kotak):
    """Bagian luas tiap kotak yang berada di dalam masker (0-1). Hanya untuk PENGUKURAN."""
    h, w = m.shape
    hasil = []
    for c, x1, y1, x2, y2 in kotak:
        a, b = max(0, int(np.floor(x1))), min(w, int(np.ceil(x2)))
        cc, d = max(0, int(np.floor(y1))), min(h, int(np.ceil(y2)))
        if b <= a or d <= cc:
            hasil.append((c, 0.0, 0))
            continue
        sub = m[cc:d, a:b]
        hasil.append((c, float(sub.mean()), sub.size))
    return hasil
