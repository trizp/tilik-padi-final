"""Bandingkan masker latar turunan (v7 vs v9, kontrol_latar_val.masker_latar) dengan masker SAM2 asli (pasca-proses sama
dengan bangun_prov_v9). Hanya foto VAL bersih. Tidak menulis apa pun ke proyek."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"D:\TELKOM\SKRIPSI\rice_yolo")
sys.path.insert(0, str(ROOT / "scripts" / "paper"))
sys.path.insert(0, str(ROOT / "scripts" / "v9_latar"))
import kontrol_latar_val as k  # noqa: E402
import umum  # noqa: E402

V7 = ROOT / "data/derived/prov_v7_valsel/images/val"
V9 = ROOT / "data/derived/prov_v9_valsel/images/val"
MASK = ROOT / "data/derived/v9_latar/sam2_v2/masker_mentah"
info = json.loads((ROOT / "data/derived/v9_latar/sam2_v2/info.json").read_text(encoding="utf-8"))
f9 = {p.name for p in V9.iterdir()}
f7 = sorted(p.name for p in V7.iterdir())
print("v7", len(f7), "v9", len(f9), "v7 tanpa pasangan v9", len([f for f in f7 if f not in f9]))
cad = [f for f in f7 if info.get(f, {}).get("cadangan")]
print("cadangan SAM2 di val bersih:", len(cad), "; di antaranya punya pasangan v9:", len([f for f in cad if f in f9]))
rng = np.random.default_rng(0)
sampel = [f for f in f7 if f in f9 and not info.get(f, {}).get("cadangan")]
sampel = [sampel[i] for i in rng.choice(len(sampel), 120, replace=False)]
for tol in (int(x) for x in sys.argv[1:] or [6]):
    salah, iou, daun_jadi_latar = [], [], []
    for f in sampel:
        a, b = cv2.imread(str(V7 / f)), cv2.imread(str(V9 / f))
        bg, _ = k.masker_latar(a, b, tol_abu=tol, tol_beda=8)
        m = umum.pasca_proses(cv2.imread(str(MASK / (Path(f).stem + ".png")), cv2.IMREAD_GRAYSCALE) >= 128)
        bg_asli = ~m
        salah.append(float((bg != bg_asli).mean()))
        daun_jadi_latar.append(float((bg & ~bg_asli).mean()))
        iou.append(float((bg & bg_asli).sum() / max((bg | bg_asli).sum(), 1)))
    s = np.array(salah)
    print(f"tol_abu={tol}: piksel tidak cocok rata-rata {100 * s.mean():.3f}% (median {100 * np.median(s):.3f}%, maks {100 * s.max():.2f}%); "
          f"IoU latar rata-rata {np.mean(iou):.4f}, min {np.min(iou):.4f}; daun dianggap latar {100 * np.mean(daun_jadi_latar):.4f}%, "
          f"latar tertinggal {100 * (s.mean() - np.mean(daun_jadi_latar)):.3f}%")
