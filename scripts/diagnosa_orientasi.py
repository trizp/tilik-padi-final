#!/usr/bin/env python3
"""
diagnosa_orientasi.py - Diagnosis penyebab label ds4 tidak cocok (HANYA BACA).

1. Tag EXIF Orientation di semua sumber: ds4 Original images, ds4 Annotated visuals, ds1, ds2 Original, ds3, dan berkas
   dataset prov_v6. Untuk ds4 dirinci per kelompok audit (kotak cocok vs tidak cocok).
2. Cara pemuat membaca foto: PIL Image.open (tanpa transpose), PIL + ImageOps.exif_transpose, cv2.imread (IMREAD_COLOR),
   cv2.imread (IMREAD_IGNORE_ORIENTATION), dan ultralytics.utils.patches.imread (dipakai dataloader train/val).
   Dicek apakah keempatnya memberi orientasi yang sama, pada contoh cocok, tidak cocok, dan semua foto bertag orientasi.
Keluaran: reports/prov_v6/audit_orientasi_ds4/diagnosa.json
"""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
OUT = ROOT / "reports/prov_v6/audit_orientasi_ds4"


def orient(p):
    try:
        return Image.open(p).getexif().get(274)
    except Exception:
        return "galat"


def semua(root, cond=lambda s: True):
    return [p for p in (ROOT / root).rglob("*") if p.suffix.lower() in IMG and cond(p.as_posix().lower())]


def muat(p):
    """Kembalikan array RGB kecil (64x64 abu) dari tiap pemuat + ukuran asli."""
    from ultralytics.utils.patches import imread as ul_imread
    out = {}
    a = Image.open(p)
    out["PIL_open"] = np.asarray(a.convert("RGB"))
    out["PIL_exif_transpose"] = np.asarray(ImageOps.exif_transpose(a).convert("RGB"))
    out["cv2_imread"] = cv2.cvtColor(cv2.imread(str(p), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    out["cv2_ignore_orientation"] = cv2.cvtColor(cv2.imread(str(p), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION), cv2.COLOR_BGR2RGB)
    out["ultralytics_imread"] = cv2.cvtColor(ul_imread(p), cv2.COLOR_BGR2RGB)
    return out


def sama(a, b):
    if a.shape != b.shape:
        return False
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean()) < 3.0     # toleransi dekode JPEG


def main():
    hasil = {}
    # ---- 1. tag EXIF per sumber
    sumber = {"ds4_original": semua("data/raw/ds4", lambda s: "/original images/" in s),
              "ds4_visuals": semua("data/raw/ds4", lambda s: "/visuals/" in s),
              "ds1": semua("data/raw/ds1"), "ds2_original": semua("data/raw/ds2", lambda s: "/original/" in s),
              "ds3": semua("data/raw/ds3"), "prov_v6_dataset": semua("data/processed/prov_v6/images")}
    tag = {}
    for k, ps in sumber.items():
        c = Counter(orient(p) for p in ps)
        tag[k] = {"n": len(ps), "orientasi": {str(o): n for o, n in c.items()},
                  "bertag_rotasi_atau_flip(2-8)": sum(n for o, n in c.items() if o in (2, 3, 4, 5, 6, 7, 8))}
        print(k, tag[k], flush=True)
    hasil["tag_exif_per_sumber"] = tag

    # ---- ds4 per kelompok audit
    audit = list(csv.DictReader(open(OUT / "orientasi_per_gambar.csv", encoding="utf-8")))
    per = defaultdict(Counter)
    for r in audit:
        if r["orientasi_label"] in ("tanpa_visual",):
            continue
        g = "tidak_cocok" if r["label_tidak_cocok"] == "True" else "cocok"
        per[g][str(orient(ROOT / r["orig_path"]))] += 1
    hasil["ds4_tag_exif_per_kelompok_audit"] = {g: dict(c) for g, c in per.items()}
    print("ds4 per kelompok audit:", hasil["ds4_tag_exif_per_kelompok_audit"], flush=True)

    # ---- 2. pemuat
    rng = np.random.default_rng(0)
    tc = [r for r in audit if r["label_tidak_cocok"] == "True"]
    co = [r for r in audit if r["label_tidak_cocok"] == "False" and r["class"] != "background"]
    contoh = [("tidak_cocok", ROOT / tc[i]["orig_path"]) for i in rng.choice(len(tc), 10, replace=False)] + \
             [("cocok", ROOT / co[i]["orig_path"]) for i in rng.choice(len(co), 10, replace=False)]
    bertag = [p for k in sumber for p in sumber[k] if orient(p) in (2, 3, 4, 5, 6, 7, 8)]
    contoh += [("bertag_exif", p) for p in bertag]
    rinci = []
    for g, p in contoh:
        L = muat(p)
        ref = L["ultralytics_imread"]
        rinci.append({"kelompok": g, "berkas": p.relative_to(ROOT).as_posix(), "exif": orient(p),
                      **{f"sama_dengan_ultralytics__{k}": sama(v, ref) for k, v in L.items() if k != "ultralytics_imread"},
                      "ukuran_PIL_open": list(L["PIL_open"].shape[:2]), "ukuran_ultralytics": list(ref.shape[:2])})
    hasil["pemuat"] = {"catatan": "ultralytics_imread = cv2.imdecode(IMREAD_COLOR) = yang dipakai dataloader train/val/predict",
                       "ringkas": {g: {k: sum(r[k] for r in rinci if r["kelompok"] == g) for k in rinci[0] if k.startswith("sama_")} |
                                   {"n": sum(1 for r in rinci if r["kelompok"] == g)} for g in ("tidak_cocok", "cocok", "bertag_exif")},
                       "rinci": rinci}
    (OUT / "diagnosa.json").write_text(json.dumps(hasil, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(hasil["pemuat"]["ringkas"], indent=1, ensure_ascii=False))
    for r in rinci:
        if r["kelompok"] == "bertag_exif":
            print(r)


if __name__ == "__main__":
    main()
