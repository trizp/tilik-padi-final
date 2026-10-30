#!/usr/bin/env python3
"""
cek_orientasi_ds1.py - Cek orientasi label ds1 (HANYA foto TRAIN; val/test tidak disentuh). Tidak memperbaiki apa pun.

Model: best.pt final v6 (baku yolo11s seed 2). Untuk tiap foto: kotak prediksi (conf >= 0.05, tanpa memandang kelas)
dibandingkan dengan kotak label yang DIPUTAR 0/90/180/270 derajat (koordinat ternormalisasi, putar berlawanan jarum jam:
(x, y) -> (y, 1 - x), lebar/tinggi bertukar). Skor rotasi = rata-rata IoU terbaik tiap kotak label terhadap prediksi.
"Jauh lebih cocok di rotasi lain" = skor terbaik non-0 >= skor 0 derajat + 0.2 DAN skor terbaik >= 0.3.
"Tidak dapat dinilai" = semua skor < 0.1 (mis. tidak ada prediksi).

Himpunan:
  ds1_train        : semua foto ds1 berlabel di split train prov_v6 (Blast/Brown spot ds1, BLB ds1_provisional)
  kontrol_ds4_fix  : 200 foto ds4 train berlabel acak, versi SUDAH diperbaiki (jpegtran)  -> harus hampir semua 0 derajat
  kontrol_ds4_salah: 100 foto ds4 train yang diketahui salah orientasi, versi prov_v6 (belum diperbaiki) -> harus tertandai
Keluaran: reports/prov_v7_prep/cek_orientasi_ds1/{per_foto.csv, ringkasan.json, montase_ds1.jpg}
"""
import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
R6 = ROOT / "data/processed/prov_v6"
FIX = ROOT / "data/interim/ds4_orientasi_v7_jpegtran"
OUT = ROOT / "reports/prov_v7_prep/cek_orientasi_ds1"


def baca_label(p):
    return [list(map(float, l.split()[1:5])) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def putar(b, k):
    x, y, w, h = b
    for _ in range(k):
        x, y, w, h = y, 1 - x, h, w
    return [x, y, w, h]


def iou(a, b, W, H):
    ax1, ay1, ax2, ay2 = (a[0] - a[2] / 2) * W, (a[1] - a[3] / 2) * H, (a[0] + a[2] / 2) * W, (a[1] + a[3] / 2) * H
    bx1, by1, bx2, by2 = b
    iw, ih = max(0, min(ax2, bx2) - max(ax1, bx1)), max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    u = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / u if u > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/final_prov_v6_yolo11s_s2/weights/best.pt")
    ap.add_argument("--conf", type=float, default=0.05)
    a = ap.parse_args()
    from ultralytics import YOLO
    OUT.mkdir(parents=True, exist_ok=True)
    man = list(csv.DictReader(open(R6 / "manifest.csv", encoding="utf-8")))
    aud = {r["file"]: r for r in csv.DictReader(open(ROOT / "reports/prov_v6/audit_orientasi_ds4/orientasi_per_gambar.csv", encoding="utf-8"))}
    rng = random.Random(20261010)
    item = []
    for r in man:
        if r["split"] == "train" and r["source"].startswith("ds1") and r["class"] != "background":
            item.append(("ds1_train", r, R6 / "images/train" / r["file"]))
    ds4tr = [r for r in man if r["split"] == "train" and r["source"] == "ds4" and r["class"] != "background"]
    for r in rng.sample(ds4tr, 200):
        item.append(("kontrol_ds4_fix", r, FIX / r["file"]))
    salah = [r for r in ds4tr if aud[r["file"]]["label_tidak_cocok"] == "True"]
    for r in rng.sample(salah, 100):
        item.append(("kontrol_ds4_salah", r, R6 / "images/train" / r["file"]))
    print({k: n for k, n in Counter(i[0] for i in item).items()}, flush=True)

    model = YOLO(str(ROOT / a.weights))
    rows = []
    B = 16
    for s in range(0, len(item), B):
        chunk = item[s:s + B]
        res = model.predict([str(p) for _, _, p in chunk], imgsz=640, conf=a.conf, iou=0.7, device=0, verbose=False)
        for (him, r, p), rr in zip(chunk, res):
            H, W = rr.orig_shape
            pred = rr.boxes.xyxy.cpu().numpy().tolist()
            lab = baca_label(R6 / "labels/train" / (Path(r["file"]).stem + ".txt"))
            skor = []
            for k in range(4):
                if not pred or not lab:
                    skor.append(0.0)
                    continue
                skor.append(float(np.mean([max(iou(putar(b, k), q, W, H) for q in pred) for b in lab])))
            kb = int(np.argmax(skor))
            best_non0 = max(skor[1:])
            if max(skor) < 0.1:
                status = "tidak_dapat_dinilai"
            elif best_non0 >= skor[0] + 0.2 and best_non0 >= 0.3:
                status = "jauh_lebih_cocok_rotasi_lain"
            else:
                status = "cocok_0_derajat_atau_tidak_jelas"
            rows.append({"himpunan": him, "file": r["file"], "class": r["class"], "source": r["source"], "n_label": len(lab),
                         "n_prediksi": len(pred), "skor_0": round(skor[0], 3), "skor_90": round(skor[1], 3), "skor_180": round(skor[2], 3),
                         "skor_270": round(skor[3], 3), "rotasi_terbaik": 90 * kb, "status": status, "path": str(p.relative_to(ROOT))})
    with open(OUT / "per_foto.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summ = {"model": a.weights, "conf": a.conf, "aturan": "jauh lebih cocok: skor non-0 terbaik >= skor 0 + 0.2 dan >= 0.3; tidak dapat dinilai: semua skor < 0.1",
            "himpunan": {}}
    for him in ("ds1_train", "kontrol_ds4_fix", "kontrol_ds4_salah"):
        for kel in sorted({r["class"] for r in rows if r["himpunan"] == him}) + ["SEMUA"]:
            rs = [r for r in rows if r["himpunan"] == him and (kel == "SEMUA" or r["class"] == kel)]
            c = Counter(r["status"] for r in rs)
            summ["himpunan"].setdefault(him, {})[kel] = {"n": len(rs), **dict(c),
                                                          "persen_rotasi_lain": round(100 * c["jauh_lebih_cocok_rotasi_lain"] / len(rs), 1),
                                                          "rotasi_terbaik_pada_yang_tertandai": dict(Counter(r["rotasi_terbaik"] for r in rs if r["status"] == "jauh_lebih_cocok_rotasi_lain"))}
    (OUT / "ringkasan.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summ, indent=1, ensure_ascii=False))

    # montase 10 contoh ds1 (yang tertandai; bila < 10, ditambah selisih skor terbesar)
    d1 = [r for r in rows if r["himpunan"] == "ds1_train"]
    tand = [r for r in d1 if r["status"] == "jauh_lebih_cocok_rotasi_lain"]
    sisa = sorted([r for r in d1 if r not in tand], key=lambda r: -(max(r["skor_90"], r["skor_180"], r["skor_270"]) - r["skor_0"]))
    pilih = (rng.sample(tand, 10) if len(tand) >= 10 else tand + sisa[:10 - len(tand)])
    T = 360
    cv = Image.new("RGB", (5 * T, 2 * (T + 34) + 26), "white")
    dd = ImageDraw.Draw(cv)
    dd.text((6, 6), "ds1 TRAIN: merah = label apa adanya (0 derajat), hijau = label diputar ke rotasi terbaik, biru = prediksi model v6", fill="black")
    for i, r in enumerate(pilih):
        im = Image.open(ROOT / r["path"]).convert("RGB")
        W, H = im.size
        d = ImageDraw.Draw(im)
        lw = max(2, W // 160)
        q = model.predict(str(ROOT / r["path"]), imgsz=640, conf=a.conf, iou=0.7, device=0, verbose=False)[0]
        for bx in q.boxes.xyxy.cpu().numpy():
            d.rectangle(list(bx), outline=(42, 120, 214), width=lw)
        for b in baca_label(R6 / "labels/train" / (Path(r["file"]).stem + ".txt")):
            for k, col in ((0, (227, 73, 72)), (r["rotasi_terbaik"] // 90, (27, 175, 122))):
                if k == 0 and col != (227, 73, 72):
                    continue
                x, y, w, h = putar(b, k)
                d.rectangle([(x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H], outline=col, width=lw)
        im.thumbnail((T, T))
        X, Y = (i % 5) * T, 26 + (i // 5) * (T + 34)
        cv.paste(im, (X, Y + 34))
        dd.text((X + 3, Y + 2), f"{i + 1}. {r['class']} | {r['status'][:22]} | terbaik {r['rotasi_terbaik']}", fill="black")
        dd.text((X + 3, Y + 17), f"skor 0/90/180/270: {r['skor_0']}/{r['skor_90']}/{r['skor_180']}/{r['skor_270']}", fill=(80, 80, 80))
    cv.save(OUT / "montase_ds1.jpg", quality=88)
    with open(OUT / "montase_ds1_daftar.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(pilih[0]))
        w.writeheader()
        w.writerows(pilih)


if __name__ == "__main__":
    main()
