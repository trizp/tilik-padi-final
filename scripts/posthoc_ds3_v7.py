#!/usr/bin/env python3
"""
posthoc_ds3_v7.py - ANALISIS POST-HOC ds3 (tidak mengubah model, ambang, atau angka utama; tidak dipakai menyetel apa pun).

1. Tabel analisis tambahan (160 foto, label asli): model final v7 (dari tabel yang sudah ada) + model web (eval_ds3_tambahan_posthoc).
2. Montase 12 contoh model web: 4 Blast -> Brown spot, 4 BLB benar, 4 Tungro -> BLB, dengan kotak prediksi (conf >= 0,29).
   Model dijalankan ulang pada 12 foto ini HANYA untuk menggambar.
3. Karakteristik foto ds3 vs ds4 vs ds1: resolusi, kecerahan, porsi latar terang-netral (proksi latar putih), porsi piksel hijau
   (proksi luas daun), ukuran relatif kotak (ds4/ds1: kotak label; ds3: kotak prediksi model web, conf >= 0,29 — tanpa label kotak).
Keluaran: reports/final_eval_v7/ds3/analisis_posthoc/
"""
import csv
import json
import random
import statistics as S
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports/final_eval_v7/ds3/analisis_posthoc"
RUNW = ROOT / "runs/v7_deploy_yolo11s_s2"
W = RUNW / "weights/last.pt"
WARNA = {0: (42, 120, 214), 1: (235, 104, 52), 2: (27, 175, 122)}
NAMA = {0: "Blast", 1: "Brown spot", 2: "BLB"}


def ciri(path):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    w, h = im.size
    a = np.asarray(im.resize((256, round(256 * h / w)) if w >= h else (round(256 * w / h), 256)), np.float32) / 255
    hsv = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).convert("HSV"), np.float32) / 255
    terang_netral = ((a.mean(2) > 0.75) & (hsv[..., 1] < 0.15)).mean()
    hijau = ((hsv[..., 0] > 0.17) & (hsv[..., 0] < 0.45) & (hsv[..., 1] > 0.25) & (hsv[..., 2] > 0.2)).mean()
    return {"W": w, "H": h, "megapiksel": w * h / 1e6, "kecerahan": float(a.mean()), "latar_terang_netral": float(terang_netral), "piksel_hijau": float(hijau)}


def main():
    from ultralytics import YOLO
    OUT.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(W))

    # ---- 1. tabel tambahan 160
    tab = [dict(r, Catatan="final v7 (ambang terkunci tiap model)") for r in csv.DictReader(open(ROOT / "reports/final_eval_v7/ds3/tabel_ds3_tambahan_per_model.csv", encoding="utf-8-sig"))]
    d = json.loads((RUNW / "eval_ds3_tambahan_posthoc/ringkasan_diagnosis.json").read_text(encoding="utf-8"))["dalam_cakupan"]
    pk = {r["kelas"]: r for r in d["per_kelas"]}
    tab.append({"Split latih": "model web (semua data, 1 run)", "Model": "YOLO11s", "Ambang": "0.35", "Akurasi": f"{d['akurasi']:.3f}", "IK95 akurasi": "-",
                "P BLB": f"{pk['bacterial_leaf_blight']['precision']:.3f}", "R BLB": f"{pk['bacterial_leaf_blight']['recall']:.3f}", "IK95 R BLB": "-",
                "F1 BLB": f"{pk['bacterial_leaf_blight']['f1']:.3f}", "P Blast": f"{pk['leaf_blast']['precision']:.3f}", "R Blast": f"{pk['leaf_blast']['recall']:.3f}",
                "IK95 R Blast": "-", "F1 Blast": f"{pk['leaf_blast']['f1']:.3f}", "Catatan": "POST-HOC; model web (pembanding)"})
    with open(OUT / "tambahan_160_final_dan_model_web.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(tab[0]))
        w.writeheader()
        w.writerows(tab)

    # ---- 2. montase
    rng = random.Random(14)
    rows_u = list(csv.DictReader(open(RUNW / "eval_ds3_tambahan_posthoc/diagnosis_dalam_cakupan_per_citra.csv", encoding="utf-8")))
    rows_t = list(csv.DictReader(open(RUNW / "eval_ds3_tungro_posthoc/diagnosis_luar_cakupan_per_citra.csv", encoding="utf-8")))
    grup = [("Blast -> Brown spot", [r for r in rows_u if r["label"] == "leaf_blast" and r["diagnosis"] == "brown_spot"]),
            ("BLB benar", [r for r in rows_u if r["label"] == "bacterial_leaf_blight" and r["diagnosis"] == "bacterial_leaf_blight"]),
            ("Tungro -> BLB", [r for r in rows_t if r["diagnosis"] == "bacterial_leaf_blight"])]
    pilih = [(g, r) for g, rs in grup for r in rng.sample(rs, min(4, len(rs)))]
    T = 300
    cv = Image.new("RGB", (4 * T, 3 * (T + 34) + 30), "white")
    dd = ImageDraw.Draw(cv)
    dd.text((6, 6), "POST-HOC - model web (YOLO11s, last.pt), ds3. Kotak prediksi conf >= 0,29: biru Blast, oranye Brown spot, hijau BLB.", fill="black")
    for i, (g, r) in enumerate(pilih):
        im = ImageOps.exif_transpose(Image.open(r["berkas"])).convert("RGB")
        res = model.predict(r["berkas"], imgsz=640, conf=0.29, iou=0.7, device=0, verbose=False)[0]
        dr = ImageDraw.Draw(im)
        lw = max(3, im.width // 150)
        for bx, c, cf in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy().astype(int), res.boxes.conf.cpu().numpy()):
            dr.rectangle(list(bx), outline=WARNA[c], width=lw)
        im.thumbnail((T, T))
        X, Y = (i % 4) * T, 30 + (i // 4) * (T + 34)
        cv.paste(im, (X, Y + 34))
        dd.text((X + 3, Y + 2), f"{g} | {Path(r['berkas']).name}", fill="black")
        dd.text((X + 3, Y + 17), f"teratas {NAMA.get({'leaf_blast': 0, 'brown_spot': 1, 'bacterial_leaf_blight': 2}.get(r['kelas_teratas'], -1), '-')} {float(r['conf_teratas']):.2f}", fill=(80, 80, 80))
    cv.save(OUT / "montase_kesalahan_ds3.jpg", quality=88)

    # ---- 3. karakteristik
    man = list(csv.DictReader(open(ROOT / "data/processed/prov_v7/manifest.csv", encoding="utf-8")))
    rr = random.Random(7)
    himp = {"ds4 (train, berlabel)": rr.sample([r for r in man if r["split"] == "train" and r["source"] == "ds4" and r["class"] != "background"], 200),
            "ds1 BLB (train)": rr.sample([r for r in man if r["split"] == "train" and r["source"] == "ds1_provisional"], 200),
            "ds1 Blast/Brown spot (train)": rr.sample([r for r in man if r["split"] == "train" and r["source"] == "ds1"], 200)}
    hasil = {}
    for nama, rows in himp.items():
        c, kotak, nk = [], [], []
        for r in rows:
            p = ROOT / "data/processed/prov_v7/images/train" / r["file"]
            c.append(ciri(p))
            lab = ROOT / "data/processed/prov_v7/labels/train" / (Path(r["file"]).stem + ".txt")
            ls = [list(map(float, x.split()[1:5])) for x in lab.read_text().splitlines() if x.strip()]
            kotak += [b[2] * b[3] for b in ls]
            nk.append(len(ls))
        hasil[nama] = (c, kotak, nk, "kotak label")
    for nama, daftar in (("ds3 blight+blast (160)", "ds3_tambahan_160.txt"), ("ds3 tungro (80)", "ds3_tungro_luar_cakupan.txt")):
        paths = [ROOT / ln.split("\t")[0] for ln in (ROOT / "data/annotation/verifikasi_ds3/uji_utama_final" / daftar).read_text(encoding="utf-8").splitlines() if ln.strip()]
        c, kotak, nk = [], [], []
        for p in paths:
            c.append(ciri(p))
            res = model.predict(str(p), imgsz=640, conf=0.29, iou=0.7, device=0, verbose=False)[0]
            wh = res.boxes.xywhn.cpu().numpy()
            kotak += [float(b[2] * b[3]) for b in wh]
            nk.append(len(wh))
        hasil[nama] = (c, kotak, nk, "kotak PREDIKSI model web (conf>=0,29)")
    tabel = []
    for nama, (c, kotak, nk, sumber) in hasil.items():
        tabel.append({"Himpunan": nama, "n foto": len(c),
                      "Resolusi median (WxH)": f"{int(S.median(x['W'] for x in c))}x{int(S.median(x['H'] for x in c))}",
                      "Potret (%)": round(100 * np.mean([x["H"] > x["W"] for x in c]), 1),
                      "Kecerahan rata-rata (0-1)": round(S.mean(x["kecerahan"] for x in c), 3),
                      "Latar terang-netral (%)": round(100 * S.mean(x["latar_terang_netral"] for x in c), 1),
                      "Piksel hijau (%)": round(100 * S.mean(x["piksel_hijau"] for x in c), 1),
                      "Kotak per foto (median)": S.median(nk) if nk else 0,
                      "Luas kotak relatif median (%)": round(100 * S.median(kotak), 2) if kotak else None,
                      "Sumber kotak": sumber})
    with open(OUT / "karakteristik_ds3_ds4_ds1.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(tabel[0]))
        w.writeheader()
        w.writerows(tabel)
    (OUT / "CATATAN.txt").write_text("ANALISIS POST-HOC ds3 (2026-10-14). Tidak mengubah model, ambang, atau angka utama; tidak dipakai menyetel apa pun.\n"
                                     "Model dijalankan ulang pada foto ds3 hanya untuk menggambar kotak (montase) dan mengukur ukuran kotak prediksi.\n", encoding="utf-8")
    for t in tabel:
        print(t)


if __name__ == "__main__":
    main()
