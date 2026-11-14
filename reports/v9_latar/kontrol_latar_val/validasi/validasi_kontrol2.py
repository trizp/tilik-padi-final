"""(a) Akurasi gray/black bila 8 foto tanpa pasangan v9 dihitung salah pada 869 foto (seperti acuan lama).
(b) Foto dengan selisih keyakinan teratas besar vs val resmi: ukuran foto dan apakah keyakinannya di bawah ambang."""
import csv
import json
from pathlib import Path

import cv2

ROOT = Path(r"D:\TELKOM\SKRIPSI\rice_yolo")
OUT = ROOT / "reports/v9_latar/kontrol_latar_val"
per = list(csv.DictReader(open(OUT / "per_foto.csv", encoding="utf-8")))
tahap1 = {r["run"]: r for r in csv.DictReader(open(ROOT / "reports/v9_latar/tahap1_per_run.csv", encoding="utf-8-sig"))}
hitam = json.load(open(ROOT / "reports/v9_latar/tahap1_isian_hitam_s_seed0.json", encoding="utf-8"))["per_run"]
kunci = {r["run"]: float(r["ambang_diagnosis"]) for r in csv.DictReader(open(ROOT / "reports/final_eval_v7/kunci_model.csv", encoding="utf-8-sig"))}
runs = sorted({r["run"] for r in per})
print("== (a) benar/869 dengan 8 foto tanpa pasangan dihitung salah")
for run in runs:
    s = []
    for k in ("gray", "black"):
        benar = sum(r["diagnosis"] == r["benar"] for r in per if r["run"] == run and r["kondisi"] == k)
        acuan = tahap1[run]["tanpa_bersih_akurasi"] if k == "gray" else (hitam[run]["akurasi"] if run in hitam else "-")
        s.append(f"{k}: {benar}/869 = {benar / 869:.4f} (acuan {acuan})")
    print("  ", run, "; ".join(s))

print("== (b) selisih keyakinan teratas original vs val resmi")
V7 = ROOT / "data/derived/prov_v7_valsel/images/val"
for run in runs:
    resmi = {r["berkas"]: r for r in csv.DictReader(open(ROOT / f"runs/{run}/eval_val/diagnosis_bersih_per_citra.csv", encoding="utf-8-sig"))}
    ori = [r for r in per if r["run"] == run and r["kondisi"] == "original"]
    d = sorted(((abs(float(r["conf"]) - float(resmi[r["foto"]]["conf_teratas"])), r["foto"], float(r["conf"]),
                float(resmi[r["foto"]]["conf_teratas"])) for r in ori), reverse=True)
    besar = [x for x in d if x[0] > 0.01]
    persegi = sum(1 for x in besar if (lambda im: im.shape[0] == im.shape[1])(cv2.imread(str(V7 / x[1]))))
    di_atas = sum(1 for x in besar if max(x[2], x[3]) >= kunci[run])
    kecil = sum(1 for x in d if x[0] <= 1e-3)
    print(f"   {run}: |selisih| <= 0,001: {kecil}/{len(d)}; > 0,01: {len(besar)} (foto persegi {persegi}; salah satu >= ambang {di_atas}); "
          f"contoh {[(f, round(a, 3), round(b, 3)) for _, f, a, b in besar[:2]]}")
