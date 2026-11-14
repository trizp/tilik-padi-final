"""Validasi hasil kontrol latar terhadap hasil tersimpan (tanpa menjalankan model):
1. original vs diagnosis val resmi (runs/*/eval_val/diagnosis_bersih_per_citra.csv), per foto;
2. akurasi per run (original, gray, black) dibanding kunci_model (val resmi), tahap1 (gray SAM, 869 foto), dan acuan hitam;
3. komposisi 8 foto tanpa pasangan v9 dan 16 foto cadangan."""
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(r"D:\TELKOM\SKRIPSI\rice_yolo")
OUT = ROOT / "reports/v9_latar/kontrol_latar_val"
per = list(csv.DictReader(open(OUT / "per_foto.csv", encoding="utf-8")))
peta_nama = {"sehat": "sehat_atau_tidak_terdeteksi"}
runs = sorted({r["run"] for r in per})
tahap1 = {r["run"]: r for r in csv.DictReader(open(ROOT / "reports/v9_latar/tahap1_per_run.csv", encoding="utf-8-sig"))}
kunci = {r["run"]: r for r in csv.DictReader(open(ROOT / "reports/final_eval_v7/kunci_model.csv", encoding="utf-8-sig"))}
hitam = json.load(open(ROOT / "reports/v9_latar/tahap1_isian_hitam_s_seed0.json", encoding="utf-8"))["per_run"]

print("== 1. original vs diagnosis val resmi (per foto)")
for run in runs:
    resmi_rows = list(csv.DictReader(open(ROOT / f"runs/{run}/eval_val/diagnosis_bersih_per_citra.csv", encoding="utf-8-sig")))
    if run == runs[0]:
        print("   kolom resmi:", list(resmi_rows[0].keys()))
    kol_f = next(c for c in resmi_rows[0] if c in ("berkas", "file", "foto"))
    kol_d = next(c for c in resmi_rows[0] if "diagnosis" in c)
    kol_c = next((c for c in resmi_rows[0] if "conf" in c or "keyakinan" in c), None)
    resmi = {r[kol_f]: r for r in resmi_rows}
    ori = {r["foto"]: r for r in per if r["run"] == run and r["kondisi"] == "original"}
    beda_d = [f for f in ori if peta_nama.get(ori[f]["diagnosis"], ori[f]["diagnosis"]) != resmi[f][kol_d]]
    dconf = [abs(float(ori[f]["conf"]) - float(resmi[f][kol_c])) for f in ori] if kol_c else []
    print(f"   {run}: foto {len(ori)}; diagnosis beda {len(beda_d)} {beda_d[:3]}; |selisih conf| maks "
          f"{max(dconf):.4f}" if dconf else f"   {run}: foto {len(ori)}; diagnosis beda {len(beda_d)}")

print("== 2. akurasi per run")
for run in runs:
    acc = {k: sum(r["diagnosis"] == r["benar"] for r in per if r["run"] == run and r["kondisi"] == k) /
           sum(1 for r in per if r["run"] == run and r["kondisi"] == k) for k in ("original", "gray", "black")}
    print(f"   {run}: original {acc['original']:.4f} (val resmi 869: {kunci[run]['akurasi_val_saat_dipilih']}); "
          f"gray {acc['gray']:.4f} (tahap1 SAM 869: {tahap1[run]['tanpa_bersih_akurasi']}); black {acc['black']:.4f}"
          + (f" (acuan hitam 869: {hitam[run]['akurasi']})" if run in hitam else ""))

print("== 3. foto tanpa pasangan dan cadangan")
v7 = {p.name for p in (ROOT / "data/derived/prov_v7_valsel/images/val").iterdir()}
v9 = {p.name for p in (ROOT / "data/derived/prov_v9_valsel/images/val").iterdir()}
print("   tanpa pasangan v9:", sorted(v7 - v9))
cad = sorted({r["foto"] for r in per if r["cadangan"] == "1"})
kel = {r["foto"]: r["benar"] for r in per}
print("   cadangan:", len(cad), Counter(kel[f] for f in cad))
for run in ("v7_final_prov_yolov8s_s0", "v7_final_prov_yolo11s_s0"):
    rr = [r for r in per if r["run"] == run and r["kondisi"] == "black"]
    nc = [r for r in rr if r["cadangan"] == "0"]
    print(f"   {run} black: akurasi 861 {sum(r['diagnosis'] == r['benar'] for r in rr) / len(rr):.4f}; "
          f"tanpa 16 cadangan (845) {sum(r['diagnosis'] == r['benar'] for r in nc) / len(nc):.4f}; "
          f"benar pada cadangan {sum(r['diagnosis'] == r['benar'] for r in rr if r['cadangan'] == '1')}/16")
    c = Counter(f"{r['benar']}->{r['diagnosis']}" for r in rr)
    print("     confusion:", dict(sorted(c.items())))
