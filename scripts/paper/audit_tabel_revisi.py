#!/usr/bin/env python3
"""Audit sel per sel Tabel I-VII paper/revisi_softt/main.tex terhadap berkas sumber (HANYA MEMBACA; main.tex tidak diubah).

  .venv\\Scripts\\python.exe scripts\\paper\\audit_tabel_revisi.py
"""
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEX = (ROOT / "paper/revisi_softt/main.tex").read_text(encoding="utf-8")
V7 = ROOT / "reports/final_eval_v7/tabel"
V9 = ROOT / "reports/final_eval_v9/tabel"
hasil = []


def baca(p):
    return list(csv.DictReader(open(p, encoding="utf-8-sig")))


def angka(s):
    s = s.replace("$-$", "-").replace("−", "-").replace("$\\pm$", " ").replace("±", " ")
    return [float(x) for x in re.findall(r"[-+]?\d+\.\d+|[-+]?\d+", s)]


def tabel(label):
    i = TEX.index(f"\\label{{{label}}}")
    a = TEX.rfind("\\begin{tabular}", 0, i)
    b = TEX.index("\\end{tabular}", a)
    rows = []
    for ln in TEX[a:b].split("\\\\"):
        ln = re.sub(r"\\begin\{tabular\}\{[^}]*\}", "", ln)
        ln = re.sub(r"\\multicolumn\{\d+\}\{[^}]*\}\{([^}]*)\}", r"\1", ln)
        ln = re.sub(r"\\hline|\\textbf\{([^}]*)\}|\\textit\{([^}]*)\}", lambda m: m.group(1) or m.group(2) or "", ln)
        ln = re.sub(r"\$\^\{\\mathrm\{[ab]\}\}\$|\^\{\\mathrm\{[ab]\}\}", "", ln).strip()
        if ln:
            rows.append([c.strip() for c in ln.split("&")])
    return rows


def cek(tab, sel, tex, src, tol=5e-4):
    ok = len(tex) == len(src) and all(abs(x - y) <= tol for x, y in zip(tex, src))
    hasil.append((tab, sel, tex, src, ok))


# Tabel I
t1 = {r["Kelas"]: r for r in baca(ROOT / "paper/figures/tabel_data_prov_v7.csv")}
peta = {"Blast": "leaf_blast", "Brown spot": "brown_spot", "BLB": "bacterial_leaf_blight", "Healthy": "background",
        "Total": "TOTAL (kelompok berbeda per split)"}
for r in tabel("tab:data")[1:]:
    k = next((v for kk, v in peta.items() if r[0].startswith(kk)), None)
    if k:
        src = [float(t1[k][f"{sp} {j}"]) for sp in ("train", "val", "test", "test_new_session") for j in ("foto", "kelompok")]
        cek("I", r[0], [x for c in r[1:] for x in angka(c)], src, 0)
# Tabel II
t10 = baca(V7 / "tabel10_ketepatan_perkiraan.csv")
rs = {r["Model"]: r for r in baca(ROOT / "reports/revisi_paper/randsplit_3seed/ringkasan.csv")}
for r in tabel("tab:leak")[1:]:
    if r[0] == "Grouped":
        b = next(x for x in t10 if x["Split latih"].startswith("split bersih") and x["Model"] == r[1])
        src = [angka(b["Selisih mAP50-95"])[0], angka(b["Selisih akurasi"])[0]]
    else:
        src = [angka(rs[r[1]]["Selisih mAP50-95"])[0], angka(rs[r[1]]["Selisih akurasi"])[0]]
    cek("II", f"{r[0]} {r[1]}", angka(r[2]) + angka(r[3]), src)
# Tabel III
t3 = baca(V7 / "tabel1_utama_per_model.csv")
spd = {r["Model"]: r for r in baca(V7 / "tabel4_kecepatan.csv")}
blok = None
for r in tabel("tab:main")[1:]:
    if len(r) == 1:
        blok = "Test bersih (tanpa BLB prov.)" if r[0].startswith("Test set") else "Test sesi baru"
        continue
    b = next(x for x in t3 if x["Set"] == blok and x["Model"] == r[0])
    src = [angka(b[k])[0] for k in ("P", "R", "F1", "mAP50")] + angka(b["mAP50-95"])[:2] + angka(b["Akurasi diagnosis"])[:2]
    tex = [angka(c)[0] for c in r[1:5]] + angka(r[5]) + angka(r[6])
    if blok.startswith("Test bersih"):
        src += [float(spd[r[0]]["Parameter (juta)"]), float(spd[r[0]]["GFLOPs"]), angka(spd[r[0]]["CPU ms (median)"])[0]]
        tex += [angka(c)[0] for c in r[7:10]]
    cek("III", f"{'test' if blok.startswith('Test bersih') else 'sesi baru'} {r[0]}", tex, src)
# Tabel IV
t8 = [x for x in baca(V7 / "tabel8_v8_vs_v11_berpasangan.csv") if x["Seed"] == "gabungan"]
for r in tabel("tab:paired")[1:]:
    p = "YOLO11n − YOLOv8n" if r[0] == "n" else "YOLO11s − YOLOv8s"
    s = "Test bersih (tanpa BLB prov.)" if r[1] == "Test" else "Test sesi baru"
    src = []
    for mm in ("mAP50_95", "akurasi"):
        x = next(z for z in t8 if z["Pasangan"] == p and z["Set"] == s and z["Metrik"] == mm)
        src += [float(x["Selisih"])] + angka(x["IK95"])
    cek("IV", f"{r[0]} {r[1]}", angka(r[2]) + angka(r[3]), src)
# Tabel V
fb = baca(ROOT / "paper/figures/fig_background.csv")
kond = {("Original", "Original"): "Orig/Orig", ("Original", "Leaf-only"): "Orig/Leaf", ("Leaf-only", "Leaf-only"): "Leaf/Leaf",
        ("Leaf-only", "Original"): "Leaf/Orig"}
model = None
for r in tabel("tab:bg")[1:]:
    if len(r) == 1:
        model = r[0]
        continue
    k = kond[(r[0], r[1])]
    src = [round(float(next(x for x in fb if x["model"] == model and x["train_test"] == k and x["metrik"] == mk)["rata2_3seed"]), 3)
           for mk in ("mAP50-95", "Diagnosis accuracy", "Blast recall")]
    cek("V", f"{model} {k}", [angka(c)[0] for c in r[2:5]], src)
# Tabel VI
kl = {(r["model"], r["kondisi"]): r for r in baca(ROOT / "reports/v9_latar/kontrol_latar_val/ringkasan.csv")}
nama = {"Original": "original", "Gray": "gray", "Black": "black", "Blurred": "blur", "Swapped": "swap"}
for r in tabel("tab:ctrl")[2:]:
    src = []
    for mm in ("yolov8s", "yolo11s"):
        x = kl[(mm, nama[r[0]])]
        # ringkasan.csv 4 desimal: dibandingkan tanpa dibulatkan lagi (hindari pembulatan ganda, mis. 0,5545 dari 0,55452)
        src += [float(x["akurasi_mean"]), float(x["blast_dR_mean"])]
    cek("VI", r[0], [angka(c)[0] for c in r[1:5]], src, tol=6e-4)
# Tabel VII
t7 = [x for x in baca(V9 / "tabel7_ds3.csv") if x["Set"].startswith("ds3 uji utama")]
for r in tabel("tab:ext")[1:]:
    if len(r) == 1:
        model = r[0]
        continue
    kd = "v7 pada ds3 asli" if r[0] == "Original" else "v9 pada ds3 tanpa latar"
    x = next(z for z in t7 if z["Model"] == model and z["Kondisi"] == kd)
    cek("VII", f"{model} {r[0]}", [angka(c)[0] for c in r[1:4]], [angka(x[k])[0] for k in ("Akurasi", "R Blast", "R BLB")])

n_ok = sum(h[4] for h in hasil)
print(f"baris tabel cocok: {n_ok}/{len(hasil)}; sel: {sum(len(h[2]) for h in hasil if h[4])}/{sum(len(h[3]) for h in hasil)}")
for t, s, a, b, ok in hasil:
    print(("OK   " if ok else "BEDA ") + f"Tabel {t} | {s} | main.tex {a} | sumber {b}" if not ok else f"OK   Tabel {t} | {s}")
