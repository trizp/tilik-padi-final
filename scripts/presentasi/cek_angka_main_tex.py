#!/usr/bin/env python3
"""Cocokkan angka di paper/main.tex (versi final) dengan bahan_presentasi: tabel I-VI sel per sel terhadap tabel/paper/*.csv, lalu setiap
angka di teks terhadap seluruh angka di tabel/ dan angka_kunci/key_numbers. Hanya membaca. Keluaran: bahan_presentasi/CEK_ANGKA_MAIN_TEX.md"""
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEX = (ROOT / "paper/main.tex").read_text(encoding="utf-8")
B = ROOT / "bahan_presentasi"
TP = B / "tabel/paper"


def baca(p):
    return list(csv.DictReader(open(p, encoding="utf-8-sig")))


def angka(s):
    s = s.replace("$-$", "-").replace("−", "-").replace("$\\pm$", "±")
    return [float(x) for x in re.findall(r"[-+]?\d+\.\d+|[-+]?\d+", s)]


def tabel(label):
    i = TEX.index(f"\\label{{{label}}}")
    a = TEX.rfind("\\begin{tabular}", 0, i)
    b = TEX.index("\\end{tabular}", a)
    rows = []
    for ln in TEX[a:b].split("\\\\"):
        ln = re.sub(r"\\hline|\\begin\{tabular\}\{[^}]*\}|\\textbf\{([^}]*)\}|\\textit\{([^}]*)\}", lambda m: m.group(1) or m.group(2) or "", ln)
        ln = re.sub(r"\\multicolumn\{\d+\}\{[^}]*\}\{([^}]*)\}", r"\1", ln).strip()
        if ln:
            rows.append([c.strip() for c in ln.split("&")])
    return rows


hasil = []   # (tabel, sel, main.tex, bahan, cocok)

# Penilaian angka teks yang tidak muncul harfiah di bahan_presentasi (dinilai 2026-10-27; sumber diverifikasi di paper/LAPORAN_PAPER.md)
PENJELASAN = {
    "75.0": "cocok: = 0.750 (akurasi YOLOv8s, T3) dalam persen",
    "0.74": "cocok: pembulatan -0.739 (kepekaan recall Blast YOLO11s, angka_kunci)",
    "0.97": "cocok: parameter metode; sama dengan label gambar 1 (DINOv2 >= 0.97)",
    "0.43": "cocok: pembulatan kappa 0.431 (angka_kunci)",
    "3050": "cocok: perangkat (RTX 3050 Ti); sama dengan label gambar 1",
    "869": "tidak ada di bahan; benar menurut KEPUTUSAN_PROYEK.md (val bersih 869 = 1259 - 390 BLB)",
    "0.02": "cocok: rentang selisih mAP50-95 0.017-0.041 di T4 dibulatkan",
    "0.7": "tidak ada di bahan; benar menurut CSV per kelas 12 run v7 (recall BLB 0.605-0.774)",
    "0.18": "tidak ada di bahan; benar menurut reports/final_eval_v7/RINGKASAN_HASIL.md (akurasi ds3 semua model 0.18-0.34)",
}


def cek(tab, sel, nilai_tex, nilai_bahan, tol=5e-4):
    ok = len(nilai_tex) == len(nilai_bahan) and all(abs(x - y) <= tol for x, y in zip(nilai_tex, nilai_bahan))
    hasil.append((tab, sel, nilai_tex, nilai_bahan, ok))


def main():
    # ---- Tabel I
    t1 = {r["Kelas"]: r for r in baca(TP / "T1_dataset_foto_kelompok_prov_v7.csv")}
    peta = {"Blast": "leaf_blast", "Brown spot": "brown_spot", "BLB": "bacterial_leaf_blight", "Healthy": "background",
            "Total": "TOTAL (kelompok berbeda per split)"}
    for r in tabel("tab:data")[1:]:
        k = next((v for kk, v in peta.items() if r[0].startswith(kk)), None)
        if k:
            bahan = [float(t1[k][f"{sp} {j}"]) for sp in ("train", "val", "test", "test_new_session") for j in ("foto", "kelompok")]
            cek("I", r[0], [x for c in r[1:] for x in angka(c)], bahan, 0)
    # ---- Tabel II
    t2 = baca(TP / "T2_selisih_uji_vs_sesi_baru.csv")
    for r in tabel("tab:leak")[1:]:
        sp = "split bersih" if r[0].startswith("Leakage") else "split acak"
        b = next(x for x in t2 if x["Split latih"].startswith(sp) and x["Model"] == r[1])
        cek("II", f"{r[0]} {r[1]}", angka(r[2]) + angka(r[3]), [angka(b["Selisih mAP50-95"])[0], angka(b["Selisih akurasi"])[0]])
    # ---- Tabel III
    t3 = baca(TP / "T3_hasil_utama.csv")
    blok = None
    for r in tabel("tab:main")[1:]:
        if len(r) == 1:
            blok = "Test bersih (tanpa BLB prov.)" if r[0].startswith("Test set") else "Test sesi baru"
            continue
        b = next(x for x in t3 if x["Set"] == blok and x["Model"] == r[0])
        bahan = [angka(b[k])[0] for k in ("P", "R", "F1", "mAP50")] + angka(b["mAP50-95"])[:2] + angka(b["Akurasi diagnosis"])[:2]
        tex = [angka(c)[0] for c in r[1:5]] + angka(r[5]) + angka(r[6])
        if blok.startswith("Test bersih"):
            bahan += [float(b["Parameter (juta)"]), float(b["GFLOPs"]), angka(b["CPU ms (median)"])[0]]
            tex += [angka(c)[0] for c in r[7:10]]
        cek("III", f"{'test' if blok.startswith('Test bersih') else 'sesi baru'} {r[0]}", tex, bahan)
    # ---- Tabel IV
    t4 = baca(TP / "T4_selisih_berpasangan_yolo11_yolov8.csv")
    for r in tabel("tab:paired")[1:]:
        p = "YOLO11n − YOLOv8n" if r[0] == "n" else "YOLO11s − YOLOv8s"
        s = "Test bersih (tanpa BLB prov.)" if r[1] == "Test" else "Test sesi baru"
        bahan = []
        for mm in ("mAP50_95", "akurasi"):
            x = next(z for z in t4 if z["Pasangan"] == p and z["Set"] == s and z["Metrik"] == mm)
            bahan += [float(x["Selisih"])] + angka(x["IK95"])
        tex = angka(re.sub(r"\^\{\\mathrm\{a\}\}", "", r[2])) + angka(re.sub(r"\^\{\\mathrm\{a\}\}", "", r[3]))
        cek("IV", f"{r[0]} {r[1]}", tex, bahan)
    # ---- Tabel V
    t5 = baca(TP / "T5_ketergantungan_latar_432_foto.csv")
    model = None
    kond = {("Original", "Original"): "Orig/Orig", ("Original", "Leaf-only"): "Orig/Leaf", ("Leaf-only", "Leaf-only"): "Leaf/Leaf",
            ("Leaf-only", "Original"): "Leaf/Orig"}
    for r in tabel("tab:bg")[1:]:
        if len(r) == 1:
            model = r[0]
            continue
        k = kond[(r[0], r[1])]
        bahan = [round(float(next(x for x in t5 if x["model"] == model and x["train_test"] == k and x["metrik"] == mk)["rata2_3seed"]), 3)
                 for mk in ("mAP50-95", "Diagnosis accuracy", "Blast recall")]
        cek("V", f"{model} {k}", [angka(c)[0] for c in r[2:5]], bahan)
    # ---- Tabel VI
    t6 = baca(TP / "T6_ds3_uji_utama.csv")
    for r in tabel("tab:ext")[1:]:
        if len(r) == 1:
            model = r[0]
            continue
        kd = "v7 pada ds3 asli" if r[0] == "Original" else "v9 pada ds3 tanpa latar"
        x = next(z for z in t6 if z["Model"] == model and z["Kondisi"] == kd)
        cek("VI", f"{model} {r[0]}", [angka(c)[0] for c in r[1:4]], [angka(x[k])[0] for k in ("Akurasi", "R Blast", "R BLB")])

    # ---- angka di teks: cari di bahan_presentasi
    teks = re.sub(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", " ", TEX, flags=re.S)
    teks = teks[teks.index("\\begin{abstract}"):teks.index("\\bibliographystyle")]
    teks = re.sub(r"\\cite\{[^}]*\}|\\ref\{[^}]*\}|\\label\{[^}]*\}|\\figorbox\{[^}]*\}\{[^}]*\}", " ", teks)
    sumber = " ".join(p.read_text(encoding="utf-8-sig", errors="replace") for p in list((B / "tabel").rglob("*.csv")) +
                      [B / "angka_kunci.md", B / "key_numbers.md", B / "README.md"])
    sumber = sumber.replace(",", " ").replace("−", "-")
    ada = set(re.findall(r"\d+\.\d+|\d+", sumber))
    ada |= {x.lstrip("0") for x in ada}
    tidak = []
    for m in re.finditer(r"(?<![\w.])(\d+\.\d+|\d{2,})(?![\w])", teks):
        x = m.group(1)
        if x in ada or x.lstrip("0") in ada:
            continue
        konteks = re.sub(r"\s+", " ", teks[max(0, m.start() - 60):m.end() + 40])
        tidak.append((x, konteks))

    md = ["# Cek angka main.tex (final) terhadap bahan_presentasi", "",
          f"main.tex: `paper/main.tex` (salinan `paper/final/main.tex`). Tabel dibandingkan sel per sel dengan `bahan_presentasi/tabel/paper/`.", "",
          f"**Tabel: {sum(h[4] for h in hasil)}/{len(hasil)} baris cocok.**", "", "| Tabel | Baris | Cocok |", "|---|---|---|"]
    md += [f"| {t} | {s} | {'ya' if ok else 'TIDAK: main.tex ' + str(a) + ' vs bahan ' + str(b)} |" for t, s, a, b, ok in hasil]
    md += ["", f"**Angka di teks yang tidak ditemukan di bahan_presentasi: {len(tidak)}** (perlu dinilai satu per satu; banyak yang memang tidak "
           "ada di bahan, misalnya parameter metode):", ""]
    md += [f"- `{x}`: …{k}… → **{PENJELASAN.get(x, 'BELUM DINILAI')}**" for x, k in tidak]
    (B / "CEK_ANGKA_MAIN_TEX.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"tabel cocok {sum(h[4] for h in hasil)}/{len(hasil)}")
    for h in hasil:
        if not h[4]:
            print("TIDAK", h)
    print("angka teks tidak ditemukan:", len(tidak))
    for x, k in tidak:
        print(f"  {x}: {k}")


if __name__ == "__main__":
    main()
