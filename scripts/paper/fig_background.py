#!/usr/bin/env python3
"""fig_background.pdf: kondisi latih/uji 2x2 + sel silang pada test bersih irisan (432 foto, 104 kelompok), YOLOv8s dan YOLO11s,
rata-rata 3 seed, IK 95% bootstrap per kelompok foto.

HANYA membaca keluaran evaluasi yang sudah ada (stats_per_citra_*.pkl dan diagnosis_*_per_citra.csv di runs/*/eval_*); tidak menjalankan
model. Bootstrap memakai fungsi dan urutan resample yang SAMA dengan scripts/analisis_akhir_v9.py (B=1000, seed 20261017, set pertama =
test bersih), sehingga IK sel diagonal harus identik dengan reports/final_eval_v9/tabel/tabel1_2x2_utama.csv dan rata-rata semua sel identik
dengan tabel4_sel_silang.csv (diperiksa; skrip berhenti bila tidak sama). IK sel silang (Orig/Leaf, Leaf/Orig) dihitung dengan cara yang
sama karena belum ada di tabel.

Keluaran: paper/figures/fig_background.{pdf,png,csv} dan paper/figures/tabel5_nilai_mutlak.csv (verifikasi Tabel V)."""
import csv
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "false")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import analisis_akhir_v9 as A  # noqa: E402
import numpy as np  # noqa: E402

OUT = ROOT / "paper" / "figures"
SP, SUB, NAMA = A.SETS[0]                       # ("test", "bersih", "Test bersih (tanpa BLB)")
KOND = [("v7_asli", "Orig/Orig"), ("v7_tanpa", "Orig/Leaf"), ("v9_tanpa", "Leaf/Leaf"), ("v9_asli", "Leaf/Orig")]
METRIK = [("mAP50_95", "mAP50-95"), ("akurasi", "Diagnosis accuracy"), ("recall_blast", "Blast recall")]


def hitung():
    rng = np.random.default_rng(A.SEED_BOOT)
    grup = {r["file"]: r["group"] for r in csv.DictReader(open(ROOT / "data/processed/prov_v7/manifest.csv", encoding="utf-8"))}
    per, _ = A.muat_stats("v9_tanpa", "yolov8s", 0, SP, SUB)
    files = sorted(p["im_name"] for p in per)
    g2i = defaultdict(list)
    for i, f in enumerate(files):
        g2i[grup[f]].append(i)
    gs = sorted(g2i)
    idx = [np.concatenate([g2i[gs[j]] for j in rng.choice(len(gs), len(gs), replace=True)]) for _ in range(A.B)]
    tasks = [(k, m, s, SP, SUB, files, idx) for k, _ in KOND for m in A.MODELS for s in A.SEEDS]
    H = {}
    with ProcessPoolExecutor(max_workers=8) as ex:
        for key, titik, penuh, boot in ex.map(A.tugas, tasks):
            H[key] = (titik, boot)
    D = {}
    for k, _ in KOND:
        for m in A.MODELS:
            for s in A.SEEDS:
                rows = A.diag_rows(k, m, s, SP, SUB)
                ada = [rows[f] for f in files if f in rows]
                benar = np.array([(rows[f]["label"] == rows[f]["diagnosis"]) if f in rows else np.nan for f in files], float)
                lab = np.array([rows[f]["label"] if f in rows else "" for f in files])
                dg = np.array([rows[f]["diagnosis"] if f in rows else "" for f in files])
                mb = lab == "leaf_blast"
                bacc = np.array([np.nanmean(benar[ix]) for ix in idx])
                brec = np.array([((dg[ix] == "leaf_blast") & mb[ix]).sum() / mb[ix].sum() if mb[ix].sum() else np.nan for ix in idx])
                D[(k, m, s)] = (A.diag_metrik(ada), bacc, brec)
    hasil = []
    for k, kl in KOND:
        for m in A.MODELS:
            for key, ml in METRIK:
                if key in ("akurasi", "recall_blast"):
                    pts = [D[(k, m, s)][0][key] for s in A.SEEDS]
                    arr = np.array([D[(k, m, s)][1 if key == "akurasi" else 2] for s in A.SEEDS])
                else:
                    pts = [H[(k, m, s, SP, SUB)][0][key] for s in A.SEEDS]
                    arr = np.array([[b[key] for b in H[(k, m, s, SP, SUB)][1]] for s in A.SEEDS])
                bt = np.nanmean(arr, axis=0)
                hasil.append({"kondisi": k, "train_test": kl, "model": A.LABEL[m], "metrik": ml, "rata2_3seed": float(np.mean(pts)),
                              "sd_3seed": float(np.std(pts, ddof=1)), "ik95_bawah": float(np.percentile(bt[~np.isnan(bt)], 2.5)),
                              "ik95_atas": float(np.percentile(bt[~np.isnan(bt)], 97.5)), "seed0": pts[0], "seed1": pts[1], "seed2": pts[2],
                              "foto": len(files), "kelompok": len(gs)})
    return hasil


def verifikasi(hasil):
    """Rata-rata = tabel4 (semua sel); IK diagonal = tabel1. Mengembalikan daftar ketidakcocokan."""
    T = ROOT / "reports/final_eval_v9/tabel"
    t4 = {r["Model"]: r for r in csv.DictReader(open(T / "tabel4_sel_silang.csv", encoding="utf-8-sig")) if r["Set"] == NAMA}
    t1 = {(r["Model"], r["Kondisi"]): r for r in csv.DictReader(open(T / "tabel1_2x2_utama.csv", encoding="utf-8-sig")) if r["Set"] == NAMA}
    kol4 = {"mAP50-95": "mAP50-95", "Diagnosis accuracy": "akurasi", "Blast recall": "recall Blast"}
    kol1 = {"mAP50-95": "IK95 mAP50-95", "Diagnosis accuracy": "IK95 Akurasi diagnosis", "Blast recall": "IK95 Recall diagnosis Blast"}
    kond1 = {"v7_asli": "Dengan latar (v7, foto asli)", "v9_tanpa": "Tanpa latar (v9, foto tanpa latar)"}
    beda = []
    for h in hasil:
        ms = f"{h['rata2_3seed']:.3f} ± {h['sd_3seed']:.3f}"
        ref = t4[h["model"]][f"{h['kondisi']} {kol4[h['metrik']]}"]
        if ms != ref:
            beda.append(("tabel4", h["model"], h["kondisi"], h["metrik"], ms, ref))
        if h["kondisi"] in kond1:
            ik = f"[{h['ik95_bawah']:.3f}, {h['ik95_atas']:.3f}]"
            ref = t1[(h["model"], kond1[h["kondisi"]])][kol1[h["metrik"]]]
            if ik != ref:
                beda.append(("tabel1", h["model"], h["kondisi"], h["metrik"], ik, ref))
    return beda


def gambar(hasil):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2,
                         "ytick.major.size": 2, "hatch.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False})
    gaya = {"YOLOv8s": {"color": "#0072B2", "hatch": ""}, "YOLO11s": {"color": "#E69F00", "hatch": "////"}}   # Okabe-Ito
    fig, axs = plt.subplots(1, 3, figsize=(3.5, 1.9))
    x = np.arange(len(KOND))
    w = 0.38
    batas = {"mAP50-95": (0, 0.32), "Diagnosis accuracy": (0, 1.0), "Blast recall": (0, 1.0)}
    for ax, (_, ml) in zip(axs, METRIK):
        for j, m in enumerate(("YOLOv8s", "YOLO11s")):
            hs = [next(h for h in hasil if h["kondisi"] == k and h["model"] == m and h["metrik"] == ml) for k, _ in KOND]
            y = np.array([h["rata2_3seed"] for h in hs])
            err = np.array([[h["rata2_3seed"] - h["ik95_bawah"] for h in hs], [h["ik95_atas"] - h["rata2_3seed"] for h in hs]])
            ax.bar(x + (j - 0.5) * w, y, w, color=gaya[m]["color"], hatch=gaya[m]["hatch"], edgecolor="black", linewidth=0.5, label=m,
                   yerr=err, error_kw={"elinewidth": 0.6, "capsize": 1.2, "capthick": 0.6, "ecolor": "black"})
        ax.set_ylabel(ml, labelpad=1.5)
        ax.set_ylim(*batas[ml])
        ax.set_xticks(x)
        ax.set_xticklabels([kl for _, kl in KOND], rotation=55, ha="right", rotation_mode="anchor")
        ax.tick_params(axis="x", pad=1.5)
        ax.tick_params(axis="y", pad=1.5)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0), handlelength=1.6, columnspacing=1.2)
    fig.text(0.5, 0.005, "Train/test images (Orig = original, Leaf = leaf-only)", ha="center", va="bottom", fontsize=7)
    fig.subplots_adjust(left=0.105, right=0.995, top=0.88, bottom=0.30, wspace=0.62)
    fig.savefig(OUT / "fig_background.pdf")
    fig.savefig(OUT / "fig_background.png", dpi=300)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    hasil = hitung()
    beda = verifikasi(hasil)
    with open(OUT / "fig_background.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(hasil[0]))
        w.writeheader()
        w.writerows(hasil)
    # Tabel V: nilai mutlak (4 desimal) per sel
    with open(OUT / "tabel5_nilai_mutlak.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "train_test", "mAP50-95", "akurasi", "recall_blast"])
        for m in ("YOLOv8s", "YOLO11s"):
            for k, kl in KOND:
                v = {h["metrik"]: h["rata2_3seed"] for h in hasil if h["model"] == m and h["kondisi"] == k}
                w.writerow([m, kl, f"{v['mAP50-95']:.4f}", f"{v['Diagnosis accuracy']:.4f}", f"{v['Blast recall']:.4f}"])
    if beda:
        for b in beda:
            print("TIDAK COCOK", b)
        sys.exit("verifikasi terhadap tabel resmi gagal; gambar tidak dibuat")
    print("verifikasi: rata-rata 24 sel = tabel4, IK 12 sel diagonal = tabel1 (identik)")
    gambar(hasil)
    print("selesai")


if __name__ == "__main__":
    main()
