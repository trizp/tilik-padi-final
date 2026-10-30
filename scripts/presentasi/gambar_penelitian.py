#!/usr/bin/env python3
"""Gambar penelitian 1-11 untuk bahan presentasi, dua bahasa (paper_en / skripsi_id). HANYA membaca hasil yang sudah ada; tidak melatih,
tidak mengevaluasi, tidak menjalankan model. Angka turunan baru (dihitung dari keluaran tersimpan) ditandai TURUNAN di komentar dan di README.

  .venv\\Scripts\\python.exe scripts\\presentasi\\gambar_penelitian.py"""
import csv
import random
import re
import sys
from pathlib import Path

import numpy as np
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle
from matplotlib.lines import Line2D
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gaya as G  # noqa: E402
from gaya import T, plt  # noqa: E402

R = G.ROOT
V7T = R / "reports/final_eval_v7/tabel"
V9T = R / "reports/final_eval_v9/tabel"
MODELS = ["YOLOv8n", "YOLO11n", "YOLOv8s", "YOLO11s"]
LAPORAN = []          # catatan angka turunan untuk README


def baca(p):
    return list(csv.DictReader(open(p, encoding="utf-8-sig")))


def ms(s):
    """'0.253 ± 0.005' -> (0.253, 0.005)."""
    a = re.findall(r"[-+]?\d*\.\d+|\d+", s)
    return float(a[0]), (float(a[1]) if len(a) > 1 else 0.0)


def ik(s):
    a = re.findall(r"[-+]?\d*\.\d+", s)
    return float(a[0]), float(a[1])


# =========================================================== 1. alur penelitian
def kotak(ax, x, y, w, h, judul, isi, no, sorot=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.008,rounding_size=0.02",
                                facecolor="#F4F4F4" if not sorot else "#FFF4DD", edgecolor=G.ABU if not sorot else G.MODEL["YOLO11"],
                                linewidth=2.2))
    ax.text(x - w / 2 + 0.004, y + h / 2 - 0.004, f"{no}", fontsize=17, fontweight="bold", color="white", ha="center", va="center",
            bbox={"boxstyle": "circle,pad=0.3", "facecolor": G.ABU, "edgecolor": "none"})
    ax.text(x, y + h / 2 - 0.075, judul, fontsize=19, fontweight="bold", ha="center", va="center")
    ax.text(x, y - 0.045, isi, fontsize=16, ha="center", va="center", linespacing=1.35, color="#222222")


def panah(ax, xy0, xy1):
    ax.annotate("", xy=xy1, xytext=xy0, arrowprops={"arrowstyle": "-|>", "lw": 2.4, "color": G.ABU, "mutation_scale": 26})


def g01_alur(b):
    fig = G.kanvas()
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.98])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    w, h = 0.305, 0.38
    xs, ys = [0.165, 0.5, 0.835], [0.74, 0.24]
    langkah = T[b]["wf"]
    for i, (jd, isi) in enumerate(langkah):
        x, y = xs[i % 3], ys[i // 3]
        kotak(ax, x, y, w, h, jd, isi, i + 1, sorot=(i == 5))
    for i in (0, 1):
        panah(ax, (xs[i] + w / 2, ys[0]), (xs[i + 1] - w / 2, ys[0]))
        panah(ax, (xs[i] + w / 2, ys[1]), (xs[i + 1] - w / 2, ys[1]))
    # dari kotak 3 ke kotak 4 (siku)
    ymid = 0.49
    ax.plot([xs[2], xs[2], xs[0]], [ys[0] - h / 2, ymid, ymid], color=G.ABU, lw=2.4)
    panah(ax, (xs[0], ymid), (xs[0], ys[1] + h / 2))
    return G.simpan(fig, b, "01_alur_penelitian" if b == "id" else "01_research_workflow")


# =========================================================== 2. komposisi dataset
def g02_dataset(b):
    rows = {r["Kelas"]: r for r in baca(R / "paper/figures/tabel_data_prov_v7.csv")}
    splits = [("train", "train"), ("val", "val"), ("test", "test"), ("test_new_session", "new")]
    kelas = ["leaf_blast", "brown_spot", "bacterial_leaf_blight", "background"]
    fig = G.kanvas(layout="constrained")
    ax = fig.add_subplot()
    y = np.arange(len(splits))[::-1]
    for yi, (sp, key) in zip(y, splits):
        kiri = 0
        for k in kelas:
            n = int(rows[k][f"{sp} foto"])
            if n == 0:
                continue
            ax.barh(yi, n, left=kiri, color=G.KELAS[k], edgecolor="white", linewidth=1.5, height=0.62)
            if n >= 600:
                ax.text(kiri + n / 2, yi, f"{n:,}".replace(",", "." if b == "id" else ","), ha="center", va="center", fontsize=15,
                        color="black" if k == "bacterial_leaf_blight" else "white", fontweight="bold")
            kiri += n
        tot = rows["TOTAL (kelompok berbeda per split)"]
        teks = (f"{int(tot[f'{sp} foto']):,}  ·  {int(tot[f'{sp} kelompok']):,} {T[b]['groups']}").replace(",", "." if b == "id" else ",")
        ax.text(kiri + 70, yi, teks, va="center", fontsize=16)
    ax.set_yticks(y)
    ax.set_yticklabels([T[b][k] for _, k in splits], fontsize=18)
    ax.set_xlabel(T[b]["images"])
    ax.set_xlim(0, 9300)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(handles=[Patch(facecolor=G.KELAS[k], label=G.nama_kelas(b, k)) for k in kelas], loc="lower right", ncol=1,
              fontsize=17, bbox_to_anchor=(1.0, 0.02))
    return G.simpan(fig, b, "02_komposisi_dataset" if b == "id" else "02_dataset_composition")


# =========================================================== 3a. contoh foto kembar (split acak)
def pilih_kembar():
    """ds4 saja: kelompok duplikat (manifest) yang anggotanya jatuh di train DAN test pada split acak. 1 pasangan per kelas, seed tetap."""
    man = [r for r in csv.DictReader(open(R / "data/processed/prov_v7_randsplit/manifest.csv", encoding="utf-8")) if r["source"] == "ds4"]
    per_grup = {}
    for r in man:
        per_grup.setdefault(r["group"], []).append(r)
    rng = random.Random(20261027)
    out = []
    for k in ("leaf_blast", "brown_spot", "background"):
        calon = []
        for g, rs in sorted(per_grup.items()):
            tr = sorted(x["file"] for x in rs if x["split"] == "train" and x["class"] == k)
            te = sorted(x["file"] for x in rs if x["split"] == "test" and x["class"] == k)
            if tr and te:
                calon.append((g, tr[0], te[0]))
        g, a, c = rng.choice(calon)
        out.append((k, g, a, c, len(calon)))
    return out


def g03a_kembar(b, pasangan):
    fig = G.kanvas()
    n = len(pasangan)
    for j, (k, g, a, c, _) in enumerate(pasangan):
        for i, (f, sp) in enumerate(((a, "train"), (c, "test"))):
            ax = fig.add_axes([0.07 + j * 0.31, 0.52 - i * 0.47, 0.29, 0.42])
            im = Image.open(R / "data/processed/prov_v7_randsplit/images" / sp / f).convert("RGB")
            im.thumbnail((900, 900))
            ax.imshow(np.asarray(im))
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(True)
                s.set_linewidth(1.5)
            if i == 0:
                ax.set_title(G.nama_kelas(b, k), fontsize=19, pad=8)
    fig.text(0.035, 0.73, T[b]["train_split"], rotation=90, ha="center", va="center", fontsize=17)
    fig.text(0.035, 0.26, T[b]["test_split"], rotation=90, ha="center", va="center", fontsize=17)
    return G.simpan(fig, b, "03a_contoh_foto_kembar" if b == "id" else "03a_duplicate_examples")


# =========================================================== 3b. selisih test - sesi baru
def g03b_bocor(b):
    rows = baca(V7T / "tabel10_ketepatan_perkiraan_per_run.csv")
    nm = {"yolov8n": "YOLOv8n", "yolo11n": "YOLO11n", "yolov8s": "YOLOv8s", "yolo11s": "YOLO11s"}
    kol = [("selisih mAP (test tanpa BLB - sesi baru)", "gap_map"), ("selisih akurasi (test tanpa BLB - sesi baru)", "gap_acc")]
    grup = [("split bersih", ["YOLOv8n", "YOLO11n", "YOLOv8s", "YOLO11s"]), ("split acak", ["YOLOv8n", "YOLO11n"])]
    fig, axs = plt.subplots(1, 2, figsize=(G.W_IN, G.H_IN), layout="constrained")
    for ax, (c, lab) in zip(axs, kol):
        x0, ticks, tl = 0, [], []
        for gi, (sp, ms_) in enumerate(grup):
            for m in ms_:
                v = [float(r[c]) for r in rows if r["split"] == sp and nm[r["model"]] == m]
                ax.bar(x0, np.mean(v), 0.75, **G.gaya_batang(m))
                ax.scatter([x0] * len(v), v, s=55, color="black", zorder=3, marker="o" if len(v) > 1 else "D")
                ax.text(x0, max(np.mean(v), max(v)) + 0.008, G.angka(np.mean(v), b), ha="center", va="bottom", fontsize=15)
                ticks.append(x0)
                tl.append(m)
                x0 += 1
            if gi == 0:
                x0 += 0.8
        ax.axhline(0, color="black", lw=1.2)
        ax.set_xticks(ticks)
        ax.set_xticklabels(tl, rotation=35, ha="right", rotation_mode="anchor")
        ax.set_ylabel(T[b][lab])
        ax.set_ylim(-0.08, 0.15)
        ax.text(1.5, -0.075, T[b]["leakfree"], ha="center", va="bottom", fontsize=16, color=G.ABU)
        ax.text(5.3, -0.075, T[b]["random"], ha="center", va="bottom", fontsize=16, color=G.ABU)
    axs[1].legend(handles=[Line2D([], [], marker="o", color="black", ls="", markersize=8, label=T[b]["seed"])], loc="upper left")
    return G.simpan(fig, b, "03b_selisih_uji_sesi_baru" if b == "id" else "03b_test_vs_new_session_gap")


# =========================================================== 4. hasil utama
def g04_utama(b):
    rows = baca(V7T / "tabel1_utama_per_model.csv")
    sets = [("Test bersih (tanpa BLB prov.)", "test_lbl"), ("Test sesi baru", "new_lbl")]
    met = [("mAP50-95", "IK95 mAP50-95", "map", (0, 0.42)), ("Akurasi diagnosis", "IK95 akurasi", "acc", (0, 1.15))]
    fig, axs = plt.subplots(1, 2, figsize=(G.W_IN, G.H_IN), layout="constrained")
    w = 0.2
    for ax, (k, kik, lab, lim) in zip(axs, met):
        for si, (s, sl) in enumerate(sets):
            for mi, m in enumerate(MODELS):
                r = next(x for x in rows if x["Set"] == s and x["Model"] == m)
                v = ms(r[k])[0]
                lo, hi = ik(r[kik])
                ax.bar(si + (mi - 1.5) * w, v, w * 0.92, yerr=[[v - lo], [hi - v]], capsize=5, error_kw={"elinewidth": 1.6, "capthick": 1.6},
                       label=m if si == 0 else None, **G.gaya_batang(m))
        ax.set_xticks([0, 1])
        ax.set_xticklabels([T[b][sl].replace(" (", "\n(") for _, sl in sets], fontsize=17)
        ax.set_ylabel(T[b][lab])
        ax.set_ylim(*lim)
    axs[0].legend(ncol=2, loc="upper left", fontsize=16, columnspacing=1.0)
    G.catatan(fig, T[b]["ci"], ha="right")
    return G.simpan(fig, b, "04_hasil_utama" if b == "id" else "04_main_results")


# =========================================================== 5. forest plot selisih berpasangan
def g05_forest(b):
    rows = baca(V7T / "tabel8_v8_vs_v11_berpasangan.csv")
    baris = [("YOLO11n − YOLOv8n", "Test bersih (tanpa BLB prov.)", "pair_n_test"), ("YOLO11n − YOLOv8n", "Test sesi baru", "pair_n_new"),
             ("YOLO11s − YOLOv8s", "Test bersih (tanpa BLB prov.)", "pair_s_test"), ("YOLO11s − YOLOv8s", "Test sesi baru", "pair_s_new")]
    fig, axs = plt.subplots(1, 2, figsize=(G.W_IN, G.H_IN), layout="constrained", sharey=True)
    for ax, (met, lab, lim) in zip(axs, (("mAP50_95", "d_map", (-0.03, 0.08)), ("akurasi", "d_acc", (-0.09, 0.15)))):
        for yi, (p, s, _) in enumerate(baris):
            r = next(x for x in rows if x["Pasangan"] == p and x["Set"] == s and x["Metrik"] == met and x["Seed"] == "gabungan")
            v = float(r["Selisih"])
            lo, hi = ik(r["IK95"])
            y = len(baris) - 1 - yi
            isi = r["Memuat nol"] == "tidak"
            c = G.MODEL["YOLO11"] if isi else G.ABU
            ax.plot([lo, hi], [y, y], color=c, lw=3)
            ax.plot(v, y, "o", markersize=14, markerfacecolor=c if isi else "white", markeredgecolor=c, markeredgewidth=2.5)
            ax.text(hi + (lim[1] - lim[0]) * 0.02, y, f"{G.angka(v, b)}  [{G.angka(lo, b)}; {G.angka(hi, b)}]", va="center", fontsize=15)
        ax.axvline(0, color="black", lw=1.2, ls="--")
        ax.set_xlim(lim[0], lim[1] + (lim[1] - lim[0]) * 0.75)
        ax.set_xlabel(T[b][lab])
        ax.set_yticks(range(len(baris)))
        ax.set_yticklabels([T[b][k] for _, _, k in baris][::-1], fontsize=17)
        ax.set_ylim(-1.5, len(baris) - 0.4)
    axs[1].legend(handles=[Line2D([], [], marker="o", color=G.MODEL["YOLO11"], markersize=12, lw=3, label=T[b]["ci_excl"]),
                           Line2D([], [], marker="o", color=G.ABU, markerfacecolor="white", markersize=12, lw=3, label=T[b]["ci_incl"])],
                  loc="lower right", fontsize=15)
    return G.simpan(fig, b, "05_selisih_berpasangan" if b == "id" else "05_paired_differences")


# =========================================================== 6. kecepatan vs akurasi
def g06_kecepatan(b):
    spd = {r["Model"]: r for r in baca(V7T / "tabel4_kecepatan.csv")}
    rows = baca(V7T / "tabel1_utama_per_model.csv")
    fig = G.kanvas(layout="constrained")
    ax = fig.add_subplot()
    for m in MODELS:
        x, sx = ms(spd[m]["CPU ms (median)"])
        prm = float(spd[m]["Parameter (juta)"])
        ya = ms(next(r for r in rows if r["Set"] == "Test bersih (tanpa BLB prov.)" and r["Model"] == m)["Akurasi diagnosis"])[0]
        yb = ms(next(r for r in rows if r["Set"] == "Test sesi baru" and r["Model"] == m)["Akurasi diagnosis"])[0]
        c = G.warna_model(m)
        ax.plot([x, x], [ya, yb], color=c, lw=2, alpha=0.6)
        ax.scatter(x, ya, s=prm * 120, color=c, edgecolor="black", linewidth=1.2, zorder=3)
        ax.scatter(x, yb, s=prm * 120, facecolor="white", edgecolor=c, linewidth=3, zorder=3)
        dx = 5 if m.endswith("s") else 3
        ax.text(x + dx, (ya + yb) / 2, f"{m}\n{G.angka(prm, b, 1)} M", va="center", fontsize=16, color=c, fontweight="bold")
    ax.set_xlabel(T[b]["cpu"])
    ax.set_ylabel(T[b]["acc"])
    ax.set_xlim(20, 120)
    ax.set_ylim(0.66, 0.80)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=G.ABU, markersize=14, label=T[b]["test_lbl"]),
                       Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=G.ABU, markeredgewidth=3,
                              markersize=14, label=T[b]["new_lbl"])], loc="lower right")
    G.catatan(fig, ("point area proportional to parameters; CPU: batch 1, FP32, 640 px" if b == "en" else
                    "luas titik sebanding jumlah parameter; CPU: batch 1, FP32, 640 px"), ha="right")
    return G.simpan(fig, b, "06_kecepatan_vs_akurasi" if b == "id" else "06_speed_vs_accuracy")


# =========================================================== 7. confusion matrix (TURUNAN: jumlah 3 seed)
KLS4 = ["leaf_blast", "brown_spot", "bacterial_leaf_blight", "sehat_atau_tidak_terdeteksi"]


def confusion(m):
    tot = np.zeros((4, 4), int)
    for s in (0, 1, 2):
        rows = list(csv.reader(open(R / f"runs/v7_final_prov_{m}_s{s}/eval_test/diagnosis_dengan_background_confusion.csv", encoding="utf-8")))
        kol = rows[0][1:]
        for r in rows[1:]:
            i = KLS4.index(r[0])
            for c, v in zip(kol, r[1:]):
                tot[i, KLS4.index(c)] += int(v)
    return tot


def g07_confusion(b, cm):
    fig, axs = plt.subplots(1, 2, figsize=(G.W_IN, G.H_IN))
    fig.subplots_adjust(left=0.21, right=0.985, top=0.90, bottom=0.25, wspace=0.06)
    lab = [G.nama_kelas(b, k).replace(" / ", " /\n") for k in KLS4]
    for ax, (m, c) in zip(axs, cm.items()):
        p = c / c.sum(1, keepdims=True)
        ax.imshow(p, cmap="Greys", vmin=0, vmax=1.25)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{100 * p[i, j]:.0f}%\n({c[i, j]})", ha="center", va="center", fontsize=15,
                        color="white" if p[i, j] > 0.6 else "black", fontweight="bold" if i == j else "normal")
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(lab, rotation=30, ha="right", rotation_mode="anchor", fontsize=15)
        ax.set_yticklabels(lab if ax is axs[0] else [""] * 4, fontsize=15)
        ax.set_xlabel(T[b]["pred"])
        ax.set_title(f"{m}  ({T[b]['pooled']})", fontsize=18, fontweight="bold", color=G.warna_model(m), pad=10)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    axs[0].set_ylabel(T[b]["true"])
    return G.simpan(fig, b, "07_confusion_matrix")


# =========================================================== 8. kurva training (TURUNAN: epoch terbaik = fitness maks)
def g08_kurva(b):
    fig, axs = plt.subplots(1, 2, figsize=(G.W_IN, G.H_IN), layout="constrained")
    for m, nm in (("yolov8s", "YOLOv8s"), ("yolo11s", "YOLO11s")):
        rows = baca(R / f"runs/v7_final_prov_{m}_s0/results.csv")
        rows = [{k.strip(): v for k, v in r.items()} for r in rows]
        ep = np.array([int(r["epoch"]) for r in rows])
        tl = np.array([float(r["train/box_loss"]) for r in rows])
        vl = np.array([float(r["val/box_loss"]) for r in rows])
        mp = np.array([float(r["metrics/mAP50-95(B)"]) for r in rows])
        m50 = np.array([float(r["metrics/mAP50(B)"]) for r in rows])
        fit = 0.1 * m50 + 0.9 * mp                 # fitness Ultralytics untuk best.pt
        best = int(ep[fit.argmax()])
        c = G.warna_model(nm)
        axs[0].plot(ep, tl, color=c, lw=2.5, label=f"{nm} · {T[b]['train_l']}")
        axs[0].plot(ep, vl, color=c, lw=2.5, ls="--", label=f"{nm} · {T[b]['val_l']}")
        axs[1].plot(ep, mp, color=c, lw=2.5, label=f"{nm} ({T[b]['best']} {best})")
        axs[1].plot(best, mp[fit.argmax()], marker="*", markersize=24, color=c, markeredgecolor="black", zorder=4)
        LAPORAN.append(f"Kurva training {nm} seed 0: epoch terbaik = {best} (fitness Ultralytics 0,1 x mAP50 + 0,9 x mAP50-95 maksimum "
                       f"pada val bersih); mAP50-95 val di epoch itu {mp[fit.argmax()]:.3f}.")
    axs[0].set_xlabel(T[b]["epoch"])
    axs[0].set_ylabel(T[b]["box_loss"])
    axs[0].legend(fontsize=15)
    axs[1].set_xlabel(T[b]["epoch"])
    axs[1].set_ylabel(T[b]["map_val"])
    axs[1].legend(loc="lower right")
    return G.simpan(fig, b, "08_kurva_training" if b == "id" else "08_training_curves")


# =========================================================== 9. contoh kualitatif (tanpa kotak prediksi)
def pilih_kualitatif():
    rows = baca(R / "runs/v7_final_prov_yolov8s_s0/eval_test/diagnosis_bersih_per_citra.csv")
    rows = [r for r in rows if r["berkas"].startswith("ds4_")]
    rng = random.Random(20261027)
    benar_b = sorted([r for r in rows if r["label"] == "leaf_blast" and r["diagnosis"] == "leaf_blast"], key=lambda r: r["berkas"])
    benar_s = sorted([r for r in rows if r["label"] == "brown_spot" and r["diagnosis"] == "brown_spot"], key=lambda r: r["berkas"])
    salah_bs = sorted([r for r in rows if r["label"] == "leaf_blast" and r["diagnosis"] == "brown_spot"], key=lambda r: r["berkas"])
    salah_lain = sorted([r for r in rows if r["label"] != r["diagnosis"] and not (r["label"] == "leaf_blast" and r["diagnosis"] == "brown_spot")],
                        key=lambda r: r["berkas"])
    return [("correct", rng.choice(benar_b)), ("correct", rng.choice(benar_s)), ("wrong", rng.choice(salah_bs)), ("wrong", rng.choice(salah_lain))]


def g09_kualitatif(b, pilih):
    """Kotak label (putih) + kotak prediksi (putus-putus, warna kelas) dari INFERENSI ULANG UNTUK ILUSTRASI
    (bahan_presentasi/_ilustrasi_prediksi_gambar9.json, scripts/presentasi/ilustrasi_prediksi.py; pengaturan resmi, kotak conf >= ambang).
    Teks diagnosis dan keyakinan tetap dari HASIL TERSIMPAN (CSV evaluasi resmi), 2 desimal."""
    import json
    ilus = json.loads((G.OUT / "_ilustrasi_prediksi_gambar9.json").read_text(encoding="utf-8"))
    pred = {h["foto"]: h for h in ilus["foto"]}
    ambang = ilus["pengaturan"]["ambang_diagnosis"]
    beda_maks = max(abs(round(h["conf_teratas"], 4) - float(h["resmi"]["conf_teratas"])) for h in ilus["foto"])
    assert beda_maks <= 1e-4 + 1e-9 and all(h["diagnosis"] == h["resmi"]["diagnosis"] for h in ilus["foto"])
    WARNA_PRED = {"leaf_blast": G.KELAS["leaf_blast"], "brown_spot": G.KELAS["brown_spot"], "bacterial_leaf_blight": "#F0E442"}
    fig = G.kanvas()
    kelas_tampil = set()
    for j, (jenis, r) in enumerate(pilih):
        ax = fig.add_axes([0.012 + j * 0.247, 0.47, 0.235, 0.43])
        f = r["berkas"]
        im = Image.open(R / "data/processed/prov_v7/images/test" / f).convert("RGB")
        W, H = im.size
        ax.imshow(np.asarray(im))
        lab = R / "data/processed/prov_v7/labels/test" / (Path(f).stem + ".txt")
        for ln in (lab.read_text().splitlines() if lab.exists() else []):
            c, x, y, w, h = map(float, ln.split()[:5])
            ax.add_patch(Rectangle(((x - w / 2) * W, (y - h / 2) * H), w * W, h * H, fill=False, edgecolor="white", linewidth=2.2))
        assert pred[f]["ukuran"] == [W, H], (f, pred[f]["ukuran"], W, H)
        for k in pred[f]["kotak_ge_ambang"]:
            x1, y1, x2, y2 = k["xyxy"]
            kelas_tampil.add(k["kelas"])
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=WARNA_PRED[k["kelas"]], linewidth=2.6, linestyle=(0, (4, 2))))
        ax.set_xticks([])
        ax.set_yticks([])
        warna = "#009E73" if jenis == "correct" else "#D55E00"
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_edgecolor(warna)
            sp.set_linewidth(5)
        ax.set_title(T[b][jenis], fontsize=20, fontweight="bold", color=warna, pad=10)
        dg = r["diagnosis"]
        lb = "background" if r["label"] == "sehat_atau_tidak_terdeteksi" else r["label"]   # label: cukup "Sehat"
        ax.text(0.5, -0.04, f"{T[b]['label']}: {G.nama_kelas(b, lb)}\n{T[b]['diag']}: {G.nama_kelas(b, dg)}\n"
                f"({T[b]['conf']} {G.angka(float(r['conf_teratas']), b, 2)})", transform=ax.transAxes, ha="center", va="top", fontsize=16,
                linespacing=1.3)
    isi = "#6E6E6E"
    h = [Patch(facecolor=isi, edgecolor="white", linewidth=3, label=T[b]["leg_label"])]
    h += [Patch(facecolor=isi, edgecolor=WARNA_PRED[k], linewidth=3, linestyle="--", label=f"{T[b]['leg_pred']}: {G.nama_kelas(b, k)}")
          for k in ("leaf_blast", "brown_spot", "bacterial_leaf_blight") if k in kelas_tampil]
    fig.legend(handles=h, loc="center", bbox_to_anchor=(0.5, 0.235), ncol=len(h), fontsize=16, frameon=False, handlelength=2.2)
    fig.text(0.5, 0.012, T[b]["boxes_note"].format(ambang=G.angka(ambang, b, 2), beda=G.angka(beda_maks, b, 4)), ha="center", va="bottom",
             fontsize=15, color=G.ABU, linespacing=1.3)
    return G.simpan(fig, b, "09_contoh_kualitatif" if b == "id" else "09_qualitative_examples")


# =========================================================== 10a/10b. latar (versi slide)
def g10a_daun(b):
    pilih = baca(R / "paper/figures/fig_leaf_examples.csv")
    fig = G.kanvas()
    for j, r in enumerate(pilih):
        f = r["terpilih"]
        for i, (d, lbl) in enumerate(((R / "data/processed/prov_v7/images/val", "orig"), (R / "data/processed/prov_v9/images/val", "leaf"))):
            ax = fig.add_axes([0.17 + j * 0.275, 0.485 - i * 0.465, 0.42 * 6 / G.W_IN, 0.42])
            im = Image.open(d / f).convert("RGB")
            ax.imshow(np.asarray(im))
            if i == 0:
                p = R / "data/processed/prov_v7/labels/val" / (Path(f).stem + ".txt")
                Wd, Hd = im.size
                for ln in (p.read_text().splitlines() if p.exists() else []):
                    c, x, y, w, h = map(float, ln.split()[:5])
                    ax.add_patch(Rectangle(((x - w / 2) * Wd, (y - h / 2) * Hd), w * Wd, h * Hd, fill=False, edgecolor="#F0E442", linewidth=2))
                ax.set_title(G.nama_kelas(b, r["kelas"]), fontsize=20, pad=8)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
    fig.text(0.13, 0.695, T[b]["orig"], fontsize=20, ha="center", va="center", rotation=90)
    fig.text(0.13, 0.23, T[b]["leaf"], fontsize=20, ha="center", va="center", rotation=90)
    return G.simpan(fig, b, "10a_contoh_tanpa_latar" if b == "id" else "10a_leaf_only_examples")


def g10b_latar(b):
    rows = baca(R / "paper/figures/fig_background.csv")
    kond = ["Orig/Orig", "Orig/Leaf", "Leaf/Leaf", "Leaf/Orig"]
    met = [("mAP50-95", "map", (0, 0.42)), ("Diagnosis accuracy", "acc", (0, 1.15)), ("Blast recall", "blast_r", (0, 1.15))]
    fig, axs = plt.subplots(1, 3, figsize=(G.W_IN, G.H_IN), layout="constrained")
    w = 0.38
    x = np.arange(4)
    for ax, (mk, lab, lim) in zip(axs, met):
        for j, m in enumerate(("YOLOv8s", "YOLO11s")):
            hs = [next(r for r in rows if r["train_test"] == k and r["model"] == m and r["metrik"] == mk) for k in kond]
            y = np.array([float(h["rata2_3seed"]) for h in hs])
            e = np.array([[y[i] - float(h["ik95_bawah"]) for i, h in enumerate(hs)], [float(h["ik95_atas"]) - y[i] for i, h in enumerate(hs)]])
            ax.bar(x + (j - 0.5) * w, y, w * 0.95, yerr=e, capsize=4, error_kw={"elinewidth": 1.5, "capthick": 1.5}, label=m,
                   **G.gaya_batang(m))
        ax.set_xticks(x)
        ax.set_xticklabels([T[b]["cond"][k] for k in kond], rotation=40, ha="right", rotation_mode="anchor", fontsize=16)
        ax.set_ylabel(T[b][lab])
        ax.set_ylim(*lim)
    axs[0].legend(loc="upper center", ncol=2, fontsize=15, columnspacing=1.0, handlelength=1.5)
    G.catatan(fig, f"{T[b]['tt']}: {T[b]['tt_note']}  ·  {T[b]['ci']}")
    return G.simpan(fig, b, "10b_ketergantungan_latar" if b == "id" else "10b_background_dependence")


# =========================================================== 11. ds3 (grafik saja)
def g11_ds3(b):
    rows = [r for r in baca(V9T / "tabel7_ds3.csv") if r["Set"].startswith("ds3 uji utama")]
    fig, axs = plt.subplots(1, 2, figsize=(G.W_IN, G.H_IN), layout="constrained")
    for ax, (k, kik, lab) in zip(axs, (("Akurasi", "IK95 akurasi", "acc"), ("R Blast", "IK95 R Blast", "blast_r"))):
        for mi, m in enumerate(("YOLOv8s", "YOLO11s")):
            for ci, (kond, ttl) in enumerate((("v7 pada ds3 asli", "v7"), ("v9 pada ds3 tanpa latar", "v9"))):
                r = next(x for x in rows if x["Model"] == m and x["Kondisi"] == kond)
                v = ms(r[k])[0]
                lo, hi = ik(r[kik])
                st = G.gaya_batang(m)
                if ci == 1:
                    st = {"facecolor": G.tint(G.warna_model(m), 0.35), "edgecolor": G.warna_model(m), "hatch": "..", "linewidth": 1.5}
                xx = mi + (ci - 0.5) * 0.36
                ax.bar(xx, v, 0.34, yerr=[[v - lo], [hi - v]], capsize=6, error_kw={"elinewidth": 1.6, "capthick": 1.6}, **st)
                ax.text(xx, hi + 0.02, G.angka(v, b), ha="center", fontsize=15)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["YOLOv8s", "YOLO11s"], fontsize=18)
        ax.set_ylabel(T[b][lab])
        ax.set_ylim(0, 0.8)
    axs[0].legend(handles=[Patch(facecolor=G.ABU, edgecolor=G.ABU, label=T[b]["v7"]),
                           Patch(facecolor=G.tint(G.ABU, 0.35), edgecolor=G.ABU, hatch="..", label=T[b]["v9"])], loc="upper left", fontsize=15)
    G.catatan(fig, T[b]["ds3_note"], ha="right")
    return G.simpan(fig, b, "11_hasil_ds3" if b == "id" else "11_external_ds3")


def main():
    kembar = pilih_kembar()
    kual = pilih_kualitatif()
    cm = {"YOLOv8s": confusion("yolov8s"), "YOLO11s": confusion("yolo11s")}
    for b in ("en", "id"):
        g01_alur(b)
        g02_dataset(b)
        g03a_kembar(b, kembar)
        g03b_bocor(b)
        g04_utama(b)
        g05_forest(b)
        g06_kecepatan(b)
        g07_confusion(b, cm)
        g08_kurva(b)
        g09_kualitatif(b, kual)
        g10a_daun(b)
        g10b_latar(b)
        g11_ds3(b)
    # catatan pemilihan dan angka turunan (untuk README)
    cat = ["# Catatan otomatis gambar_penelitian.py", ""]
    cat += [f"- Foto kembar (3a), kelas {k}: kelompok {g}; latih `{a}`, uji `{c}` (dipilih acak seed 20261027 dari {n} kelompok ds4 yang "
            f"anggotanya ada di train dan test split acak)." for k, g, a, c, n in kembar]
    cat += [f"- Contoh kualitatif (9): {j} `{r['berkas']}` label {r['label']}, diagnosis {r['diagnosis']}, conf {r['conf_teratas']}"
            for j, r in kual]
    for m, c in cm.items():
        cat.append(f"- Confusion {m} (test semua kelas, 3 seed digabung; TURUNAN = jumlah 3 CSV confusion): baris = {KLS4}; "
                   f"{c.tolist()}; total {int(c.sum())}")
    cat += [f"- {x}" for x in dict.fromkeys(LAPORAN)]
    (G.OUT / "_catatan_gambar_penelitian.md").write_text("\n".join(cat) + "\n", encoding="utf-8")
    print("\n".join(cat))


if __name__ == "__main__":
    main()
