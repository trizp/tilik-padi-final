"""Gaya bersama bahan presentasi: kanvas 16:9 (3200x1800 px pada 300 dpi = 10,67 x 6 in), sans-serif besar, palet Okabe-Ito tetap,
latar putih, tanpa judul di dalam gambar. Ditampilkan penuh di slide 16:9 (13,33 in), skala x1,25: huruf 15 pt di gambar = 18,75 pt di slide.
Semua teks label lewat kamus T[bahasa][kunci] agar isi gambar bahasa Inggris dan Indonesia identik."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "bahan_presentasi"
FOLDER = {"en": OUT / "paper_en", "id": OUT / "skripsi_id"}
W_IN, H_IN, DPI = 3200 / 300, 1800 / 300, 300
HURUF_MIN = 15          # pt di gambar (>= 18 pt saat tampil penuh)

# Okabe-Ito. Model: YOLOv8 selalu biru, YOLO11 selalu jingga. Kelas: tidak memakai biru/jingga agar tidak tertukar dengan model.
MODEL = {"YOLOv8": "#0072B2", "YOLO11": "#E69F00"}
KELAS = {"leaf_blast": "#D55E00", "brown_spot": "#CC79A7", "bacterial_leaf_blight": "#F0E442", "background": "#009E73"}
ABU = "#4D4D4D"
ABU_MUDA = "#BDBDBD"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 17,
    "axes.labelsize": 18, "axes.titlesize": 18, "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 16,
    "svg.fonttype": "none", "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 1.2, "xtick.major.width": 1.2, "ytick.major.width": 1.2, "xtick.major.size": 5, "ytick.major.size": 5,
    "hatch.linewidth": 1.2, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "legend.frameon": False,
})


def warna_model(nama):
    """'YOLOv8s' -> biru, 'YOLO11n' -> jingga."""
    return MODEL["YOLO11"] if nama.startswith("YOLO11") else MODEL["YOLOv8"]


def tint(c, f=0.55):
    r, g, b = to_rgb(c)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


def gaya_batang(nama):
    """Varian n: warna muda + arsir; varian s: warna penuh."""
    c = warna_model(nama)
    if nama.endswith("n"):
        return {"facecolor": tint(c), "edgecolor": c, "hatch": "///", "linewidth": 1.5}
    return {"facecolor": c, "edgecolor": c, "linewidth": 1.5}


def kanvas(**kw):
    return plt.figure(figsize=(W_IN, H_IN), dpi=DPI, **kw)


def _koma(fig):
    """Sumbu numerik versi Indonesia: desimal koma, ribuan titik."""
    from matplotlib.ticker import FuncFormatter, ScalarFormatter
    import numpy as np
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            if not isinstance(axis.get_major_formatter(), ScalarFormatter):
                continue
            locs = [v for v in axis.get_majorticklocs() if np.isfinite(v)]
            d = next((k for k in range(4) if all(abs(round(v, k) - v) < 1e-9 for v in locs)), 3)
            axis.set_major_formatter(FuncFormatter(lambda v, p, d=d: (f"{v:,.{d}f}".replace(",", "_").replace(".", ",")
                                                                      .replace("_", ".")).replace("-", "−")))


def simpan(fig, bahasa, nama):
    if bahasa == "id":
        _koma(fig)
    d = FOLDER[bahasa]
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f"{nama}.png", dpi=DPI)
    fig.savefig(d / f"{nama}.svg")
    plt.close(fig)
    return d / f"{nama}.png"


# ------------------------------------------------------------------ teks
T = {
    "en": {
        "blast": "Blast", "brown": "Brown spot", "blb": "BLB (prov.)", "healthy": "Healthy",
        "healthy_none": "Healthy / none",
        "train": "Train", "val": "Val", "test": "Test", "new": "New session", "test_lbl": "Test (434 images)",
        "new_lbl": "New session (299 images)",
        "images": "Images", "groups": "groups",
        "map": "mAP50-95", "acc": "Diagnosis accuracy", "blast_r": "Blast recall", "blb_r": "BLB recall",
        "d_map": "ΔmAP50-95 (YOLO11 − YOLOv8)", "d_acc": "ΔAccuracy (YOLO11 − YOLOv8)",
        "gap_map": "Test − new session\n(mAP50-95)", "gap_acc": "Test − new session\n(diagnosis accuracy)",
        "leakfree": "Leakage-free split\n(3 seeds)", "random": "Random split\n(1 seed)",
        "seed": "individual seed", "mean": "mean",
        "cpu": "CPU time per image (ms)", "params": "parameters",
        "true": "True class", "pred": "Diagnosis", "pooled": "3 seeds pooled",
        "epoch": "Epoch", "box_loss": "Box loss", "map_val": "Validation mAP50-95", "train_l": "train", "val_l": "validation",
        "best": "best epoch",
        "label": "Label", "diag": "Diagnosis", "correct": "Correct", "wrong": "Wrong", "conf": "conf.",
        "boxes_note": ("Dashed boxes: YOLOv8s (seed 0) predictions, confidence ≥ {ambang}, re-run for illustration only\n"
                       "with the official settings; confidences differ by at most {beda} from stored results (likely GPU batch size).\n"
                       "Diagnosis and confidence text: stored evaluation results."),
        "leg_label": "Label (white box)", "leg_pred": "Prediction",
        "orig": "Original", "leaf": "Leaf-only",
        "tt": "Train/test images", "cond": {"Orig/Orig": "Orig/Orig", "Orig/Leaf": "Orig/Leaf", "Leaf/Leaf": "Leaf/Leaf",
                                             "Leaf/Orig": "Leaf/Orig"},
        "tt_note": "Orig = original photo, Leaf = background removed",
        "v7": "Trained on original", "v9": "Trained leaf-only (tested leaf-only)",
        "ds3_note": "External Indonesian set (70 detached leaves); error bars: 95% CI",
        "ci": "error bars: 95% bootstrap CI (by image group)", "sd_note": "dots: individual seeds",
        "ci_excl": "CI excludes 0", "ci_incl": "CI includes 0",
        "pair_n_test": "n · Test", "pair_n_new": "n · New session", "pair_s_test": "s · Test", "pair_s_new": "s · New session",
        "train_split": "Train (random split)", "test_split": "Test (random split)", "same_group": "same duplicate group",
        # alur penelitian
        "wf": [
            ("Data sources", "RiceLeafDiseaseBD V3\n(field photos, Bangladesh)\nKaggle BLB · Kaggle healthy\nIndonesian set (external test)"),
            ("Duplicate grouping", "pHash · DINOv2 ≥ 0.97\nSIFT ≥ 21 inliers\nburst shots ≤ 5 s apart\n→ 1,544 image groups"),
            ("Group-wise split", "70 : 20 : 10 by group\n+ held-out capture session\n(299 images)"),
            ("Training", "YOLOv8n/s · YOLO11n/s\n× 3 seeds, 100 epochs, 640 px\nlaptop GPU (RTX 3050 Ti)"),
            ("Evaluation", "threshold locked on validation\neach test set evaluated once\ngroup bootstrap 95% CI"),
            ("Background experiment", "SAM 2.1 leaf-only images\n2 × 2 train/test design\nYOLOv8s and YOLO11s"),
        ],
    },
    "id": {
        "blast": "Blast", "brown": "Brown spot", "blb": "BLB (prov.)", "healthy": "Sehat",
        "healthy_none": "Sehat / tidak terdeteksi",
        "train": "Latih", "val": "Validasi", "test": "Uji", "new": "Sesi baru", "test_lbl": "Uji (434 foto)",
        "new_lbl": "Sesi baru (299 foto)",
        "images": "Jumlah foto", "groups": "kelompok",
        "map": "mAP50-95", "acc": "Akurasi diagnosis", "blast_r": "Recall Blast", "blb_r": "Recall BLB",
        "d_map": "ΔmAP50-95 (YOLO11 − YOLOv8)", "d_acc": "ΔAkurasi (YOLO11 − YOLOv8)",
        "gap_map": "Uji − sesi baru\n(mAP50-95)", "gap_acc": "Uji − sesi baru\n(akurasi diagnosis)",
        "leakfree": "Split bebas kebocoran\n(3 seed)", "random": "Split acak\n(1 seed)",
        "seed": "tiap seed", "mean": "rata-rata",
        "cpu": "Waktu CPU per foto (ms)", "params": "parameter",
        "true": "Kelas sebenarnya", "pred": "Diagnosis", "pooled": "3 seed digabung",
        "epoch": "Epoch", "box_loss": "Box loss", "map_val": "mAP50-95 validasi", "train_l": "latih", "val_l": "validasi",
        "best": "epoch terbaik",
        "label": "Label", "diag": "Diagnosis", "correct": "Benar", "wrong": "Salah", "conf": "keyakinan",
        "boxes_note": ("Kotak putus-putus: prediksi YOLOv8s (seed 0), keyakinan ≥ {ambang}, dari inferensi ulang khusus ilustrasi\n"
                       "dengan pengaturan resmi; keyakinan berbeda maks. {beda} dari hasil tersimpan (diduga ukuran batch GPU).\n"
                       "Teks diagnosis dan keyakinan: hasil evaluasi tersimpan."),
        "leg_label": "Label (kotak putih)", "leg_pred": "Prediksi",
        "orig": "Asli", "leaf": "Tanpa latar",
        "tt": "Foto latih/uji", "cond": {"Orig/Orig": "Asli/Asli", "Orig/Leaf": "Asli/Daun", "Leaf/Leaf": "Daun/Daun",
                                         "Leaf/Orig": "Daun/Asli"},
        "tt_note": "Asli = foto asli, Daun = latar dihapus",
        "v7": "Dilatih foto asli", "v9": "Dilatih tanpa latar (diuji tanpa latar)",
        "ds3_note": "Set Indonesia eksternal (70 daun petik); garis galat: IK 95%",
        "ci": "garis galat: IK 95% bootstrap (per kelompok foto)", "sd_note": "titik: tiap seed",
        "ci_excl": "IK tidak memuat 0", "ci_incl": "IK memuat 0",
        "pair_n_test": "n · Uji", "pair_n_new": "n · Sesi baru", "pair_s_test": "s · Uji", "pair_s_new": "s · Sesi baru",
        "train_split": "Latih (split acak)", "test_split": "Uji (split acak)", "same_group": "kelompok duplikat yang sama",
        "wf": [
            ("Sumber data", "RiceLeafDiseaseBD V3\n(foto lapangan, Bangladesh)\nKaggle BLB · Kaggle sehat\nSet Indonesia (uji eksternal)"),
            ("Pengelompokan duplikat", "pHash · DINOv2 ≥ 0,97\nSIFT ≥ 21 inlier\nfoto beruntun ≤ 5 detik\n→ 1.544 kelompok foto"),
            ("Split per kelompok", "70 : 20 : 10 per kelompok\n+ satu sesi pemotretan ditahan\n(299 foto)"),
            ("Pelatihan", "YOLOv8n/s · YOLO11n/s\n× 3 seed, 100 epoch, 640 px\nGPU laptop (RTX 3050 Ti)"),
            ("Evaluasi", "ambang dikunci di validasi\ntiap set uji dievaluasi sekali\nIK 95% bootstrap per kelompok"),
            ("Eksperimen latar", "foto tanpa latar (SAM 2.1)\ndesain latih/uji 2 × 2\nYOLOv8s dan YOLO11s"),
        ],
    },
}


def nama_kelas(bahasa, k):
    return T[bahasa][{"leaf_blast": "blast", "brown_spot": "brown", "bacterial_leaf_blight": "blb", "background": "healthy",
                      "sehat_atau_tidak_terdeteksi": "healthy_none"}[k]]


def angka(x, bahasa, d=3):
    """Desimal koma untuk bahasa Indonesia."""
    s = f"{x:.{d}f}"
    return s.replace(".", ",") if bahasa == "id" else s


def catatan(fig, teks, ha="center"):
    """Catatan kecil di bawah gambar, dengan ruang yang disisihkan dari tata letak otomatis agar tidak bertumpuk dengan label sumbu."""
    eng = fig.get_layout_engine()
    if eng is not None:
        eng.set(rect=(0, 0.055, 1, 0.945))
    x = {"center": 0.5, "right": 0.99, "left": 0.01}[ha]
    fig.text(x, 0.008, teks, ha=ha, va="bottom", fontsize=HURUF_MIN, color=ABU)
