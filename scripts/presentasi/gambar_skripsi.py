#!/usr/bin/env python3
"""Gambar khusus skripsi (bahasa Indonesia saja, skripsi_id/): 12 arsitektur aplikasi, 13 forward chaining + tabel aturan R001-R003,
14 tangkapan layar aplikasi (HP dan desktop), 15 ringkasan pengujian aplikasi, 16 flowchart penelitian versi aktual.
Hanya membaca berkas aplikasi dan hasil uji yang sudah ada (app/config.yaml, app/rbs/rules.yaml, app/docs/uji/*, app/docs/tangkapan_layar/*).

  .venv\\Scripts\\python.exe scripts\\presentasi\\gambar_skripsi.py"""
import json
import shutil
import sys
import textwrap
from pathlib import Path

import numpy as np
import yaml
from matplotlib.patches import FancyBboxPatch, Patch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gaya as G  # noqa: E402
from gaya import plt  # noqa: E402

R = G.ROOT
APP = R / "app"
B = "id"
ABU_KOTAK = "#F4F4F4"
HIJAU, MERAH = "#009E73", "#D55E00"


def kotak(ax, x, y, w, h, judul, isi="", tepi=G.ABU, isian=ABU_KOTAK, garis="-", fs_j=18, fs_i=15):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.006,rounding_size=0.015", facecolor=isian,
                                edgecolor=tepi, linewidth=2.4, linestyle=garis))
    if isi:
        ax.text(x, y + h / 2 - 0.045, judul, fontsize=fs_j, fontweight="bold", ha="center", va="center")
        ax.text(x, y - 0.03, isi, fontsize=fs_i, ha="center", va="center", linespacing=1.3, color="#222222")
    else:
        ax.text(x, y, judul, fontsize=fs_j, fontweight="bold", ha="center", va="center")


def panah(ax, a, b, teks=None, warna=G.ABU, gaya="-|>"):
    ax.annotate("", xy=b, xytext=a, arrowprops={"arrowstyle": gaya, "lw": 2.4, "color": warna, "mutation_scale": 24})
    if teks:
        ax.text((a[0] + b[0]) / 2 + 0.008, (a[1] + b[1]) / 2, teks, fontsize=15, color=warna, ha="left", va="center")


def kanvas_kosong():
    fig = G.kanvas()
    ax = fig.add_axes([0.005, 0.005, 0.99, 0.99])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


# =========================================================== 12. arsitektur aplikasi
def g12_arsitektur():
    cfg = yaml.safe_load((APP / "config.yaml").read_text(encoding="utf-8"))
    tau, t_det, t_ren = cfg["pemisah_daun"]["tau"], cfg["ambang"]["terdeteksi"], cfg["ambang"]["rendah"]
    k = lambda v: f"{v:.2f}".replace(".", ",")  # noqa: E731
    fig, ax = kanvas_kosong()
    w, h = 0.22, 0.36
    xs = [0.125, 0.375, 0.625, 0.875]
    y1, y2 = 0.76, 0.29
    kotak(ax, xs[0], y1, w, h, "1  Unggah foto", "kamera HP atau berkas\nJPG/PNG ≤ 10 MB\ndibaca di memori,\norientasi dikoreksi")
    kotak(ax, xs[1], y1, w, h, "2  Pemisah daun", f"YOLOv8n-seg (ONNX)\npilih satu helai daun\n(keyakinan ≥ {k(tau)})\nlatar diganti abu")
    kotak(ax, xs[2], y1, w, h, "3  Deteksi bercak", "YOLOv8s v9 (ONNX)\ndilatih tanpa latar\nkotak berkeyakinan\ntertinggi = diagnosis")
    kotak(ax, xs[3], y1, w, h, "4  Tiga tingkat", f"≥ {k(t_det)}: terdeteksi\n{k(t_ren)}–{k(t_det)}: kemungkinan\n< {k(t_ren)}: tidak terdeteksi",
          tepi=G.MODEL["YOLO11"], isian="#FFF4DD")
    kotak(ax, xs[3], y2, w, h, "5  Sistem aturan", "Rule-Based System,\nforward chaining\nrules.yaml (Tabel 3.1)\nsaran + jejak penalaran")
    kotak(ax, xs[2], y2, w, h, "6  Tampilan hasil", "foto asli + kotak bercak\npita keyakinan, saran,\n\"Mengapa sistem\nmenyimpulkan ini\"")
    kotak(ax, xs[1], y2, w, h, "Pesan foto ulang", "daun tidak jelas:\ntidak didiagnosis,\npengguna diminta\nmemotret ulang", tepi=MERAH, garis="--")
    for i in range(3):
        panah(ax, (xs[i] + w / 2, y1), (xs[i + 1] - w / 2, y1))
    panah(ax, (xs[3], y1 - h / 2), (xs[3], y2 + h / 2))
    panah(ax, (xs[3] - w / 2, y2), (xs[2] + w / 2, y2))
    panah(ax, (xs[1], y1 - h / 2), (xs[1], y2 + h / 2), "daun tidak jelas", warna=MERAH)
    panah(ax, (xs[1] + w / 2, y2), (xs[2] - w / 2, y2), warna=MERAH)
    ax.text(0.01, 0.015, "Aplikasi Streamlit \"Tilik Padi\"; kedua model dimuat sekali, berjalan di CPU (app/config.yaml).",
            fontsize=15, color=G.ABU, ha="left", va="bottom")
    return G.simpan(fig, B, "12_arsitektur_aplikasi")


# =========================================================== 13a. forward chaining
def g13a_forward():
    fig, ax = kanvas_kosong()
    x0 = 0.215
    kotak(ax, x0, 0.88, 0.40, 0.14, "Fakta awal", "penyakit · tingkat · keyakinan", fs_i=16)
    ax.add_patch(plt.Polygon([[x0, 0.72], [x0 + 0.20, 0.585], [x0, 0.45], [x0 - 0.20, 0.585]], closed=True, facecolor="#FFF4DD",
                             edgecolor=G.MODEL["YOLO11"], linewidth=2.4))
    ax.text(x0, 0.585, "Ada aturan belum aktif\nyang semua syaratnya\nbenar?", fontsize=16, ha="center", va="center", fontweight="bold")
    kotak(ax, x0, 0.25, 0.40, 0.17, "Aktifkan aturan pertama", "tambah fakta baru, saran,\ndan catat jejak", fs_i=16)
    kotak(ax, 0.535, 0.585, 0.15, 0.17, "Berhenti", "tampilkan saran\ndan jejak", fs_i=16)
    panah(ax, (x0, 0.81), (x0, 0.72))
    panah(ax, (x0, 0.45), (x0, 0.335), "ya")
    panah(ax, (x0 + 0.20, 0.585), (0.46, 0.585))
    ax.text(0.437, 0.605, "tidak", fontsize=16, color=G.ABU, ha="center")
    ax.plot([x0 - 0.20, 0.015, 0.015, x0 - 0.20], [0.25, 0.25, 0.585, 0.585], color=G.ABU, lw=2.4)
    panah(ax, (0.015, 0.585), (x0 - 0.20, 0.585))
    # rantai per tingkat
    ax.text(0.83, 0.93, "Rantai yang terjadi (2 putaran)", fontsize=18, fontweight="bold", ha="center")
    rantai = [("terdeteksi", "D01 → R001 / R002 / R003", "saran khusus penyakit (Tabel 3.1)"),
              ("kemungkinan", "K01 → K02", "konfirmasi ke penyuluh +\npencegahan umum"),
              ("tidak terdeteksi", "N01 → N02", "foto ulang + tanya penyuluh +\npencegahan umum")]
    for i, (t, r, s) in enumerate(rantai):
        y = 0.76 - i * 0.25
        kotak(ax, 0.835, y, 0.31, 0.21, f"tingkat: {t}", f"{r}\n{s}", fs_j=17, fs_i=15)
    ax.text(0.01, 0.015, "Sumber: app/rbs/rules.yaml dan app/rbs/mesin.py (aturan dicoba berurutan).",
            fontsize=15, color=G.ABU, ha="left", va="bottom")
    return G.simpan(fig, B, "13a_forward_chaining")


# =========================================================== 13b. tabel ringkas aturan + 13c. teks lengkap R001-R003
NAMA_NILAI = {"leaf_blast": "Blast", "brown_spot": "Brown spot", "bacterial_leaf_blight": "BLB", "terdeteksi": "terdeteksi",
              "kemungkinan": "kemungkinan", "tidak_terdeteksi": "tidak terdeteksi", "tidak_ada": "tidak ada", "ya": "ya"}


def syarat(a):
    op = {"==": "=", "!=": "≠", ">=": "≥", "<": "<"}
    return " DAN ".join(f"{j['fakta'].replace('_', ' ')} {op.get(j['op'], j['op'])} {NAMA_NILAI.get(str(j['nilai']), j['nilai'])}"
                        for j in a["jika"])


def g13b_aturan():
    rb = yaml.safe_load((APP / "rbs/rules.yaml").read_text(encoding="utf-8"))
    fig, ax = kanvas_kosong()
    kol = [0.015, 0.095, 0.43, 0.70]
    lebar = {1: 30, 2: 26, 3: 26}
    for x, t in zip(kol, ("ID", "JIKA", "MAKA (fakta baru)", "Saran")):
        ax.text(x, 0.94, t, fontsize=18, fontweight="bold", va="center")
    ax.plot([0.01, 0.99], [0.905, 0.905], color=G.ABU, lw=1.8)
    y = 0.855
    for i, a in enumerate(rb["aturan"]):
        if i % 2 == 0:
            ax.add_patch(plt.Rectangle((0.01, y - 0.048), 0.98, 0.096, facecolor="#F4F4F4", edgecolor="none"))
        maka = ", ".join(f"{k.replace('_', ' ')} = {str(v).replace('{penyakit}', 'penyakit')}" for k, v in a["maka"].items())
        sr = a.get("saran", [])
        if sr and isinstance(sr[0], dict):
            nS, nP = sum(x["kelompok"] == "S" for x in sr), sum(x["kelompok"] == "P" for x in sr)
            saran = f"{nS} tindakan sekarang + {nP} pencegahan"
        elif sr:
            saran = f"{len(sr)} saran umum"
        else:
            saran = "–"
        ax.text(kol[0], y, a["id"], fontsize=17, fontweight="bold", va="center")
        for c, t in ((1, syarat(a)), (2, maka), (3, saran)):
            ax.text(kol[c], y, "\n".join(textwrap.wrap(t, lebar[c])), fontsize=16, va="center", linespacing=1.1)
        y -= 0.098
    ax.text(0.01, 0.015, "Sumber: app/rbs/rules.yaml. Teks lengkap R001–R003 (Tabel 3.1 proposal): gambar 13c_*.", fontsize=15, color=G.ABU,
            va="bottom")
    return G.simpan(fig, B, "13b_tabel_aturan")


def g13c_rinci():
    rb = yaml.safe_load((APP / "rbs/rules.yaml").read_text(encoding="utf-8"))
    at = {a["id"]: a for a in rb["aturan"]}
    nama = {"R001": "Blast (blas daun)", "R002": "BLB (hawar daun bakteri)", "R003": "Brown spot (bercak cokelat)"}
    for rid in ("R001", "R002", "R003"):
        a = at[rid]
        fig, ax = kanvas_kosong()
        ax.text(0.015, 0.93, f"{rid}  {nama[rid]}", fontsize=24, fontweight="bold", va="center")
        ax.text(0.015, 0.865, f"JIKA {syarat(a)}  (setelah D01: tingkat terdeteksi)", fontsize=17, color=G.ABU, va="center")
        for j, (kel, x) in enumerate((("S", 0.015), ("P", 0.51))):
            ax.text(x, 0.79, rb["kelompok"][kel], fontsize=19, fontweight="bold", va="top")
            y = 0.72
            for t in [z["teks"] for z in a["saran"] if z["kelompok"] == kel]:
                baris = textwrap.wrap(t, 46)
                ax.text(x, y, "• " + "\n   ".join(baris), fontsize=17, va="top", linespacing=1.25)
                y -= 0.052 * len(baris) + 0.03
        ax.plot([0.495, 0.495], [0.10, 0.80], color=G.ABU_MUDA, lw=1.5)
        ax.text(0.015, 0.075, f"Catatan yang selalu ditampilkan: {rb['catatan_tetap']}", fontsize=15, color=G.ABU, va="bottom")
        ax.text(0.015, 0.015, "Kata-kata butir persis Tabel 3.1 proposal (hlm. 18; Kementerian Pertanian RI, 2023).", fontsize=15,
                color=G.ABU, va="bottom")
        G.simpan(fig, B, f"13c_aturan_{rid}")


# =========================================================== 14. tangkapan layar
LAYAR = [("awal", "Halaman awal"), ("terdeteksi", "Terdeteksi"), ("kemungkinan", "Kemungkinan"),
         ("kemungkinan_lain", "Kemungkinan lain"), ("foto_ulang", "Foto ulang"), ("tentang", "Halaman Tentang")]


def potong(p, lebar_css, x0, y0, w, h):
    """Potong dalam ukuran CSS; tangkapan layar 2x (scripts/presentasi/tangkap_layar_2x.py) otomatis dikali skalanya."""
    im = Image.open(p).convert("RGB")
    k = im.width / lebar_css
    x0, y0, w, h = (round(v * k) for v in (x0, y0, w, h))
    kanvas = Image.new("RGB", (w, h), "white")
    kanvas.paste(im.crop((x0, y0, min(x0 + w, im.width), min(y0 + h, im.height))), (0, 0))
    return kanvas


def g14_layar():
    asli = G.FOLDER[B] / "14_tangkapan_layar_asli"          # tangkapan layar 2x dari scripts/presentasi/tangkap_layar_2x.py
    out = []
    for kode, judul in LAYAR:
        hp, ds = asli / f"{kode}_390.png", asli / f"{kode}_1280.png"
        assert Image.open(hp).width == 780 and Image.open(ds).width == 2560, "jalankan tangkap_layar_2x.py dulu"
        fig = G.kanvas()
        # dua layar HP (390 x 844: layar pertama dan kedua halaman) + layar desktop pertama (1280 x 800)
        for i in range(2):
            a = fig.add_axes([0.012 + i * 0.235, 0.03, 0.225, 0.86])
            a.imshow(np.asarray(potong(hp, 390, 0, i * 844, 390, 844)), interpolation="lanczos")
            a.set_xticks([])
            a.set_yticks([])
            for s in a.spines.values():
                s.set_visible(True)
                s.set_color(G.ABU_MUDA)
                s.set_linewidth(1.5)
            a.set_title(f"HP, layar {i + 1}", fontsize=17, pad=6)
        a = fig.add_axes([0.49, 0.17, 0.50, 0.62])
        a.imshow(np.asarray(potong(ds, 1280, 0, 0, 1280, 800)), interpolation="lanczos")
        a.set_xticks([])
        a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(True)
            s.set_color(G.ABU_MUDA)
            s.set_linewidth(1.5)
        a.set_title("Desktop 1280 px, layar pertama", fontsize=17, pad=6)
        fig.text(0.74, 0.10, judul, fontsize=22, fontweight="bold", ha="center")
        out.append(G.simpan(fig, B, f"14_layar_{kode}"))
    return out


# =========================================================== 15. ringkasan pengujian aplikasi
def g15_pengujian():
    U = APP / "docs/uji"
    bb = json.loads((U / "blackbox.json").read_text(encoding="utf-8"))
    sd = json.loads((U / "sumber_daya.json").read_text(encoding="utf-8"))
    pd = json.loads((U / "pemisah_daun.json").read_text(encoding="utf-8"))
    kt = json.loads((APP / "docs/uji_kesetaraan.json").read_text(encoding="utf-8"))
    n_test = UNIT_TEST
    fmt = lambda v, d=0: (f"{v:,.{d}f}".replace(",", "_").replace(".", ",").replace("_", "."))  # noqa: E731
    baris = [
        ("Black-box: perilaku sistem", f"foto VAL acak {bb['foto_sesuai']}/10 · masukan khusus {bb['khusus_sesuai']}/4 · "
                                       f"fitur penyampaian {bb['fitur_sesuai']}/{bb['fitur_n']} sesuai"),
        ("Unit test (pytest)", f"{n_test} test lulus (termasuk 1 test black-box)"),
        ("Kesetaraan detektor ONNX", f"tingkat jawaban sama {kt['tingkat_sama']}/{kt['n']} · kelas sama "
                                     f"{kt['kelas_identik_keyakinan_acuan_ge_0_01']} (keyakinan ≥ 0,01) · selisih maks. "
                                     f"{fmt(kt['selisih_maks_vs_onnx_ultralytics'], 4)}"),
        ("Kesetaraan pemisah daun", f"keputusan foto ulang sama {fmt(pd['keputusan_cadangan_sama'])}/{fmt(pd['foto_val'])} foto VAL · "
                                    f"IoU masker median {fmt(pd['iou_masker']['median'], 3)}"),
        ("RAM (Streamlit + 2 model)", f"diam {fmt(sd['rss_diam_setelah_model_dimuat_MB'], 1)} MB · puncak "
                                      f"{fmt(sd['rss_puncak_saat_memproses_MB'], 1)} MB · setelah 11 foto "
                                      f"{fmt(sd['rss_diam_setelah_11_foto_MB'], 1)} MB"),
        ("Waktu respons (CPU)", f"inti {fmt(sd['waktu_inti_ms_median_20_val'])} ms · ujung ke ujung "
                                       f"{fmt(sd['waktu_ujung_ke_ujung_ms_median_val'])} ms · foto 4000×3000 "
                                       f"{fmt(sd['waktu_ujung_ke_ujung_ms_median_4000x3000'])} ms"),
    ]
    fig, ax = kanvas_kosong()
    y = 0.90
    tinggi = 0.135
    for i, (a, b) in enumerate(baris):
        if i % 2 == 0:
            ax.add_patch(plt.Rectangle((0.01, y - tinggi / 2), 0.98, tinggi, facecolor="#F4F4F4", edgecolor="none"))
        ax.text(0.025, y, a, fontsize=18, fontweight="bold", va="center")
        ax.text(0.37, y, "\n".join(textwrap.wrap(b, 62)), fontsize=16, va="center", linespacing=1.3)
        y -= tinggi + 0.01
    ax.text(0.01, 0.015, "Sumber: app/docs/uji/*.json dan app/docs/uji_kesetaraan.json (perilaku sistem, bukan ketepatan diagnosis).",
            fontsize=15, color=G.ABU, va="bottom")
    return G.simpan(fig, B, "15_ringkasan_pengujian_aplikasi")


# =========================================================== 16. flowchart penelitian versi aktual
def g16_flowchart():
    SAMA, UBAH, BARU = ("-", G.ABU, ABU_KOTAK), ("-", HIJAU, "#E6F5F0"), ("--", MERAH, "#FBEBE3")
    langkah = [
        ("1  Studi literatur", "YOLOv8, YOLO11, RBS,\npenyakit padi, aplikasi web", SAMA),
        ("2  Pengumpulan dataset", "4 sumber publik: Mendeley V3,\n2 Kaggle, set Indonesia", UBAH),
        ("3  Audit data", "orientasi 931 foto diperbaiki;\nBLB satu sumber (provisional)", BARU),
        ("4  Split bebas kebocoran", "kelompok duplikat, split per\nkelompok + sesi baru", UBAH),
        ("5  Pelatihan", "YOLOv8n/s dan YOLO11n/s\n× 3 seed, GPU laptop", UBAH),
        ("6  Evaluasi dan komparasi", "ambang dikunci, uji sekali,\nsplit acak, set Indonesia", UBAH),
        ("7  Eksperimen latar", "foto tanpa latar (SAM 2.1),\ndesain latih/uji 2 × 2", BARU),
        ("8  Rule-Based System", "forward chaining, Tabel 3.1,\ntiga tingkat jawaban", SAMA),
        ("9  Aplikasi web", "Streamlit: pemisah daun\n+ YOLOv8s v9 + RBS", UBAH),
        ("10  Pengujian aplikasi", "black-box, unit test,\nRAM dan waktu respons", BARU),
        ("11  Penulisan", "laporan skripsi\ndan naskah paper", SAMA),
    ]
    fig, ax = kanvas_kosong()
    w, h = 0.30, 0.175
    xs, ys = [0.165, 0.5, 0.835], [0.865, 0.64, 0.415, 0.19]
    urut = [(0, 0), (1, 0), (2, 0), (2, 1), (1, 1), (0, 1), (0, 2), (1, 2), (2, 2), (2, 3), (1, 3)]
    pos = [(xs[c], ys[r]) for c, r in urut]
    for (jd, isi, (gs, tp, isian)), (x, y) in zip(langkah, pos):
        kotak(ax, x, y, w, h, jd, isi, tepi=tp, isian=isian, garis=gs, fs_j=17, fs_i=15)
    for i in range(len(pos) - 1):
        (xa, ya), (xb, yb) = pos[i], pos[i + 1]
        if ya == yb:
            sg = 1 if xb > xa else -1
            panah(ax, (xa + sg * w / 2, ya), (xb - sg * w / 2, yb))
        else:
            panah(ax, (xa, ya - h / 2), (xb, yb + h / 2))
    ax.legend(handles=[Patch(facecolor=ABU_KOTAK, edgecolor=G.ABU, linewidth=2, label="sesuai proposal"),
                       Patch(facecolor="#E6F5F0", edgecolor=HIJAU, linewidth=2, label="diubah dari proposal"),
                       Patch(facecolor="#FBEBE3", edgecolor=MERAH, linewidth=2, linestyle="--", label="tahap baru")],
              loc="center", bbox_to_anchor=(xs[0], ys[3]), fontsize=16)
    ax.text(0.01, 0.015, "Dibandingkan dengan Gambar 3.1 proposal (BAB3_Metodologi); rincian perubahan di README.", fontsize=15,
            color=G.ABU, va="bottom")
    return G.simpan(fig, B, "16_flowchart_penelitian_aktual")


UNIT_TEST = None


def main():
    global UNIT_TEST
    import subprocess
    r = subprocess.run([str(APP / ".venv/Scripts/python.exe"), "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", "tests"],
                       cwd=APP, capture_output=True, text=True, env={**__import__("os").environ, "PYTHONDONTWRITEBYTECODE": "1"})
    UNIT_TEST = int([l for l in r.stdout.splitlines() if "tests collected" in l or "test collected" in l][-1].split()[0])
    for f in (g12_arsitektur, g13a_forward, g13b_aturan, g13c_rinci, g14_layar, g15_pengujian, g16_flowchart):
        f()
    print("unit test terkumpul:", UNIT_TEST)


if __name__ == "__main__":
    main()
