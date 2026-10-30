#!/usr/bin/env python3
"""fig_leaf_examples.pdf: 2 baris x 3 kolom (Blast, Brown spot, Healthy). Baris atas: foto VAL asli (prov_v7) dengan kotak label tipis;
baris bawah: versi tanpa latar (prov_v9, SAM2 + aturan v2). Hanya foto VAL ds4 (RiceLeafDiseaseBD, CC BY 4.0); tidak memakai ds1, ds2, ds3.

Pemilihan: kerangka = foto VAL prov_v7 bersumber ds4 pada kelasnya, yang ada di VAL prov_v9 dengan latar_dihapus == "ya" (foto cadangan
tidak dipisah dikeluarkan, karena gambar ini memperlihatkan hasil penghapusan latar). Satu foto per kelas dipilih acak dengan
random.Random(SEED) dari daftar terurut, urutan kelas tetap. Tidak menjalankan model.
Keluaran: paper/figures/fig_leaf_examples.{pdf,png,csv}"""
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
V7, V9 = ROOT / "data/processed/prov_v7", ROOT / "data/processed/prov_v9"
OUT = ROOT / "paper/figures"
SEED = 20261026
KELAS = [("leaf_blast", "Blast"), ("brown_spot", "Brown spot"), ("background", "Healthy")]
SISI_MAKS = 450          # piksel sisi terpanjang foto yang ditanam di PDF (sel ±1,1 in -> ±400 dpi; IEEE: >= 300 dpi untuk foto)


def pilih():
    v7 = [r for r in csv.DictReader(open(V7 / "manifest.csv", encoding="utf-8")) if r["split"] == "val" and r["source"] == "ds4"]
    v9 = {r["file"]: r for r in csv.DictReader(open(V9 / "manifest.csv", encoding="utf-8")) if r["split"] == "val"}
    rng = random.Random(SEED)
    out, info = [], []
    for k, nama in KELAS:
        semua = sorted(r["file"] for r in v7 if r["class"] == k)
        di_v9 = [f for f in semua if f in v9]
        kerangka = [f for f in di_v9 if v9[f]["latar_dihapus"] == "ya"]
        f = rng.choice(kerangka)
        out.append((k, nama, f))
        info.append({"kelas": k, "foto_val_ds4_v7": len(semua), "ada_di_v9": len(di_v9),
                     "cadangan_dikeluarkan": len(di_v9) - len(kerangka), "kerangka": len(kerangka), "terpilih": f,
                     "asal": next(r["orig_path"] for r in v7 if r["file"] == f).split("Original images/")[-1],
                     "luas_masker": v9[f]["luas_masker"], "kotak_v7": next(r["n_boxes"] for r in v7 if r["file"] == f),
                     "kotak_v9": v9[f]["n_boxes"]})
    return out, info


def kecil(p):
    im = Image.open(p).convert("RGB")
    s = SISI_MAKS / max(im.size)
    return im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS) if s < 1 else im


def kotak(f, w, h):
    p = V7 / "labels/val" / (Path(f).stem + ".txt")
    out = []
    for baris in p.read_text().split("\n") if p.exists() else []:
        if baris.strip():
            c, cx, cy, bw, bh = map(float, baris.split()[:5])
            out.append(((cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h))
    return out


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"], "font.size": 8, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "image.interpolation": "none"})
    pilihan, info = pilih()
    fig, axs = plt.subplots(2, 3, figsize=(3.5, 2.4))
    for j, (k, nama, f) in enumerate(pilihan):
        a7, a9 = Image.open(V7 / "images/val" / f), Image.open(V9 / "images/val" / f)
        assert a7.size == a9.size, (f, a7.size, a9.size)
        info[j]["ukuran"] = f"{a7.width}x{a7.height}"
        im7, im9 = kecil(V7 / "images/val" / f), kecil(V9 / "images/val" / f)
        for i, im in enumerate((im7, im9)):
            ax = axs[i, j]
            ax.imshow(np.asarray(im))
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.4)
        for x, y, bw, bh in kotak(f, im7.width, im7.height):
            axs[0, j].add_patch(Rectangle((x, y), bw, bh, fill=False, edgecolor="#F0E442", linewidth=0.6))
        axs[0, j].set_title(nama, fontsize=8, pad=2)
    axs[0, 0].set_ylabel("Original", fontsize=8, labelpad=2)
    axs[1, 0].set_ylabel("Leaf-only", fontsize=8, labelpad=2)
    fig.subplots_adjust(left=0.06, right=0.995, top=0.93, bottom=0.005, wspace=0.04, hspace=0.04)
    fig.savefig(OUT / "fig_leaf_examples.pdf")
    fig.savefig(OUT / "fig_leaf_examples.png", dpi=300)
    plt.close(fig)
    with open(OUT / "fig_leaf_examples.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(info[0]))
        w.writeheader()
        w.writerows(info)
    for r in info:
        print(r)


if __name__ == "__main__":
    main()
