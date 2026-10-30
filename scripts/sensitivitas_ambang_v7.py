#!/usr/bin/env python3
"""
sensitivitas_ambang_v7.py - ANALISIS POST-HOC (setelah test v7 dibuka): akurasi diagnosis per kelas terhadap ambang 0,05-0,60.

TIDAK dipakai untuk memilih model, mengganti ambang terkunci, atau mengganti angka utama. Tidak menjalankan model:
diagnosis pada ambang t dihitung ulang dari berkas per citra evaluate.py (kelas dan confidence kotak teratas tiap citra,
diprediksi dengan conf 0,001): diagnosis = kelas teratas bila conf_teratas >= t, selain itu "sehat/tidak terdeteksi".
Akurasi per kelas = proporsi citra kelas itu yang didiagnosis benar (recall per kelas, level citra).
Himpunan: test v7 (semua citra) dan val v7 (semua citra, termasuk BLB provisional yang tidak dipakai memilih ambang).
Keluaran: reports/final_eval_v7/posthoc_sensitivitas_ambang/{kurva.csv, gambar_sensitivitas_ambang.png, CATATAN.txt}
"""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports/final_eval_v7/posthoc_sensitivitas_ambang"
MODELS = ["yolov8n", "yolo11n", "yolov8s", "yolo11s"]
LABEL = {"yolov8n": "YOLOv8n", "yolo11n": "YOLO11n", "yolov8s": "YOLOv8s", "yolo11s": "YOLO11s"}
KELAS = [("bacterial_leaf_blight", "BLB (prov.)"), ("leaf_blast", "Blast"), ("brown_spot", "Brown spot"),
         ("sehat_atau_tidak_terdeteksi", "Sehat/tdk terdeteksi")]
GRID = np.round(np.arange(0.05, 0.6001, 0.01), 2)
WARNA = {"yolov8n": "#2a78d6", "yolov8s": "#2a78d6", "yolo11n": "#eb6834", "yolo11s": "#eb6834"}
GARIS = {"yolov8n": "--", "yolo11n": "--", "yolov8s": "-", "yolo11s": "-"}


def kurva(rows):
    lab = np.array([r["label"] for r in rows])
    cls = np.array([r["kelas_teratas"] or "" for r in rows])
    cf = np.array([float(r["conf_teratas"]) for r in rows])
    out = {}
    for t in GRID:
        d = np.where((cf >= t) & (cls != ""), cls, "sehat_atau_tidak_terdeteksi")
        out[t] = {k: (float((d[lab == k] == k).mean()) if (lab == k).any() else None) for k, _ in KELAS}
        out[t]["SEMUA"] = float((d == lab).mean())
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows_csv, data = [], {}
    for st, folder in (("test", "eval_test"), ("val", "eval_val")):
        for m in MODELS:
            for s in (0, 1, 2):
                run = f"v7_final_prov_{m}_s{s}"
                per = list(csv.DictReader(open(ROOT / "runs" / run / folder / "diagnosis_dengan_background_per_citra.csv", encoding="utf-8")))
                amb = json.loads((ROOT / "runs" / run / "ambang_diagnosis.json").read_text(encoding="utf-8"))["ambang"]
                k = kurva(per)
                data[(st, m, s)] = (k, amb)
                for t in GRID:
                    rows_csv.append({"ANALISIS": "POST-HOC (tidak dipakai memilih model/ambang)", "set": st, "model": m, "seed": s,
                                     "ambang_terkunci": amb, "ambang": t, **{f"akurasi_{n}": (round(k[t][kk], 4) if k[t][kk] is not None else "")
                                                                            for kk, n in KELAS + [("SEMUA", "SEMUA")]}})
    with open(OUT / "kurva.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0]))
        w.writeheader()
        w.writerows(rows_csv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#8a8984", "xtick.color": "#52514e", "ytick.color": "#52514e"})
    kol = KELAS + [("SEMUA", "Semua citra")]
    fig, axes = plt.subplots(2, len(kol), figsize=(10.5, 4.6), sharex=True, sharey=True)
    for i, st in enumerate(("test", "val")):
        for j, (kk, nama) in enumerate(kol):
            ax = axes[i, j]
            for m in MODELS:
                ys = np.array([[data[(st, m, s)][0][t][kk] for t in GRID] for s in (0, 1, 2)], float)
                ax.plot(GRID, ys.mean(0), GARIS[m], color=WARNA[m], lw=1.6, label=LABEL[m])
                amb = np.mean([data[(st, m, s)][1] for s in (0, 1, 2)])
                ax.axvline(amb, color=WARNA[m], ls=":", lw=1)
            ax.set_ylim(0, 1)
            ax.grid(color="#e6e5e0", lw=0.6)
            ax.set_axisbelow(True)
            if i == 0:
                ax.set_title(nama, fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{'TEST v7' if st == 'test' else 'VAL v7'}\nakurasi per kelas")
            if i == 1:
                ax.set_xlabel("ambang confidence")
    axes[0, 0].legend(loc="lower left", fontsize=6.5, frameon=False)
    fig.suptitle("ANALISIS POST-HOC (setelah test dibuka): sensitivitas ambang diagnosis. Tidak dipakai untuk memilih model atau mengganti angka utama.",
                 fontsize=8)
    fig.text(0.01, 0.005, "Garis = rata-rata 3 seed; garis titik-titik vertikal = rata-rata ambang terkunci tiap model (dipilih di val bersih tanpa BLB).",
             fontsize=6.5, color="#52514e")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(OUT / "gambar_sensitivitas_ambang.png", dpi=300)

    # ringkasan angka: pada ambang terkunci vs ambang 0,10 (contoh) untuk tiap kelas, rata-rata 3 seed
    ring = {}
    for st in ("test", "val"):
        for m in MODELS:
            r = {}
            for kk, nama in kol:
                pada_kunci = np.mean([data[(st, m, s)][0][min(GRID, key=lambda g: abs(g - data[(st, m, s)][1]))][kk] for s in (0, 1, 2)])
                r[nama] = {"pada_ambang_terkunci": round(float(pada_kunci), 3),
                           "pada_0.10": round(float(np.mean([data[(st, m, s)][0][0.1][kk] for s in (0, 1, 2)])), 3),
                           "maksimum_kurva": round(float(max(np.mean([data[(st, m, s)][0][t][kk] for s in (0, 1, 2)]) for t in GRID)), 3)}
            ring[f"{st}|{m}"] = r
    (OUT / "ringkasan.json").write_text(json.dumps({"ANALISIS": "POST-HOC; tidak dipakai memilih model/ambang/angka utama", "hasil": ring},
                                                   indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "CATATAN.txt").write_text(
        "ANALISIS POST-HOC (dibuat setelah test v7 dibuka).\n"
        "Tujuan: memahami sensitivitas akurasi diagnosis per kelas terhadap ambang confidence.\n"
        "TIDAK dipakai untuk memilih model, mengganti ambang terkunci (runs/<run>/ambang_diagnosis.json), atau mengganti angka utama\n"
        "di tabel/ dan RINGKASAN_HASIL.md. Diagnosis dihitung ulang dari berkas per citra evaluate.py; model tidak dijalankan ulang.\n",
        encoding="utf-8")
    print(json.dumps({k: {n: v["pada_ambang_terkunci"] for n, v in r.items()} for k, r in ring.items() if k.startswith("test")}, indent=1, ensure_ascii=False))
    print(json.dumps({k: {n: v["pada_0.10"] for n, v in r.items()} for k, r in ring.items() if k.startswith("test")}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
