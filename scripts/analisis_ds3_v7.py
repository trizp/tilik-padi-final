#!/usr/bin/env python3
"""
analisis_ds3_v7.py - Analisis uji eksternal ds3 (level citra) untuk 12 model final v7 + 2 model randsplit v7.
Hanya MEMBACA hasil evaluate.py (runs/<run>/eval_ds3_<himpunan>/); tidak menjalankan model.

Per himpunan (utama = 70 foto "sesuai" kedua penilai; tambahan = 160 foto label asli):
  akurasi diagnosis, P/R/F1 BLB dan Blast (level citra), confusion matrix (label BLB/Blast x diagnosis BLB/Blast/Brown spot/
  tidak terdeteksi, gabungan 3 seed); rata-rata +- std 3 seed; IK 95% bootstrap per kelompok foto (group_id, 1.000 resample,
  resample sama untuk semua model); YOLO11 - YOLOv8 berpasangan; randsplit - bersih (ds3 segar bagi keduanya).
Tungro (di luar cakupan): sebaran diagnosis.
Keluaran: reports/final_eval_v7/ds3/ (tabel_ds3_*.csv/.tex, angka_ds3.json)
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analisis_akhir_v7 import ms, tulis_tex  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports/final_eval_v7/ds3"
B, SEED = 1000, 20261013
MODELS = ["yolov8n", "yolo11n", "yolov8s", "yolo11s"]
LABEL = {"yolov8n": "YOLOv8n", "yolo11n": "YOLO11n", "yolov8s": "YOLOv8s", "yolo11s": "YOLO11s"}
DIAG = ["bacterial_leaf_blight", "leaf_blast", "brown_spot", "sehat_atau_tidak_terdeteksi"]
NAMA = {"bacterial_leaf_blight": "BLB", "leaf_blast": "Blast", "brown_spot": "Brown spot", "sehat_atau_tidak_terdeteksi": "Tidak terdeteksi"}
HIMP = {"utama": "Uji utama (70 foto, sesuai kedua penilai)", "tambahan": "Analisis tambahan (160 foto, label asli)"}


def run_f(m, s):
    return f"v7_final_prov_{m}_s{s}"


def run_r(m):
    return f"v7_randsplit_prov_{m}_s0"


def baca(run, h):
    p = ROOT / "runs" / run / f"eval_ds3_{h}" / "diagnosis_dalam_cakupan_per_citra.csv"
    return {Path(r["berkas"]).as_posix().split("data/raw/")[-1]: r for r in csv.DictReader(open(p, encoding="utf-8"))}


def metrik(rows):
    lab = np.array([r["label"] for r in rows])
    dg = np.array([r["diagnosis"] for r in rows])
    out = {"akurasi": float((lab == dg).mean())}
    for k in ("bacterial_leaf_blight", "leaf_blast"):
        tp = int(((lab == k) & (dg == k)).sum())
        pp, ap = int((dg == k).sum()), int((lab == k).sum())
        p = tp / pp if pp else 0.0
        r = tp / ap if ap else 0.0
        out[f"P_{NAMA[k]}"], out[f"R_{NAMA[k]}"], out[f"F1_{NAMA[k]}"] = p, r, (2 * p * r / (p + r) if p + r else 0.0)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    angka = {"bootstrap": {"B": B, "seed": SEED, "unit": "group_id ds3"}}
    semua_tabel = {}
    for h in ("utama", "tambahan"):
        ref = baca(run_f("yolov8n", 0), h)
        files = sorted(ref)
        grp = {f: ref[f]["group_id"] for f in files}
        g2i = defaultdict(list)
        for i, f in enumerate(files):
            g2i[grp[f]].append(i)
        gs = sorted(g2i)
        idx = [np.concatenate([g2i[gs[j]] for j in rng.choice(len(gs), len(gs), replace=True)]) for _ in range(B)]
        angka[h] = {"foto": len(files), "kelompok": len(gs)}

        def vek(run):
            d = baca(run, h)
            rows = [d[f] for f in files]
            benar = np.array([r["label"] == r["diagnosis"] for r in rows], float)
            lab = np.array([r["label"] for r in rows])
            dg = np.array([r["diagnosis"] for r in rows])
            return rows, benar, lab, dg

        def boot_acc(benar):
            return np.array([benar[ix].mean() for ix in idx])

        def boot_rec(lab, dg, k):
            m = (lab == k)
            return np.array([((dg[ix] == k) & m[ix]).sum() / max(m[ix].sum(), 1) for ix in idx])

        V = {}
        for m in MODELS:
            for s in (0, 1, 2):
                V[run_f(m, s)] = vek(run_f(m, s))
        for m in ("yolov8n", "yolo11n"):
            V[run_r(m)] = vek(run_r(m))

        # ---- tabel utama per model
        tab, cmrows = [], []
        for m in MODELS:
            runs = [run_f(m, s) for s in (0, 1, 2)]
            mets = [metrik(V[r][0]) for r in runs]
            ba = np.mean([boot_acc(V[r][1]) for r in runs], axis=0)
            rb = np.mean([boot_rec(V[r][2], V[r][3], "bacterial_leaf_blight") for r in runs], axis=0)
            rl = np.mean([boot_rec(V[r][2], V[r][3], "leaf_blast") for r in runs], axis=0)
            amb = [json.loads((ROOT / "runs" / r / "ambang_diagnosis.json").read_text(encoding="utf-8"))["ambang"] for r in runs]
            tab.append({"Split latih": "bersih (3 seed)", "Model": LABEL[m], "Ambang": "/".join(f"{a:.2f}" for a in amb),
                        "Akurasi": ms([x["akurasi"] for x in mets]), "IK95 akurasi": f"[{np.percentile(ba, 2.5):.3f}, {np.percentile(ba, 97.5):.3f}]",
                        "P BLB": ms([x["P_BLB"] for x in mets]), "R BLB": ms([x["R_BLB"] for x in mets]),
                        "IK95 R BLB": f"[{np.percentile(rb, 2.5):.3f}, {np.percentile(rb, 97.5):.3f}]", "F1 BLB": ms([x["F1_BLB"] for x in mets]),
                        "P Blast": ms([x["P_Blast"] for x in mets]), "R Blast": ms([x["R_Blast"] for x in mets]),
                        "IK95 R Blast": f"[{np.percentile(rl, 2.5):.3f}, {np.percentile(rl, 97.5):.3f}]", "F1 Blast": ms([x["F1_Blast"] for x in mets])})
            cm = {l: {d: sum(int(((V[r][2] == l) & (V[r][3] == d)).sum()) for r in runs) for d in DIAG} for l in ("bacterial_leaf_blight", "leaf_blast")}
            for l in cm:
                cmrows.append({"Model": LABEL[m], "Label": NAMA[l], **{f"Diagnosis {NAMA[d]}": cm[l][d] for d in DIAG}, "Total (3 seed)": sum(cm[l].values())})
        for m in ("yolov8n", "yolo11n"):
            r = run_r(m)
            x = metrik(V[r][0])
            ba = boot_acc(V[r][1])
            amb = json.loads((ROOT / "runs" / r / "ambang_diagnosis.json").read_text(encoding="utf-8"))["ambang"]
            rb, rl = boot_rec(V[r][2], V[r][3], "bacterial_leaf_blight"), boot_rec(V[r][2], V[r][3], "leaf_blast")
            tab.append({"Split latih": "acak (1 seed)", "Model": LABEL[m], "Ambang": f"{amb:.2f}", "Akurasi": f"{x['akurasi']:.3f}",
                        "IK95 akurasi": f"[{np.percentile(ba, 2.5):.3f}, {np.percentile(ba, 97.5):.3f}]",
                        "P BLB": f"{x['P_BLB']:.3f}", "R BLB": f"{x['R_BLB']:.3f}", "IK95 R BLB": f"[{np.percentile(rb, 2.5):.3f}, {np.percentile(rb, 97.5):.3f}]",
                        "F1 BLB": f"{x['F1_BLB']:.3f}", "P Blast": f"{x['P_Blast']:.3f}", "R Blast": f"{x['R_Blast']:.3f}",
                        "IK95 R Blast": f"[{np.percentile(rl, 2.5):.3f}, {np.percentile(rl, 97.5):.3f}]", "F1 Blast": f"{x['F1_Blast']:.3f}"})
            cm = {l: {d: int(((V[r][2] == l) & (V[r][3] == d)).sum()) for d in DIAG} for l in ("bacterial_leaf_blight", "leaf_blast")}
            for l in cm:
                cmrows.append({"Model": LABEL[m] + " (acak)", "Label": NAMA[l], **{f"Diagnosis {NAMA[d]}": cm[l][d] for d in DIAG}, "Total (3 seed)": sum(cm[l].values())})

        # ---- perbandingan berpasangan
        banding = []
        def pasang(nama, a_runs, b_runs):   # selisih b - a pada akurasi, recall BLB, recall Blast
            for met, f in (("akurasi", lambda r: (V[r][1].mean(), boot_acc(V[r][1]))),
                           ("recall BLB", lambda r: (((V[r][2] == "bacterial_leaf_blight") & (V[r][3] == "bacterial_leaf_blight")).sum() / (V[r][2] == "bacterial_leaf_blight").sum(),
                                                     boot_rec(V[r][2], V[r][3], "bacterial_leaf_blight"))),
                           ("recall Blast", lambda r: (((V[r][2] == "leaf_blast") & (V[r][3] == "leaf_blast")).sum() / (V[r][2] == "leaf_blast").sum(),
                                                       boot_rec(V[r][2], V[r][3], "leaf_blast")))):
                A = [f(r) for r in a_runs]
                Bv = [f(r) for r in b_runs]
                d0 = float(np.mean([x[0] for x in Bv]) - np.mean([x[0] for x in A]))
                db = np.mean([x[1] for x in Bv], axis=0) - np.mean([x[1] for x in A], axis=0)
                lo, hi = np.percentile(db, [2.5, 97.5])
                banding.append({"Perbandingan": nama, "Metrik": met, "Selisih": f"{d0:+.3f}", "IK95": f"[{lo:+.3f}, {hi:+.3f}]",
                                "Memuat nol": "ya" if lo <= 0 <= hi else "tidak"})
        for v8, v11 in (("yolov8n", "yolo11n"), ("yolov8s", "yolo11s")):
            pasang(f"{LABEL[v11]} − {LABEL[v8]} (rata-rata 3 seed, berpasangan)", [run_f(v8, s) for s in (0, 1, 2)], [run_f(v11, s) for s in (0, 1, 2)])
        for m in ("yolov8n", "yolo11n"):
            pasang(f"{LABEL[m]} acak − bersih s0", [run_f(m, 0)], [run_r(m)])
            pasang(f"{LABEL[m]} acak − bersih (rata-rata 3 seed)", [run_f(m, s) for s in (0, 1, 2)], [run_r(m)])

        for nama, rows in ((f"tabel_ds3_{h}_per_model", tab), (f"tabel_ds3_{h}_confusion", cmrows), (f"tabel_ds3_{h}_perbandingan", banding)):
            with open(OUT / f"{nama}.csv", "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
        tulis_tex_ds3 = lambda nama, rows, kol, cap, lab, cat=None: tulis_tex_ke(OUT, nama, rows, kol, cap, lab, cat)
        tulis_tex_ds3(f"tabel_ds3_{h}_per_model", tab, ["Split latih", "Model", "Akurasi", "IK95 akurasi", "R BLB", "F1 BLB", "R Blast", "F1 Blast"],
                      f"Uji eksternal ds3 — {HIMP[h]}: diagnosis per citra dengan ambang terkunci tiap model (rata-rata ± simpangan baku 3 seed; IK 95% bootstrap per kelompok foto).",
                      f"tab:ds3_{h}", "ds3: daun dipetik difoto di atas latar putih (berbeda dengan data latih foto lapangan). prov_v7, BLB provisional.")
        tulis_tex_ds3(f"tabel_ds3_{h}_perbandingan", banding, list(banding[0]),
                      f"Perbandingan berpasangan pada ds3 — {HIMP[h]} (IK 95% bootstrap per kelompok foto).", f"tab:ds3_{h}_banding")
        semua_tabel[h] = {"per_model": tab, "confusion": cmrows, "perbandingan": banding}

    # ---- tungro (di luar cakupan)
    tg = []
    for m in MODELS:
        rs = [json.loads((ROOT / "runs" / run_f(m, s) / "eval_ds3_tungro" / "ringkasan_diagnosis.json").read_text(encoding="utf-8"))["luar_cakupan"] for s in (0, 1, 2)]
        n = rs[0]["n_citra"]
        tg.append({"Split latih": "bersih (3 seed)", "Model": LABEL[m], "Foto": n, "% didiagnosis penyakit": ms([r["persen_didiagnosis_salah_satu_penyakit"] for r in rs], 1),
                   **{f"% {NAMA[d]}": ms([100 * r["sebaran_diagnosis"].get(d, 0) / n for r in rs], 1) for d in DIAG}})
    for m in ("yolov8n", "yolo11n"):
        r = json.loads((ROOT / "runs" / run_r(m) / "eval_ds3_tungro" / "ringkasan_diagnosis.json").read_text(encoding="utf-8"))["luar_cakupan"]
        n = r["n_citra"]
        tg.append({"Split latih": "acak (1 seed)", "Model": LABEL[m], "Foto": n, "% didiagnosis penyakit": f"{r['persen_didiagnosis_salah_satu_penyakit']:.1f}",
                   **{f"% {NAMA[d]}": f"{100 * r['sebaran_diagnosis'].get(d, 0) / n:.1f}" for d in DIAG}})
    with open(OUT / "tabel_ds3_tungro.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(tg[0]))
        w.writeheader()
        w.writerows(tg)
    tulis_tex_ke(OUT, "tabel_ds3_tungro", tg, list(tg[0]), "ds3 Tungro (di luar cakupan): sebaran diagnosis (%).", "tab:ds3_tungro")
    semua_tabel["tungro"] = tg
    angka["tabel"] = semua_tabel
    (OUT / "angka_ds3.json").write_text(json.dumps(angka, indent=2, ensure_ascii=False), encoding="utf-8")
    for h in ("utama", "tambahan"):
        print("==", h, angka[h])
        for r in semua_tabel[h]["per_model"]:
            print(" ", r["Split latih"], r["Model"], "akurasi", r["Akurasi"], r["IK95 akurasi"], "| R BLB", r["R BLB"], "| R Blast", r["R Blast"], "| P Blast", r["P Blast"])
        for r in semua_tabel[h]["perbandingan"]:
            print("  ", r)
    for r in tg:
        print(" tungro", r)
    for r in semua_tabel["utama"]["confusion"]:
        print(" cm", r)


def tulis_tex_ke(out, nama, rows, kol, cap, lab, cat=None):
    import analisis_akhir_v7 as A
    lama = A.OUT
    A.OUT = out.parent if out.name == "tabel" else out
    try:
        (A.OUT / "tabel").mkdir(exist_ok=True)
        tulis_tex(nama, rows, kol, cap, lab, cat)
    finally:
        A.OUT = lama


if __name__ == "__main__":
    main()
