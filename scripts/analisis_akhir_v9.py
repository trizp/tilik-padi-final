#!/usr/bin/env python3
"""
analisis_akhir_v9.py - Analisis evaluasi v9 (latar dihapus) sesuai rencana yang dicatat SEBELUM training (KEPUTUSAN_PROYEK.md 2026-10-16).
Hanya MEMBACA hasil evaluate.py (tidak menjalankan model).

Kondisi (model x data uji), varian s, 3 seed:
  v7_asli  : model v7 pada foto asli (prov_v7)            = "dengan latar"   (runs/v7_final_prov_*/eval_<split>)
  v9_tanpa : model v9 pada foto tanpa latar (prov_v9)     = "tanpa latar"    (runs/v9_final_prov_*/eval_<split>)
  v7_tanpa : model v7 pada foto tanpa latar (silang)                         (runs/v7_final_prov_*/eval_v9tl_<split>)
  v9_asli  : model v9 pada foto asli (silang)                                (runs/v9_final_prov_*/eval_v7asli_<split>)
Semua dihitung pada IRISAN foto (foto test/test sesi baru yang ada di prov_v7 DAN prov_v9). Label mengikuti data uji (asli = label v7;
tanpa latar = label v9 setelah buang-kotak). Metrik deteksi dari statistik per gambar (ap_per_class, sama dengan evaluate.py); diagnosis dari
CSV per citra (ambang terkunci masing-masing model). Bootstrap 1.000 resample per KELOMPOK foto (group manifest prov_v7), resample sama
untuk semua model dan kondisi pada set yang sama. BLB (ds1_provisional) dilaporkan terpisah.
ds3: model v9 pada ds3 tanpa latar vs model v7 pada ds3 asli (hasil v7 yang sudah ada); pembanding "sudah terbuka".
Keluaran: reports/final_eval_v9/ (tabel/*.csv, tabel/*.tex, grafik/*.png, angka_kunci_v9.json)
"""
import csv
import json
import pickle
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "final_eval_v9"
B, SEED_BOOT = 1000, 20261017
MODELS = ["yolov8s", "yolo11s"]
LABEL = {"yolov8s": "YOLOv8s", "yolo11s": "YOLO11s"}
SEEDS = (0, 1, 2)
KELAS = ["leaf_blast", "brown_spot", "bacterial_leaf_blight"]
DIAG = KELAS + ["sehat_atau_tidak_terdeteksi"]
NK = {"leaf_blast": "Blast", "brown_spot": "Brown spot", "bacterial_leaf_blight": "BLB (prov.)", "sehat_atau_tidak_terdeteksi": "Sehat/tdk terdeteksi"}
KOND = {"v7_asli": ("v7", "eval_{sp}", "Dengan latar (v7, foto asli)"),
        "v9_tanpa": ("v9", "eval_{sp}", "Tanpa latar (v9, foto tanpa latar)"),
        "v7_tanpa": ("v7", "eval_v9tl_{sp}", "Silang: v7 pada foto tanpa latar"),
        "v9_asli": ("v9", "eval_v7asli_{sp}", "Silang: v9 pada foto asli")}
UTAMA = ["v7_asli", "v9_tanpa"]
SETS = [("test", "bersih", "Test bersih (tanpa BLB)"), ("test_new_session", "dengan_background", "Test sesi baru"),
        ("test", "blb_provisional", "Test BLB provisional (terpisah)")]
KUNCI = ["mAP50", "mAP50_95", "akurasi", "recall_blast"]


def run(versi, m, s):
    return f"{versi}_final_prov_{m}_s{s}"


def eval_dir(kond, m, s, sp):
    versi, pola, _ = KOND[kond]
    return ROOT / "runs" / run(versi, m, s) / pola.format(sp=sp)


def muat_stats(kond, m, s, sp, sub):
    d = pickle.load(open(eval_dir(kond, m, s, sp) / f"stats_per_citra_{sub}.pkl", "rb"))
    return d["per_citra"], d["names"]


def metrik(per, idx, names):
    from ultralytics.utils.metrics import ap_per_class
    tp = np.concatenate([per[i]["tp"] for i in idx])
    cf = np.concatenate([per[i]["conf"] for i in idx])
    pc = np.concatenate([per[i]["pred_cls"] for i in idx])
    tc = np.concatenate([per[i]["target_cls"] for i in idx])
    if len(tc) == 0:
        return None
    _, _, p, r, f1, ap, uc, *_ = ap_per_class(tp, cf, pc, tc)
    P, R = float(p.mean()), float(r.mean())
    out = {"mAP50": float(ap[:, 0].mean()), "mAP50_95": float(ap.mean()), "P": P, "R": R, "F1": 2 * P * R / (P + R) if P + R else 0.0}
    for k, c in enumerate(uc):
        n = names[int(c)]
        out[f"{n}|AP50"], out[f"{n}|AP50_95"] = float(ap[k, 0]), float(ap[k].mean())
        out[f"{n}|P"], out[f"{n}|R"] = float(p[k]), float(r[k])
        out[f"{n}|F1"] = float(2 * p[k] * r[k] / (p[k] + r[k])) if p[k] + r[k] else 0.0
    return out


def tugas(arg):
    kond, m, s, sp, sub, files, idx_list = arg
    per, names = muat_stats(kond, m, s, sp, sub)
    pos = {p["im_name"]: i for i, p in enumerate(per)}
    hilang = [f for f in files if f not in pos]
    if hilang:
        raise RuntimeError(f"{kond} {m} s{s} {sp}/{sub}: {len(hilang)} foto irisan tidak ada, mis. {hilang[:2]}")
    peta = np.array([pos[f] for f in files])
    titik = metrik(per, peta, names)
    penuh = metrik(per, list(range(len(per))), names)
    boot = [metrik(per, peta[ix], names) for ix in idx_list]
    return (kond, m, s, sp, sub), titik, penuh, boot


def diag_rows(kond, m, s, sp, sub):
    p = eval_dir(kond, m, s, sp) / f"diagnosis_{sub}_per_citra.csv"
    return {r["berkas"]: r for r in csv.DictReader(open(p, encoding="utf-8"))}


def diag_metrik(rows):
    lab = np.array([r["label"] for r in rows])
    dg = np.array([r["diagnosis"] for r in rows])
    out = {"akurasi": float((lab == dg).mean()) if len(lab) else float("nan")}
    for k in DIAG:
        tp, pp, ap = int(((lab == k) & (dg == k)).sum()), int((dg == k).sum()), int((lab == k).sum())
        p, r = (tp / pp if pp else 0.0), (tp / ap if ap else float("nan"))
        out[f"{k}|dP"], out[f"{k}|dR"] = p, r
        out[f"{k}|dF1"] = (2 * p * r / (p + r) if (p + r) and not np.isnan(r) else (0.0 if ap else float("nan")))
        out[f"{k}|n"] = ap
    out["recall_blast"] = out["leaf_blast|dR"]
    return out


def ms(v, d=3):
    v = [x for x in v if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if not v:
        return "-"
    return f"{np.mean(v):.{d}f} ± {np.std(v, ddof=1) if len(v) > 1 else 0:.{d}f}"


def ik(a):
    a = np.asarray(a, float)
    a = a[~np.isnan(a)]
    return f"[{np.percentile(a, 2.5):.3f}, {np.percentile(a, 97.5):.3f}]" if len(a) else "-"


def nol(a):
    a = np.asarray(a, float)
    a = a[~np.isnan(a)]
    if not len(a):
        return "-"
    return "ya" if len(a) and np.percentile(a, 2.5) <= 0 <= np.percentile(a, 97.5) else "tidak"


def tulis_csv(nama, rows):
    (OUT / "tabel").mkdir(parents=True, exist_ok=True)
    with open(OUT / "tabel" / f"{nama}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def tulis_tex(nama, rows, kolom, caption, label, catatan=None):
    esc = lambda s: str(s).replace("_", "\\_").replace("±", "$\\pm$").replace("%", "\\%").replace("&", "\\&").replace("−", "$-$")  # noqa: E731
    t = ["\\begin{table}[!t]", "\\centering", f"\\caption{{{esc(caption)}}}", f"\\label{{{label}}}", "\\footnotesize",
         "\\setlength{\\tabcolsep}{3pt}", "\\begin{tabular}{" + "l" * 2 + "c" * (len(kolom) - 2) + "}", "\\hline",
         " & ".join(f"\\textbf{{{esc(k)}}}" for k in kolom) + " \\\\", "\\hline"]
    for r in rows:
        t.append(" & ".join(esc(r[k]) for k in kolom) + " \\\\")
    t += ["\\hline", "\\end{tabular}"]
    if catatan:
        t.append(f"\\\\[2pt]\\parbox{{\\columnwidth}}{{\\scriptsize {esc(catatan)}}}")
    t.append("\\end{table}")
    (OUT / "tabel" / f"{nama}.tex").write_text("\n".join(t) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "grafik").mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED_BOOT)
    grup = {r["file"]: r["group"] for r in csv.DictReader(open(ROOT / "data/processed/prov_v7/manifest.csv", encoding="utf-8"))}
    v9man = list(csv.DictReader(open(ROOT / "data/processed/prov_v9/manifest.csv", encoding="utf-8")))
    angka = {"bootstrap": {"B": B, "seed": SEED_BOOT, "unit": "kelompok foto (group manifest prov_v7)"}, "irisan": {}}

    # ---- irisan foto per set: foto pada subset itu di prov_v9 (semuanya juga ada di prov_v7)
    rs, tasks = {}, []
    for sp, sub, nama in SETS:
        per, _ = muat_stats("v9_tanpa", "yolov8s", 0, sp, sub)
        files = sorted(p["im_name"] for p in per)
        g2i = defaultdict(list)
        for i, f in enumerate(files):
            g2i[grup[f]].append(i)
        gs = sorted(g2i)
        idx = [np.concatenate([g2i[gs[j]] for j in rng.choice(len(gs), len(gs), replace=True)]) for _ in range(B)]
        rs[(sp, sub)] = (files, idx)
        per7, _ = muat_stats("v7_asli", "yolov8s", 0, sp, sub)
        angka["irisan"][nama] = {"foto_irisan": len(files), "kelompok": len(gs), "foto_v7_asli": len(per7),
                                 "foto_v7_tidak_di_v9": len(per7) - len(files)}
        for kond in KOND:
            for m in MODELS:
                for s in SEEDS:
                    tasks.append((kond, m, s, sp, sub, files, idx))
    print(f"bootstrap: {len(tasks)} tugas x {B}", flush=True)
    H = {}
    with ProcessPoolExecutor(max_workers=8) as ex:
        for key, titik, penuh, boot in ex.map(tugas, tasks):
            H[key] = (titik, penuh, boot)

    # ---- verifikasi: metrik penuh dari statistik = angka resmi evaluate.py
    sel_maks = 0.0
    for (kond, m, s, sp, sub), (_, penuh, _) in H.items():
        ref = {r["kelas"]: r for r in csv.DictReader(open(eval_dir(kond, m, s, sp) / "metrik_per_kelas.csv", encoding="utf-8")) if r["subset"] == sub}
        for a, b in (("mAP50", "mAP50"), ("mAP50_95", "mAP50_95"), ("precision", "P"), ("recall", "R")):
            sel_maks = max(sel_maks, abs(float(ref["SEMUA"][a]) - penuh[b]))
    angka["verifikasi_selisih_maks_vs_resmi"] = sel_maks
    print("selisih maks vs resmi:", sel_maks, flush=True)
    assert sel_maks < 6e-4

    # ---- diagnosis per citra (irisan)
    D = {}
    for sp, sub, _ in SETS:
        files, idx = rs[(sp, sub)]
        for kond in KOND:
            for m in MODELS:
                for s in SEEDS:
                    rows = diag_rows(kond, m, s, sp, sub)
                    ada = [rows[f] for f in files if f in rows]          # foto multi-kelas tidak ada di CSV
                    benar = np.array([(rows[f]["label"] == rows[f]["diagnosis"]) if f in rows else np.nan for f in files], float)
                    lab = np.array([rows[f]["label"] if f in rows else "" for f in files])
                    dg = np.array([rows[f]["diagnosis"] if f in rows else "" for f in files])
                    mb = lab == "leaf_blast"
                    bacc = np.array([np.nanmean(benar[ix]) for ix in idx])
                    brec = np.array([((dg[ix] == "leaf_blast") & mb[ix]).sum() / mb[ix].sum() if mb[ix].sum() else np.nan for ix in idx])
                    D[(kond, m, s, sp, sub)] = (diag_metrik(ada), bacc, brec, ada)

    def nilai(kond, m, sp, sub, k):
        """titik per seed dan bootstrap rata-rata seed untuk kunci k."""
        if k in ("akurasi", "recall_blast"):
            pts = [D[(kond, m, s, sp, sub)][0][k] for s in SEEDS]
            arr = np.array([D[(kond, m, s, sp, sub)][1 if k == "akurasi" else 2] for s in SEEDS])
        else:
            pts = [H[(kond, m, s, sp, sub)][0][k] if H[(kond, m, s, sp, sub)][0] else np.nan for s in SEEDS]
            arr = np.array([[b[k] if b and k in b else np.nan for b in H[(kond, m, s, sp, sub)][2]] for s in SEEDS])
        return pts, np.nanmean(arr, axis=0)

    # ================= Tabel 1: 2x2 utama =================
    t1 = []
    for sp, sub, nama in SETS:
        for m in MODELS:
            for kond in (UTAMA if sub != "blb_provisional" else UTAMA):
                row = {"Set": nama, "Model": LABEL[m], "Kondisi": KOND[kond][2]}
                for k, lab in (("mAP50", "mAP50"), ("mAP50_95", "mAP50-95")):
                    pts, bt = nilai(kond, m, sp, sub, k)
                    row[lab], row[f"IK95 {lab}"] = ms(pts), ik(bt)
                pts_h = [H[(kond, m, s, sp, sub)][0] for s in SEEDS]
                for k in ("P", "R", "F1"):
                    row[k] = ms([p[k] for p in pts_h])
                for k, lab in (("akurasi", "Akurasi diagnosis"), ("recall_blast", "Recall diagnosis Blast")):
                    pts, bt = nilai(kond, m, sp, sub, k)
                    row[lab], row[f"IK95 {lab}"] = ms(pts), ik(bt)
                t1.append(row)
    BLBSET = SETS[2][2]
    tulis_csv("tabel1_2x2_utama", [r for r in t1 if r["Set"] != BLBSET])
    tulis_csv("tabel6_blb_terpisah", [r for r in t1 if r["Set"] == BLBSET])
    for sp, sub, nama in SETS[:2]:
        sel = [r for r in t1 if r["Set"] == nama]
        tulis_tex(f"tabel1_2x2_{sp}_{sub}", sel, ["Model", "Kondisi", "mAP50", "mAP50-95", "IK95 mAP50-95", "Akurasi diagnosis",
                                                  "IK95 Akurasi diagnosis", "Recall diagnosis Blast"],
                  f"Dengan vs tanpa latar pada {nama} (irisan foto prov\\_v7 dan prov\\_v9; rata-rata ± simpangan baku 3 seed; IK 95% bootstrap per kelompok foto).",
                  f"tab:v9_2x2_{sp}", "Dengan latar = model v7 pada foto asli; tanpa latar = model v9 pada foto tanpa latar (SAM2 + aturan v2). PROVISIONAL.")
    tulis_tex("tabel6_blb_terpisah", [r for r in t1 if r["Set"] == BLBSET], ["Model", "Kondisi", "mAP50-95", "IK95 mAP50-95", "R", "Akurasi diagnosis"],
              "BLB provisional (ds1), dilaporkan terpisah dan tidak dipakai untuk kesimpulan tentang latar.", "tab:v9_blb")

    # ================= Tabel 2: selisih YOLO11s - YOLOv8s berpasangan per sel =================
    t2 = []
    for sp, sub, nama in SETS:
        for kond in KOND:
            row = {"Set": nama, "Kondisi": KOND[kond][2]}
            for k, lab in (("mAP50", "mAP50"), ("mAP50_95", "mAP50-95"), ("akurasi", "akurasi"), ("recall_blast", "recall Blast")):
                p11, b11 = nilai(kond, "yolo11s", sp, sub, k)
                p8, b8 = nilai(kond, "yolov8s", sp, sub, k)
                d = b11 - b8
                dt = np.nanmean(p11) - np.nanmean(p8)
                row[f"Selisih {lab}"] = "-" if np.isnan(dt) else f"{dt:+.3f} {ik(d)}"
                row[f"Memuat nol ({lab})"] = nol(d)
            t2.append(row)
    tulis_csv("tabel2_selisih_yolo11s_yolov8s", t2)
    tulis_tex("tabel2_selisih_yolo11s_yolov8s", [r for r in t2 if r["Set"] != BLBSET],
              ["Set", "Kondisi", "Selisih mAP50-95", "Selisih akurasi", "Selisih recall Blast"],
              "Selisih YOLO11s − YOLOv8s berpasangan (rata-rata 3 seed; IK 95% bootstrap per kelompok foto, resample sama).", "tab:v9_selisih")

    # ================= Tabel 3: per kelas (deteksi + diagnosis) =================
    t3 = []
    for sp, sub, nama in SETS[:2]:
        for m in MODELS:
            for kond in UTAMA:
                pts_h = [H[(kond, m, s, sp, sub)][0] for s in SEEDS]
                pts_d = [D[(kond, m, s, sp, sub)][0] for s in SEEDS]
                for k in DIAG:
                    if k == "bacterial_leaf_blight":
                        continue
                    row = {"Set": nama, "Model": LABEL[m], "Kondisi": KOND[kond][2], "Kelas": NK[k]}
                    if f"{k}|AP50" in pts_h[0]:
                        row.update({"AP50": ms([p[f"{k}|AP50"] for p in pts_h]), "AP50-95": ms([p[f"{k}|AP50_95"] for p in pts_h]),
                                    "P kotak": ms([p[f"{k}|P"] for p in pts_h]), "R kotak": ms([p[f"{k}|R"] for p in pts_h]),
                                    "F1 kotak": ms([p[f"{k}|F1"] for p in pts_h])})
                    else:
                        row.update({"AP50": "-", "AP50-95": "-", "P kotak": "-", "R kotak": "-", "F1 kotak": "-"})
                    row.update({"P diagnosis": ms([p[f"{k}|dP"] for p in pts_d]), "R diagnosis": ms([p[f"{k}|dR"] for p in pts_d]),
                                "F1 diagnosis": ms([p[f"{k}|dF1"] for p in pts_d])})
                    t3.append(row)
    tulis_csv("tabel3_per_kelas", t3)
    tulis_tex("tabel3_per_kelas_test_bersih", [r for r in t3 if r["Set"] == SETS[0][2]],
              ["Model", "Kondisi", "Kelas", "AP50-95", "P kotak", "R kotak", "F1 kotak", "P diagnosis", "R diagnosis", "F1 diagnosis"],
              "Per kelas pada test bersih (irisan; rata-rata ± simpangan baku 3 seed).", "tab:v9_per_kelas")

    # ================= Tabel 4: sel silang (4 kondisi) =================
    t4 = []
    for sp, sub, nama in SETS:
        for m in MODELS:
            row = {"Set": nama, "Model": LABEL[m]}
            for kond in ("v7_asli", "v7_tanpa", "v9_asli", "v9_tanpa"):
                for k, lab in (("mAP50_95", "mAP50-95"), ("akurasi", "akurasi"), ("recall_blast", "recall Blast")):
                    pts, _ = nilai(kond, m, sp, sub, k)
                    row[f"{kond} {lab}"] = ms(pts)
            t4.append(row)
    tulis_csv("tabel4_sel_silang", t4)
    tulis_tex("tabel4_sel_silang_test_bersih", [{"Model": r["Model"], **{f"{a} {b}": r[f"{a} {b}"] for a in ("v7_asli", "v7_tanpa", "v9_asli", "v9_tanpa")
                                                                          for b in ("mAP50-95", "akurasi")}} for r in t4 if r["Set"] == SETS[0][2]],
              ["Model"] + [f"{a} {b}" for a in ("v7_asli", "v7_tanpa", "v9_asli", "v9_tanpa") for b in ("mAP50-95", "akurasi")],
              "Sel silang pada test bersih (analisis tambahan): model v7/v9 pada foto asli/tanpa latar.", "tab:v9_silang")

    # ================= Tabel 5: kontras berpasangan =================
    t5 = []
    kontras = [("v9_tanpa", "v7_asli", "v9 tanpa latar − v7 dengan latar"), ("v7_tanpa", "v7_asli", "Kepekaan v7: tanpa − asli"),
               ("v9_asli", "v9_tanpa", "Kepekaan v9: asli − tanpa")]
    for sp, sub, nama in SETS:
        for a, b, lab in kontras:
            row = {"Set": nama, "Kontras": lab}
            for k, kl in (("mAP50_95", "mAP50-95"), ("akurasi", "akurasi"), ("recall_blast", "recall Blast")):
                dd = {}
                for m in MODELS:
                    pa, ba = nilai(a, m, sp, sub, k)
                    pb, bb = nilai(b, m, sp, sub, k)
                    dd[m] = (np.nanmean(pa) - np.nanmean(pb), ba - bb)
                    row[f"{LABEL[m]} {kl}"] = f"{dd[m][0]:+.3f} {ik(dd[m][1])}"
                dod = dd["yolo11s"][1] - dd["yolov8s"][1]
                row[f"YOLO11s−YOLOv8s {kl}"] = f"{dd['yolo11s'][0] - dd['yolov8s'][0]:+.3f} {ik(dod)}"
                row[f"Memuat nol YOLO11s−YOLOv8s {kl}"] = nol(dod)
            t5.append(row)
    tulis_csv("tabel5_kontras_latar", t5)
    tulis_tex("tabel5_kontras_latar_test_bersih", [r for r in t5 if r["Set"] == SETS[0][2]],
              ["Kontras", "YOLOv8s mAP50-95", "YOLO11s mAP50-95", "YOLO11s−YOLOv8s mAP50-95", "YOLOv8s akurasi", "YOLO11s akurasi",
               "YOLO11s−YOLOv8s akurasi"], "Kontras berpasangan pada test bersih (irisan; IK 95% bootstrap per kelompok foto).", "tab:v9_kontras")

    # ================= confusion matrix diagnosis =================
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for sp, sub, nama in SETS[:2]:
        fig, axs = plt.subplots(2, 2, figsize=(9, 8.2))
        for i, m in enumerate(MODELS):
            for j, kond in enumerate(UTAMA):
                cm = np.zeros((4, 4), int)
                for s in SEEDS:
                    for r in D[(kond, m, s, sp, sub)][3]:
                        cm[DIAG.index(r["label"]), DIAG.index(r["diagnosis"])] += 1
                keep = [q for q in range(4) if cm[q].sum() > 0]
                c = cm[keep]
                pr = c / c.sum(1, keepdims=True)
                ax = axs[i, j]
                ax.imshow(pr, cmap="Greens", vmin=0, vmax=1)
                for a_ in range(c.shape[0]):
                    for b_ in range(4):
                        ax.text(b_, a_, f"{c[a_, b_]}\n{100 * pr[a_, b_]:.0f}%", ha="center", va="center", fontsize=8,
                                color="white" if pr[a_, b_] > 0.6 else "black")
                ax.set_xticks(range(4), [NK[d].replace(" ", "\n", 1) for d in DIAG], fontsize=7)
                ax.set_yticks(range(len(keep)), [NK[DIAG[q]] for q in keep], fontsize=7)
                ax.set_title(f"{LABEL[m]} - {KOND[kond][2]}", fontsize=8)
                ax.set_xlabel("Diagnosis", fontsize=7)
        fig.suptitle(f"Confusion matrix diagnosis, {nama} (irisan, jumlah 3 seed)", fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT / "grafik" / f"confusion_diagnosis_{sp}_{sub}.png", dpi=150)
        plt.close(fig)

    # ================= grafik kondisi =================
    for sp, sub, nama in SETS[:2]:
        fig, axs = plt.subplots(1, 3, figsize=(12, 3.8))
        urut = ["v7_asli", "v7_tanpa", "v9_asli", "v9_tanpa"]
        lbl = ["v7\nasli", "v7\ntanpa latar", "v9\nasli", "v9\ntanpa latar"]
        for ax, (k, kl) in zip(axs, (("mAP50_95", "mAP50-95"), ("akurasi", "Akurasi diagnosis"), ("recall_blast", "Recall diagnosis Blast"))):
            for i, m in enumerate(MODELS):
                y, lo, hi = [], [], []
                for kond in urut:
                    pts, bt = nilai(kond, m, sp, sub, k)
                    y.append(np.nanmean(pts))
                    lo.append(np.nanmean(pts) - np.nanpercentile(bt, 2.5))
                    hi.append(np.nanpercentile(bt, 97.5) - np.nanmean(pts))
                x = np.arange(4) + (i - 0.5) * 0.36
                ax.bar(x, y, 0.34, yerr=[lo, hi], capsize=3, color="#2a78d6" if m == "yolov8s" else "#eb6834", label=LABEL[m])
            ax.set_xticks(range(4), lbl, fontsize=8)
            ax.set_title(kl, fontsize=9)
            ax.grid(axis="y", alpha=0.3)
        axs[0].legend(fontsize=8)
        fig.suptitle(f"{nama}: model x data uji (irisan; rata-rata 3 seed, IK 95%)", fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT / "grafik" / f"grafik_kondisi_{sp}_{sub}.png", dpi=150)
        plt.close(fig)

    # ================= ds3 =================
    cad = json.loads((OUT / "ds3_pemisahan/ringkasan.json").read_text(encoding="utf-8"))["per_himpunan"]
    t7, ds3_angka = [], {}
    rng3 = np.random.default_rng(SEED_BOOT + 3)

    def baca_ds3(r, nama_eval):
        p = ROOT / "runs" / r / nama_eval / "diagnosis_dalam_cakupan_per_citra.csv"
        return {"/".join(Path(x["berkas"]).parts[-2:]): x for x in csv.DictReader(open(p, encoding="utf-8"))}
    for h, hn in (("utama", "ds3 uji utama (70)"), ("tambahan", "ds3 tambahan (160)")):
        ref = baca_ds3(run("v7", "yolov8s", 0), f"eval_ds3_{h}")
        files = sorted(ref)
        g2i = defaultdict(list)
        for i, f in enumerate(files):
            g2i[ref[f]["group_id"]].append(i)
        gs = sorted(g2i)
        idx = [np.concatenate([g2i[gs[j]] for j in rng3.choice(len(gs), len(gs), replace=True)]) for _ in range(B)]
        V = {}
        for m in MODELS:
            for s in SEEDS:
                for versi, ne in (("v7", f"eval_ds3_{h}"), ("v9", f"eval_ds3tl_{h}")):
                    d = baca_ds3(run(versi, m, s), ne)
                    rows = [d[f] for f in files]
                    lab = np.array([x["label"] for x in rows])
                    dg = np.array([x["diagnosis"] for x in rows])
                    V[(versi, m, s)] = (lab, dg)

        def stat(lab, dg, ix=None):
            ix = np.arange(len(lab)) if ix is None else ix
            L, G = lab[ix], dg[ix]
            out = {"akurasi": float((L == G).mean())}
            for k, nk in (("bacterial_leaf_blight", "BLB"), ("leaf_blast", "Blast")):
                tp, pp, ap = ((L == k) & (G == k)).sum(), (G == k).sum(), (L == k).sum()
                p, r = (tp / pp if pp else 0.0), (tp / ap if ap else np.nan)
                out[f"P {nk}"], out[f"R {nk}"], out[f"F1 {nk}"] = p, r, (2 * p * r / (p + r) if p + r else 0.0)
            return out
        for m in MODELS:
            for versi, kl in (("v7", "v7 pada ds3 asli"), ("v9", "v9 pada ds3 tanpa latar")):
                pts = [stat(*V[(versi, m, s)]) for s in SEEDS]
                bt = {k: np.mean([[stat(*V[(versi, m, s)], ix)[k] for ix in idx] for s in SEEDS], axis=0) for k in ("akurasi", "R Blast", "R BLB")}
                row = {"Set": hn, "Model": LABEL[m], "Kondisi": kl,
                       "Cadangan (tidak dipisah)": f"{cad[h]['persen_cadangan']}%" if versi == "v9" else "-",
                       "Akurasi": ms([p["akurasi"] for p in pts]), "IK95 akurasi": ik(bt["akurasi"]),
                       "R BLB": ms([p["R BLB"] for p in pts]), "R Blast": ms([p["R Blast"] for p in pts]), "IK95 R Blast": ik(bt["R Blast"]),
                       "F1 BLB": ms([p["F1 BLB"] for p in pts]), "F1 Blast": ms([p["F1 Blast"] for p in pts])}
                t7.append(row)
                ds3_angka[f"{h}|{m}|{versi}"] = {"akurasi": float(np.mean([p["akurasi"] for p in pts])), "R_Blast": float(np.mean([p["R Blast"] for p in pts])),
                                                 "R_BLB": float(np.mean([p["R BLB"] for p in pts])), "bt_akurasi": bt["akurasi"], "bt_RBlast": bt["R Blast"]}
            d_acc = ds3_angka[f"{h}|{m}|v9"]["bt_akurasi"] - ds3_angka[f"{h}|{m}|v7"]["bt_akurasi"]
            d_bl = ds3_angka[f"{h}|{m}|v9"]["bt_RBlast"] - ds3_angka[f"{h}|{m}|v7"]["bt_RBlast"]
            t7.append({"Set": hn, "Model": LABEL[m], "Kondisi": "Selisih v9 − v7", "Cadangan (tidak dipisah)": "-",
                       "Akurasi": f"{ds3_angka[f'{h}|{m}|v9']['akurasi'] - ds3_angka[f'{h}|{m}|v7']['akurasi']:+.3f}", "IK95 akurasi": ik(d_acc),
                       "R BLB": f"{ds3_angka[f'{h}|{m}|v9']['R_BLB'] - ds3_angka[f'{h}|{m}|v7']['R_BLB']:+.3f}",
                       "R Blast": f"{ds3_angka[f'{h}|{m}|v9']['R_Blast'] - ds3_angka[f'{h}|{m}|v7']['R_Blast']:+.3f}", "IK95 R Blast": ik(d_bl),
                       "F1 BLB": "-", "F1 Blast": "-"})
        # confusion ds3 (gabungan 3 seed)
        if h == "utama":
            fig, axs = plt.subplots(2, 2, figsize=(9, 6.2))
            for i, m in enumerate(MODELS):
                for j, versi in enumerate(("v7", "v9")):
                    cm = np.zeros((2, 4), int)
                    for s in SEEDS:
                        lab, dg = V[(versi, m, s)]
                        for a_, b_ in zip(lab, dg):
                            if a_ in ("bacterial_leaf_blight", "leaf_blast"):
                                cm[("bacterial_leaf_blight", "leaf_blast").index(a_), DIAG.index(b_)] += 1
                    pr = cm / cm.sum(1, keepdims=True)
                    ax = axs[i, j]
                    ax.imshow(pr, cmap="Greens", vmin=0, vmax=1)
                    for a_ in range(2):
                        for b_ in range(4):
                            ax.text(b_, a_, f"{cm[a_, b_]}\n{100 * pr[a_, b_]:.0f}%", ha="center", va="center", fontsize=8,
                                    color="white" if pr[a_, b_] > 0.6 else "black")
                    ax.set_xticks(range(4), [NK[d].replace(" ", "\n", 1) for d in DIAG], fontsize=7)
                    ax.set_yticks(range(2), ["BLB", "Blast"], fontsize=7)
                    ax.set_title(f"{LABEL[m]} - {'v7, ds3 asli' if versi == 'v7' else 'v9, ds3 tanpa latar'}", fontsize=8)
            fig.suptitle("ds3 uji utama (70 foto, sudah terbuka): confusion diagnosis, jumlah 3 seed", fontsize=10)
            fig.tight_layout()
            fig.savefig(OUT / "grafik" / "confusion_ds3_utama.png", dpi=150)
            plt.close(fig)
    # Tungro
    for m in MODELS:
        for versi, ne, kl in (("v7", "eval_ds3_tungro", "v7 pada ds3 asli"), ("v9", "eval_ds3tl_tungro", "v9 pada ds3 tanpa latar")):
            ps = [json.loads((ROOT / "runs" / run(versi, m, s) / ne / "ringkasan_diagnosis.json").read_text(encoding="utf-8"))["sebaran_diagnosis_per_folder"]["tungro"]
                  for s in SEEDS]
            t7.append({"Set": "ds3 Tungro (80, di luar cakupan)", "Model": LABEL[m], "Kondisi": kl,
                       "Cadangan (tidak dipisah)": f"{cad['tungro']['persen_cadangan']}%" if versi == "v9" else "-",
                       "Akurasi": "-", "IK95 akurasi": "-", "R BLB": f"disebut BLB {ms([100 * p['sebaran'].get('bacterial_leaf_blight', 0) / p['n'] for p in ps], 1)}%",
                       "R Blast": "-", "IK95 R Blast": "-", "F1 BLB": "-",
                       "F1 Blast": f"disebut penyakit {ms([p['persen_didiagnosis_penyakit'] for p in ps], 1)}%"})
    tulis_csv("tabel7_ds3", t7)
    tulis_tex("tabel7_ds3_utama", [r for r in t7 if r["Set"].startswith("ds3 uji utama")],
              ["Model", "Kondisi", "Cadangan (tidak dipisah)", "Akurasi", "IK95 akurasi", "R BLB", "R Blast", "IK95 R Blast"],
              "ds3 uji utama (70 foto; PEMBANDING, ds3 sudah terbuka): v7 pada foto asli vs v9 pada foto tanpa latar. Rata-rata ± simpangan baku 3 seed.",
              "tab:v9_ds3")
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, (k, kl) in zip(axs, (("akurasi", "Akurasi"), ("R_Blast", "Recall Blast"), ("R_BLB", "Recall BLB"))):
        for i, m in enumerate(MODELS):
            y = [ds3_angka[f"utama|{m}|{v}"][k] for v in ("v7", "v9")]
            ax.bar(np.arange(2) + (i - 0.5) * 0.36, y, 0.34, color="#2a78d6" if m == "yolov8s" else "#eb6834", label=LABEL[m])
        ax.set_xticks(range(2), ["v7, ds3 asli", "v9, ds3 tanpa latar"], fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(kl, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    axs[0].legend(fontsize=8)
    fig.suptitle(f"ds3 uji utama (70 foto, sudah terbuka); cadangan v9 {cad['utama']['persen_cadangan']}%", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "grafik" / "grafik_ds3_utama.png", dpi=150)
    plt.close(fig)

    # ================= audit =================
    au = json.loads((OUT.parent / "v9_latar/audit_masker/ringkasan_audit.json").read_text(encoding="utf-8"))
    t8 = [{"Kelompok": g, "n": v["n"], "Baik": v["baik"], "Salah objek": v["salah_objek"], "Terpotong": v["terpotong"], "Cadangan": v["cadangan"],
           "% baik [IK95 Wilson]": f"{v['persen_baik']}% [{v['IK95_baik_wilson'][0]}; {v['IK95_baik_wilson'][1]}]"} for g, v in au["per_kelompok"].items()]
    tulis_csv("tabel8_audit_pemisahan", t8)
    tulis_tex("tabel8_audit_pemisahan", t8, list(t8[0]), "Audit manusia pemisahan daun (50 foto VAL acak di luar sampel tahap 0).", "tab:v9_audit")

    # ================= angka kunci =================
    ringkas = {}
    for sp, sub, nama in SETS:
        for kond in KOND:
            for m in MODELS:
                ringkas[f"{nama}|{kond}|{m}"] = {k: float(np.nanmean(nilai(kond, m, sp, sub, k)[0])) for k in KUNCI}
    angka["rata2_3seed"] = ringkas
    angka["ds3"] = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("bt_")} for k, v in ds3_angka.items()}
    angka["ds3_cadangan"] = cad
    (OUT / "angka_kunci_v9.json").write_text(json.dumps(angka, indent=2, ensure_ascii=False), encoding="utf-8")
    print("selesai")


if __name__ == "__main__":
    main()
