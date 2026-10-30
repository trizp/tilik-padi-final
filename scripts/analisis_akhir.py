#!/usr/bin/env python3
"""
analisis_akhir.py - Langkah 2-4 evaluasi akhir: agregasi 3 seed, bootstrap per KELOMPOK foto (group_id), perbandingan
berpasangan YOLOv8 vs YOLO11, tabel CSV + LaTeX (IEEE), grafik, contoh kualitatif. Hanya MEMBACA hasil evaluate.py
(tidak mengevaluasi ulang). Pengecualian: contoh kualitatif memprediksi ulang 8 gambar test hanya untuk digambar.

Metrik deteksi dihitung ulang dari statistik per gambar yang direkam evaluate.py (stats_per_citra_*.pkl) dengan
ultralytics.utils.metrics.ap_per_class -> titik taksiran SAMA dengan angka resmi (diverifikasi otomatis).
Keluaran: reports/final_eval/ (tabel/*.csv, tabel/*.tex, grafik/*.png, angka_kunci.json)
"""
import csv
import json
import pickle
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "final_eval"
B = 1000
SEED_BOOT = 20261009
MODELS = ["yolov8n", "yolo11n", "yolov8s", "yolo11s"]
LABEL = {"yolov8n": "YOLOv8n", "yolo11n": "YOLO11n", "yolov8s": "YOLOv8s", "yolo11s": "YOLO11s"}
SEEDS = (0, 1, 2)
KELAS = ["leaf_blast", "brown_spot", "bacterial_leaf_blight"]
NAMA_KELAS = {"leaf_blast": "Blast", "brown_spot": "Brown spot", "bacterial_leaf_blight": "BLB (prov.)",
              "sehat_atau_tidak_terdeteksi": "Sehat/tdk terdeteksi"}
# (set, subset) yang dianalisis untuk run final
SETS = [("test", "dengan_background"), ("test", "bersih"), ("test", "blb_provisional"), ("test_new_session", "dengan_background")]
NAMA_SET = {("test", "dengan_background"): "Test (semua)", ("test", "bersih"): "Test bersih (tanpa BLB prov.)",
            ("test", "blb_provisional"): "Test BLB provisional", ("test_new_session", "dengan_background"): "Test sesi baru",
            ("test_acak", "dengan_background"): "Test acak (sendiri)"}
BIRU, ORANYE = "#2a78d6", "#eb6834"
WARNA = {"yolov8n": BIRU, "yolov8s": BIRU, "yolo11n": ORANYE, "yolo11s": ORANYE}


def run_final(m, s):
    return f"final_prov_v6_{m}_s{s}"


def eval_dir(run, st):
    return ROOT / "runs" / run / f"eval_{'test' if st in ('test', 'test_acak') else st}"


# ------------------------------------------------------------------ metrik dari statistik per gambar
def muat_stats(run, st, sub):
    d = pickle.load(open(eval_dir(run, st) / f"stats_per_citra_{sub}.pkl", "rb"))
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


def kelompok(st):
    man = ROOT / ("data/processed/prov_v6_randsplit/manifest.csv" if st == "test_acak" else "data/processed/prov_v6/manifest.csv")
    return {r["file"]: r["group"] for r in csv.DictReader(open(man, encoding="utf-8"))}


def resample_grup(files, grup, rng):
    """Indeks gambar per resample (ambil kelompok dengan pengembalian). Sama untuk semua model pada set yang sama."""
    g2i = defaultdict(list)
    for i, f in enumerate(files):
        g2i[grup[f]].append(i)
    gs = sorted(g2i)
    out = []
    for _ in range(B):
        pick = rng.choice(len(gs), len(gs), replace=True)
        out.append(np.concatenate([g2i[gs[j]] for j in pick]))
    return out, len(gs)


def tugas_boot(arg):
    run, st, sub, idx_list, urutan = arg
    per, names = muat_stats(run, st, sub)
    pos = {p["im_name"]: i for i, p in enumerate(per)}
    peta = np.array([pos[f] for f in urutan])           # urutan berkas bersama -> indeks di stats model ini
    titik = metrik(per, list(range(len(per))), names)
    boot = [metrik(per, peta[ix], names) for ix in idx_list]
    return (run, st, sub), titik, boot


def diag_csv(run, st, sub):
    p = eval_dir(run, st) / f"diagnosis_{sub}_per_citra.csv"
    return list(csv.DictReader(open(p, encoding="utf-8")))


# ------------------------------------------------------------------ tabel
def f3(x):
    return "-" if x is None else f"{x:.3f}"


def ms(vals, d=3):
    v = [x for x in vals if x is not None]
    if not v:
        return "-"
    return f"{np.mean(v):.{d}f} ± {np.std(v, ddof=1) if len(v) > 1 else 0:.{d}f}"


def tulis_csv(nama, rows):
    (OUT / "tabel").mkdir(parents=True, exist_ok=True)
    with open(OUT / "tabel" / f"{nama}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def tulis_tex(nama, rows, kolom, caption, label, catatan=None):
    esc = lambda s: str(s).replace("_", "\\_").replace("±", "$\\pm$").replace("%", "\\%").replace("&", "\\&")
    t = ["\\begin{table}[!t]", "\\centering", f"\\caption{{{esc(caption)}}}", f"\\label{{{label}}}", "\\footnotesize",
         "\\begin{tabular}{" + "l" + "c" * (len(kolom) - 1) + "}", "\\hline",
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
    rng = np.random.default_rng(SEED_BOOT)
    grup = {st: kelompok(st) for st in ("test", "test_new_session", "test_acak")}

    # ---- resample kelompok bersama per (set, subset) + daftar tugas
    tasks, info_resample = [], {}
    rs_idx = {}
    for st, sub in SETS + [("test_acak", "dengan_background")]:
        contoh = run_final("yolov8n", 0) if st != "test_acak" else "randsplit_prov_v6_yolov8n_s0"
        per, _ = muat_stats(contoh, st, sub)
        files = [p["im_name"] for p in per]
        idx, ng = resample_grup(files, grup[st], rng)
        rs_idx[(st, sub)] = (files, idx)
        info_resample[f"{st}|{sub}"] = {"gambar": len(files), "kelompok": ng}
    for st, sub in SETS:
        files, idx = rs_idx[(st, sub)]
        tasks += [(run_final(m, s), st, sub, idx, files) for m in MODELS for s in SEEDS]
    files_tns, idx_tns = rs_idx[("test_new_session", "dengan_background")]
    files_ac, idx_ac = rs_idx[("test_acak", "dengan_background")]
    for m in ("yolov8n", "yolo11n"):
        tasks.append((f"randsplit_prov_v6_{m}_s0", "test_acak", "dengan_background", idx_ac, files_ac))
        tasks.append((f"randsplit_prov_v6_{m}_s0", "test_new_session", "dengan_background", idx_tns, files_tns))
    print(f"bootstrap: {len(tasks)} tugas x {B} resample", flush=True)
    hasil = {}
    with ProcessPoolExecutor(max_workers=12) as ex:
        for key, titik, boot in ex.map(tugas_boot, tasks):
            hasil[key] = (titik, boot)
            print("  selesai", key, flush=True)

    # ---- verifikasi: titik taksiran = angka resmi evaluate.py
    selisih_maks = 0.0
    for (run, st, sub), (titik, _) in hasil.items():
        ref = {r["kelas"]: r for r in csv.DictReader(open(eval_dir(run, st) / "metrik_per_kelas.csv", encoding="utf-8")) if r["subset"] == sub}
        for k_ref, k_ in (("mAP50", "mAP50"), ("mAP50_95", "mAP50_95"), ("precision", "P"), ("recall", "R")):
            selisih_maks = max(selisih_maks, abs(float(ref["SEMUA"][k_ref]) - titik[k_]))
    print(f"selisih maks titik vs resmi: {selisih_maks:.5f}", flush=True)
    assert selisih_maks < 6e-4, "statistik per gambar tidak cocok dengan angka resmi"

    # ---- diagnosis per citra: akurasi titik + bootstrap pada resample yang sama
    def diag_boot(run, st, sub):
        rows = diag_csv(run, st, sub)
        benar = {r["berkas"]: r["label"] == r["diagnosis"] for r in rows}
        files, idx = rs_idx[(st, sub)]
        v = np.array([benar.get(f, np.nan) for f in files], float)     # citra multi-kelas tidak ada di csv -> NaN
        titik = float(np.nanmean(v))
        return titik, np.array([np.nanmean(v[ix]) for ix in idx]), rows
    diag = {}
    for st, sub in SETS:
        for m in MODELS:
            for s in SEEDS:
                diag[(run_final(m, s), st, sub)] = diag_boot(run_final(m, s), st, sub)
    for m in ("yolov8n", "yolo11n"):
        r_ = f"randsplit_prov_v6_{m}_s0"
        diag[(r_, "test_acak", "dengan_background")] = diag_boot(r_, "test_acak", "dengan_background")
        diag[(r_, "test_new_session", "dengan_background")] = diag_boot(r_, "test_new_session", "dengan_background")

    def ci_mean_seed(m, st, sub, kunci):
        arr = np.array([[b[kunci] if b and kunci in b else np.nan for b in hasil[(run_final(m, s), st, sub)][1]] for s in SEEDS])
        mean_b = np.nanmean(arr, axis=0)
        return np.nanpercentile(mean_b, 2.5), np.nanpercentile(mean_b, 97.5), arr

    def ci_diag(m, st, sub):
        arr = np.array([diag[(run_final(m, s), st, sub)][1] for s in SEEDS])
        mb = arr.mean(axis=0)
        return np.percentile(mb, 2.5), np.percentile(mb, 97.5), arr

    angka = {"bootstrap": {"B": B, "seed": SEED_BOOT, "unit": "kelompok foto (group_id manifest)", "resample": info_resample},
             "verifikasi_selisih_maks_vs_resmi": selisih_maks}

    # ================= Tabel 1: utama per model =================
    t1 = []
    for st, sub in SETS:
        for m in MODELS:
            pts = [hasil[(run_final(m, s), st, sub)][0] for s in SEEDS]
            lo95, hi95, _ = ci_mean_seed(m, st, sub, "mAP50_95")
            lo50, hi50, _ = ci_mean_seed(m, st, sub, "mAP50")
            dl, dh, _ = ci_diag(m, st, sub)
            acc = [diag[(run_final(m, s), st, sub)][0] for s in SEEDS]
            t1.append({"Set": NAMA_SET[(st, sub)], "Model": LABEL[m],
                       "mAP50": ms([p["mAP50"] for p in pts]), "IK95 mAP50": f"[{lo50:.3f}, {hi50:.3f}]",
                       "mAP50-95": ms([p["mAP50_95"] for p in pts]), "IK95 mAP50-95": f"[{lo95:.3f}, {hi95:.3f}]",
                       "P": ms([p["P"] for p in pts]), "R": ms([p["R"] for p in pts]), "F1": ms([p["F1"] for p in pts]),
                       "Akurasi diagnosis": ms(acc), "IK95 akurasi": f"[{dl:.3f}, {dh:.3f}]"})
    tulis_csv("tabel1_utama_per_model", t1)
    for st, sub in SETS:
        sel = [r for r in t1 if r["Set"] == NAMA_SET[(st, sub)]]
        tag = f"{st}_{sub}"
        tulis_tex(f"tabel1_utama_{tag}", sel, ["Model", "mAP50", "IK95 mAP50", "mAP50-95", "IK95 mAP50-95", "P", "R", "F1"],
                  f"Kinerja deteksi pada {NAMA_SET[(st, sub)]} (rata-rata ± simpangan baku 3 seed; IK 95% bootstrap per kelompok foto). PROVISIONAL: BLB dari ds1.",
                  f"tab:utama_{tag}")

    # ================= Tabel 2: per kelas =================
    t2 = []
    for st, sub in [("test", "dengan_background"), ("test_new_session", "dengan_background")]:
        for m in MODELS:
            for k in KELAS:
                pts = [hasil[(run_final(m, s), st, sub)][0] for s in SEEDS]
                if f"{k}|AP50" not in pts[0]:
                    continue
                lo, hi, _ = ci_mean_seed(m, st, sub, f"{k}|AP50_95")
                t2.append({"Set": NAMA_SET[(st, sub)], "Model": LABEL[m], "Kelas": NAMA_KELAS[k],
                           "AP50": ms([p[f"{k}|AP50"] for p in pts]), "AP50-95": ms([p[f"{k}|AP50_95"] for p in pts]),
                           "IK95 AP50-95": f"[{lo:.3f}, {hi:.3f}]", "P": ms([p[f"{k}|P"] for p in pts]),
                           "R": ms([p[f"{k}|R"] for p in pts]), "F1": ms([p[f"{k}|F1"] for p in pts])})
    tulis_csv("tabel2_per_kelas", t2)
    tulis_tex("tabel2_per_kelas_test", [r for r in t2 if r["Set"] == NAMA_SET[("test", "dengan_background")]],
              ["Model", "Kelas", "AP50", "AP50-95", "IK95 AP50-95", "P", "R", "F1"],
              "Kinerja per kelas pada test (rata-rata ± simpangan baku 3 seed). BLB provisional (ds1).", "tab:per_kelas_test")
    tulis_tex("tabel2_per_kelas_sesi_baru", [r for r in t2 if r["Set"] == NAMA_SET[("test_new_session", "dengan_background")]],
              ["Model", "Kelas", "AP50", "AP50-95", "IK95 AP50-95", "P", "R", "F1"],
              "Kinerja per kelas pada test sesi baru (rata-rata ± simpangan baku 3 seed).", "tab:per_kelas_sesi_baru")

    # ================= Tabel 3: diagnosis per citra =================
    t3, cm_gab = [], {}
    for st, sub in [("test", "dengan_background"), ("test", "bersih"), ("test_new_session", "dengan_background")]:
        for m in MODELS:
            per_seed = []
            cm = defaultdict(lambda: defaultdict(int))
            for s in SEEDS:
                rows = diag[(run_final(m, s), st, sub)][2]
                pk = list(csv.DictReader(open(eval_dir(run_final(m, s), st) / f"diagnosis_{sub}_per_kelas.csv", encoding="utf-8")))
                per_seed.append({r["kelas"]: r for r in pk})
                for r in rows:
                    cm[r["label"]][r["diagnosis"]] += 1
            cm_gab[(m, st, sub)] = cm
            dl, dh, _ = ci_diag(m, st, sub)
            row = {"Set": NAMA_SET[(st, sub)], "Model": LABEL[m],
                   "Ambang (s0/s1/s2)": "/".join(f"{json.loads((ROOT / 'runs' / run_final(m, s) / 'ambang_diagnosis.json').read_text(encoding='utf-8'))['ambang']:.2f}" for s in SEEDS),
                   "Akurasi": ms([diag[(run_final(m, s), st, sub)][0] for s in SEEDS]), "IK95 akurasi": f"[{dl:.3f}, {dh:.3f}]"}
            for k in KELAS + ["sehat_atau_tidak_terdeteksi"]:
                if any(int(ps.get(k, {}).get("n_benar_kelas", 0) or 0) > 0 for ps in per_seed):
                    row[f"F1 {NAMA_KELAS[k]}"] = ms([float(ps[k]["f1"]) for ps in per_seed])
                else:
                    row[f"F1 {NAMA_KELAS[k]}"] = "-"
            t3.append(row)
    tulis_csv("tabel3_diagnosis_per_citra", t3)
    for st, sub in [("test", "dengan_background"), ("test", "bersih"), ("test_new_session", "dengan_background")]:
        sel = [r for r in t3 if r["Set"] == NAMA_SET[(st, sub)]]
        kol = ["Model", "Akurasi", "IK95 akurasi"] + [k for k in sel[0] if k.startswith("F1 ") and any(r[k] != "-" for r in sel)]
        tulis_tex(f"tabel3_diagnosis_{st}_{sub}", sel, kol,
                  f"Diagnosis per citra pada {NAMA_SET[(st, sub)]} (kelas kotak dengan confidence tertinggi; ambang dikunci pada val). Rata-rata ± simpangan baku 3 seed.",
                  f"tab:diagnosis_{st}_{sub}")

    # ================= Tabel 4: kecepatan =================
    t4 = []
    for m in MODELS:
        kec = [list(csv.DictReader(open(eval_dir(run_final(m, s), "test") / "kecepatan.csv", encoding="utf-8"))) for s in SEEDS]
        info = json.loads((eval_dir(run_final(m, 0), "test") / "ringkasan.json").read_text(encoding="utf-8"))["model"]
        gpu = [[r for r in k if r["perangkat"].startswith("GPU")][0] for k in kec]
        cpu = [[r for r in k if r["perangkat"].startswith("CPU")][0] for k in kec]
        t4.append({"Model": LABEL[m], "Parameter (juta)": f"{info['parameter_fused'] / 1e6:.2f}", "GFLOPs": f"{info['GFLOPs_fused']:.1f}",
                   "GPU ms (median)": ms([float(g["ms_total_median"]) for g in gpu], 1), "GPU FPS": ms([float(g["fps_dari_median"]) for g in gpu], 1),
                   "CPU ms (median)": ms([float(c["ms_total_median"]) for c in cpu], 1), "CPU FPS": ms([float(c["fps_dari_median"]) for c in cpu], 1)})
    perangkat = {"GPU": gpu[0]["perangkat"], "CPU": cpu[0]["perangkat"]}
    tulis_csv("tabel4_kecepatan", t4)
    tulis_tex("tabel4_kecepatan", t4, list(t4[0]),
              f"Kecepatan inferensi (batch 1, FP32, ujung ke ujung, imgsz 640; rata-rata ± simpangan baku 3 seed). {perangkat['GPU']}; {perangkat['CPU']}. Parameter dan GFLOPs setelah fuse.",
              "tab:kecepatan")

    # ================= Tabel 5: split acak vs bersih =================
    kb = json.loads((ROOT / "reports/prov_v6_randsplit/kebocoran_ketat.json").read_text(encoding="utf-8"))
    t5 = []
    for m in ("yolov8n", "yolo11n"):
        for jenis, run, st in (("split acak", f"randsplit_prov_v6_{m}_s0", "test_acak"), ("split acak", f"randsplit_prov_v6_{m}_s0", "test_new_session"),
                               ("split bersih (final s0)", run_final(m, 0), "test"), ("split bersih (final s0)", run_final(m, 0), "test_new_session")):
            p = hasil[(run, st, "dengan_background")][0]
            bt = hasil[(run, st, "dengan_background")][1]
            v = np.array([b["mAP50_95"] for b in bt])
            t5.append({"Model": LABEL[m], "Split latih": jenis,
                       "Set uji": {"test_acak": "test acak (sendiri)", "test": "test bersih", "test_new_session": "test sesi baru*"}[st],
                       "mAP50": f3(p["mAP50"]), "mAP50-95": f3(p["mAP50_95"]),
                       "IK95 mAP50-95": f"[{np.percentile(v, 2.5):.3f}, {np.percentile(v, 97.5):.3f}]",
                       "Akurasi diagnosis": f3(diag[(run, st, "dengan_background")][0])})
    tulis_csv("tabel5_split_acak_vs_bersih", t5)
    cat5 = ("*Test sesi baru = satu-satunya set yang sebanding (sama untuk kedua model, tanpa kembaran di train mana pun). "
            f"Kebocoran split acak (val+test): batas atas {kb['val+test']['batas_atas_kelompok_berantai_persen']}% (kelompok berantai), "
            f"batas bawah {kb['val+test']['batas_bawah_kembar_langsung_persen']}% (kembaran langsung). Satu seed (s0).")
    tulis_tex("tabel5_split_acak_vs_bersih", t5, list(t5[0]), "Split acak vs split bersih (YOLO n, seed 0).", "tab:split_acak", cat5)
    t5b = [{"Himpunan": f"split {k}", "Gambar": v["gambar"], "Batas atas (kelompok berantai)": f"{v['batas_atas_kelompok_berantai_persen']}%",
            "Batas bawah (kembaran langsung)": f"{v['batas_bawah_kembar_langsung_persen']}%"} for k, v in kb["per_split"].items()]
    t5b += [{"Himpunan": f"sumber {k}", "Gambar": v["gambar"], "Batas atas (kelompok berantai)": f"{v['batas_atas_kelompok_berantai_persen']}%",
             "Batas bawah (kembaran langsung)": f"{v['batas_bawah_kembar_langsung_persen']}%"} for k, v in kb["per_sumber"].items()]
    tulis_csv("tabel5b_kebocoran_split_acak", t5b)
    tulis_tex("tabel5b_kebocoran_split_acak", t5b, list(t5b[0]), "Kebocoran split acak: gambar val/test dengan kembaran di train.", "tab:kebocoran")

    # ================= Tabel 6: di luar cakupan =================
    t6 = []
    folders = None
    for m in MODELS:
        rs = [json.loads((ROOT / "runs" / run_final(m, s) / "eval_luar_cakupan_ds4" / "ringkasan_diagnosis.json").read_text(encoding="utf-8")) for s in SEEDS]
        folders = sorted(rs[0]["sebaran_diagnosis_per_folder"])
        for fo in folders:
            ps = [r["sebaran_diagnosis_per_folder"][fo] for r in rs]
            n = ps[0]["n"]
            row = {"Model": LABEL[m], "Penyakit (ds4)": fo, "Foto": n, "% salah alarm": ms([p["persen_didiagnosis_penyakit"] for p in ps], 1)}
            for k in KELAS:
                row[f"% ditebak {NAMA_KELAS[k]}"] = ms([100 * p["sebaran"].get(k, 0) / n for p in ps], 1)
            t6.append(row)
    tulis_csv("tabel6_luar_cakupan", t6)
    tulis_tex("tabel6_luar_cakupan", t6, list(t6[0]),
              "Uji di luar cakupan (ds4 Tungro, Sheath blight, Leaf smut): persentase foto yang didiagnosis salah satu penyakit (salah alarm) dan kelas yang ditebak. Rata-rata ± simpangan baku 3 seed.",
              "tab:luar_cakupan")

    # ================= Tabel 7: false positive background =================
    t7 = []
    for st in ("test", "test_new_session"):
        for m in MODELS:
            fps = [list(csv.DictReader(open(eval_dir(run_final(m, s), st) / "fp_background.csv", encoding="utf-8"))) for s in SEEDS]
            for i, c in enumerate(("0.25", "0.5")):
                t7.append({"Set": NAMA_SET[(st, "dengan_background")], "Model": LABEL[m], "conf": c, "Gambar background": fps[0][i]["n_gambar_background"],
                           "% gambar dengan FP": ms([float(f[i]["persen_gambar_fp"]) for f in fps], 1),
                           "FP per gambar": ms([float(f[i]["box_fp_per_gambar"]) for f in fps], 3)})
    tulis_csv("tabel7_fp_background", t7)
    tulis_tex("tabel7_fp_background", t7, list(t7[0]), "Laju false positive pada gambar background (daun sehat). Rata-rata ± simpangan baku 3 seed.", "tab:fp_background")

    # ================= Tabel 8: YOLOv8 vs YOLO11 berpasangan =================
    t8 = []
    for pasangan in (("yolov8n", "yolo11n"), ("yolov8s", "yolo11s")):
        v8, v11 = pasangan
        for st, sub in [("test", "dengan_background"), ("test", "bersih"), ("test_new_session", "dengan_background")]:
            for met in ("mAP50_95", "mAP50", "akurasi"):
                def nilai(m, s):
                    if met == "akurasi":
                        return diag[(run_final(m, s), st, sub)][0], diag[(run_final(m, s), st, sub)][1]
                    t_, bt = hasil[(run_final(m, s), st, sub)]
                    return t_[met], np.array([b[met] for b in bt])
                ds_, bs_ = [], []
                for s in SEEDS:
                    a0, ab = nilai(v8, s)
                    b0, bb = nilai(v11, s)
                    d0, db = b0 - a0, bb - ab
                    ds_.append(d0)
                    bs_.append(db)
                    lo, hi = np.percentile(db, [2.5, 97.5])
                    t8.append({"Pasangan": f"{LABEL[v11]} − {LABEL[v8]}", "Set": NAMA_SET[(st, sub)], "Metrik": met, "Seed": str(s),
                               "Selisih": f"{d0:+.3f}", "IK95": f"[{lo:+.3f}, {hi:+.3f}]", "Memuat nol": "ya" if lo <= 0 <= hi else "tidak"})
                db = np.mean(bs_, axis=0)
                lo, hi = np.percentile(db, [2.5, 97.5])
                t8.append({"Pasangan": f"{LABEL[v11]} − {LABEL[v8]}", "Set": NAMA_SET[(st, sub)], "Metrik": met, "Seed": "gabungan",
                           "Selisih": f"{np.mean(ds_):+.3f}", "IK95": f"[{lo:+.3f}, {hi:+.3f}]", "Memuat nol": "ya" if lo <= 0 <= hi else "tidak"})
    tulis_csv("tabel8_v8_vs_v11_berpasangan", t8)
    tulis_tex("tabel8_v8_vs_v11_gabungan", [r for r in t8 if r["Seed"] == "gabungan"], ["Pasangan", "Set", "Metrik", "Selisih", "IK95", "Memuat nol"],
              "Selisih YOLO11 − YOLOv8 (rata-rata 3 seed), IK 95% bootstrap berpasangan per kelompok foto pada citra uji yang sama.", "tab:v8_vs_v11")

    # ---- pemilihan model terbaik TANPA melihat test: rata-rata mAP50-95 val bersih (best.pt) 3 seed
    val_best = {}
    for m in MODELS:
        v = []
        for s in SEEDS:
            ep = [json.loads(ln) for ln in (ROOT / "runs" / run_final(m, s) / "epoch_stats.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
            v.append(max(e["metrik_val"]["metrics/mAP50-95(B)"] for e in ep))
        val_best[m] = v
    terbaik_kel = {"YOLOv8": max(("yolov8n", "yolov8s"), key=lambda m: np.mean(val_best[m])),
                   "YOLO11": max(("yolo11n", "yolo11s"), key=lambda m: np.mean(val_best[m]))}
    terbaik = max(MODELS, key=lambda m: np.mean(val_best[m]))
    seed_terbaik = int(np.argmax(val_best[terbaik]))
    angka["pemilihan_terbaik"] = {"kriteria": "rata-rata mAP50-95 val bersih best.pt (tanpa melihat test)", "val": val_best,
                                  "terbaik_per_keluarga": terbaik_kel, "terbaik": terbaik, "seed_terbaik": seed_terbaik}

    # ================= Grafik =================
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": "#8a8984", "axes.labelcolor": "#0b0b0b",
                         "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.spines.top": False, "axes.spines.right": False})
    (OUT / "grafik").mkdir(parents=True, exist_ok=True)

    def batang(kunci_fn, judul_y, nama_file, sets):
        fig, axes = plt.subplots(1, len(sets), figsize=(2.4 * len(sets), 2.6), sharey=True)
        for ax, (st, sub) in zip(np.atleast_1d(axes), sets):
            for i, m in enumerate(MODELS):
                pts, lo, hi = kunci_fn(m, st, sub)
                mean = np.mean(pts)
                ax.bar(i, mean, width=0.6, color=WARNA[m], alpha=0.85 if m.endswith("s") else 0.55, edgecolor="none")
                ax.errorbar(i, mean, yerr=[[mean - lo], [hi - mean]], fmt="none", ecolor="#0b0b0b", elinewidth=1, capsize=3)
                ax.scatter([i] * len(pts), pts, s=9, color="#0b0b0b", zorder=3, linewidths=0)
            ax.set_xticks(range(len(MODELS)), [LABEL[m] for m in MODELS], rotation=30, ha="right")
            ax.set_title(NAMA_SET[(st, sub)], fontsize=8)
            ax.grid(axis="y", color="#e6e5e0", linewidth=0.6)
            ax.set_axisbelow(True)
        np.atleast_1d(axes)[0].set_ylabel(judul_y)
        from matplotlib.patches import Patch
        fig.legend(handles=[Patch(color=BIRU, label="YOLOv8"), Patch(color=ORANYE, label="YOLO11")], loc="upper right", frameon=False, ncol=2)
        fig.text(0.01, 0.01, "Batang = rata-rata 3 seed; garis = IK 95% bootstrap per kelompok; titik = tiap seed.", fontsize=6, color="#52514e")
        fig.tight_layout(rect=(0, 0.04, 1, 0.92))
        fig.savefig(OUT / "grafik" / nama_file, dpi=300)
        plt.close(fig)

    sets3 = [("test", "dengan_background"), ("test", "bersih"), ("test_new_session", "dengan_background")]

    def k_map(m, st, sub):
        lo, hi, arr = ci_mean_seed(m, st, sub, "mAP50_95")
        return [hasil[(run_final(m, s), st, sub)][0]["mAP50_95"] for s in SEEDS], lo, hi

    def k_acc(m, st, sub):
        lo, hi, _ = ci_diag(m, st, sub)
        return [diag[(run_final(m, s), st, sub)][0] for s in SEEDS], lo, hi
    batang(k_map, "mAP50-95", "gambar1_map50_95_per_model.png", sets3)
    batang(k_acc, "Akurasi diagnosis per citra", "gambar2_akurasi_diagnosis_per_model.png", sets3)

    # confusion matrix diagnosis (gabungan 3 seed) untuk model terbaik tiap keluarga, test (semua)
    urut = KELAS + ["sehat_atau_tidak_terdeteksi"]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0))
    for ax, (kel, m) in zip(axes, terbaik_kel.items()):
        cm = cm_gab[(m, "test", "dengan_background")]
        M = np.array([[cm[t][p] for p in urut] for t in urut], float)
        Mn = M / np.maximum(M.sum(1, keepdims=True), 1)
        ax.imshow(Mn, cmap="Blues", vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{int(M[i, j])}\n{Mn[i, j]:.0%}", ha="center", va="center", fontsize=6.5,
                        color="#ffffff" if Mn[i, j] > 0.55 else "#0b0b0b")
        ax.set_xticks(range(4), [NAMA_KELAS[k] for k in urut], rotation=30, ha="right")
        ax.set_yticks(range(4), [NAMA_KELAS[k] for k in urut])
        ax.set_xlabel("Diagnosis model")
        ax.set_ylabel("Label")
        ax.set_title(f"{LABEL[m]} (gabungan 3 seed)", fontsize=8)
        ax.spines[:].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "grafik" / "gambar3_confusion_diagnosis_terbaik.png", dpi=300)
    plt.close(fig)

    # contoh kualitatif: 4 benar + 4 salah dari test (semua), model terbaik (seed dipilih dari val)
    from PIL import Image, ImageDraw
    from ultralytics import YOLO
    run_b = run_final(terbaik, seed_terbaik)
    ambang_b = json.loads((ROOT / "runs" / run_b / "ambang_diagnosis.json").read_text(encoding="utf-8"))["ambang"]
    rows = diag[(run_b, "test", "dengan_background")][2]
    rr = random.Random(7)
    benar = [r for r in rows if r["label"] == r["diagnosis"]]
    salah = [r for r in rows if r["label"] != r["diagnosis"]]
    pilih_b = []
    for k in urut:                                      # satu contoh benar per kelas
        c = [r for r in benar if r["label"] == k]
        if c:
            pilih_b.append(rr.choice(c))
    pilih_s = rr.sample(salah, 4)
    model_b = YOLO(str(ROOT / "runs" / run_b / "weights" / "best.pt"))
    warna_k = {0: (42, 120, 214), 1: (235, 104, 52), 2: (27, 175, 122)}
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.2))
    for ax, r, jenis in zip(axes.ravel(), pilih_b[:4] + pilih_s, ["benar"] * 4 + ["salah"] * 4):
        f = ROOT / "data/processed/prov_v6/images/test" / r["berkas"]
        im = Image.open(f).convert("RGB")
        d = ImageDraw.Draw(im)
        lab = ROOT / "data/processed/prov_v6/labels/test" / (Path(r["berkas"]).stem + ".txt")
        W, H = im.size
        lw = max(2, W // 200)
        if lab.exists():
            for ln in lab.read_text().splitlines():
                c, x, y, w, h = map(float, ln.split()[:5])
                d.rectangle([(x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H], outline=(255, 255, 255), width=lw + 1)
        res = model_b.predict(str(f), imgsz=640, conf=ambang_b, iou=0.7, device=0, verbose=False)[0]
        for bx, c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy().astype(int)):
            d.rectangle(list(bx), outline=warna_k[c], width=lw)
        ax.imshow(im)
        ax.set_axis_off()
        ax.set_title(f"{'BENAR' if jenis == 'benar' else 'SALAH'}: label {NAMA_KELAS[r['label']]}\ndiagnosis {NAMA_KELAS[r['diagnosis']]}", fontsize=6.5)
    fig.text(0.01, 0.01, f"{LABEL[terbaik]} seed {seed_terbaik}, ambang {ambang_b:.2f}. Putih = kotak label; biru = Blast, oranye = Brown spot, hijau = BLB (prediksi).",
             fontsize=6, color="#52514e")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "grafik" / "gambar4_contoh_kualitatif.png", dpi=300)
    plt.close(fig)
    angka["contoh_kualitatif"] = {"run": run_b, "ambang": ambang_b, "benar": [r["berkas"] for r in pilih_b[:4]], "salah": [r["berkas"] for r in pilih_s]}

    # ---- angka kunci
    angka["tabel1"] = t1
    angka["tabel8_gabungan"] = [r for r in t8 if r["Seed"] == "gabungan"]
    angka["tabel5"] = t5
    angka["luar_cakupan"] = t6
    angka["kecepatan"] = t4
    (OUT / "angka_kunci.json").write_text(json.dumps(angka, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print("selesai", flush=True)


if __name__ == "__main__":
    main()
