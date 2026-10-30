#!/usr/bin/env python3
"""
analisis_alur_aplikasi_v9.py - Analisis fase 5 (KEPUTUSAN_PROYEK.md 2026-10-20). Label: "evaluasi alur aplikasi". Hanya MEMBACA hasil evaluate.py.

Butir 3 (model final YOLOv8s v9 s0-2; irisan foto test bersih / test sesi baru / BLB terpisah):
  app  = alur aplikasi (YOLOv8n-seg; foto cadangan = diminta foto ulang: TANPA prediksi kotak dan TIDAK didiagnosis)  runs/v9_*/eval_alurapp_<split>
  sam2 = alur SAM2 (hasil v9 yang ada)                                                                               runs/v9_*/eval_<split>
  v7   = model v7 pada foto asli (hasil v7 yang ada, dibatasi pada irisan)                                             runs/v7_*/eval_<split>
  mAP50, mAP50-95, P/R/F1, akurasi diagnosis (cadangan dihitung salah), akurasi pada foto yang didiagnosis, recall Blast, % cadangan;
  rata-rata ± std 3 seed; IK 95% bootstrap 1.000 per kelompok foto; selisih berpasangan app - sam2 dan app - v7.
Butir 4 (PEMBANDING, bukan hasil utama): model aplikasi v9_deploy (alur aplikasi, ambang aplikasi) vs model aplikasi v7 YOLOv8s
  pada test sesi baru (irisan), ds3 uji utama, Tungro ds3.
Keluaran: reports/v9_latar/aplikasi_v9/fase5_*.csv/json, RINGKASAN_FASE4_5.md ditulis terpisah.
"""
import csv
import json
import pickle
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from analisis_akhir_v9 import ik, metrik, ms, nol  # noqa: E402

REP = ROOT / "reports/v9_latar/aplikasi_v9"
B, SEED = 1000, 20261020
SEEDS = (0, 1, 2)
TAK = "sehat_atau_tidak_terdeteksi"
SETS = [("test", "bersih", "Test bersih (tanpa BLB)"), ("test_new_session", "dengan_background", "Test sesi baru"),
        ("test", "blb_provisional", "Test BLB provisional (terpisah)")]
KOND = {"app": ("v9", "eval_alurapp_{sp}", "Alur aplikasi (YOLOv8n-seg)"), "sam2": ("v9", "eval_{sp}", "Alur SAM2 (v9)"),
        "v7": ("v7", "eval_{sp}", "v7 (foto asli)")}
ALUR_MAN = ROOT / "data/derived/v9_alur_aplikasi/manifest.csv"


def edir(kond, s, sp):
    v, pola, _ = KOND[kond]
    return ROOT / "runs" / f"{v}_final_prov_yolov8s_s{s}" / pola.format(sp=sp)


def cadangan():
    return {r["file"] for r in csv.DictReader(open(ALUR_MAN, encoding="utf-8")) if r["seg_cadangan"] == "ya"}


def per_citra(d, sub, kosongkan):
    per = pickle.load(open(d / f"stats_per_citra_{sub}.pkl", "rb"))
    names = per["names"]
    out = []
    for p in per["per_citra"]:
        if p["im_name"] in kosongkan:                      # foto ulang: tidak ada prediksi kotak
            p = dict(p, tp=p["tp"][:0], conf=p["conf"][:0], pred_cls=p["pred_cls"][:0])
        out.append(p)
    return out, names


def tugas(arg):
    kond, s, sp, sub, files, idx, kos = arg
    per, names = per_citra(edir(kond, s, sp), sub, kos if kond == "app" else set())
    pos = {p["im_name"]: i for i, p in enumerate(per)}
    peta = np.array([pos[f] for f in files])
    return (kond, s, sp, sub), metrik(per, peta, names), [metrik(per, peta[ix], names) for ix in idx]


def diag(kond, s, sp, sub, files, kos):
    rows = {r["berkas"]: r for r in csv.DictReader(open(edir(kond, s, sp) / f"diagnosis_{sub}_per_citra.csv", encoding="utf-8"))}
    lab = np.array([rows[f]["label"] if f in rows else "" for f in files])
    dg = np.array([("FOTO_ULANG" if (kond == "app" and f in kos) else rows[f]["diagnosis"]) if f in rows else "" for f in files])
    ada = lab != ""
    return lab, dg, ada


def main():
    kos = cadangan()
    grup = {r["file"]: r["group"] for r in csv.DictReader(open(ROOT / "data/processed/prov_v7/manifest.csv", encoding="utf-8"))}
    rng = np.random.default_rng(SEED)
    rs, tasks = {}, []
    for sp, sub, _ in SETS:
        per, _ = per_citra(edir("sam2", 0, sp), sub, set())
        files = sorted(p["im_name"] for p in per)
        g2i = defaultdict(list)
        for i, f in enumerate(files):
            g2i[grup[f]].append(i)
        gs = sorted(g2i)
        idx = [np.concatenate([g2i[gs[j]] for j in rng.choice(len(gs), len(gs), replace=True)]) for _ in range(B)]
        rs[(sp, sub)] = (files, idx, len(gs))
        tasks += [(k, s, sp, sub, files, idx, kos) for k in KOND for s in SEEDS]
    H = {}
    with ProcessPoolExecutor(max_workers=8) as ex:
        for key, titik, boot in ex.map(tugas, tasks):
            H[key] = (titik, boot)
    D = {}
    for sp, sub, _ in SETS:
        files, idx, _ = rs[(sp, sub)]
        for k in KOND:
            for s in SEEDS:
                lab, dg, ada = diag(k, s, sp, sub, files, kos)
                benar = np.where(ada, (lab == dg).astype(float), np.nan)
                didiag = ada & (dg != "FOTO_ULANG")
                mb = lab == "leaf_blast"
                f = lambda ix: (np.nanmean(benar[ix]),  # noqa: E731
                                ((dg[ix] == "leaf_blast") & mb[ix]).sum() / mb[ix].sum() if mb[ix].sum() else np.nan)
                pt = f(np.arange(len(files)))
                bt = np.array([f(ix) for ix in idx])
                D[(k, s, sp, sub)] = {"akurasi": pt[0], "recall_blast": pt[1], "akurasi_didiagnosis": float((lab == dg)[didiag].mean()),
                                      "persen_cadangan": 100 * float((ada & (dg == "FOTO_ULANG")).sum() / ada.sum()), "bt_akurasi": bt[:, 0], "bt_blast": bt[:, 1]}

    def nilai(k, sp, sub, kk):
        if kk in ("akurasi", "recall_blast"):
            return [D[(k, s, sp, sub)][kk] for s in SEEDS], np.nanmean([D[(k, s, sp, sub)]["bt_akurasi" if kk == "akurasi" else "bt_blast"] for s in SEEDS], axis=0)
        return ([H[(k, s, sp, sub)][0][kk] for s in SEEDS],
                np.nanmean([[b[kk] if b and kk in b else np.nan for b in H[(k, s, sp, sub)][1]] for s in SEEDS], axis=0))

    t1, t2 = [], []
    for sp, sub, nama in SETS:
        files, _, ng = rs[(sp, sub)]
        for k in KOND:
            row = {"Set": nama, "Foto / kelompok": f"{len(files)} / {ng}", "Alur": KOND[k][2]}
            for kk, lab_ in (("mAP50", "mAP50"), ("mAP50_95", "mAP50-95")):
                p, b = nilai(k, sp, sub, kk)
                row[lab_], row[f"IK95 {lab_}"] = ms(p), ik(b)
            for kk in ("P", "R", "F1"):
                row[kk] = ms([H[(k, s, sp, sub)][0][kk] for s in SEEDS])
            for kk, lab_ in (("akurasi", "Akurasi diagnosis"), ("recall_blast", "Recall diagnosis Blast")):
                p, b = nilai(k, sp, sub, kk)
                row[lab_], row[f"IK95 {lab_}"] = ms(p), ik(b)
            row["Akurasi pada foto yang didiagnosis"] = ms([D[(k, s, sp, sub)]["akurasi_didiagnosis"] for s in SEEDS])
            row["% cadangan (foto ulang)"] = f"{D[(k, 0, sp, sub)]['persen_cadangan']:.1f}%"
            t1.append(row)
        for a, b_, lab_ in (("app", "sam2", "Alur aplikasi − alur SAM2"), ("app", "v7", "Alur aplikasi − v7")):
            row = {"Set": nama, "Selisih": lab_}
            for kk, kl in (("mAP50_95", "mAP50-95"), ("akurasi", "akurasi"), ("recall_blast", "recall Blast")):
                pa, ba = nilai(a, sp, sub, kk)
                pb, bb = nilai(b_, sp, sub, kk)
                d = np.nanmean(pa) - np.nanmean(pb)
                row[kl] = "-" if np.isnan(d) else f"{d:+.3f} {ik(ba - bb)}"
                row[f"memuat nol ({kl})"] = nol(ba - bb)
            t2.append(row)
    for nama, rows in (("fase5_alur_aplikasi_final", t1), ("fase5_selisih_alur", t2)):
        with open(REP / f"{nama}.csv", "w", newline="", encoding="utf-8-sig") as fo:
            w = csv.DictWriter(fo, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    # ---------- butir 4: model aplikasi v9 vs v7 (pembanding)
    def satu(d, sub, files, kos_):
        per = {p["im_name"]: p for p in pickle.load(open(d / f"stats_per_citra_{sub}.pkl", "rb"))["per_citra"]}
        names = pickle.load(open(d / f"stats_per_citra_{sub}.pkl", "rb"))["names"]
        lst = [dict(per[f], tp=per[f]["tp"][:0], conf=per[f]["conf"][:0], pred_cls=per[f]["pred_cls"][:0]) if f in kos_ else per[f] for f in files]
        m = metrik(lst, list(range(len(lst))), names)
        rows = {r["berkas"]: r for r in csv.DictReader(open(d / f"diagnosis_{sub}_per_citra.csv", encoding="utf-8"))}
        fs = [f for f in files if f in rows]
        dg = {f: ("FOTO_ULANG" if f in kos_ else rows[f]["diagnosis"]) for f in fs}
        acc = np.mean([dg[f] == rows[f]["label"] for f in fs])
        dd = [f for f in fs if dg[f] != "FOTO_ULANG"]
        bl = [f for f in fs if rows[f]["label"] == "leaf_blast"]
        return {"mAP50": m["mAP50"], "mAP50_95": m["mAP50_95"], "akurasi": float(acc), "akurasi_didiagnosis": float(np.mean([dg[f] == rows[f]["label"] for f in dd])),
                "recall_blast": float(np.mean([dg[f] == "leaf_blast" for f in bl])), "foto": len(fs), "cadangan": sum(f in kos_ for f in fs)}
    files_tns = rs[("test_new_session", "dengan_background")][0]
    dep = {"tns_v9": satu(ROOT / "runs/v9_deploy_yolov8s_s2/eval_alurapp_test_new_session", "dengan_background", files_tns, kos),
           "tns_v7": satu(ROOT / "runs/v7_deploy_yolov8s_s2/eval_test_new_session", "dengan_background", files_tns, set())}
    cad3 = json.loads((ROOT / "data/derived/v9_latar/seg_n_ds3/info.json").read_text(encoding="utf-8"))["cadangan"]

    def ds3(d, kos_rel):
        rows = list(csv.DictReader(open(d / "diagnosis_dalam_cakupan_per_citra.csv", encoding="utf-8")))
        for r in rows:
            r["k"] = "/".join(Path(r["berkas"]).parts[-2:])
            if r["k"] in kos_rel:
                r["diagnosis"] = "FOTO_ULANG"
        dd = [r for r in rows if r["diagnosis"] != "FOTO_ULANG"]
        rk = lambda k: np.mean([r["diagnosis"] == k for r in rows if r["label"] == k])  # noqa: E731
        return {"foto": len(rows), "cadangan": len(rows) - len(dd), "akurasi": float(np.mean([r["diagnosis"] == r["label"] for r in rows])),
                "akurasi_didiagnosis": float(np.mean([r["diagnosis"] == r["label"] for r in dd])) if dd else None,
                "recall_blast": float(rk("leaf_blast")), "recall_blb": float(rk("bacterial_leaf_blight"))}
    kos3 = {k for k, v in cad3.items() if v}
    dep["ds3_utama_v9"] = ds3(ROOT / "runs/v9_deploy_yolov8s_s2/eval_alurapp_ds3_utama", kos3)
    dep["ds3_utama_v7"] = ds3(ROOT / "runs/v7_deploy_yolov8s_s2/eval_ds3_utama", set())

    def tungro(p, kos_rel):
        rows = list(csv.DictReader(open(p / "diagnosis_luar_cakupan_per_citra.csv", encoding="utf-8")))
        ks = ["/".join(Path(r["berkas"]).parts[-2:]) for r in rows]
        fu = [k in kos_rel for k in ks]
        blb = [r["diagnosis"] == "bacterial_leaf_blight" and not f for r, f in zip(rows, fu)]
        n_d = len(rows) - sum(fu)
        return {"foto": len(rows), "cadangan": sum(fu), "disebut_BLB": sum(blb), "persen_BLB_dari_semua": round(100 * sum(blb) / len(rows), 1),
                "persen_BLB_dari_yang_didiagnosis": round(100 * sum(blb) / n_d, 1) if n_d else None}
    dep["tungro_v9"] = tungro(ROOT / "runs/v9_deploy_yolov8s_s2/eval_alurapp_ds3_tungro", kos3)
    dep["tungro_v7"] = tungro(ROOT / "runs/v7_deploy_yolov8s_s2/eval_ds3_tungro", set())
    dep["LABEL"] = "PEMBANDING, bukan hasil utama; model aplikasi v9 (alur aplikasi, ambang 0,36) vs model aplikasi v7 YOLOv8s (foto asli, ambang 0,37)"
    (REP / "fase5_model_aplikasi_v9_vs_v7.json").write_text(json.dumps(dep, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"t1": t1, "t2": t2, "dep": dep}, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
