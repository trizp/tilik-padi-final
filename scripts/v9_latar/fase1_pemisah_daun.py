#!/usr/bin/env python3
"""
fase1_pemisah_daun.py - Fase 1 rencana aplikasi v9 (KEPUTUSAN_PROYEK.md 2026-10-20): pemisah daun YOLOv8n-seg (distilasi SAM2), HANYA VAL.

  statistik : prediksi .pt pada semua foto VAL prov_v7 (1.259; foto asli) -> ciri tiap instance + IoU (setelah pasca-proses) terhadap
              masker alur SAM2 (cadangan SAM2 = masker penuh)                          -> data/derived/v9_latar/seg_n_val/statistik.pkl
  pilih_tau : grid 0,05-0,95; kriteria = rata-rata "IoU alur" tertinggi pada VAL prov_v9 (1.138 foto); cadangan = masker penuh; seri -> terkecil
  akhir     : masker akhir dengan tau terpilih (semua VAL) + ukuran fase 1 + montase 20 contoh + kecocokan dengan penilaian manual
  onnx      : ekspor ONNX; ukuran berkas; kesetaraan ONNX (numpy) vs .pt pada 20 foto VAL; waktu dan RAM per foto di CPU (proses baru)
Keluaran laporan: reports/v9_latar/aplikasi_v9/
"""
import csv
import json
import os
import pickle
import random
import statistics as st
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

os.environ["YOLO_AUTOINSTALL"] = "false"
import cv2  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import pemisah_daun_seg as pd  # noqa: E402
import umum  # noqa: E402

ROOT = umum.ROOT
RUN = ROOT / "runs/v9_pemisah_daun_yolov8n_seg"
BOBOT = RUN / "weights/best.pt"
KERJA = umum.KERJA / "seg_n_val"
REP = ROOT / "reports/v9_latar/aplikasi_v9"
SAM2 = umum.KERJA / "sam2_v2"
GRID = [round(0.05 * i, 2) for i in range(1, 20)]


def val_v9():
    return {r["file"]: r for r in csv.DictReader(open(ROOT / "data/processed/prov_v9/manifest.csv", encoding="utf-8")) if r["split"] == "val"}


def masker_sam2(f, shape, info):
    if info.get(f, {}).get("cadangan"):
        return np.ones(shape, bool), True
    mm = cv2.imread(str(SAM2 / "masker_mentah" / (Path(f).stem + ".png")), cv2.IMREAD_GRAYSCALE)
    return umum.pasca_proses(mm >= 128), False


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 1.0


def statistik():
    from ultralytics import YOLO
    KERJA.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(BOBOT))
    info = json.loads((SAM2 / "info.json").read_text(encoding="utf-8"))
    hasil = {}
    rows = umum.manifest_val()
    for i, r in enumerate(rows):
        f = r["file"]
        bgr = cv2.imread(str(umum.VAL_IMG / f))
        S, s_cad = masker_sam2(f, bgr.shape[:2], info)
        masks, confs = pd.prediksi_pt(model, bgr)
        ciri = pd.ciri_kandidat(bgr, masks, confs)
        iou_pp = [iou(umum.pasca_proses(m), S) for m in masks]
        hasil[f] = {"ciri": ciri, "iou_pp": iou_pp, "iou_penuh": float(S.mean()), "sam2_cadangan": s_cad}
        if i % 200 == 0:
            print(i, len(rows), len(masks), flush=True)
    pickle.dump(hasil, open(KERJA / "statistik.pkl", "wb"))


def iou_alur(h, tau):
    j = pd.pilih_indeks(h["ciri"], tau)
    return (h["iou_penuh"] if j is None else h["iou_pp"][j]), j is None


def pilih_tau():
    REP.mkdir(parents=True, exist_ok=True)
    H = pickle.load(open(KERJA / "statistik.pkl", "rb"))
    v9 = val_v9()
    tabel = []
    for t in GRID:
        v = [iou_alur(H[f], t) for f in v9]
        tabel.append({"tau": t, "rata2_iou_alur": round(float(np.mean([x for x, _ in v])), 4),
                      "median_iou_alur": round(float(np.median([x for x, _ in v])), 4), "cadangan": sum(c for _, c in v)})
    terbaik = max(tabel, key=lambda r: (r["rata2_iou_alur"], -r["tau"]))
    out = {"kriteria": "rata-rata IoU alur tertinggi pada VAL prov_v9 (1.138 foto); cadangan = masker penuh; seri -> tau terkecil",
           "tau_terpilih": terbaik["tau"], "n_foto": len(v9), "tabel": tabel}
    (REP / "fase1_pilih_tau.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


def ubin(img, s=230):
    h, w = img.shape[:2]
    k = s / max(h, w)
    kecil = cv2.resize(img, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
    kan = np.full((s, s, 3), 255, np.uint8)
    y0, x0 = (s - kecil.shape[0]) // 2, (s - kecil.shape[1]) // 2
    kan[y0:y0 + kecil.shape[0], x0:x0 + kecil.shape[1]] = kecil
    return kan


def akhir():
    from ultralytics import YOLO
    tau = json.loads((REP / "fase1_pilih_tau.json").read_text(encoding="utf-8"))["tau_terpilih"]
    model = YOLO(str(BOBOT))
    info = json.loads((SAM2 / "info.json").read_text(encoding="utf-8"))
    (KERJA / "masker").mkdir(parents=True, exist_ok=True)
    v9 = val_v9()
    per, cad_seg = [], {}
    for i, r in enumerate(umum.manifest_val()):
        f = r["file"]
        bgr = cv2.imread(str(umum.VAL_IMG / f))
        masks, confs = pd.prediksi_pt(model, bgr)
        M, c = pd.pilih_daun(bgr, masks, confs, tau)
        cad_seg[f] = c
        cv2.imwrite(str(KERJA / "masker" / (Path(f).stem + ".png")), M.astype(np.uint8) * 255)
        S, sc = masker_sam2(f, bgr.shape[:2], info)
        h, w = bgr.shape[:2]
        kotak = umum.baca_kotak(f, w, h)
        per.append({"berkas": f, "sumber": r["source"], "kelas": r["class"], "di_v9": f in v9, "sam2_cadangan": sc, "seg_cadangan": c,
                    "iou": round(iou(M, S), 4), "kotak": len(kotak),
                    "kotak_ge50_seg": sum(v >= 0.5 for _, v, _ in umum.retensi_kotak(M, kotak)),
                    "kotak_ge50_sam2": sum(v >= 0.5 for _, v, _ in umum.retensi_kotak(S, kotak)), "luas_seg": round(float(M.mean()), 4)})
        if i % 200 == 0:
            print(i, flush=True)
    (KERJA / "info.json").write_text(json.dumps({"tau": tau, "cadangan": cad_seg}, indent=0), encoding="utf-8")
    with open(REP / "fase1_per_foto.csv", "w", newline="", encoding="utf-8") as fo:
        wr = csv.DictWriter(fo, fieldnames=list(per[0]))
        wr.writeheader()
        wr.writerows(per)
    # ---- ukuran (VAL prov_v9)
    p9 = [p for p in per if p["di_v9"]]
    keduanya = [p for p in p9 if not p["sam2_cadangan"] and not p["seg_cadangan"]]
    grup = {"semua VAL prov_v9": p9, "ds4": [p for p in p9 if p["sumber"] == "ds4"], "ds1 BLB": [p for p in p9 if p["sumber"] == "ds1_provisional"],
            "ds2 sehat": [p for p in p9 if p["sumber"] == "ds2"]}
    ring = {"tau": tau, "per_kelompok": {}}
    for g, ps in grup.items():
        kd = [p for p in ps if not p["sam2_cadangan"] and not p["seg_cadangan"]]
        nk = sum(p["kotak"] for p in ps)
        ring["per_kelompok"][g] = {
            "n": len(ps), "iou_alur_rata2": round(float(np.mean([p["iou"] for p in ps])), 4), "iou_alur_median": round(float(np.median([p["iou"] for p in ps])), 4),
            "n_keduanya_dipisah": len(kd), "iou_keduanya_dipisah_rata2": round(float(np.mean([p["iou"] for p in kd])), 4) if kd else None,
            "iou_keduanya_dipisah_median": round(float(np.median([p["iou"] for p in kd])), 4) if kd else None,
            "kotak": nk, "persen_kotak_ge50_seg": round(100 * sum(p["kotak_ge50_seg"] for p in ps) / nk, 1) if nk else None,
            "persen_kotak_ge50_sam2": round(100 * sum(p["kotak_ge50_sam2"] for p in ps) / nk, 1) if nk else None,
            "cadangan_sam2": sum(p["sam2_cadangan"] for p in ps), "cadangan_seg": sum(p["seg_cadangan"] for p in ps),
            "cadangan_cocok": sum(p["sam2_cadangan"] == p["seg_cadangan"] for p in ps),
            "matriks_cadangan_sam2_x_seg": {f"sam2 {'ya' if a else 'tidak'} / seg {'ya' if b else 'tidak'}": sum(p["sam2_cadangan"] == a and p["seg_cadangan"] == b for p in ps)
                                             for a in (True, False) for b in (True, False)}}
    # ---- kecocokan dengan penilaian manual (50 foto VAL v7)
    nil = {r["berkas"]: r["penilaian"] for r in csv.DictReader(open(ROOT / "reports/v9_latar/audit_masker/penilaian.csv", encoding="utf-8"))}
    idx = {p["berkas"]: p for p in per}
    aud = Counter()
    for f, n in nil.items():
        p = idx[f]
        hasil_seg = "cadangan" if p["seg_cadangan"] else ("mirip SAM2 (IoU>=0,8)" if p["iou"] >= 0.8 else "berbeda dari SAM2 (IoU<0,8)")
        aud[f"{n}|{hasil_seg}"] += 1
    ring["penilaian_manual_x_seg"] = dict(sorted(aud.items()))
    ring["catatan_penilaian"] = "penilaian manual menilai keluaran SAM2; 'mirip SAM2' pada foto 'baik' = setuju; pada 'salah_objek' = ikut salah"
    (REP / "fase1_ringkasan.json").write_text(json.dumps(ring, indent=1, ensure_ascii=False), encoding="utf-8")
    # ---- montase 20 contoh VAL prov_v9 (acak berstrata, seed tetap): asli | SAM2 | YOLOv8n-seg
    rng = random.Random(20261020)
    contoh = []
    for g, n in (("ds4", 14), ("ds1 BLB", 4), ("ds2 sehat", 2)):
        contoh += rng.sample(sorted(grup[g], key=lambda p: p["berkas"]), n)
    sel = []
    for p in contoh:
        f = p["berkas"]
        bgr = cv2.imread(str(umum.VAL_IMG / f))
        S, _ = masker_sam2(f, bgr.shape[:2], info)
        M = cv2.imread(str(KERJA / "masker" / (Path(f).stem + ".png")), cv2.IMREAD_GRAYSCALE) > 0
        a, b = bgr.copy(), bgr.copy()
        a[~S] = umum.ABU[::-1]
        b[~M] = umum.ABU[::-1]
        trip = np.hstack([ubin(bgr), ubin(a), ubin(b)])
        t = f"{p['sumber']} IoU {p['iou']:.2f}{' S2cad' if p['sam2_cadangan'] else ''}{' SEGcad' if p['seg_cadangan'] else ''}"
        cv2.putText(trip, t, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(trip, t, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        sel.append(np.pad(trip, ((4, 4), (4, 4), (0, 0)), constant_values=255))
    baris = [np.hstack(sel[i:i + 2]) for i in range(0, len(sel), 2)]
    judul = np.full((36, baris[0].shape[1], 3), 255, np.uint8)
    cv2.putText(judul, f"VAL: asli | SAM2 aturan v2 | YOLOv8n-seg (tau {tau}). Latar abu 114.", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.imwrite(str(REP / "fase1_montase_20.jpg"), np.vstack([judul] + baris), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(json.dumps(ring, indent=1, ensure_ascii=False))


UKUR = r'''
import sys, time, json, statistics as st, psutil, os, cv2
sys.path.insert(0, sys.argv[1])
import pemisah_daun_seg as pd
p = psutil.Process()
r0 = p.memory_info().rss
m = pd.OnnxSeg(sys.argv[2], threads=int(sys.argv[4]) if len(sys.argv) > 4 else None)
r1 = p.memory_info().rss
fs = json.loads(open(sys.argv[3], encoding="utf-8").read())
puncak, t = r1, []
for i, f in enumerate(fs):
    bgr = cv2.imread(f)
    a = time.perf_counter()
    masks, confs = m.prediksi(bgr)
    M, c = pd.pilih_daun(bgr, masks, confs, float(sys.argv[5]))
    if i:
        t.append(1000 * (time.perf_counter() - a))
    puncak = max(puncak, p.memory_info().rss)
print(json.dumps({"rss_awal_MB": round(r0 / 2**20, 1), "rss_model_dimuat_MB": round(r1 / 2**20, 1), "rss_puncak_MB": round(puncak / 2**20, 1),
                  "ms_median": round(st.median(t), 1), "ms_maks": round(max(t), 1), "n": len(t)}))
'''


def onnx():
    from ultralytics import YOLO
    tau = json.loads((REP / "fase1_pilih_tau.json").read_text(encoding="utf-8"))["tau_terpilih"]
    op = RUN / "weights/best.onnx"
    if not op.exists():
        YOLO(str(BOBOT)).export(format="onnx", imgsz=640, opset=12, simplify=True)
    ukuran = op.stat().st_size / 2**20
    # kesetaraan ONNX (numpy) vs .pt pada 20 foto VAL (acak, seed tetap)
    rng = random.Random(7)
    fs = rng.sample(sorted(r["file"] for r in umum.manifest_val()), 20)
    model = YOLO(str(BOBOT))
    om = pd.OnnxSeg(op)
    ious, cad_sama = [], 0
    for f in fs:
        bgr = cv2.imread(str(umum.VAL_IMG / f))
        Mp, cp = pd.pilih_daun(bgr, *pd.prediksi_pt(model, bgr), tau)
        Mo, co = pd.pilih_daun(bgr, *om.prediksi(bgr), tau)
        ious.append(iou(Mp, Mo))
        cad_sama += int(cp == co)
    daftar = KERJA / "daftar_ukur.json"
    jalur = [str(umum.VAL_IMG / f) for f in rng.sample(sorted(r["file"] for r in umum.manifest_val()), 21)]
    daftar.write_text(json.dumps(jalur), encoding="utf-8")
    skrip = KERJA / "_ukur_onnx.py"
    skrip.write_text(UKUR, encoding="utf-8")
    ukur = {}
    for th in (None, 2):
        args = [sys.executable, str(skrip), str(Path(__file__).parent), str(op), str(daftar), str(th or 0), str(tau)]
        if th is None:
            args[5] = "0"
        out = subprocess.run(args, capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1"))
        ukur["semua inti" if th is None else f"{th} thread"] = json.loads(out.stdout.strip().splitlines()[-1]) if out.returncode == 0 else out.stderr[-500:]
    res = {"tau": tau, "onnx": str(op.relative_to(ROOT)), "ukuran_MB": round(ukuran, 2),
           "kesetaraan_onnx_vs_pt_20_val": {"iou_rata2": round(float(np.mean(ious)), 4), "iou_min": round(float(min(ious)), 4), "cadangan_sama": f"{cad_sama}/20"},
           "cpu": ukur, "catatan": "waktu = letterbox + inferensi ONNX Runtime CPU + pasca-proses numpy + aturan pemilih daun (tanpa baca berkas); 20 foto VAL setelah 1 pemanasan; proses baru tanpa GPU"}
    (REP / "fase1_onnx.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    {"statistik": statistik, "pilih_tau": pilih_tau, "akhir": akhir, "onnx": onnx}[sys.argv[1]]()
