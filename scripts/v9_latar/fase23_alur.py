#!/usr/bin/env python3
"""
fase23_alur.py - Fase 2 dan 3 rencana aplikasi v9 (KEPUTUSAN_PROYEK.md 2026-10-20), HANYA VAL prov_v9.

  fase2 : alur baru (masker YOLOv8n-seg dari fase 1; cadangan = foto asli) + model final YOLOv8s v9 seed 0-2 (best.pt, ambang terkunci v9)
          vs alur SAM2 (gambar prov_v9 val = hasil yang ada; dihitung ulang dengan kode yang sama dan dicek terhadap eval_val tersimpan).
          Label = label prov_v9 yang sama. mAP50-95 dan akurasi diagnosis pada subset bersih (tanpa ds1_provisional); recall diagnosis Blast.
          Kriteria LAYAK (dicatat sebelum dijalankan): penurunan akurasi rata-rata 3 seed <= 2 poin DAN penurunan recall Blast <= 3 poin.
  fase3 : foto yang menjadi cadangan pada alur baru (subset bersih): (v9) model v9 pada foto asli [perilaku fase 2], (a) model v7 YOLOv8s
          (ambang terkunci v7) pada foto asli, (b) tidak didiagnosis (foto ulang). Akurasi keseluruhan, akurasi pada cadangan, cakupan;
          RAM ONNX Runtime CPU untuk satu vs dua detektor + pemisah daun (proses baru).
Keluaran: reports/v9_latar/aplikasi_v9/fase2_*.json/md, fase3_*.json/md
"""
import csv
import json
import os
import shutil
import statistics as S
import subprocess
import sys
from pathlib import Path

os.environ["YOLO_AUTOINSTALL"] = "false"
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import umum  # noqa: E402

ROOT = umum.ROOT
V9 = ROOT / "data/processed/prov_v9"
SEG = umum.KERJA / "seg_n_val"
ALUR = umum.KERJA / "alur_seg_val"
EV = umum.KERJA / "eval_fase2"
REP = ROOT / "reports/v9_latar/aplikasi_v9"
NAMES = {0: "leaf_blast", 1: "brown_spot", 2: "bacterial_leaf_blight"}
TAK = "sehat_atau_tidak_terdeteksi"
RUN9 = [f"v9_final_prov_yolov8s_s{s}" for s in (0, 1, 2)]
RUN7 = [f"v7_final_prov_yolov8s_s{s}" for s in (0, 1, 2)]


def val9():
    return [r for r in csv.DictReader(open(V9 / "manifest.csv", encoding="utf-8")) if r["split"] == "val"]


def label_citra(lbl):
    cl = {int(b.split()[0]) for b in lbl.read_text().split("\n") if b.strip()} if lbl.exists() else set()
    if not cl:
        return TAK
    return NAMES[cl.pop()] if len(cl) == 1 else None


def bangun_alur():
    if ALUR.exists():
        return
    cad = json.loads((SEG / "info.json").read_text(encoding="utf-8"))["cadangan"]
    (ALUR / "images").mkdir(parents=True)
    for r in val9():
        f = r["file"]
        src = ROOT / "data/processed/prov_v7/images/val" / f
        if cad[f]:
            os.link(src, ALUR / "images" / f)
        else:
            bgr = cv2.imread(str(src))
            M = cv2.imread(str(SEG / "masker" / (Path(f).stem + ".png")), cv2.IMREAD_GRAYSCALE) > 0
            bgr[~M] = umum.ABU[::-1]
            cv2.imwrite(str(ALUR / "images" / f), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])


def subset(nama, img_dir, pilih):
    d = EV / nama
    if d.exists():
        shutil.rmtree(d)
    (d / "images").mkdir(parents=True)
    (d / "labels").mkdir(parents=True)
    for r in val9():
        if pilih(r):
            f = r["file"]
            os.link(img_dir / f, d / "images" / f)
            lbl = V9 / "labels/val" / (Path(f).stem + ".txt")
            if lbl.exists():
                shutil.copyfile(lbl, d / "labels" / lbl.name)
    y = EV / f"_{nama}.yaml"
    y.write_text(yaml.safe_dump({"path": str(d), "train": "images", "val": "images", "names": NAMES}), encoding="utf-8")
    return y, d


def diagnosa(model, d, files, amb):
    out = {}
    for i in range(0, len(files), 16):
        part = files[i:i + 16]
        for f, res in zip(part, model.predict([str(d / "images" / f) for f in part], imgsz=640, conf=0.001, iou=0.7, device=0, verbose=False)):
            cf = res.boxes.conf.cpu().numpy()
            out[f] = NAMES[int(res.boxes.cls[int(cf.argmax())].item())] if len(cf) and cf.max() >= amb else TAK
    return out


def metrik_diag(lab, dg, files):
    fs = [f for f in files if lab[f] is not None]
    acc = sum(dg[f] == lab[f] for f in fs) / len(fs)
    bl = [f for f in fs if lab[f] == "leaf_blast"]
    return acc, sum(dg[f] == "leaf_blast" for f in bl) / len(bl), len(fs)


def fase2():
    from ultralytics import YOLO
    REP.mkdir(parents=True, exist_ok=True)
    bangun_alur()
    bersih = lambda r: r["source"] != "ds1_provisional"  # noqa: E731
    blb = lambda r: r["source"] == "ds1_provisional"  # noqa: E731
    sub = {"sam2_bersih": subset("sam2_bersih", V9 / "images/val", bersih), "seg_bersih": subset("seg_bersih", ALUR / "images", bersih),
           "sam2_blb": subset("sam2_blb", V9 / "images/val", blb), "seg_blb": subset("seg_blb", ALUR / "images", blb)}
    lab = {r["file"]: label_citra(V9 / "labels/val" / (Path(r["file"]).stem + ".txt")) for r in val9()}
    fb = sorted(r["file"] for r in val9() if bersih(r))
    hasil, diag_simpan = [], {}
    for run in RUN9:
        amb = json.loads((ROOT / "runs" / run / "ambang_diagnosis.json").read_text(encoding="utf-8"))
        model = YOLO(str(ROOT / "runs" / run / "weights/best.pt"))
        b = {"run": run, "ambang": amb["ambang"], "akurasi_val_tersimpan": amb["akurasi_val_saat_dipilih"]}
        for nama, (y, d) in sub.items():
            r = model.val(data=str(y), split="val", imgsz=640, batch=8, device=0, conf=0.001, iou=0.7, plots=False,
                          project=str(EV / "_val"), name=f"{run}_{nama}", exist_ok=True, verbose=False)
            b[f"{nama}_mAP50"], b[f"{nama}_mAP50_95"] = round(float(r.box.map50), 4), round(float(r.box.map), 4)
        for alur in ("sam2", "seg"):
            dg = diagnosa(model, sub[f"{alur}_bersih"][1], fb, amb["ambang"])
            diag_simpan[f"{run}|{alur}"] = dg
            acc, rb, n = metrik_diag(lab, dg, fb)
            b[f"{alur}_akurasi"], b[f"{alur}_recall_blast"], b["n_diagnosis"] = round(acc, 4), round(rb, 4), n
        ref = next(m for m in json.loads((ROOT / "runs" / run / "eval_val/ringkasan.json").read_text(encoding="utf-8"))["metrik"]
                   if m["subset"] == "bersih" and m["kelas"] == "SEMUA")
        b["reproduksi_mAP50_95_tersimpan"] = ref["mAP50_95"]
        hasil.append(b)
        print(b, flush=True)
    json.dump({"diag": diag_simpan, "label": lab}, open(EV / "diag_fase2.json", "w"))
    m = lambda k: float(np.mean([h[k] for h in hasil]))  # noqa: E731
    d_acc, d_bl = m("sam2_akurasi") - m("seg_akurasi"), m("sam2_recall_blast") - m("seg_recall_blast")
    ring = {"kriteria": "LAYAK bila penurunan akurasi diagnosis rata-rata 3 seed <= 0,02 DAN penurunan recall Blast <= 0,03 (alur SAM2 - alur seg)",
            "per_run": hasil,
            "rata2": {k: round(m(k), 4) for k in ("sam2_bersih_mAP50_95", "seg_bersih_mAP50_95", "sam2_akurasi", "seg_akurasi", "sam2_recall_blast",
                                                   "seg_recall_blast", "sam2_blb_mAP50_95", "seg_blb_mAP50_95")},
            "penurunan_akurasi": round(d_acc, 4), "penurunan_recall_blast": round(d_bl, 4),
            "penurunan_mAP50_95": round(m("sam2_bersih_mAP50_95") - m("seg_bersih_mAP50_95"), 4),
            "reproduksi_selisih_maks": {"mAP50_95": round(max(abs(h["sam2_bersih_mAP50_95"] - h["reproduksi_mAP50_95_tersimpan"]) for h in hasil), 4),
                                        "akurasi": round(max(abs(h["sam2_akurasi"] - h["akurasi_val_tersimpan"]) for h in hasil), 4)},
            "LAYAK": bool(d_acc <= 0.02 and d_bl <= 0.03)}
    (REP / "fase2_ringkasan.json").write_text(json.dumps(ring, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(ring, indent=1, ensure_ascii=False))


RAM = r'''
import sys, json, psutil, numpy as np, onnxruntime as ort
p = psutil.Process(); r0 = p.memory_info().rss
so = ort.SessionOptions(); so.enable_cpu_mem_arena = False
ses = [ort.InferenceSession(m, so, providers=["CPUExecutionProvider"]) for m in sys.argv[1:]]
r1 = p.memory_info().rss; puncak = r1
for s in ses:
    i = s.get_inputs()[0]
    s.run(None, {i.name: np.random.rand(1, 3, 640, 640).astype(np.float32)})
    puncak = max(puncak, p.memory_info().rss)
print(json.dumps({"model": len(ses), "rss_awal_MB": round(r0/2**20,1), "rss_dimuat_MB": round(r1/2**20,1), "rss_setelah_inferensi_MB": round(p.memory_info().rss/2**20,1), "rss_puncak_MB": round(puncak/2**20,1)}))
'''


def fase3():
    from ultralytics import YOLO
    D = json.load(open(EV / "diag_fase2.json"))
    lab = D["label"]
    cad = json.loads((SEG / "info.json").read_text(encoding="utf-8"))["cadangan"]
    fb = sorted(r["file"] for r in val9() if r["source"] != "ds1_provisional" and lab[r["file"]] is not None)
    fc = [f for f in fb if cad[f]]
    hasil = []
    for s, (r9, r7) in enumerate(zip(RUN9, RUN7)):
        dg9 = D["diag"][f"{r9}|seg"]                      # alur baru; cadangan = model v9 pada foto asli
        amb7 = json.loads((ROOT / "runs" / r7 / "ambang_diagnosis.json").read_text(encoding="utf-8"))["ambang"]
        m7 = YOLO(str(ROOT / "runs" / r7 / "weights/best.pt"))
        dg7 = {}
        for i in range(0, len(fc), 16):
            part = fc[i:i + 16]
            for f, res in zip(part, m7.predict([str(umum.VAL_IMG / f) for f in part], imgsz=640, conf=0.001, iou=0.7, device=0, verbose=False)):
                cf = res.boxes.conf.cpu().numpy()
                dg7[f] = NAMES[int(res.boxes.cls[int(cf.argmax())].item())] if len(cf) and cf.max() >= amb7 else TAK
        ok = lambda dgc: sum((dgc[f] if f in dgc else dg9[f]) == lab[f] for f in fb) / len(fb)  # noqa: E731
        benar_c9 = sum(dg9[f] == lab[f] for f in fc)
        benar_c7 = sum(dg7[f] == lab[f] for f in fc)
        benar_non = sum(dg9[f] == lab[f] for f in fb if not cad[f])
        hasil.append({"seed": s, "n_bersih": len(fb), "n_cadangan": len(fc),
                      "akurasi_cadangan_v9_foto_asli": round(benar_c9 / len(fc), 4) if fc else None,
                      "akurasi_cadangan_a_v7": round(benar_c7 / len(fc), 4) if fc else None,
                      "akurasi_total_v9_foto_asli": round(ok({}), 4), "akurasi_total_a_v7": round(ok(dg7), 4),
                      "akurasi_total_b_tidak_didiagnosis": round(benar_non / len(fb), 4),
                      "akurasi_b_pada_yang_didiagnosis": round(benar_non / (len(fb) - len(fc)), 4), "cakupan_b": round((len(fb) - len(fc)) / len(fb), 4)})
    rata = {k: round(float(np.mean([h[k] for h in hasil])), 4) for k in hasil[0] if k not in ("seed", "n_bersih", "n_cadangan") and hasil[0][k] is not None}
    per_kelas_cad = {}
    for f in fc:
        per_kelas_cad[lab[f]] = per_kelas_cad.get(lab[f], 0) + 1
    # RAM ONNX Runtime CPU (proses baru): pemisah daun + 1 atau 2 detektor YOLOv8s (ONNX deploy v7 sebagai pengganti berukuran sama)
    seg = ROOT / "runs/v9_pemisah_daun_yolov8n_seg/weights/best.onnx"
    det = ROOT / "app/models/v7_deploy_yolov8s_s2.onnx"          # hanya dibaca; arsitektur YOLOv8s sama dengan model v9
    sk = EV / "_ram.py"
    sk.write_text(RAM, encoding="utf-8")
    ram = {}
    for nama, ms in (("pemisah daun + 1 detektor", [seg, det]), ("pemisah daun + 2 detektor", [seg, det, det]), ("1 detektor (aplikasi v7)", [det])):
        o = subprocess.run([sys.executable, str(sk)] + [str(x) for x in ms], capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1"))
        ram[nama] = json.loads(o.stdout.strip().splitlines()[-1]) if o.returncode == 0 else o.stderr[-400:]
    ring = {"per_seed": hasil, "rata2_3_seed": rata, "cadangan_per_kelas_label": per_kelas_cad, "ram_onnx_cpu": ram,
            "ukuran_berkas_MB": {"pemisah_daun_onnx": round(seg.stat().st_size / 2**20, 2), "detektor_yolov8s_onnx": round(det.stat().st_size / 2**20, 2)},
            "catatan": "subset bersih VAL prov_v9 (tanpa ds1_provisional); foto multi-kelas tidak dihitung; RAM = RSS proses Python + ONNX Runtime saja (tanpa Streamlit)"}
    (REP / "fase3_ringkasan.json").write_text(json.dumps(ring, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(ring, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    {"fase2": fase2, "fase3": fase3}[sys.argv[1]]()
