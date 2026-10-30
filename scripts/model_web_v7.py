#!/usr/bin/env python3
"""
model_web_v7.py - Langkah setelah training model web v7_deploy_yolo11s_s2 (last.pt; disetujui dospem 2026-10-13).

  python scripts/model_web_v7.py kunci     # tulis ambang_diagnosis.json = 0,35 (DITETAPKAN, bukan dipilih) untuk last.pt
  python scripts/model_web_v7.py evaluasi  # SEKALI: test_new_session dan ds3 uji utama (ambang 0,35) -> pembanding
  python scripts/model_web_v7.py banding   # tabel berdampingan dengan v7_final_prov_yolo11s_s2 (hasil yang sudah ada)
  python scripts/model_web_v7.py onnx      # ekspor ONNX + bukti kesamaan prediksi (50 gambar val) + kecepatan CPU

Label semua keluaran: PEMBANDING, bukan hasil utama paper. Keluaran: reports/final_eval_v7/aplikasi/
"""
import csv
import hashlib
import json
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
RUN = "v7_deploy_yolo11s_s2"
FINAL = "v7_final_prov_yolo11s_s2"
W = ROOT / "runs" / RUN / "weights" / "last.pt"
OUT = ROOT / "reports/final_eval_v7/aplikasi"
DATA = "data/processed/prov_v7/data.yaml"
UTAMA = "data/annotation/verifikasi_ds3/uji_utama_final/ds3_uji_utama.txt"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def kunci():
    rs = json.loads((ROOT / "runs" / RUN / "run_stats.json").read_text(encoding="utf-8"))
    if len(rs.get("epoch", [])) < 63:
        sys.exit("[GAGAL] training model web belum 63 epoch")
    k = {"ambang": 0.35, "akurasi_val_saat_dipilih": None,
         "dipilih_pada": "DITETAPKAN (tidak dipilih ulang; tidak ada val): ambang terkunci model web v7_final_prov_yolo11s_s2, keputusan 2026-10-13",
         "kriteria": "ditetapkan sebelum evaluasi", "weights": str(W), "weights_sha256": sha(W), "dikunci": datetime.now().isoformat(timespec="seconds"),
         "catatan": "checkpoint = last.pt (63 epoch); hasil evaluasi = PEMBANDING, bukan hasil utama paper"}
    (ROOT / "runs" / RUN / "ambang_diagnosis.json").write_text(json.dumps(k, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(k, indent=1, ensure_ascii=False))


def evaluasi():
    OUT.mkdir(parents=True, exist_ok=True)
    tugas = [("test_new_session", ["--split", "test_new_session", "--izinkan-test", "--tanpa-benchmark"], "eval_test_new_session/ringkasan.json"),
             ("ds3_utama", ["--set-folder", "data/raw/ds3", "--izinkan-uji-eksternal", "--peta-folder", "blight=bacterial_leaf_blight", "blast=leaf_blast",
                            "--daftar", UTAMA, "--out", f"runs/{RUN}/eval_ds3_utama"], "eval_ds3_utama/ringkasan_diagnosis.json")]
    for nama, args, jejak in tugas:
        if (ROOT / "runs" / RUN / jejak).exists():
            print(f"{nama}: sudah dievaluasi, dilewati")
            continue
        cmd = [PY, "scripts/evaluate.py", "--weights", str(W.relative_to(ROOT)), "--data", DATA] + args
        with open(ROOT / "runs" / RUN / f"eval_{nama}.log", "w", encoding="utf-8") as f:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
        print(f"{nama}: kode {rc}")
        if rc:
            sys.exit(f"[GAGAL] {nama}")


def ringkas_run(run):
    t = json.loads((ROOT / "runs" / run / "eval_test_new_session/ringkasan.json").read_text(encoding="utf-8"))
    m = {(r["subset"], r["kelas"]): r for r in t["metrik"]}
    d = json.loads((ROOT / "runs" / run / "eval_ds3_utama/ringkasan_diagnosis.json").read_text(encoding="utf-8"))["dalam_cakupan"]
    pk = {r["kelas"]: r for r in d["per_kelas"]}
    return {"run": run, "sesi_baru_mAP50": m[("dengan_background", "SEMUA")]["mAP50"], "sesi_baru_mAP50_95": m[("dengan_background", "SEMUA")]["mAP50_95"],
            "sesi_baru_akurasi_diagnosis": t["diagnosis_per_citra"]["subset"]["dengan_background"]["akurasi"],
            "ds3_utama_akurasi": d["akurasi"], "ds3_utama_recall_BLB": pk["bacterial_leaf_blight"]["recall"], "ds3_utama_recall_Blast": pk["leaf_blast"]["recall"],
            "ds3_utama_akurasi_per_kelompok": d.get("akurasi_rata2_per_kelompok")}


def banding():
    rows = [dict(ringkas_run(FINAL), model="final (split bersih, best.pt dari val bersih, 100 epoch)"),
            dict(ringkas_run(RUN), model="model web (train+val+test, last.pt, 63 epoch)")]
    for r in rows:
        r["LABEL"] = "PEMBANDING, bukan hasil utama paper"
    with open(OUT / "model_web_vs_final.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(rows, indent=1, ensure_ascii=False))


def onnx():
    import numpy as np
    from ultralytics import YOLO
    OUT.mkdir(parents=True, exist_ok=True)
    m = YOLO(str(W))
    f_onnx = Path(m.export(format="onnx", imgsz=640, simplify=True, dynamic=False, device="cpu"))
    val = sorted((ROOT / "data/processed/prov_v7/images/val").iterdir())
    pilih = random.Random(0).sample(val, 50)
    mpt, mox = YOLO(str(W)), YOLO(str(f_onnx), task="detect")
    rows, waktu = [], {"pt": [], "onnx": []}
    for nama, mod in (("pt", mpt), ("onnx", mox)):
        for p in pilih[:10]:                                   # warmup
            mod.predict(str(p), imgsz=640, conf=0.001, iou=0.7, device="cpu", verbose=False)
    hasil = {}
    for nama, mod in (("pt", mpt), ("onnx", mox)):
        hasil[nama] = []
        for p in pilih:
            t0 = time.perf_counter()
            r = mod.predict(str(p), imgsz=640, conf=0.001, iou=0.7, device="cpu", verbose=False)[0]
            waktu[nama].append(1000 * (time.perf_counter() - t0))
            c = r.boxes.conf.cpu().numpy()
            hasil[nama].append((int(r.boxes.cls[int(c.argmax())].item()), float(c.max())) if len(c) else (-1, 0.0))
    for p, a, b in zip(pilih, hasil["pt"], hasil["onnx"]):
        rows.append({"gambar": p.name, "kelas_pt": a[0], "conf_pt": round(a[1], 5), "kelas_onnx": b[0], "conf_onnx": round(b[1], 5),
                     "kelas_identik": a[0] == b[0], "selisih_conf": round(abs(a[1] - b[1]), 5)})
    with open(OUT / "onnx_vs_pt_50_val.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    import onnxruntime
    res = {"model": RUN, "checkpoint": "last.pt", "sha256_pt": sha(W), "onnx": str(f_onnx.relative_to(ROOT)), "sha256_onnx": sha(f_onnx),
           "ukuran_MB": {"pt": round(W.stat().st_size / 2 ** 20, 2), "onnx": round(f_onnx.stat().st_size / 2 ** 20, 2)},
           "versi": {"onnxruntime": onnxruntime.__version__, "onnx": __import__("onnx").__version__},
           "n_gambar": len(rows), "kelas_teratas_identik": sum(r["kelas_identik"] for r in rows),
           "selisih_conf_maks": max(r["selisih_conf"] for r in rows), "selisih_conf_rata2": round(statistics.mean(r["selisih_conf"] for r in rows), 6),
           "LOLOS": all(r["kelas_identik"] for r in rows) and max(r["selisih_conf"] for r in rows) < 0.01,
           "cpu_ms_median": {k: round(statistics.median(v), 1) for k, v in waktu.items()}, "cpu_ms_rata2": {k: round(statistics.mean(v), 1) for k, v in waktu.items()},
           "catatan": "keduanya di CPU, batch 1, imgsz 640, conf 0,001, iou 0,7; 50 gambar val prov_v7 acak (seed 0), warmup 10; waktu = predict ujung ke ujung termasuk baca berkas"}
    (OUT / "onnx_bukti.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    {"kunci": kunci, "evaluasi": evaluasi, "banding": banding, "onnx": onnx}[sys.argv[1]]()
