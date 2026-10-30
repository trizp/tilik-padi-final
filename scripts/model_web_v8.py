#!/usr/bin/env python3
"""
model_web_v8.py - Langkah setelah training model web YOLOv8 v7_deploy_yolov8s_s2 (last.pt, 60 epoch; KEPUTUSAN 2026-10-15).
Alasan: kesesuaian dengan judul dan rumusan masalah proposal (YOLOv8), BUKAN kinerja. Semua hasil = PEMBANDING, bukan hasil utama.

  python scripts/model_web_v8.py kunci     # ambang_diagnosis.json = 0,37 (DITETAPKAN = ambang terkunci v7_final_prov_yolov8s_s2)
  python scripts/model_web_v8.py evaluasi  # SEKALI: test_new_session, ds3 uji utama, Tungro ds3 (ambang 0,37)
  python scripts/model_web_v8.py banding   # model web YOLOv8s vs model web YOLO11s (masing-masing dengan ambangnya)
  python scripts/model_web_v8.py onnx      # ekspor ONNX, bersihkan metadata, kesetaraan ONNX vs .pt (20 gambar val), kecepatan CPU

Keluaran: reports/final_eval_v7/aplikasi/*_yolov8s.* (berkas model web YOLO11s tidak ditimpa)
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
RUN, FINAL, EPOCH, AMBANG = "v7_deploy_yolov8s_s2", "v7_final_prov_yolov8s_s2", 60, 0.37
RUN11 = "v7_deploy_yolo11s_s2"
W = ROOT / "runs" / RUN / "weights" / "last.pt"
OUT = ROOT / "reports/final_eval_v7/aplikasi"
DATA = "data/processed/prov_v7/data.yaml"
UD = "data/annotation/verifikasi_ds3/uji_utama_final"
LABEL = "PEMBANDING, bukan hasil utama paper"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def kunci():
    rs = json.loads((ROOT / "runs" / RUN / "run_stats.json").read_text(encoding="utf-8"))
    if len(rs.get("epoch", [])) < EPOCH:
        sys.exit(f"[GAGAL] training model web belum {EPOCH} epoch")
    fin = json.loads((ROOT / "runs" / FINAL / "ambang_diagnosis.json").read_text(encoding="utf-8"))
    if abs(float(fin["ambang"]) - AMBANG) > 1e-9:
        sys.exit(f"[GAGAL] ambang {FINAL} = {fin['ambang']}, bukan {AMBANG}")
    k = {"ambang": AMBANG, "akurasi_val_saat_dipilih": None,
         "dipilih_pada": f"DITETAPKAN (tidak dipilih ulang; tidak ada val): ambang terkunci {FINAL}, keputusan 2026-10-15",
         "kriteria": "ditetapkan sebelum evaluasi", "weights": str(W), "weights_sha256": sha(W), "dikunci": datetime.now().isoformat(timespec="seconds"),
         "catatan": f"checkpoint = last.pt ({EPOCH} epoch); alasan model web YOLOv8 = kesesuaian judul, bukan kinerja; hasil = {LABEL}"}
    p = ROOT / "runs" / RUN / "ambang_diagnosis.json"
    if p.exists():
        sys.exit(f"[DITOLAK] {p} sudah ada")
    p.write_text(json.dumps(k, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(k, indent=1, ensure_ascii=False))


def evaluasi():
    tugas = [("test_new_session", ["--split", "test_new_session", "--izinkan-test", "--tanpa-benchmark"], "eval_test_new_session/ringkasan.json"),
             ("ds3_utama", ["--set-folder", "data/raw/ds3", "--izinkan-uji-eksternal", "--peta-folder", "blight=bacterial_leaf_blight", "blast=leaf_blast",
                            "--daftar", f"{UD}/ds3_uji_utama.txt", "--out", f"runs/{RUN}/eval_ds3_utama"],
              "eval_ds3_utama/ringkasan_diagnosis.json"),
             ("ds3_tungro", ["--set-folder", "data/raw/ds3", "--izinkan-uji-eksternal", "--peta-folder", "blight=bacterial_leaf_blight", "blast=leaf_blast",
                             "tungro=luar_cakupan", "--daftar", f"{UD}/ds3_tungro_luar_cakupan.txt", "--out", f"runs/{RUN}/eval_ds3_tungro"],
              "eval_ds3_tungro/ringkasan_diagnosis.json")]
    if not (ROOT / "runs" / RUN / "ambang_diagnosis.json").exists():
        sys.exit("[GAGAL] jalankan 'kunci' dulu")
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


def ringkas(run, tungro_dir):
    t = json.loads((ROOT / "runs" / run / "eval_test_new_session/ringkasan.json").read_text(encoding="utf-8"))
    m = {(r["subset"], r["kelas"]): r for r in t["metrik"]}
    d = json.loads((ROOT / "runs" / run / "eval_ds3_utama/ringkasan_diagnosis.json").read_text(encoding="utf-8"))["dalam_cakupan"]
    pk = {r["kelas"]: r for r in d["per_kelas"]}
    tg = json.loads((ROOT / "runs" / run / tungro_dir / "ringkasan_diagnosis.json").read_text(encoding="utf-8"))["luar_cakupan"]
    sb = tg["sebaran_diagnosis"]
    amb = json.loads((ROOT / "runs" / run / "ambang_diagnosis.json").read_text(encoding="utf-8"))["ambang"]
    return {"run": run, "ambang": amb,
            "sesi_baru_mAP50": m[("dengan_background", "SEMUA")]["mAP50"], "sesi_baru_mAP50_95": m[("dengan_background", "SEMUA")]["mAP50_95"],
            "sesi_baru_akurasi_diagnosis": t["diagnosis_per_citra"]["subset"]["dengan_background"]["akurasi"],
            "ds3_utama_akurasi": d["akurasi"], "ds3_utama_recall_BLB": pk["bacterial_leaf_blight"]["recall"],
            "ds3_utama_recall_Blast": pk["leaf_blast"]["recall"], "ds3_utama_akurasi_per_kelompok": d.get("akurasi_rata2_per_kelompok"),
            "tungro_n": tg["n_citra"], "tungro_didiagnosis_BLB": sb.get("bacterial_leaf_blight", 0),
            "tungro_persen_BLB": round(100 * sb.get("bacterial_leaf_blight", 0) / tg["n_citra"], 1), "tungro_sebaran": json.dumps(sb, ensure_ascii=False)}


def banding():
    rows = [dict(ringkas(RUN11, "eval_ds3_tungro_posthoc"), model="model web YOLO11s (aplikasi sekarang)"),
            dict(ringkas(RUN, "eval_ds3_tungro"), model="model web YOLOv8s (disiapkan; kesesuaian judul)")]
    for r in rows:
        r["LABEL"] = LABEL
    p = OUT / "model_web_yolov8s_vs_yolo11s.csv"
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(rows, indent=1, ensure_ascii=False))


def onnx():
    import onnx as onnxlib
    import onnxruntime
    from ultralytics import YOLO
    m = YOLO(str(W))
    f_onnx = Path(m.export(format="onnx", imgsz=640, simplify=True, dynamic=False, device="cpu"))
    # bersihkan metadata: jalur lokal dan nama aplikasi tidak boleh ada (graf dan bobot tidak diubah)
    mo = onnxlib.load(str(f_onnx))
    g0 = mo.graph.SerializeToString()
    for p in mo.metadata_props:
        if p.key == "description":
            p.value = "Ultralytics YOLOv8s model trained on prov_v7_deploy"
    onnxlib.save(mo, str(f_onnx))
    assert onnxlib.load(str(f_onnx)).graph.SerializeToString() == g0
    sisa = [p.key for p in onnxlib.load(str(f_onnx)).metadata_props if ":\\" in p.value or "TELKOM" in p.value or "Tilik" in p.value]
    val = sorted((ROOT / "data/processed/prov_v7/images/val").iterdir())
    pilih = random.Random(0).sample(val, 20)
    mpt, mox = YOLO(str(W)), YOLO(str(f_onnx), task="detect")
    for mod in (mpt, mox):
        for p in pilih[:5]:
            mod.predict(str(p), imgsz=640, conf=0.001, iou=0.7, device="cpu", verbose=False)
    hasil, waktu = {}, {"pt": [], "onnx": []}
    for nama, mod in (("pt", mpt), ("onnx", mox)):
        hasil[nama] = []
        for p in pilih:
            t0 = time.perf_counter()
            r = mod.predict(str(p), imgsz=640, conf=0.001, iou=0.7, device="cpu", verbose=False)[0]
            waktu[nama].append(1000 * (time.perf_counter() - t0))
            c = r.boxes.conf.cpu().numpy()
            hasil[nama].append((int(r.boxes.cls[int(c.argmax())].item()), float(c.max())) if len(c) else (-1, 0.0))
    rows = [{"gambar": p.name, "kelas_pt": a[0], "conf_pt": round(a[1], 5), "kelas_onnx": b[0], "conf_onnx": round(b[1], 5),
             "kelas_identik": a[0] == b[0], "selisih_conf": round(abs(a[1] - b[1]), 5)} for p, a, b in zip(pilih, hasil["pt"], hasil["onnx"])]
    with open(OUT / "onnx_vs_pt_20_val_yolov8s.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    res = {"model": RUN, "checkpoint": "last.pt", "sha256_pt": sha(W), "onnx": str(f_onnx.relative_to(ROOT)), "sha256_onnx": sha(f_onnx),
           "metadata_description": "Ultralytics YOLOv8s model trained on prov_v7_deploy", "metadata_berisi_jalur_atau_nama_aplikasi": sisa,
           "ukuran_MB": {"pt": round(W.stat().st_size / 2 ** 20, 2), "onnx": round(f_onnx.stat().st_size / 2 ** 20, 2)},
           "versi": {"onnxruntime": onnxruntime.__version__, "onnx": onnxlib.__version__},
           "n_gambar": len(rows), "kelas_teratas_identik": sum(r["kelas_identik"] for r in rows),
           "selisih_conf_maks": max(r["selisih_conf"] for r in rows),
           "LOLOS": all(r["kelas_identik"] for r in rows) and max(r["selisih_conf"] for r in rows) < 0.01,
           "cpu_ms_median": {k: round(statistics.median(v), 1) for k, v in waktu.items()},
           "catatan": "CPU, batch 1, imgsz 640, conf 0,001, iou 0,7; 20 gambar val prov_v7 acak (seed 0), warmup 5; graf ONNX identik sebelum/sesudah ganti metadata"}
    (OUT / "onnx_bukti_yolov8s.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    {"kunci": kunci, "evaluasi": evaluasi, "banding": banding, "onnx": onnx}[sys.argv[1]]()
