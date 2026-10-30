#!/usr/bin/env python3
"""
evaluasi_ds3_v7.py - Evaluasi ds3 (uji eksternal, level citra) untuk 12 model final v7 + 2 model randsplit v7.
Setiap himpunan dievaluasi SEKALI (evaluate.py menolak evaluasi ulang). Ambang = ambang terkunci masing-masing model.
Himpunan (dibekukan 2026-10-13 sebelum model dijalankan pada ds3, data/annotation/verifikasi_ds3/uji_utama_final/):
  utama     : ds3_uji_utama.txt (70 foto, "sesuai" menurut kedua penilai)
  tambahan  : ds3_tambahan_160.txt (semua 160 foto blight + blast, label asli ds3)
  tungro    : ds3_tungro_luar_cakupan.txt (80 foto, di luar cakupan: sebaran diagnosis)
Keluaran per run: runs/<run>/eval_ds3_<himpunan>/ ; log: reports/final_eval_v7/ds3/evaluasi_ds3.log
"""
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
OUT = ROOT / "reports/final_eval_v7/ds3"
UD = "data/annotation/verifikasi_ds3/uji_utama_final"
HIMPUNAN = {"utama": f"{UD}/ds3_uji_utama.txt", "tambahan": f"{UD}/ds3_tambahan_160.txt", "tungro": f"{UD}/ds3_tungro_luar_cakupan.txt"}
PETA = ["blight=bacterial_leaf_blight", "blast=leaf_blast", "tungro=luar_cakupan"]
RUNS = ([(f"v7_final_prov_{m}_s{s}", "data/processed/prov_v7/data.yaml") for m in ("yolov8n", "yolo11n", "yolov8s", "yolo11s") for s in (0, 1, 2)] +
        [(f"v7_randsplit_prov_{m}_s0", "data/processed/prov_v7_randsplit/data.yaml") for m in ("yolov8n", "yolo11n")])


def log(msg):
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with open(OUT / "evaluasi_ds3.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for run, data in RUNS:
        for nama, daftar in HIMPUNAN.items():
            out = ROOT / "runs" / run / f"eval_ds3_{nama}"
            if (out / "ringkasan_diagnosis.json").exists():
                log(f"{run} {nama}: sudah dievaluasi, dilewati")
                continue
            args = [PY, "scripts/evaluate.py", "--weights", f"runs/{run}/weights/best.pt", "--data", data, "--set-folder", "data/raw/ds3",
                    "--izinkan-uji-eksternal", "--peta-folder"] + PETA + ["--daftar", daftar, "--out", str(out.relative_to(ROOT))]
            with open(ROOT / "runs" / run / f"eval_ds3_{nama}.log", "w", encoding="utf-8") as f:
                rc = subprocess.run(args, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
            log(f"{run} {nama}: selesai kode {rc}")
            if rc != 0:
                sys.exit(f"[GAGAL] {run} {nama}")
    log("evaluasi ds3 selesai")


if __name__ == "__main__":
    main()
