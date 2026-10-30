#!/usr/bin/env python3
"""
latih_pemisah_daun.py - Latih YOLOv8n-seg satu kelas "daun" (distilasi SAM2; KEPUTUSAN_PROYEK.md 2026-10-20, fase 1). Satu run, resep standar
ultralytics: yolov8n-seg.pt, 100 epoch, imgsz 640, batch 8 (GPU 4 GB), workers 2, seed 0. Cek memori virtual sama dengan train.py.
Bila terputus dan last.pt ada, jalankan ulang skrip ini: dilanjutkan (resume).

  .venv/Scripts/python scripts/v9_latar/latih_pemisah_daun.py
Keluaran: runs/v9_pemisah_daun_yolov8n_seg/
"""
import os
import sys
from pathlib import Path

os.environ["YOLO_AUTOINSTALL"] = "false"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from train import cek_memori_virtual  # noqa: E402

NAMA = "v9_pemisah_daun_yolov8n_seg"
DATA = ROOT / "data/derived/v9_daun_seg/data.yaml"


def main():
    from ultralytics import YOLO
    cek_memori_virtual()
    rd = ROOT / "runs" / NAMA
    last = rd / "weights" / "last.pt"
    if last.exists():
        YOLO(str(last)).train(resume=True)
        return
    if rd.exists():
        sys.exit(f"[DITOLAK] {rd} ada tanpa last.pt; periksa dulu")
    os.chdir(ROOT / "tools/_model_latar")           # unduhan yolov8n-seg.pt ke tools/_model_latar
    YOLO("yolov8n-seg.pt").train(data=str(DATA), epochs=100, imgsz=640, batch=8, workers=2, seed=0, deterministic=True,
                                 project=str(ROOT / "runs"), name=NAMA, exist_ok=False, plots=True, device=0)


if __name__ == "__main__":
    main()
