#!/usr/bin/env bash
# Tahap B - kurva latihan: yolov8n dan yolo11n, seed 0, 150 epoch, batch 8, patience 0 (tanpa early stopping),
# close_mosaic baku (10), hyperparameter identik (HP di train.py). Bila run terputus, dilanjutkan dari last.pt (maks 2x).
# Hanya train dan val yang dipakai. Jalankan dari folder rice_yolo:  bash scripts/tahap_b.sh
set -u
PY=.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
mkdir -p reports/prov_v6/tahap_b_logs
for M in yolov8n yolo11n; do
  RUN=runs/prov_v6_${M}_s0
  LOG=reports/prov_v6/tahap_b_logs/prov_v6_${M}_s0.log
  echo "== mulai $M $(date -Iseconds)"
  $PY scripts/train.py --data data/processed/prov_v6/data.yaml --model ${M}.pt --epochs 150 --seed 0 --batch 8 --patience 0 >> "$LOG" 2>&1
  RC=$?
  for TRY in 1 2; do
    [ $RC -eq 0 ] && break
    [ -f "$RUN/weights/last.pt" ] || break
    echo "== $M gagal (exit $RC), lanjut dari last.pt (percobaan $TRY) $(date -Iseconds)"
    $PY scripts/train.py --resume "$RUN/weights/last.pt" >> "$LOG" 2>&1
    RC=$?
  done
  echo "== selesai $M exit $RC $(date -Iseconds)"
done
