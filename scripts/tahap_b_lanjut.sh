#!/usr/bin/env bash
# Penjaga Tahap B: tunggu scripts/tahap_b.sh selesai, lalu lanjutkan (resume dari last.pt) run yang belum 150 epoch.
# Berurutan (tidak pernah dua training bersamaan). Maks 8 kali lanjut per model. Hyperparameter tidak diubah.
set -u
PY=.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8
until grep -q "selesai yolo11n" reports/prov_v6/tahap_b_driver.log 2>/dev/null; do sleep 60; done
n_epoch() { $PY -c "import json,sys;p=sys.argv[1];print(len({json.loads(l)['epoch'] for l in open(p,encoding='utf-8') if l.strip()}))" "$1/epoch_stats.jsonl" 2>/dev/null || echo 0; }
for M in yolov8n yolo11n; do
  RUN=runs/prov_v6_${M}_s0
  LOG=reports/prov_v6/tahap_b_logs/prov_v6_${M}_s0.log
  for TRY in 1 2 3 4 5 6 7 8; do
    N=$(n_epoch "$RUN")
    [ "$N" -ge 150 ] && break
    [ -f "$RUN/weights/last.pt" ] || { echo "== $M: last.pt tidak ada"; break; }
    echo "== penjaga: $M baru $N epoch, lanjut dari last.pt (percobaan penjaga $TRY) $(date -Iseconds)"
    $PY scripts/train.py --resume "$RUN/weights/last.pt" >> "$LOG" 2>&1
    echo "== penjaga: $M exit $? $(date -Iseconds)"
  done
  echo "== penjaga: $M akhir $(n_epoch "$RUN") epoch $(date -Iseconds)"
done
echo "== penjaga selesai $(date -Iseconds)"
