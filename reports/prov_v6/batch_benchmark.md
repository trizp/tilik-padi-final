# Pengukuran batch (Tahap A)

1 epoch, imgsz 640, AMP, mosaic menyala (close_mosaic=0 hanya untuk pengukuran). `gpu_terpakai_puncak` = memori SELURUH GPU (termasuk proses lain dan konteks CUDA) dari NVML.

| model | batch | batch aktual | VRAM PyTorch (GB) | GPU terpakai puncak (GB) | sisa min (GB) | detik epoch 1 | detik train stabil + val | 100 epoch (jam) | exit |
|---|---|---|---|---|---|---|---|---|---|
| yolov8n | 8 | 8 | 1.309 | 2.197 | 1.803 | 94.0 | 86.4 | 2.4 | 0 |
| yolo11n | 8 | 8 | 1.574 | 2.474 | 1.526 | 102.1 | 91.3 | 2.54 | 0 |
| yolov8s | 8 | 8 | 2.551 | 3.427 | 0.573 | 145.2 | 137.6 | 3.82 | 0 |
| yolo11s | 8 | 8 | 2.168 | 3.068 | 0.932 | 148.3 | 138.2 | 3.84 | 0 |
| yolov8n | 16 | 16 | 2.25 | 3.097 | 0.903 | 86.4 | 78.9 | 2.19 | 0 |
| yolo11n | 16 | 16 | 2.408 | 3.302 | 0.698 | 90.9 | 81.8 | 2.27 | 0 |
| yolov8s | 16 | 16 | 3.785 | 3.943 | 0.057 | 244.6 | 239.1 | 6.64 | 0 |
| yolo11s | 16 | 16 | 4.02 | 3.934 | 0.066 | 354.7 | 340.2 | 9.45 | 0 |
