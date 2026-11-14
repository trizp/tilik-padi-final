# Split acak multi-seed (revisi paper SOFTT, Tabel II baris Random)

Definisi sama dengan `reports/final_eval_v7/tabel/tabel10_ketepatan_perkiraan.csv`: test acak tanpa BLB (subset bersih) vs test sesi baru (dengan background). Rata-rata ± simpangan baku sampel atas seed.

## Per run

| Run | Ambang | mAP test acak | mAP sesi baru | Selisih mAP | Akurasi test acak | Akurasi sesi baru | Selisih akurasi | Kenaikan relatif |
|---|---|---|---|---|---|---|---|---|
| v7_randsplit_prov_yolov8n_s0 | 0.29 | 0.3898 | 0.2712 | 0.1186 | 0.9264 | 0.8194 | 0.1070 | 43.7% |
| v7_randsplit_prov_yolov8n_s1 | 0.2 | 0.3851 | 0.2730 | 0.1121 | 0.9114 | 0.7860 | 0.1255 | 41.1% |
| v7_randsplit_prov_yolov8n_s2 | 0.27 | 0.3917 | 0.2701 | 0.1216 | 0.9234 | 0.7759 | 0.1475 | 45.0% |
| v7_randsplit_prov_yolo11n_s0 | 0.23 | 0.3895 | 0.2834 | 0.1061 | 0.9204 | 0.8662 | 0.0542 | 37.4% |
| v7_randsplit_prov_yolo11n_s1 | 0.16 | 0.3992 | 0.2871 | 0.1121 | 0.9144 | 0.8495 | 0.0649 | 39.0% |
| v7_randsplit_prov_yolo11n_s2 | 0.19 | 0.3901 | 0.2787 | 0.1114 | 0.9054 | 0.8161 | 0.0894 | 40.0% |

## Ringkasan per model

| Model | Seed | n_seed | mAP50-95 test acak tanpa BLB | mAP50-95 sesi baru | Selisih mAP50-95 | Akurasi test acak tanpa BLB | Akurasi sesi baru | Selisih akurasi | Kenaikan relatif mAP50-95 (rata-rata per seed) | Kenaikan relatif mAP50-95 (dari rata-rata) |
|---|---|---|---|---|---|---|---|---|---|---|
| YOLOv8n | 0,1,2 | 3 | 0.389 ± 0.003 | 0.271 ± 0.001 | 0.117 ± 0.005 | 0.920 ± 0.008 | 0.794 ± 0.023 | 0.127 ± 0.020 | 43.3 ± 2.0% | 43.3% |
| YOLO11n | 0,1,2 | 3 | 0.393 ± 0.005 | 0.283 ± 0.004 | 0.110 ± 0.003 | 0.913 ± 0.008 | 0.844 ± 0.026 | 0.069 ± 0.018 | 38.8 ± 1.3% | 38.8% |

## Baris LaTeX (Tabel II)

```
Random & YOLOv8n & 0.117 $\pm$ 0.005 & 0.127 $\pm$ 0.020 \\
Random & YOLO11n & 0.110 $\pm$ 0.003 & 0.069 $\pm$ 0.018 \\
```
