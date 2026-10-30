# Batas tampilan bercak berkeyakinan rendah (v7_final_prov_yolov8s_s2)

**Parameter tampilan dari VAL v7, bukan ambang diagnosis.** Ambang terdeteksi 0.37; 369 dari 1259 foto val tidak terdeteksi.

Aturan: batas terendah dalam [0,10; ambang) dengan proporsi sehat <= 25% di antara foto tidak terdeteksi yang menampilkan bercak berkeyakinan rendah.

**Batas terpilih: 0.35**

| Batas | Sehat | Sakit (kelas bercak benar) | Proporsi sehat |
|---|---|---|---|
| 0.10 | 67 | 68 (57) | 49.6% |
| 0.11 | 62 | 65 (54) | 48.8% |
| 0.12 | 58 | 62 (51) | 48.3% |
| 0.13 | 55 | 62 (51) | 47.0% |
| 0.14 | 55 | 57 (47) | 49.1% |
| 0.15 | 47 | 56 (46) | 45.6% |
| 0.16 | 46 | 55 (45) | 45.5% |
| 0.17 | 44 | 53 (43) | 45.4% |
| 0.18 | 44 | 50 (40) | 46.8% |
| 0.19 | 39 | 49 (39) | 44.3% |
| 0.20 | 36 | 49 (39) | 42.4% |
| 0.21 | 36 | 48 (38) | 42.9% |
| 0.22 | 35 | 45 (36) | 43.8% |
| 0.23 | 33 | 44 (35) | 42.9% |
| 0.24 | 27 | 44 (35) | 38.0% |
| 0.25 | 23 | 43 (34) | 34.8% |
| 0.26 | 20 | 42 (33) | 32.3% |
| 0.27 | 17 | 40 (31) | 29.8% |
| 0.28 | 15 | 38 (29) | 28.3% |
| 0.29 | 13 | 36 (27) | 26.5% |
| 0.30 | 10 | 29 (21) | 25.6% |
| 0.31 | 9 | 24 (18) | 27.3% |
| 0.32 | 9 | 23 (17) | 28.1% |
| 0.33 | 8 | 16 (11) | 33.3% |
| 0.34 | 7 | 13 (8) | 35.0% |
| 0.35 | 3 | 11 (7) | 21.4% <- terpilih |
| 0.36 | 1 | 6 (5) | 14.3% |
