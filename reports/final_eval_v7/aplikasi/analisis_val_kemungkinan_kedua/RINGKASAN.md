# Analisis val: kelas benar sebagai kemungkinan kedua

**ANALISIS VAL, BUKAN HASIL UTAMA.** VAL prov_v7 (1259 citra), model final 3 seed,
ambang terkunci tiap run; t_rendah dengan prosedur tiga tingkat yang sama per run. Test, test sesi baru, dan ds3 tidak dipakai.

| Run | Ambang terdeteksi | t_rendah | Kotak teratas beda dari eval_val |
|---|---|---|---|
| v7_final_prov_yolo11s_s0 | 0.39 | 0.39 | 0/1259 |
| v7_final_prov_yolo11s_s1 | 0.27 | 0.27 | 0/1259 |
| v7_final_prov_yolo11s_s2 | 0.35 | 0.29 | 0/1259 |
| v7_final_prov_yolov8s_s0 | 0.38 | 0.34 | 0/1259 |
| v7_final_prov_yolov8s_s1 | 0.39 | 0.37 | 0/1259 |
| v7_final_prov_yolov8s_s2 | 0.37 | 0.37 | 0/1259 |

## Kelas benar pada kasus salah diagnosis (gabungan 3 seed)

| Model | Kasus | Kasus / foto kelas | Kelas benar >= t_rendah | Keyakinan kelas benar: median [Q1; Q3] (min-maks) | Tanpa kotak kelas benar |
|---|---|---|---|---|---|
| yolo11s | Blast -> Brown spot | 77 / 666 | 21/77 (27.3%) | 0.1964 [0.0367; 0.3067] (0.0-0.4638) | 13.0% |
| yolo11s | Blast -> BLB | 0 / 666 | - | - | - |
| yolo11s | Blast -> tidak terdeteksi | 18 / 666 | 2/18 (11.1%) | 0.1927 [0.106; 0.2876] (0.0355-0.3467) | 0.0% |
| yolo11s | Brown spot -> Blast | 84 / 990 | 15/84 (17.9%) | 0.1059 [0.012; 0.2556] (0.0-0.5621) | 6.0% |
| yolov8s | Blast -> Brown spot | 72 / 666 | 38/72 (52.8%) | 0.3759 [0.1315; 0.4833] (0.0-0.6491) | 5.6% |
| yolov8s | Blast -> BLB | 1 / 666 | 0/1 (0.0%) | 0.2175 [0.2175; 0.2175] (0.2175-0.2175) | 0.0% |
| yolov8s | Blast -> tidak terdeteksi | 20 / 666 | 3/20 (15.0%) | 0.3197 [0.2467; 0.3568] (0.0184-0.3894) | 0.0% |
| yolov8s | Brown spot -> Blast | 86 / 990 | 44/86 (51.2%) | 0.3628 [0.1817; 0.5073] (0.0017-0.6658) | 0.0% |

Catatan: pada kasus Blast -> tidak terdeteksi, kotak teratas < ambang; kelas benar >= t_rendah berarti foto itu sudah berada di tingkat
"kemungkinan" (bukan kemungkinan kedua).

## Sebaran keyakinan kelas benar per seed

- yolo11s|leaf_blast -> brown_spot, seed 0 (n=28): [0.00, 0.05): 6; [0.05, 0.10): 2; [0.10, 0.20): 6; [0.20, 0.39): 8; [0.39, 1.00): 6
- yolo11s|leaf_blast -> brown_spot, seed 1 (n=33): [0.00, 0.05): 11; [0.05, 0.10): 1; [0.10, 0.20): 3; [0.20, 0.27): 5; [0.27, 1.00): 13
- yolo11s|leaf_blast -> brown_spot, seed 2 (n=16): [0.00, 0.05): 5; [0.05, 0.10): 1; [0.10, 0.20): 5; [0.20, 0.29): 3; [0.29, 0.35): 0; [0.35, 1.00): 2
- yolo11s|leaf_blast -> sehat_atau_tidak_terdeteksi, seed 0 (n=6): [0.00, 0.05): 1; [0.05, 0.10): 1; [0.10, 0.20): 2; [0.20, 0.39): 2; [0.39, 1.00): 0
- yolo11s|leaf_blast -> sehat_atau_tidak_terdeteksi, seed 1 (n=1): [0.00, 0.05): 0; [0.05, 0.10): 0; [0.10, 0.20): 0; [0.20, 0.27): 1; [0.27, 1.00): 0
- yolo11s|leaf_blast -> sehat_atau_tidak_terdeteksi, seed 2 (n=11): [0.00, 0.05): 1; [0.05, 0.10): 2; [0.10, 0.20): 3; [0.20, 0.29): 3; [0.29, 0.35): 2; [0.35, 1.00): 0
- yolo11s|brown_spot -> leaf_blast, seed 0 (n=28): [0.00, 0.05): 13; [0.05, 0.10): 3; [0.10, 0.20): 2; [0.20, 0.39): 7; [0.39, 1.00): 3
- yolo11s|brown_spot -> leaf_blast, seed 1 (n=27): [0.00, 0.05): 8; [0.05, 0.10): 5; [0.10, 0.20): 5; [0.20, 0.27): 3; [0.27, 1.00): 6
- yolo11s|brown_spot -> leaf_blast, seed 2 (n=29): [0.00, 0.05): 11; [0.05, 0.10): 2; [0.10, 0.20): 4; [0.20, 0.29): 6; [0.29, 0.35): 4; [0.35, 1.00): 2
- yolov8s|leaf_blast -> brown_spot, seed 0 (n=27): [0.00, 0.05): 5; [0.05, 0.10): 2; [0.10, 0.20): 0; [0.20, 0.34): 4; [0.34, 0.38): 1; [0.38, 1.00): 15
- yolov8s|leaf_blast -> brown_spot, seed 1 (n=25): [0.00, 0.05): 6; [0.05, 0.10): 1; [0.10, 0.20): 0; [0.20, 0.37): 4; [0.37, 0.39): 2; [0.39, 1.00): 12
- yolov8s|leaf_blast -> brown_spot, seed 2 (n=20): [0.00, 0.05): 3; [0.05, 0.10): 1; [0.10, 0.20): 3; [0.20, 0.37): 5; [0.37, 1.00): 8
- yolov8s|leaf_blast -> bacterial_leaf_blight, seed 0 (n=1): [0.00, 0.05): 0; [0.05, 0.10): 0; [0.10, 0.20): 0; [0.20, 0.34): 1; [0.34, 0.38): 0; [0.38, 1.00): 0
- yolov8s|leaf_blast -> sehat_atau_tidak_terdeteksi, seed 0 (n=1): [0.00, 0.05): 0; [0.05, 0.10): 0; [0.10, 0.20): 0; [0.20, 0.34): 1; [0.34, 0.38): 0; [0.38, 1.00): 0
- yolov8s|leaf_blast -> sehat_atau_tidak_terdeteksi, seed 1 (n=15): [0.00, 0.05): 0; [0.05, 0.10): 1; [0.10, 0.20): 1; [0.20, 0.37): 10; [0.37, 0.39): 3; [0.39, 1.00): 0
- yolov8s|leaf_blast -> sehat_atau_tidak_terdeteksi, seed 2 (n=4): [0.00, 0.05): 1; [0.05, 0.10): 0; [0.10, 0.20): 0; [0.20, 0.37): 3; [0.37, 1.00): 0
- yolov8s|brown_spot -> leaf_blast, seed 0 (n=27): [0.00, 0.05): 3; [0.05, 0.10): 1; [0.10, 0.20): 2; [0.20, 0.34): 6; [0.34, 0.38): 2; [0.38, 1.00): 13
- yolov8s|brown_spot -> leaf_blast, seed 1 (n=29): [0.00, 0.05): 1; [0.05, 0.10): 2; [0.10, 0.20): 6; [0.20, 0.37): 8; [0.37, 0.39): 1; [0.39, 1.00): 11
- yolov8s|brown_spot -> leaf_blast, seed 2 (n=30): [0.00, 0.05): 5; [0.05, 0.10): 1; [0.10, 0.20): 2; [0.20, 0.37): 5; [0.37, 1.00): 17

## Seberapa sering "kemungkinan lain" muncul pada diagnosis yang benar (biaya tampilan)

| Model | Kelompok | n (3 seed) | Muncul kemungkinan lain |
|---|---|---|---|
| yolo11s | leaf_blast benar | 571 | 39 (6.8%) |
| yolo11s | brown_spot benar | 876 | 12 (1.4%) |
| yolo11s | bacterial_leaf_blight benar | 905 | 18 (2.0%) |
| yolo11s | sehat salah terdeteksi | 104 | 2 (1.9%) |
| yolov8s | leaf_blast benar | 573 | 105 (18.3%) |
| yolov8s | brown_spot benar | 872 | 61 (7.0%) |
| yolov8s | bacterial_leaf_blight benar | 888 | 55 (6.2%) |
| yolov8s | sehat salah terdeteksi | 115 | 7 (6.1%) |

## Yang akan tampil di aplikasi pada kasus salah diagnosis

"Kemungkinan lain" menampilkan SATU penyakit lain dengan keyakinan tertinggi (>= t_rendah); bisa saja bukan kelas benar.

| Model | Kasus | Kemungkinan lain = kelas benar |
|---|---|---|
| yolo11s | Blast -> Brown spot | 21/77 (27.3%) |
| yolo11s | Blast -> BLB | 0/0 |
| yolo11s | Brown spot -> Blast | 15/84 (17.9%) |
| yolov8s | Blast -> Brown spot | 38/72 (52.8%) |
| yolov8s | Blast -> BLB | 0/1 (0.0%) |
| yolov8s | Brown spot -> Blast | 44/86 (51.2%) |

## Bercak berkeyakinan rendah (tidak terdeteksi, kotak teratas >= 0,10)

| Model | Kelompok | Foto (3 seed) | Tidak terdeteksi | Bercak rendah tampil | % dari tidak terdeteksi | % dari semua foto kelompok |
|---|---|---|---|---|---|---|
| yolo11s | daun sehat | 951 | 847 | 155 | 18.3% | 16.3% |
| yolo11s | daun sakit | 2826 | 284 | 205 (kelas bercak = kelas benar: 167) | 72.2% | 7.3% |
| yolov8s | daun sehat | 951 | 836 | 215 | 25.7% | 22.6% |
| yolov8s | daun sakit | 2826 | 307 | 253 (kelas bercak = kelas benar: 200) | 82.4% | 9.0% |

Sumber: per_kasus.csv (satu baris per citra per run); ringkasan.json. Skrip: scripts/analisis_kemungkinan_kedua_v7.py.
