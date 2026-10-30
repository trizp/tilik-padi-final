# Tahap 1: kepekaan model final v7 terhadap penghapusan latar (ANALISIS VAL, tanpa training)

VAL prov_v7; latar diganti abu-abu (114,114,114) dengan SAM2 aturan v2 (tanpa label). 4 model x 3 seed, best.pt dan ambang
diagnosis terkunci. Bersih = tanpa ds1_provisional (BLB), seperti tabel utama. Rata-rata ± simpangan baku 3 seed.

Uji reproduksi VAL asli vs eval_val tersimpan: selisih maks. mAP50-95 bersih 0.0000; akurasi diagnosis 0.0012.

| Model | mAP50 bersih: asli / tanpa latar / tanpa latar + buang-kotak | mAP50-95 bersih: asli / tanpa / tanpa + buang | mAP50-95 dengan BLB prov.: asli / tanpa | Akurasi diagnosis bersih: asli / tanpa latar |
|---|---|---|---|---|
| yolov8n | 0,557 ± 0,012 / 0,419 ± 0,012 / 0,439 ± 0,013 | 0,228 ± 0,005 / 0,158 ± 0,007 / 0,166 ± 0,007 | 0,216 ± 0,005 / 0,159 ± 0,007 | 0,860 ± 0,005 / 0,777 ± 0,012 |
| yolo11n | 0,596 ± 0,004 / 0,352 ± 0,030 / 0,368 ± 0,030 | 0,246 ± 0,001 / 0,114 ± 0,011 / 0,119 ± 0,011 | 0,230 ± 0,003 / 0,130 ± 0,012 | 0,865 ± 0,013 / 0,742 ± 0,032 |
| yolov8s | 0,564 ± 0,002 / 0,464 ± 0,013 / 0,486 ± 0,014 | 0,232 ± 0,001 / 0,176 ± 0,007 / 0,184 ± 0,007 | 0,214 ± 0,002 / 0,162 ± 0,009 | 0,875 ± 0,006 / 0,807 ± 0,008 |
| yolo11s | 0,599 ± 0,007 / 0,334 ± 0,025 / 0,350 ± 0,026 | 0,247 ± 0,001 / 0,113 ± 0,014 / 0,118 ± 0,015 | 0,230 ± 0,007 / 0,132 ± 0,011 | 0,880 ± 0,007 / 0,697 ± 0,047 |

Selisih rata-rata (tanpa latar - asli), subset bersih:

- yolov8n: mAP50-95 -0,0691; akurasi diagnosis -0,0832
- yolo11n: mAP50-95 -0,1318; akurasi diagnosis -0,1235
- yolov8s: mAP50-95 -0,0555; akurasi diagnosis -0,0675
- yolo11s: mAP50-95 -0,1339; akurasi diagnosis -0,1834

Catatan: model final dilatih dengan latar asli; foto tanpa latar disimpan ulang sebagai JPEG q95. Label sama persis kecuali kolom
"buang-kotak". Sumber: tahap1_per_run.csv.
