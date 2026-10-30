# Uji kesetaraan pemisah daun (aplikasi vs evaluasi skripsi)

Semua 1259 foto VAL prov_v7 asli. Acuan: masker YOLOv8n-seg fase 1 yang dipakai di evaluasi alur aplikasi (tau 0,35).
Jalur aplikasi: `gambar.baca_foto` (PIL, koreksi EXIF, sisi maks. 2048 px) lalu `core.pemisah_daun.PemisahDaun.pisahkan`.

| Ukuran | Hasil |
|---|---|
| Keputusan cadangan (foto ulang) sama | 1255/1259 |
| Foto cadangan: aplikasi / acuan | 86 / 86 |
| IoU masker (foto bukan cadangan, n = 1171) | median 0.99504; minimum 0.04076; 86.51% >= 0,99; 97.78% >= 0,98 |
| Waktu pisahkan per foto (median, CPU) | 102.6 ms |

Foto yang berbeda: `ds1_provisional_r6k-train-blight-orig-023-jpg-rf-d8780d1_ef18f774.jpg`, `ds1_provisional_r6k-train-blight-orig-044-jpg-rf-2d7d29d_33e90cb4.jpg`, `ds1_provisional_r6k-train-img-blb-66-jpeg-jpg-rf-d1bd621_001af78a.jpg`, `ds2_Healthy-rice-leaf-62_d88b6613.jpg`.
