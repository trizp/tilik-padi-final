# Kualitas pemisahan daun pada ds3 (sebelum model dijalankan)

Gambar saja; model deteksi belum dijalankan pada ds3. SAM2 aturan v2 + cadangan (sama dengan prov_v9).

| Himpunan | Foto | Cadangan (tidak dipisah) | Median luas masker (yang dipisah) |
|---|---|---|---|
| utama | 70 | 2 (2.9%) | 8.1% foto |
| tambahan | 160 | 2 (1.2%) | 8.1% foto |
| tungro | 80 | 4 (5.0%) | 7.3% foto |

| Folder | Foto | % cadangan | Median luas masker |
|---|---|---|---|
| blast | 80 | 2.5% | 8.2% foto |
| blight | 80 | 0.0% | 8.0% foto |
| tungro | 80 | 5.0% | 7.3% foto |

Montase 20 contoh: `montase_ds3_20.jpg`.

Penilaian visual montase (20 contoh: 7 blast, 7 blight, 6 tungro): 20/20 daun utama terpisah baik (daun utuh, latar kertas terbuang);
tidak ada contoh cadangan di montase. Latar kertas polos jauh lebih mudah dipisah daripada foto lapangan.
Ditulis 2026-10-17T21:20, SEBELUM model deteksi apa pun dijalankan pada ds3 tanpa latar.
