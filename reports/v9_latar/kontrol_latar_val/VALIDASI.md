# Validasi kontrol latar alternatif (ANALISIS VAL POST-HOC untuk revisi paper)

Dijalankan 13 November 2026 dengan `scripts/paper/kontrol_latar_val.py` (pengaturan bawaan, setelah perbaikan di bawah). Hanya foto
VALIDASI bersih; test, test_new_session, dan ds3 tidak disentuh; tidak ada model yang dilatih. Skrip pemeriksa ada di `validasi/`
(hanya membaca berkas tersimpan; tidak menjalankan model).

## Perubahan pada skrip (sekecil mungkin)

1. `conf` prediksi 0,05 -> 0,001, sama dengan evaluasi resmi (`scripts/evaluate.py` baris 239: `conf=0.001, iou=0.7`, imgsz dari
   checkpoint = 640). Diagnosis tidak terpengaruh (semua ambang >= 0,27), tetapi kolom `conf` kini sebanding dengan hasil resmi.
2. Bawaan `--tol-abu` 6 -> 20. Foto v9 disimpan JPEG kualitas 95 (`scripts/bangun_prov_v9.py` baris 116), jadi abu-abu 114 di dekat
   tepi daun bergeser. Diukur terhadap masker SAM2 asli pada 120 foto val (`validasi/cek_tol_abu.py`):

   | tol_abu | latar tertinggal (rata-rata) | daun ikut dianggap latar | IoU latar (min) |
   |---|---|---|---|
   | 6 | 0,254% | 0,0000% | 0,9933 |
   | 10 | 0,185% | 0,0002% | 0,9943 |
   | 14 | 0,128% | 0,0009% | 0,9951 |
   | 20 | 0,073% | 0,0043% | 0,9958 |
   | 30 | 0,027% | 0,0185% | 0,9962 |

   20 dipilih: sisa latar tinggal 0,07% tanpa mengikis daun berarti; di atas itu daun mulai ikut terhapus.
3. `--limit N` mengambil sampel berjarak rata, bukan N foto pertama. Daftar terurut nama dimulai dengan foto sehat ds2, sehingga
   `--limit 20` lama hanya berisi satu kelas dan `statistics.stdev` gagal pada recall NaN. Jalan penuh tidak terpengaruh.

Yang sudah sesuai tanpa perubahan: `data_valsel.yaml` v7 dan v9 (`val: images/val`, salinan 869 dan 861 foto), konvensi label
`images/` -> `labels/`, nama kelas, bobot `runs/v7_final_prov_*/weights/best.pt`, dan pembacaan ambang dari `kunci_model.json`
(cocok dengan `kunci_model.csv` untuk keenam run: 0,38/0,39/0,37 YOLOv8s; 0,39/0,27/0,35 YOLO11s).

## Hasil pemeriksaan

- Foto: 869 val bersih v7; 861 berpasangan dengan v9; 8 tanpa pasangan (semuanya blas ds4: blast-175, 729, 730, 731, 733, 734, 735,
  736; dikeluarkan dari prov_v9 karena semua kotaknya < 50% di dalam masker daun). Sebaran: blas 214, bercak cokelat 330, sehat 317.
- Foto cadangan (masker kosong, v9 = foto asli): 16 (13 sehat, 2 blas, 1 bercak cokelat). Pada foto ini kelima kondisi = foto asli.
- Selisih abu-abu dari masker vs foto v9: rata-rata 0,14 (skala 0--255).
- **original vs evaluasi val resmi** (`runs/*/eval_val/diagnosis_bersih_per_citra.csv`): diagnosis identik pada 861/861 foto untuk
  keenam run. Akurasi lebih rendah 0,0011--0,0012 dari val resmi (869) karena 8 foto blas tanpa pasangan, yang di val resmi semuanya
  benar, tidak ikut.
- **gray vs tahap 1 v9** (`reports/v9_latar/tahap1_per_run.csv`, 869 foto, masker SAM2): bila 8 foto tanpa pasangan dihitung salah,
  jumlah benar/869 cocok PERSIS untuk keenam run (mis. YOLOv8s s0 704/869 = 0,8101; YOLO11s s2 558/869 = 0,6421).
- **black vs acuan hitam seed 0** (`tahap1_isian_hitam_s_seed0.json`): skrip melaporkan COCOK (0,7573 vs 0,7491; 0,6690 vs 0,6617),
  tetapi selisihnya hampir menyentuh batas 0,01. Penyebab: 8 foto tanpa pasangan. Dihitung pada 869 dengan 8 foto itu salah:
  652/869 = 0,7503 vs 0,7491 dan 576/869 = 0,6628 vs 0,6617, sisa selisih tepat 1 foto per model, sesuai dengan 16 foto cadangan
  yang di sini tetap asli (15/16 benar) sedangkan di acuan latarnya ditutup dengan masker SAM2.
- **Keyakinan teratas**: |selisih| <= 0,001 pada 841--843 dari 861 foto per run. Selisih > 0,01 hanya pada 7--10 foto ds2 yang
  tidak persegi, semuanya di bawah ambang (diagnosis sama). Dugaan penyebab: bantalan letterbox. Evaluasi resmi mem-batch 16 jalur
  berukuran campuran (dipad ke 640x640), skrip mem-batch 5 varian berukuran sama (bantalan minimal).
- `contoh_kondisi.jpg` (6 baris x 5 kolom: original, gray, black, blur, swap): daun utuh dan latar terganti benar di kelima kondisi.
  Catatan: satu bintik sisa latar sangat kecil di kolom black baris 5; kondisi blur menyisakan pendar warna daun di latar dekat tepi
  daun (latar asli diburamkan bersama daunnya); 4 baris terbawah adalah foto beruntun yang sangat mirip (6 foto pertama berpenyakit
  menurut urutan nama).

## Catatan penafsiran

- Akurasi di sini dihitung pada 861 foto, bukan 869; 8 foto blas yang tidak punya versi tanpa latar tidak ikut.
- Kondisi swap memakai seluruh foto daun sehat lain sebagai latar (termasuk daun pada foto donor), dipilih acak dengan seed 0 dari
  kelompok foto yang berbeda.
- Ini analisis VALIDASI post-hoc: angka tidak dipakai untuk memilih model atau ambang.
