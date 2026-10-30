# Ringkasan hasil evaluasi akhir tahap 1 (PROVISIONAL)

Tanggal: 10 Oktober 2026. Dataset prov_v6 (BLB masih dari ds1, provisional). ds3 belum dievaluasi.
Ambang diagnosis dikunci di val sebelum test dibuka (lihat KEPUTUSAN_PROYEK.md, 9 Okt 23.13). Setiap set uji dievaluasi sekali.

## PERINGATAN UTAMA: label kotak ds4 tidak cocok dengan sebagian foto

Setelah test dibuka ditemukan bahwa sebagian foto di folder "Original images" ds4 **terputar 90°** dibanding gambar yang
dipakai penulis ds4 saat membuat label. Kotak label tetap benar untuk gambar versi penulis, tetapi pada foto yang kita pakai
kotaknya jatuh di tempat yang salah (sering di rumput atau di samping bercak). Diperiksa otomatis untuk semua foto ds4
(`reports/prov_v6/audit_orientasi_ds4/`, pemisahan sangat jelas):

| split | foto ds4 berlabel | kotak tidak cocok |
|---|---|---|
| train | 2.313 | 458 (19,8%) |
| val | 552 | 182 (33,0%) |
| test | 276 | 89 (32,2%) |
| test sesi baru | 243 | 202 (83,1%) |

Akibatnya:
- **Semua angka mAP/AP/P/R/F1 tingkat kotak pada foto ds4 terlalu rendah dan tidak sah**, terutama di test sesi baru
  (mAP50-95 hanya 0,006–0,011 karena 83% labelnya bergeser; model sebenarnya menandai bercak dengan benar di contoh yang dilihat).
- Model juga dilatih dengan ~20% label kotak ds4 yang salah tempat.
- **Diagnosis per citra relatif tidak terpengaruh pada sisi penilaian** (label citra = kelas kotak, dan kelasnya benar), tetapi
  modelnya tetap dilatih dengan label yang sebagian salah.
- Hasil ini **tidak layak untuk tabel paper**. Tidak ada model, ambang, atau dataset yang diubah; keputusan perbaikan
  (dataset versi baru dengan orientasi diperbaiki dan pelatihan ulang) ada di tangan pemilik skripsi.

## Temuan (dengan catatan di atas)

1. **Diagnosis per citra (paling dapat dipercaya):** pada test (semua) akurasi 0,70–0,75. Terbaik YOLO11s 0,750 ± 0,012
   (IK 95% 0,670–0,824). BLB provisional mudah dikenali (F1 0,94–0,96), tetapi ini dari ds1 yang mirip satu sama lain.
   Kesalahan terbesar: daun sehat didiagnosis brown spot (42–47% foto sehat pada model terbaik tiap keluarga) dan blast
   tertukar dengan brown spot.
2. **YOLO11 vs YOLOv8:** pada varian **s**, YOLO11s lebih baik daripada YOLOv8s di test: mAP50-95 +0,025 (IK 95% +0,006 s.d.
   +0,041) dan akurasi diagnosis +0,046 (+0,022 s.d. +0,071); interval tidak memuat nol. Pada varian **n** tidak ada beda yang
   meyakinkan di test (interval memuat nol). Di test sesi baru, akurasi diagnosis YOLO11 lebih tinggi (+0,14 untuk n,
   +0,17 untuk s; interval tidak memuat nol).
3. **Split acak terlihat jauh lebih bagus padahal bocor:** model split acak mendapat mAP50-95 0,45 di test acaknya sendiri,
   sekitar 3× model split bersih di test bersih (0,14–0,15). Minimal 77% gambar val/test acak punya kembaran langsung di train
   (batas atas 92%). Pada test sesi baru (satu-satunya set yang sebanding) akurasi diagnosis split acak 0,52 (n v8) dan 0,65
   (n v11) vs split bersih 0,42 dan 0,62: tidak terlihat keunggulan split bersih di set ini (satu seed; kotak test sesi baru
   bermasalah seperti di atas).
4. **Penyakit di luar cakupan tidak ditolak:** foto Leaf smut 82–91% dan Tungro 82–87% tetap didiagnosis salah satu dari tiga
   penyakit (paling sering brown spot); Sheath blight 26–42%. Ini akibat ambang diagnosis rendah (0,04–0,12) dan model
   tidak pernah belajar kelas "lainnya".
5. **False positive pada daun sehat (test):** pada conf 0,25, 28–34% gambar sehat mendapat setidaknya satu kotak; pada conf
   0,5 turun menjadi 6–11%.
6. **Kecepatan (laptop ini, batch 1):** GPU RTX 3050 Ti 7,5–12 ms per gambar (84–137 FPS); CPU 34–92 ms (11–30 FPS).
   Varian n ~2,5× lebih cepat di CPU daripada s. YOLO11 memiliki parameter dan GFLOPs lebih kecil (n: 2,6 M / 6,4; s: 9,4 M / 21,4)
   daripada YOLOv8 (n: 3,0 M / 8,1; s: 11,1 M / 28,4).

## Keterbatasan lain

- Interval kepercayaan lebar karena unit sampel adalah KELOMPOK foto: test 125 kelompok, BLB test hanya 21 kelompok.
- BLB masih provisional (ds1, anotasi area, banyak augmentasi); angka BLB tidak boleh dibandingkan dengan kelas lain.
- Ambang diagnosis dipilih di val bersih yang tidak memuat BLB.
- Contoh kualitatif dibuat dengan memprediksi ulang 8 gambar test hanya untuk digambar (bukan evaluasi ulang).
- Grafik contoh kualitatif: judul baris kedua sedikit menimpa foto baris pertama (kosmetik).

## Berkas

- `tabel/` CSV dan LaTeX (IEEE): tabel1 utama, tabel2 per kelas, tabel3 diagnosis, tabel4 kecepatan, tabel5 split acak,
  tabel5b kebocoran, tabel6 di luar cakupan, tabel7 false positive background, tabel8 YOLOv8 vs YOLO11 berpasangan.
- `grafik/` gambar1 mAP50-95, gambar2 akurasi diagnosis, gambar3 confusion matrix, gambar4 contoh kualitatif.
- `kunci_model.csv` (sha256 dan ambang), `angka_kunci.json`, `evaluasi_akhir.log`, `analisis_akhir.log`.
- Bootstrap: 1.000 resample per kelompok foto; titik taksiran dihitung ulang dari statistik per gambar dan sama dengan angka
  resmi Ultralytics (selisih maksimum 0,00005).
