# Uji black-box

Jalur: `core.alur.periksa_foto` (sama dengan tampilan). 10 foto VAL prov_v7 dipilih acak per kelas (seed 20261014),
bukan berdasarkan hasil.

Yang diuji adalah **perilaku sistem**: hasil tampil, tingkat dan saran sesuai aturan, kotak hanya pada hasil yang bukan "tidak
terdeteksi", penyampaian tambahan konsisten, foto ulang diminta saat daun tidak jelas, dan masukan buruk ditolak dengan pesan.
**Kebenaran diagnosis tidak dinilai di sini**: label foto ditulis di kolom masukan hanya sebagai informasi. Ketepatan diagnosis
dilaporkan pada hasil evaluasi model (README, bagian "Hasil evaluasi alur aplikasi"), yang memakai foto uji yang tidak dipakai
untuk melatih model. Model aplikasi dilatih dengan train+val+test, jadi foto VAL di sini termasuk data latihnya.

Model: YOLOv8s (`v9_deploy_yolov8s_s2.onnx`), pemisah daun `v9_pemisah_daun_yolov8n_seg.onnx` (tau 0,35). Foto yang tidak memperlihatkan satu helai daun yang jelas diminta foto ulang dan tidak didiagnosis.
Skenario 15-20 memakai foto ds4 VAL yang sengaja dipilih dari hasil pemindaian seluruh foto VAL (pindai_val.csv,
hanya di folder proyek skripsi) karena memicu fitur penyampaian (kemungkinan lain, cara membedakan, tingkat kemungkinan, tidak
terdeteksi, foto ulang).

**Foto VAL: perilaku sesuai 10/10. Skenario masukan khusus: 4/4 sesuai. Skenario fitur penyampaian: 6/6 sesuai.**

**Perilaku umum (P)**, harapan untuk foto no. 1-10: Hasil tampil dengan tingkat dan saran sesuai aturan: terdeteksi = kotak + D01 dan aturan penyakit yang tampil (R001/R002/R003); kemungkinan = kotak + K01, K02; tidak terdeteksi = tanpa kotak + N01, N02. Bila daun tidak jelas: foto ulang diminta, tanpa diagnosis dan tanpa aturan. Penyampaian tambahan konsisten: kemungkinan lain hanya pada hasil yang bukan tidak terdeteksi dan berbeda dari diagnosis; cara membedakan terbuka untuk pasangan Blast dan Brown spot, terlipat bila Blast atau Brown spot muncul tanpa pasangan itu.

| No | Masukan | Hasil diharapkan (perilaku sistem) | Hasil aktual | Sesuai |
|---|---|---|---|---|
| 1 | VAL ds4_blast-839_8a6c8034.jpg (label: Blas daun) | Perilaku umum (P) | Blas daun, terdeteksi (74%); aturan D01, R001; cara membedakan terlipat | ya |
| 2 | VAL ds4_blast-613_400ca9e5.jpg (label: Blas daun) | Perilaku umum (P) | Blas daun, terdeteksi (61%); aturan D01, R001; cara membedakan terlipat | ya |
| 3 | VAL ds4_blast-150_0c785c74.jpg (label: Blas daun) | Perilaku umum (P) | Blas daun, terdeteksi (76%); aturan D01, R001; cara membedakan terlipat | ya |
| 4 | VAL ds4_brown-spot-1389_94d0c232.jpg (label: Bercak cokelat) | Perilaku umum (P) | Bercak cokelat, terdeteksi (70%); aturan D01, R003; cara membedakan terlipat | ya |
| 5 | VAL ds4_brown-spot-602_232fbc8f.jpg (label: Bercak cokelat) | Perilaku umum (P) | Bercak cokelat, terdeteksi (56%); aturan D01, R003; cara membedakan terlipat | ya |
| 6 | VAL ds4_brown-spot-94_25f548a2.jpg (label: Bercak cokelat) | Perilaku umum (P) | Bercak cokelat, terdeteksi (63%); aturan D01, R003; cara membedakan terlipat | ya |
| 7 | VAL ds1_provisional_r6k-train-img-blb-59-jpg-rf-a150c63f058e_57fba38a.jpg (label: Hawar daun bakteri) | Perilaku umum (P) | Bercak cokelat, terdeteksi (47%); aturan D01, R003; cara membedakan terlipat | ya |
| 8 | VAL ds1_provisional_r6k-train-img-blb-128-jpeg-jpg-rf-24c1c9_2bfac114.jpg (label: Hawar daun bakteri) | Perilaku umum (P) | Hawar daun bakteri, terdeteksi (70%); aturan D01, R002 | ya |
| 9 | VAL ds4_Healthy-328_f5e3fc92.jpg (label: daun sehat) | Perilaku umum (P) | Tidak terdeteksi (keyakinan tertinggi 0,2%); aturan N01, N02 | ya |
| 10 | VAL ds4_Healthy-269_5c2ded00.jpg (label: daun sehat) | Perilaku umum (P) | Tidak terdeteksi (keyakinan tertinggi 0,6%); aturan N01, N02 | ya |
| 11 | Foto no. 1 disimpan ulang sebagai PNG | Hasil sama dengan foto no. 1 (kelas dan tingkat) | Blas daun, terdeteksi (74%) | ya |
| 12 | Foto no. 1 diputar 90 derajat dengan tag EXIF orientasi 6 | Orientasi dikoreksi; kelas dan tingkat sama dengan foto no. 1 | Blas daun, terdeteksi (74%) | ya |
| 13 | Berkas teks berakhiran .jpg | Ditolak dengan pesan cara memperbaiki | Ditolak: "Berkas ini bukan foto yang bisa dibaca. Unggah foto berformat JPG atau PNG." | ya |
| 14 | Berkas 10,5 MB | Ditolak karena melebihi 10 MB | Ditolak: "Ukuran foto 10,5 MB, lebih dari batas 10 MB. Kecilkan foto atau ambil ulang dengan kamera HP." | ya |
| 15 | VAL ds4_blast-321_7c914ca8.jpg (label: Blas daun; dipilih karena memicu fitur) | Terdeteksi dengan kemungkinan lain: penyakit lain (>= ambang rendah, berbeda dari diagnosis) tampil; cara membedakan TERBUKA karena pasangan Blast dan Brown spot; saran tetap mengikuti diagnosis utama (D01 + aturan penyakitnya) | Blas daun, terdeteksi (67%); aturan D01, R001; kemungkinan lain Bercak cokelat (52%); cara membedakan terbuka | ya |
| 16 | VAL ds4_blast-104_67def77f.jpg (label: Blas daun; dipilih karena memicu fitur) | Terdeteksi Blast atau Brown spot tanpa kemungkinan lain: cara membedakan TERLIPAT; kotak digambar; D01 + aturan penyakitnya | Blas daun, terdeteksi (69%); aturan D01, R001; cara membedakan terlipat | ya |
| 17 | VAL ds4_blast-414_94feb6e6.jpg (label: Blas daun; dipilih karena memicu fitur) | Tingkat KEMUNGKINAN (keyakinan antara ambang rendah dan ambang terdeteksi): kotak digambar; K01, K02 (konfirmasi ke penyuluh, tanpa saran obat) | Blas daun, kemungkinan (34%); aturan K01, K02; cara membedakan terlipat | ya |
| 18 | VAL ds4_brown-spot-2173_3657de07.jpg (label: Bercak cokelat; dipilih karena memicu fitur) | Tidak terdeteksi dengan keyakinan teratas tepat di bawah ambang rendah: tanpa kotak dan tanpa kotak putus-putus (tampilan bercak berkeyakinan rendah dinonaktifkan di config); N01, N02; tanpa cara membedakan | Tidak terdeteksi (keyakinan tertinggi 29,6%); aturan N01, N02 | ya |
| 19 | VAL ds4_Healthy-1014_c5aa8fd9.jpg (label: daun sehat; dipilih karena memicu fitur) | Tidak terdeteksi: tanpa kotak, tanpa bercak berkeyakinan rendah, tanpa cara membedakan; N01, N02 | Tidak terdeteksi (keyakinan tertinggi 0,2%); aturan N01, N02 | ya |
| 20 | VAL ds4_Healthy-1432_5998a115.jpg (label: daun sehat; dipilih karena memicu fitur) | Daun tidak jelas (pemisah daun tidak menemukan satu helai daun): foto ulang diminta; tidak didiagnosis, tanpa aturan dan kotak | Diminta foto ulang (tidak didiagnosis) | ya |
