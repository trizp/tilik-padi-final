# Uji black-box

Jalur: `core.alur.periksa_foto` (sama dengan tampilan). 10 foto VAL prov_v7 dipilih acak per kelas (seed 20261014),
bukan berdasarkan hasil. Model aplikasi dilatih dengan train+val+test, jadi foto VAL termasuk data latihnya: uji ini memeriksa
perilaku sistem, bukan ketepatan pada foto baru.

Model: YOLOv8s (`v7_deploy_yolov8s_s2.onnx`). Skenario 15-18 memakai foto ds4 VAL yang sengaja dipilih dari
hasil pemindaian seluruh foto VAL (pindai_val.csv, hanya di folder proyek skripsi) karena memicu fitur penyampaian
(kemungkinan lain, cara membedakan, bercak berkeyakinan rendah).

**Foto VAL: 10/10 sesuai harapan. Skenario masukan khusus: 4/4 sesuai. Skenario fitur penyampaian: 4/4 sesuai.**

| No | Masukan | Hasil diharapkan | Hasil aktual | Sesuai |
|---|---|---|---|---|
| 1 | VAL ds4_blast-839_8a6c8034.jpg (label: Blas daun) | Blas daun, terdeteksi; saran R001 (D01, R001) | Blas daun, terdeteksi (75%); aturan D01, R001; cara membedakan terlipat | ya |
| 2 | VAL ds4_blast-613_400ca9e5.jpg (label: Blas daun) | Blas daun, terdeteksi; saran R001 (D01, R001) | Blas daun, terdeteksi (69%); aturan D01, R001; cara membedakan terlipat | ya |
| 3 | VAL ds4_blast-150_0c785c74.jpg (label: Blas daun) | Blas daun, terdeteksi; saran R001 (D01, R001) | Blas daun, terdeteksi (79%); aturan D01, R001; cara membedakan terlipat | ya |
| 4 | VAL ds4_brown-spot-1389_94d0c232.jpg (label: Bercak cokelat) | Bercak cokelat, terdeteksi; saran R003 (D01, R003) | Bercak cokelat, terdeteksi (75%); aturan D01, R003; cara membedakan terlipat | ya |
| 5 | VAL ds4_brown-spot-602_232fbc8f.jpg (label: Bercak cokelat) | Bercak cokelat, terdeteksi; saran R003 (D01, R003) | Bercak cokelat, terdeteksi (64%); aturan D01, R003; cara membedakan terlipat | ya |
| 6 | VAL ds4_brown-spot-94_25f548a2.jpg (label: Bercak cokelat) | Bercak cokelat, terdeteksi; saran R003 (D01, R003) | Bercak cokelat, terdeteksi (67%); aturan D01, R003; cara membedakan terlipat | ya |
| 7 | VAL ds1_provisional_r6k-train-img-blb-59-jpg-rf-a150c63f058e_57fba38a.jpg (label: Hawar daun bakteri) | Hawar daun bakteri, terdeteksi; saran R002 (D01, R002) | Hawar daun bakteri, terdeteksi (69%); aturan D01, R002 | ya |
| 8 | VAL ds1_provisional_r6k-train-img-blb-128-jpeg-jpg-rf-24c1c9_2bfac114.jpg (label: Hawar daun bakteri) | Hawar daun bakteri, terdeteksi; saran R002 (D01, R002) | Hawar daun bakteri, terdeteksi (89%); aturan D01, R002 | ya |
| 9 | VAL ds4_Healthy-328_f5e3fc92.jpg (label: daun sehat) | Tidak terdeteksi; saran foto ulang dan pencegahan umum (N01, N02) | Tidak terdeteksi (keyakinan tertinggi 0,4%); aturan N01, N02 | ya |
| 10 | VAL ds4_Healthy-269_5c2ded00.jpg (label: daun sehat) | Tidak terdeteksi; saran foto ulang dan pencegahan umum (N01, N02) | Tidak terdeteksi (keyakinan tertinggi 0,0%); aturan N01, N02 | ya |
| 11 | Foto no. 1 disimpan ulang sebagai PNG | Hasil sama dengan foto no. 1 (kelas dan tingkat) | Blas daun, terdeteksi (75%) | ya |
| 12 | Foto no. 1 diputar 90 derajat dengan tag EXIF orientasi 6 | Orientasi dikoreksi; kelas dan tingkat sama dengan foto no. 1 | Blas daun, terdeteksi (75%) | ya |
| 13 | Berkas teks berakhiran .jpg | Ditolak dengan pesan cara memperbaiki | Ditolak: "Berkas ini bukan foto yang bisa dibaca. Unggah foto berformat JPG atau PNG." | ya |
| 14 | Berkas 10,5 MB | Ditolak karena melebihi 10 MB | Ditolak: "Ukuran foto 10,5 MB, lebih dari batas 10 MB. Kecilkan foto atau ambil ulang dengan kamera HP." | ya |
| 15 | VAL ds4_blast-321_7c914ca8.jpg (label: Blas daun; dipilih karena memicu fitur) | Blas daun, terdeteksi; kemungkinan lain Bercak cokelat (>= ambang rendah); cara membedakan TERBUKA (pasangan Blast dan Brown spot); saran tetap R001 (D01, R001) | Blas daun, terdeteksi (73%); aturan D01, R001; kemungkinan lain Bercak cokelat (62%); cara membedakan terbuka | ya |
| 16 | VAL ds4_blast-104_67def77f.jpg (label: Blas daun; dipilih karena memicu fitur) | Blas daun, terdeteksi; tanpa kemungkinan lain; cara membedakan TERLIPAT; saran R001 (D01, R001) | Blas daun, terdeteksi (68%); aturan D01, R001; cara membedakan terlipat | ya |
| 17 | VAL ds4_brown-spot-477_a8966380.jpg (label: Bercak cokelat; dipilih karena memicu fitur) | Tidak terdeteksi (keyakinan teratas tepat di bawah ambang); tanpa kotak putus-putus karena tampilan bercak berkeyakinan rendah dinonaktifkan di config; N01, N02; tanpa cara membedakan | Tidak terdeteksi (keyakinan tertinggi 36,7%); aturan N01, N02 | ya |
| 18 | VAL ds4_Healthy-1014_c5aa8fd9.jpg (label: daun sehat; dipilih karena memicu fitur) | Tidak terdeteksi; tanpa kotak, tanpa bercak berkeyakinan rendah, tanpa cara membedakan; N01, N02 | Tidak terdeteksi (keyakinan tertinggi 0,0%); aturan N01, N02 | ya |
