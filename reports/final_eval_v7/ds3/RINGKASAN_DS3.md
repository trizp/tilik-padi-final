# Ringkasan uji eksternal ds3 (PROVISIONAL)

Tanggal: 13 Oktober 2026. Model: 12 run final v7 (split bersih) + 2 run randsplit v7. Diagnosis per citra dengan ambang terkunci
masing-masing model; setiap himpunan dievaluasi sekali. Daftar uji utama dibekukan sebelum model dijalankan pada ds3
(KEPUTUSAN_PROYEK.md, 13 Okt 16:26). ds3 adalah set segar bagi model split bersih maupun split acak (tidak ada kembaran di ds1/ds2/ds4).

## Apa itu ds3

Foto daun padi yang **dipetik lalu difoto di atas latar putih** (bingkai potret, daun kecil di tengah). Data latih berupa foto
daun di lapangan. Jadi ds3 menguji pergeseran domain yang besar, bukan hanya "foto baru".

Verifikasi label (dua penilai non-pakar, 2 putaran): kappa gabungan 0,431 (po 0,684). **Uji utama = 70 foto** yang dinilai
"sesuai" oleh kedua penilai (BLB 35 foto/35 kelompok, Blast 35 foto/33 kelompok). Analisis tambahan = semua 160 foto dengan
label asli. Tungro (80 foto) = di luar cakupan.

## Hasil utama (uji utama, 70 foto)

| Split latih | Model | Akurasi (IK 95%) | Recall BLB | Recall Blast |
|---|---|---|---|---|
| bersih (3 seed) | YOLOv8n | 0,338 ± 0,150 (0,249–0,433) | 0,667 ± 0,315 | 0,010 |
| bersih (3 seed) | YOLO11n | 0,181 ± 0,166 (0,127–0,243) | 0,352 ± 0,334 | 0,010 |
| bersih (3 seed) | YOLOv8s | 0,224 ± 0,066 (0,152–0,304) | 0,400 ± 0,076 | 0,048 |
| bersih (3 seed) | YOLO11s | 0,262 ± 0,240 (0,198–0,333) | 0,524 ± 0,480 | 0,000 |
| acak (1 seed) | YOLOv8n | 0,486 (0,371–0,614) | 0,886 | 0,086 |
| acak (1 seed) | YOLO11n | 0,500 (0,383–0,618) | 1,000 | 0,000 |

1. **Kinerja di ds3 rendah untuk semua model.** Akurasi split bersih 0,18–0,34 (test v7: 0,73–0,77).
2. **Blast hampir tidak pernah dikenali** (recall 0–0,05). Foto blast ds3 hampir semua didiagnosis brown spot (contoh YOLO11s:
   89 dari 105 diagnosis brown spot, 0 Blast, gabungan 3 seed). Bercak blast di ds3 tampak sebagai banyak bintik kecil pada daun
   petik, yang di data latih lebih mirip brown spot.
3. **BLB sangat bergantung pada ambang dan seed** (simpangan baku recall BLB sampai 0,48). Kotak teratas sering berkelas BLB
   tetapi confidence-nya di bawah ambang terkunci (0,27–0,39), sehingga jatuh ke "tidak terdeteksi".
4. **Tungro (di luar cakupan) juga sering didiagnosis BLB** (split bersih 45–78% foto; split acak 94–98%). Jadi diagnosis BLB pada
   daun petik yang menguning tidak spesifik; recall BLB yang tinggi di ds3 belum tentu berarti BLB benar-benar dikenali.
5. **YOLO11 vs YOLOv8 (berpasangan):** varian n — YOLO11n lebih rendah dari YOLOv8n (akurasi −0,157, IK −0,216 s.d. −0,099), terutama
   karena recall BLB; varian s — tidak ada beda akurasi yang meyakinkan (+0,038, IK memuat nol). Ini kebalikan dari hasil test v7,
   sehingga keunggulan YOLO11 di test tidak terbawa ke domain ds3.
6. **Split acak vs bersih:** model split acak lebih tinggi daripada rata-rata 3 seed split bersih (YOLOv8n +0,148; YOLO11n +0,319; IK
   tidak memuat nol), hampir seluruhnya lewat recall BLB; terhadap seed 0 saja, YOLOv8n tidak berbeda (+0,000). Karena tungro juga
   didiagnosis BLB oleh model split acak (94–98%), kelebihan ini bisa berasal dari kecenderungan menyebut BLB pada daun petik,
   bukan dari pengenalan BLB yang lebih baik. Satu seed.

Analisis tambahan (160 foto, label asli) memberi pola yang sama (akurasi split bersih 0,12–0,29; split acak 0,47–0,48).

## Analisis tambahan dan POST-HOC (14 Okt; tidak mengubah model, ambang, atau angka utama; tidak dipakai menyetel apa pun)

Folder: `analisis_posthoc/` (lihat `CATATAN.txt`).

- **Model web pada Tungro ds3 (80 foto, sekali, ambang 0,35):** **76 foto (95%) didiagnosis BLB**, 4 "tidak terdeteksi"; dengan tiga
  tingkat: 76 "terdeteksi", 4 "kemungkinan" — semuanya berkelas BLB. Jadi recall BLB 35/35 model web di uji utama **tidak**
  menunjukkan BLB dikenali secara spesifik: model menyebut BLB pada hampir semua daun petik yang menguning/mengering.
- Model web juga mendiagnosis foto **Blast** ds3 paling sering sebagai **BLB** (uji utama 18/35; 160 foto: 36/80), lalu Brown spot.
- **Analisis tambahan (160 foto, label asli):** model final v7 akurasi 0,12–0,29 (rata-rata 3 seed per jenis); model web 0,513
  (recall BLB 0,875 dengan presisi 0,660; recall Blast 0,150 dengan presisi 0,923). `analisis_posthoc/tambahan_160_final_dan_model_web.csv`.
- **Montase kesalahan** (`montase_kesalahan_ds3.jpg`, model web): pada BLB dan Tungro model membuat SATU kotak BLB yang membingkai
  seluruh helai daun; pada Blast model menandai bercak satu per satu tetapi memberinya kelas Brown spot.
- **Karakteristik foto** (`karakteristik_ds3_ds4_ds1.csv`, 200 foto ds4/ds1 train acak, semua ds3):

  | Himpunan | Resolusi | Kecerahan | Piksel hijau | Kotak per foto | Luas kotak relatif (median) |
  |---|---|---|---|---|---|
  | ds4 (lapangan, Bangladesh) | 1024×1024 | 0,41 | 50% | 3 (label) | 0,28% |
  | ds1 BLB (sumber seluruh BLB latih) | 587×587 | 0,42 | 46% | 1 (label) | 20,2% |
  | ds1 Blast/Brown spot | 587×587 | 0,43 | 62% | 3 (label) | 0,39% |
  | ds3 blight+blast (daun petik) | 1440×1920 (potret) | 0,76 | 4% | 1 (prediksi) | 8,7% |
  | ds3 tungro | 1503×1920 (potret) | 0,81 | 2% | 1 (prediksi) | 14,6% |

  ds3 jauh lebih terang, hampir tanpa latar hijau, dan berformat potret; kotak prediksinya besar dan tunggal seperti gaya kotak BLB ds1.

## Penafsiran

1. **Pergeseran domain.** Data latih berasal dari foto lapangan Bangladesh (ds4) dan kumpulan Roboflow (ds1) berlatar vegetasi hijau;
   ds3 adalah daun padi Indonesia yang dipetik dan difoto di atas latar putih. Model belajar mengenali penyakit dalam konteks lapangan,
   sehingga kinerjanya turun tajam di ds3 (test v7 0,73–0,77 -> ds3 0,18–0,34 untuk model final).
2. **Kemungkinan shortcut learning pada BLB.** Seluruh BLB latih berasal dari ds1 dengan kotak area besar (median 20% luas foto) yang
   sering membingkai daun memanjang. Di ds3, model memberi label BLB pada hampir semua daun petik yang menguning/mengering — termasuk
   Tungro (95% pada model web) dan sebagian Blast — dengan satu kotak yang membingkai seluruh daun. Ini konsisten dengan model yang
   belajar "daun memanjang menguning dengan kotak besar = BLB", bukan ciri BLB itu sendiri. Ini dugaan berdasarkan pola, bukan bukti
   kausal; perlu diuji ulang setelah BLB final (anotasi ulang ds2, lesion-level) tersedia.
3. **Blast vs Brown spot.** Bercak Blast ds3 ditemukan (kotak di bercak), tetapi diberi kelas Brown spot; kemungkinan karena di data
   latih bercak kecil pada latar terang/daun terisolasi lebih menyerupai contoh Brown spot.
4. **ds3 tidak boleh dipakai untuk menyetel model** (ambang, epoch, augmentasi, pilihan model). ds3 hanya untuk melaporkan ketahanan
   terhadap domain lain; setiap perbaikan berdasarkan temuan ini harus diuji pada set uji baru yang belum dilihat.

## Keterbatasan

- Pergeseran domain (daun petik, latar putih, bingkai potret) — hasil ini mengukur ketahanan terhadap domain lain, bukan kinerja
  pada foto lapangan Indonesia secara umum.
- Label ds3 diverifikasi oleh dua penilai non-pakar dengan kesepakatan sedang (kappa 0,43); 2 foto tidak dinilai ulang oleh penilai 1.
- BLB latih masih provisional (ds1); ambang dipilih di val tanpa BLB.
- Sampel kecil: 68 kelompok di uji utama; IK lebar.

## Berkas

`tabel_ds3_utama_*.csv`, `tabel_ds3_tambahan_*.csv` (per model, confusion, perbandingan), `tabel_ds3_tungro.csv`, `tabel/*.tex`,
`angka_ds3.json`, `evaluasi_ds3.log`. Per citra: `runs/<run>/eval_ds3_{utama,tambahan,tungro}/`.
