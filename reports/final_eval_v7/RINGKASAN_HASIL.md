# Ringkasan hasil evaluasi akhir v7 (PROVISIONAL)

Tanggal: 11 Oktober 2026. Dataset prov_v7 = prov_v6 dengan orientasi 931 foto ds4 diperbaiki (lossless); BLB masih dari ds1
(provisional). Ambang diagnosis dan model web dikunci/dipilih di val sebelum test dibuka
(KEPUTUSAN_PROYEK.md, 11 Okt 15:46–15:48). Setiap set uji dievaluasi sekali (40 evaluasi, tanpa evaluasi ulang).
Semua angka: rata-rata ± simpangan baku 3 seed, IK 95% bootstrap per kelompok foto (1.000 resample).

**Status (14 Okt 2026): v7 adalah MODEL FINAL.** Hasil paper = 12 run final v7; model aplikasi = `v7_deploy_yolo11s_s2`
(pembanding, `aplikasi/RINGKASAN_MODEL_WEB.md`). Rencana v8 dihentikan sebelum training karena tidak ada set uji Indonesia yang segar;
prov_v8 diarsipkan sebagai persiapan penelitian lanjutan (KEPUTUSAN_PROYEK.md 14 Okt). Uji eksternal ds3 (13 Okt) dilaporkan sebagai
temuan pergeseran domain (bagian "Uji eksternal ds3" di bawah; rincian `ds3/RINGKASAN_DS3.md`).

## Model pilihan web (dipilih dari val, bukan test)

**YOLO11s seed 2, ambang diagnosis 0,35.** YOLO11s dan YOLOv8s hampir sama di val (akurasi diagnosis 0,880 vs 0,875;
selisih < 0,01), lalu YOLO11s dipilih karena lebih cepat di CPU (81 vs 103 ms per gambar).
Hasil test model ini: akurasi diagnosis 0,771 (test), 0,726 (test sesi baru); mAP50-95 0,234 (test), 0,247 (test sesi baru).

## Temuan utama

1. **Diagnosis per citra (test, semua kelas):** akurasi 0,73–0,77. Terbaik YOLO11s 0,771 ± 0,002 (IK 0,707–0,834).
   Di test sesi baru (sesi pemotretan yang tidak pernah dilihat) 0,69–0,76; terbaik YOLO11n 0,764 dan YOLO11s 0,749.
2. **Deteksi kotak (test, semua kelas):** mAP50 0,49–0,55, mAP50-95 0,20–0,23. Tanpa BLB provisional: mAP50 0,58–0,63,
   mAP50-95 0,25–0,28. Test sesi baru: mAP50-95 0,23–0,27 — kini sebanding dengan test biasa (di v6 hampir nol karena bug label).
3. **YOLO11 vs YOLOv8 (bootstrap berpasangan):**
   - varian n: YOLO11n lebih baik pada mAP di semua set (mis. test sesi baru mAP50-95 +0,041, IK +0,026 s.d. +0,054) dan pada
     akurasi diagnosis test sesi baru (+0,071); akurasi diagnosis di test biasa tidak berbeda (interval memuat nol);
   - varian s: YOLO11s lebih baik pada akurasi diagnosis di semua set (+0,031 s.d. +0,060) dan mAP test bersih/sesi baru;
     mAP50-95 test semua kelas +0,016 dengan interval tepat menyentuh nol (−0,000 s.d. +0,034).
   - YOLO11 juga lebih kecil (n: 2,6 M vs 3,0 M parameter; s: 9,4 M vs 11,1 M) dan lebih cepat di CPU pada varian s.
4. **Pesan utama perbandingan split: split bersih memberi perkiraan kinerja yang jujur; split acak melebih-lebihkan skor test.**
   Ukurannya = selisih skor di test split masing-masing dengan skor di test sesi baru (sesi pemotretan yang tidak pernah dilihat
   kedua model), dengan test dibandingkan TANPA BLB agar kelasnya sama (`tabel/tabel10_ketepatan_perkiraan.csv`):

   | Split latih | Selisih mAP50-95 (test − sesi baru) | Selisih akurasi diagnosis | Selisih akurasi seimbang |
   |---|---|---|---|
   | split bersih (4 model × 3 seed) | +0,007 s.d. +0,025 | −0,026 s.d. +0,061 | −0,057 s.d. +0,018 |
   | split acak (2 model × 1 seed) | **+0,106 dan +0,119** | +0,054 dan +0,107 | +0,025 dan +0,083 |

   **mAP adalah bukti utama**: test split bersih hampir tepat memperkirakan sesi baru, sedangkan test acak melebih-lebihkan ~0,11
   (sekitar 40% di atas skor sesi baru). Akurasi diagnosis biasa dipengaruhi perbedaan komposisi kelas: test tanpa BLB berisi
   36% foto sehat (158/434), sesi baru hanya 19% (56/299), sehingga selisih +0,06 pada YOLOv8 sebagian berasal dari komposisi, bukan
   kebocoran. Akurasi seimbang (rata-rata recall per kelas, tidak dipengaruhi komposisi) menunjukkan split bersih jujur (−0,06 s.d.
   +0,02), sedangkan split acak +0,03 dan +0,08 — arahnya sama dengan mAP tetapi pemisahannya lebih lemah. Di test semua kelas
   (dengan BLB), mAP test acak 0,48–0,49 vs bersih 0,20–0,23; minimal 77% gambar val/test acak punya kembaran langsung di train
   (batas atas 92%).

   **Temuan (bukan bukti, 1 seed): model split acak mungkin menggeneralisasi lebih baik ke sesi baru**, kemungkinan karena keragaman
   adegan di data latih. Di test sesi baru, model split acak lebih tinggi dan selisihnya di luar IK 95% bootstrap berpasangan
   (`tabel/tabel9_split_acak_vs_bersih_sesi_baru_berpasangan.csv`):
   - YOLOv8n acak vs bersih s0: mAP50-95 0,271 vs 0,205 (+0,066, IK +0,044 s.d. +0,085); akurasi diagnosis 0,819 vs 0,649 (+0,171).
   - YOLO11n acak vs bersih s0: mAP50-95 0,283 vs 0,263 (+0,021, IK +0,003 s.d. +0,039); akurasi diagnosis 0,866 vs 0,756 (+0,110).
   - Terhadap rata-rata 3 seed split bersih: YOLOv8n tetap lebih baik di semua metrik; YOLO11n mAP50-95 +0,015 (IK memuat nol),
     akurasi diagnosis +0,103 (IK tidak memuat nol).
   - IK ini hanya menangkap variasi gambar uji; split acak hanya 1 seed sehingga variasi pelatihannya belum terukur.
   Penjelasan "lebih banyak data" TIDAK tepat: data latih split acak justru lebih sedikit (5.863 vs 6.772). Perbedaan yang terukur
   (belum terbukti sebagai penyebab):
   - **Keragaman ds4:** jumlah foto ds4 berlabel hampir sama (2.224 vs 2.313; blast 826 vs 833, brown spot 1.398 vs 1.480), tetapi
     split acak memuat **536 kelompok foto ds4 berbeda vs 421** (semua sumber 1.209 vs 1.041 kelompok) karena kelompok dipecah lintas
     split; split bersih menjaga kelompok utuh sehingga train melihat lebih sedikit adegan berbeda.
   - **Komposisi ds1:** salinan augmentasi ds1 (blast/brown spot) di train split acak lebih sedikit (1.641 vs 2.381).
   - **Aturan pemilihan checkpoint:** split acak memilih best.pt dari val acak semua kelas (bocor) -> epoch 100 pada kedua model;
     split bersih dari val bersih (tanpa BLB) -> epoch 39 (v8n s0) dan 74 (11n s0). Ambang diagnosis juga dipilih di val yang berbeda
     (randsplit 0,29/0,23; bersih s0 0,27/0,33).
5. **Penyakit di luar cakupan tetap tidak ditolak:** Leaf smut 85–88% dan Tungro 76–78% foto didiagnosis salah satu dari tiga
   penyakit; Sheath blight 21–31%.
6. **False positive pada daun sehat:** conf 0,25: 35–39% gambar sehat test mendapat kotak; conf 0,5: 9–16%.
7. **Kecepatan (laptop ini, batch 1, FP32):** GPU 7,4–11,6 ms (86–137 FPS); CPU 34–95 ms (10,5–30 FPS).

## Uji eksternal ds3: temuan pergeseran domain

ds3 = foto daun padi Indonesia yang dipetik lalu difoto di atas latar putih (data latih: foto lapangan). Uji utama 70 foto yang
dinilai "sesuai" oleh dua penilai non-pakar (kappa 0,431); dievaluasi sekali dengan ambang terkunci (`ds3/RINGKASAN_DS3.md`).
- Akurasi diagnosis model final split bersih 0,18–0,34 (test v7: 0,73–0,77); recall Blast 0–0,05 (bercak blast dibaca brown spot).
- Recall BLB sangat bergantung pada ambang dan seed. Tungro (di luar cakupan) juga sering didiagnosis BLB: 45–78% pada model final
  split bersih, 76/80 pada model aplikasi (post-hoc). Jadi diagnosis BLB pada daun petik yang menguning tidak spesifik.
- Analisis POST-HOC (`ds3/analisis_posthoc/`, tidak mengubah model/ambang/angka utama): ds3 jauh lebih terang (0,76–0,81 vs
  0,41–0,43), hampir tanpa piksel hijau (2–4% vs 46–62%), berformat potret; kotak BLB prediksi membingkai seluruh daun, mirip gaya
  kotak BLB ds1 (median 20% luas foto). Penafsiran: pergeseran domain + kemungkinan shortcut learning BLB (dugaan, bukan bukti kausal).
- ds3 tidak dipakai untuk menyetel apa pun; sekarang berstatus "sudah terbuka".

## v7 dibanding v6 (lampiran)

Metrik kotak v6 tidak valid (bug orientasi). Dengan label benar, mAP50-95 test naik dari 0,13–0,15 (v6) ke 0,20–0,23 (v7);
test sesi baru dari 0,006–0,011 ke 0,23–0,27. Akurasi diagnosis test naik sedikit (mis. YOLO11s 0,750 → 0,771), di test
sesi baru naik jelas (mis. YOLO11s 0,629 → 0,749). `tabel/lampiran_v6_vs_v7.csv`.

## Keterbatasan

1. **Penyakit di luar cakupan tidak ditolak.** Model hanya mengenal tiga penyakit dan "sehat"; foto Leaf smut (85–88%), Tungro
   (76–78%), dan Sheath blight (21–31%) sering didiagnosis sebagai salah satu dari tiga penyakit (paling sering brown spot/blast).
   Aplikasi web perlu peringatan bahwa penyakit lain tidak dikenali.
2. **False positive pada daun sehat.** Pada conf 0,25, 35–39% gambar sehat test mendapat setidaknya satu kotak; pada diagnosis per
   citra, sekitar 24% foto sehat didiagnosis brown spot pada model terbaik tiap keluarga. Sebagian mungkin karena **anomali label ds4**
   yang terlihat di contoh kualitatif: (a) bercak brown spot yang tidak berlabel (model mendeteksi bercak kedua yang tidak dikotak);
   (b) foto di folder "Healthy" ds4 yang tampak memiliki bercak kecil. Belum diaudit menyeluruh; angka FP bisa sedikit terlalu tinggi.
3. **BLB provisional dari satu sumber.** Seluruh BLB berasal dari ds1 (unit anotasi "area gejala per segmen daun", banyak augmentasi;
   dua salinan augmentasi dari foto yang sama ada di test dengan kotak label yang hanya menutup sebagian daun bergejala). Anotasi ulang
   ds2 TIDAK dilakukan dalam penelitian ini, jadi BLB tetap provisional di hasil final. Angka BLB tidak sebanding dengan Blast/Brown
   spot (unit bercak) dan dilaporkan dengan catatan ini.
4. **Interval kepercayaan lebar.** Unit sampel adalah kelompok foto: test 125 kelompok, BLB test hanya 21 kelompok, test sesi baru
   133 kelompok. Selisih kecil antarmodel perlu dibaca bersama IK-nya.
5. **Efek ambang tanpa BLB.** Ambang diagnosis dipilih di val bersih yang TIDAK memuat BLB, sehingga ambang v7 (0,27–0,39) terlalu
   tinggi untuk BLB: diagnosis BLB turun dari ~0,91 (v6) ke 0,67–0,75 (v7); sebagian besar foto BLB yang didiagnosis "sehat" punya
   kotak teratas berkelas BLB di bawah ambang (YOLO11s: 132 dari 147). Ambang tidak diubah karena test sudah dibuka.
   Analisis sensitivitas **POST-HOC** (`posthoc_sensitivitas_ambang/`, tidak dipakai memilih model atau mengganti angka utama)
   menunjukkan trade-off: pada ambang 0,10, diagnosis BLB test naik ke 0,85–0,93 tetapi daun sehat turun ke 0,41–0,53, dan akurasi
   total hampir sama (0,74–0,76 vs 0,73–0,77 pada ambang terkunci).
6. **Pergeseran domain dan dugaan shortcut BLB.** Kinerja turun tajam pada daun petik berlatar putih (ds3: akurasi 0,18–0,34), Blast
   hampir tidak dikenali di sana, dan BLB juga muncul pada Tungro. Hasil test v7 berlaku untuk foto lapangan bergaya ds4/ds1, bukan
   untuk semua cara memotret.
7. **Tidak ada set uji Indonesia yang segar.** Data latih dan test berasal dari Bangladesh (ds4) dan kumpulan Roboflow (ds1). Satu-satunya
   set Indonesia (ds3) sudah terbuka. Kandidat Indonesia baru tidak layak: satu memuat foto ds3 (gymvi), satu hanya berisi rentetan
   foto dari ~1 helai daun per kelas (aditia). Pengumpulan foto sendiri tidak memungkinkan dalam penelitian ini. Karena itu rencana
   v8 (data tambahan) dihentikan sebelum training: tanpa set uji segar, perbaikannya tidak bisa diukur secara jujur.

## Saran penelitian lanjutan

1. **Data lokal Indonesia** untuk latih DAN uji: foto lapangan (daun di tanaman) dan daun petik berlatar polos, dari beberapa lokasi
   dan hari, dengan semua kelas difoto pada setiap hari (agar kelas tidak terkacaukan dengan hari atau latar), maksimal beberapa foto
   per helai daun, dan label dari petugas POPT/penyuluh. Usulan protokol foto ada di KEPUTUSAN_PROYEK.md (Rencana v8, butir 8). Set
   uji dibekukan sebelum training.
2. **BLB dari beberapa sumber** dengan unit anotasi yang konsisten dan dinyatakan jelas. Hindari seluruh BLB berasal dari satu
   dataset dengan satu gaya kotak, karena itu yang diduga memicu shortcut "daun memanjang menguning = BLB".
3. **Contoh negatif daun menguning bukan-BLB** (Tungro, defisiensi hara, daun tua) sebagai background tanpa kotak. Ukur laju
   diagnosis BLB pada Tungro dari sumber yang berbeda dengan data latih, sebagai uji langsung hipotesis shortcut.
4. **Rancangan prov_v8 siap dipakai** (BEKU, `reports/prov_v8/LAPORAN_PROV_V8.md`): prov_v7 identik + daunpadi (daun petik berlatar
   polos dengan bercak kecil; train 725, val 172) + 600 foto Tungro ds4 sebagai background negatif. Semua audit kebocoran lolos.
   Yang kurang hanyalah set uji segar (butir 1). Resep training identik dengan v7 agar selisihnya bisa dikaitkan dengan data.
5. **Pemeriksaan wajib sebelum memakai dataset publik**:
   - cek kembaran dua arah terhadap set uji, termasuk unggahan ulang di Roboflow/Kaggle;
   - periksa durasi dan jeda pemotretan per kelas (rentetan foto beruntun dari sedikit daun);
   - cek orientasi label terhadap visual penulis dataset.
6. Penolakan penyakit di luar cakupan (kelas "lainnya" atau ambang ketidakpastian), karena Leaf smut, Tungro, dan Sheath blight
   saat ini sering didiagnosis sebagai salah satu dari tiga penyakit.

Catatan lain: kebocoran split acak diambil dari prov_v6_randsplit (pembagian identik; kriteria kembaran tidak terpengaruh rotasi);
contoh kualitatif dibuat dengan memprediksi ulang 8 gambar test hanya untuk digambar; split acak hanya 1 seed.

## Berkas

`tabel/` (CSV + LaTeX IEEE: tabel1–8, lampiran_v6_vs_v7; tabel9 split acak vs bersih di sesi baru, berpasangan;
tabel10 ketepatan perkiraan test vs sesi baru, ringkas dan per run),
`grafik/` (gambar1–4), `kunci_model.csv`, `pilihan_web.json`, `kecepatan_cpu_pemilihan_web.json`, `angka_kunci.json`,
`posthoc_sensitivitas_ambang/` (ANALISIS POST-HOC: kurva.csv, gambar, CATATAN.txt), log evaluasi dan analisis.
`ds3/` (uji eksternal; `RINGKASAN_DS3.md`, `analisis_posthoc/`), `aplikasi/` (model aplikasi). Arsip penelitian lanjutan:
`reports/prov_v8/`, `reports/kandidat_v8/`.
