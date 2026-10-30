# Rancangan v9: latar dihapus (dan alternatif "latar diacak")

Status: RANCANGAN, belum dijalankan. Menunggu persetujuan pemilik skripsi (KEPUTUSAN_PROYEK.md 2026-10-16). Tidak ada training, dan
test, test sesi baru, serta ds3 belum disentuh.

Dasar: tahap 0 (`tahap0_ringkasan.md`, montase `montase_tahap0_*.jpg`) dan tahap 1 (`tahap1_ringkasan.md`, ANALISIS VAL).

## 1. Metode pemisahan daun yang dipakai

- **SAM2.1-b mode otomatis** ("segment everything", grid 16x16 titik, batch 16) lewat ultralytics 8.4.171, ditambah **aturan v2 memilih
  daun** tanpa label:
  - kandidat = masker dengan luas 3-95% foto dan >= 50% piksel berwarna vegetasi (HSV H 8-100, S >= 40, V >= 40);
  - skor = sqrt(luas) x (0,2 + cakupan kotak tengah 40% x 40% foto); skor tertinggi = daun utama;
  - pasca-proses: buang komponen < 10% komponen terbesar, tutup celah kecil, isi lubang (bercak di tengah daun ikut masuk);
  - **cadangan**: tidak ada kandidat, atau terpilih < 5% foto -> foto TIDAK dipisah (masker penuh), ditandai di manifest.
- Latar diganti abu-abu (114,114,114), disimpan JPEG q95. Koordinat kotak tidak diubah.
- **Aturan buang-kotak:** kotak dengan < 50% luasnya di dalam masker daun dibuang dari label.
- Masker disimpan (PNG) agar dataset dapat diaudit dan dibangun ulang persis.

## 2. Dataset prov_v9

| Butir | Rancangan |
|---|---|
| Sumber | `data/processed/prov_v7` (beku, manifest sha `b467d8c8…`); 8.959 foto: train 6.772, val 1.259, test 629, test sesi baru 299 |
| ID foto, split, kelompok | identik dengan prov_v7 (nama berkas, kolom `group`, split tidak berubah) |
| Pemisahan | metode bagian 1 untuk SEMUA split (train, val, test, test sesi baru); masker dibuat tanpa label |
| Label | kotak tidak digeser; kotak < 50% di dalam daun dibuang; jumlah kotak dibuang dilaporkan per kelas dan per sumber |
| Foto yang kehilangan SEMUA kotak | train: dikeluarkan dari latih (dilaporkan per kelas); val/test: tetap ada, kotak kosong untuk mAP, kelas citra untuk diagnosis = label asli (perlu persetujuan) |
| Kolom manifest baru | `latar_dihapus` (ya / cadangan), `luas_masker`, `kotak_asli`, `kotak_dibuang` |
| Beku | `FROZEN.txt` + sha manifest, seperti prov_v7 |
| Biaya pembuatan | masker median 1,4 detik per foto (sampel 100) sampai 2,9 detik (VAL penuh; memori GPU 4 GB hampir penuh) -> ±3,5-7 jam GPU untuk 8.959 foto; ruang disk ±2-3 GB (D: tersisa ±12 GB) |

Perkiraan dari VAL penuh (tahap 1, `buang_kotak_val.json`; 1.259 foto):

| Kelompok VAL | Kotak dibuang (< 50% di daun) | Foto kehilangan semua kotak | Foto cadangan (tidak dipisah) |
|---|---|---|---|
| ds4 Blast | 58 dari 796 (7,3%) | 8 dari 222 | 2 dari 222 |
| ds4 Brown spot | 33 dari 1.433 (2,3%) | 0 dari 330 | 1 dari 330 |
| ds1 BLB provisional | 133 dari 453 (29,4%) | 113 dari 390 | 142 dari 390 (36%) |
| ds4 sehat / ds2 sehat | - | - | 9 dari 293 / 4 dari 24 |

BLB paling terdampak: sepertiga foto tidak dapat dipisah dan hampir sepertiga kotak terbuang. Akibatnya angka BLB v9 (sudah provisional)
semakin tidak sebanding; usul: BLB tetap dilaporkan terpisah dan tidak dipakai untuk kesimpulan latar.

## 3. Training

- Resep identik v7 (`runs/v7_final_prov_*/run_config.json`): 100 epoch, imgsz 640, batch 8, SGD lr0 0,01, augmentasi bawaan v7,
  patience 0, AMP; best.pt dipilih hanya dari skor val bersih (tanpa ds1_provisional), val juga tanpa latar.
- Ambang diagnosis dikunci ulang di val bersih tanpa latar per run (prosedur sama dengan v7), sebelum test dibuka.
- Pilihan jumlah run (jam GPU dari run_stats v7: n ±1,9-2,0 jam/run, s ±3,35-3,5 jam/run):

| Pilihan | Run | Perkiraan jam GPU | Catatan |
|---|---|---|---|
| A. 4 model x 3 seed | 12 | ±32 jam | setara penuh dengan v7; semua sel tabel bisa diuji berpasangan |
| **B. YOLOv8s dan YOLO11s x 3 seed (usulan)** | 6 | ±20,5 jam | cukup untuk tabel 2x2 inti paper (varian s = terbaik tiap keluarga); IK per seed tetap ada |
| C. YOLOv8n dan YOLO11n x 3 seed | 6 | ±11,7 jam | murah, tetapi bukan varian terbaik |

  Ditambah pembuatan masker ±3,5-7 jam dan evaluasi ±1-2 jam. Usulan: **B**, lalu tambah varian n bila waktu memungkinkan.

## 4. Evaluasi

- **Test dan test sesi baru tanpa latar** dievaluasi SEKALI per model v9 (evaluate.py, kunci sebelum test), bootstrap per kelompok seperti v7.
- **ds3 tanpa latar** sebagai pembanding "sudah terbuka" (ds3 sudah dilihat; bukan uji bebas). Ini uji utama hipotesis: daun petik di
  atas kertas putih (ds3) seharusnya lebih mirip data latih bila latar dihapus. Masker ds3 dibuat dengan metode yang sama.
- **Tabel 2x2** (varian s; rata-rata 3 seed, IK 95%):

| | Dengan latar (v7, test asli) | Tanpa latar (v9, test tanpa latar) |
|---|---|---|
| YOLOv8s | ada (tabel utama v7) | baru |
| YOLO11s | ada (tabel utama v7) | baru |

  Selisih berpasangan (v9 - v7 per seed, bootstrap per kelompok) untuk mAP50, mAP50-95, dan akurasi diagnosis; dan selisih
  YOLO11s - YOLOv8s di dalam tiap kolom (pertanyaan inti paper: apakah keunggulan YOLO11 bertahan tanpa latar).
- Catatan: kolom "tanpa latar" mengubah data latih DAN data uji sekaligus, jadi yang diukur adalah efek seluruh alur (pemisahan + model).
  Sel silang (model v7 pada test tanpa latar, model v9 pada test asli) berarti membuka test lagi; hanya dikerjakan bila disetujui dan
  dilabeli post-hoc. Tahap 1 sudah memberi gambaran sel silang di VAL (model v7 pada VAL tanpa latar,
  ANALISIS VAL): mAP50-95 bersih turun YOLOv8s 0,232 -> 0,176 dan YOLO11s 0,247 -> 0,113; akurasi diagnosis YOLOv8s 0,875 -> 0,807 dan
  YOLO11s 0,880 -> 0,697. Penurunan hampir seluruhnya dari Blast yang bergeser menjadi Brown spot (YOLO11s seed 0: Blast benar 188 -> 43);
  Brown spot tetap, dan alarm palsu pada daun sehat sedikit berkurang. Isian hitam lebih buruk daripada abu-abu, jadi penyebabnya bukan
  warna abu-abu yang mirip pusat bercak Blast. Artinya model v7, terutama YOLO11, memakai konteks di luar daun untuk Blast. v9 menguji
  apakah model yang dilatih tanpa latar bisa mengenali Blast dari bercaknya saja.

## 5. Konsekuensi untuk aplikasi (nanti)

- Model v9 hanya melihat daun di atas latar abu-abu, jadi **aplikasi wajib menjalankan pemisahan daun yang sama sebelum deteksi**. Tanpa itu
  latar asli menjadi pergeseran domain baru.
- SAM2.1-b (bobot 154 MB, ±1-1,4 detik per foto di GPU) terlalu berat untuk Streamlit Community Cloud (CPU, RAM terbatas): perkiraan
  puluhan detik per foto di CPU dan RAM > 1 GB, belum diukur. Perlu pengganti ringan (mis. MobileSAM atau model segmentasi daun kecil),
  yang hasilnya berbeda dari masker latih.
- Kegagalan pemisahan (tangan/kaki/malai terpilih, daun terpotong) langsung menjadi diagnosis salah atau "tidak terdeteksi"; foto cadangan
  (tidak dipisah) memberi model masukan yang tidak pernah ia lihat saat latih (bila cadangan jarang di train).
- Aplikasi saat ini (YOLOv8s v7) tidak diubah; keputusan model aplikasi ditunda sampai skripsi.

## 6. Alternatif: "latar diacak" (augmentasi saat training, tanpa pemisahan di aplikasi)

- Masker dari langkah yang sama dipakai HANYA untuk data latih: daun dipotong lalu ditempel ke latar lain, sebagai augmentasi luring
  (salinan tambahan dengan seed tetap) atau daring (hook augmentasi).
- Jenis latar: sawah (latar foto latih lain, daun utamanya dihapus/ditutup), kertas putih (sintetis dengan tekstur dan bayangan halus),
  gelap, dan tanah. Sumber latar harus berlisensi jelas: utamakan latar dari data latih sendiri dan latar sintetis.
- Val, test, test sesi baru, dan ds3 TETAP foto asli (tanpa pemisahan), sehingga evaluasi langsung mencerminkan aplikasi.
- Kotak ikut digeser bersama daun (bila daun dipindah/diubah skala), atau daun ditempel di posisi yang sama (kotak tetap).

## 7. Perbandingan kedua rancangan

| Aspek | v9 latar dihapus | Latar diacak (augmentasi) |
|---|---|---|
| Pertanyaan yang dijawab | apakah model lebih baik bila hanya melihat daun | apakah model lebih tahan terhadap latar yang berbeda |
| Biaya data | masker untuk 8.959 foto (±3,5-7 jam GPU) + 2-3 GB | masker untuk train saja (±2,5-5,5 jam GPU) + komposisi; data tambahan bila luring |
| Biaya training (pilihan B) | ±20,5 jam GPU | ±20,5 jam GPU (lebih bila jumlah foto latih digandakan) |
| Risiko utama | kegagalan masker membuang bercak (kotak dibuang, terutama BLB ds1); evaluasi bergantung pada kualitas masker; jalur aplikasi lebih berat | artefak tepi tempelan dipelajari sebagai petunjuk; latar sintetis kurang realistis; tidak menghapus bias latar pada data uji |
| Dampak ke aplikasi | wajib pemisahan daun di aplikasi (lebih lambat, sumber galat baru) | tidak ada perubahan alur aplikasi; hanya model yang diganti |
| Keterbandingan dengan v7 | test berubah (tanpa latar), sehingga 2x2 mencampur efek data dan model | test sama persis dengan v7, sehingga selisih murni efek pelatihan |
| ds3 (kertas putih) | latar ds3 juga dihapus -> lebih mirip latih | latar putih termasuk jenis latar latih -> diuji langsung pada foto asli |

Catatan rekomendasi:
- Untuk pertanyaan paper (YOLOv8 vs YOLO11), temuan tahap 1 membuat **v9 latar dihapus** layak dijalankan lebih dulu: ia menguji langsung
  apakah keunggulan YOLO11 di v7 bergantung pada konteks latar (YOLO11 turun jauh lebih tajam tanpa latar). Usulan: pilihan B (YOLOv8s dan
  YOLO11s x 3 seed, ±20,5 jam GPU + masker ±3,5-7 jam), BLB dilaporkan terpisah dan tidak dipakai untuk kesimpulan latar.
- **Latar diacak** lebih cocok untuk aplikasi (alur aplikasi tidak berubah, test sama dengan v7) dan bisa menjadi eksperimen lanjutan
  dengan biaya training serupa bila v9 menunjukkan bahwa model bisa belajar dari bercak saja.
- Keputusan yang perlu disetujui: (1) metode SAM2 aturan v2 + cadangan; (2) aturan buang-kotak 50% dan perlakuan foto yang kehilangan
  semua kotak; (3) pilihan jumlah run; (4) apakah sel silang di test boleh dibuka sebagai post-hoc; (5) urutan v9 vs latar diacak.
