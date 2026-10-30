# Penyaringan kandidat dataset v8 (2026-10-14)

Tahap ini hanya penyaringan. Tidak ada training dan **tidak ada model deteksi yang dijalankan pada kandidat**. Piksel kandidat
hanya dibaca untuk profil (warna/latar), montase label, dan deteksi duplikat (phash, embedding DINOv2, SIFT), sama seperti
penyaringan ds3.

Skrip: `scripts/unduh_kandidat_v8.py` (venv alat), `scripts/profil_kandidat_v8.py`, `scripts/kembar_kandidat_v8.py`.
Data mentah: `data/raw/kandidat_v8/<nama>/`. Hasil: `reports/kandidat_v8/` (profil_*.json, profil_ringkas.csv, montase_*.jpg,
kembar/ringkasan.json, kembar/pasangan*.csv, kembar/per_gambar.csv, kembar/pasangan_teratas_*.jpg).

EXIF/GPS: 0 dari 8.650 gambar punya EXIF, GPS, atau data kamera (ekspor Roboflow/Kaggle sudah membuangnya). Tidak ada yang perlu
dihapus. Akibatnya juga tidak ada petunjuk kamera.

## 1. Profil

| | rf_gymvi_5class | rf_aditia_4ued0 | rf_daunpadi_b1cjw | kg_hadiur_ricespot |
|---|---|---|---|---|
| Sumber | Roboflow dataset-gymvi, versi 2 (tanpa augmentasi), 2021-12-02 | Roboflow aditia-putra, versi 8 (tanpa augmentasi), 2023-12-22 | Roboflow daun-padi-8ckhf, versi 25 (tanpa augmentasi), 2024-12-12 | Kaggle hadiurrahmannabil (ekspor Roboflow "Rice-SwOG", **sudah beraugmentasi 3x**: flip) |
| Lisensi | CC BY 4.0 | CC BY 4.0 | CC BY 4.0 | **CC BY-NC-SA 4.0** |
| Gambar | 1.584 (416x416, diregangkan) | 1.500 (640x640) | 1.999 (640x640) | 3.567 (640x640) |
| Kelas (gambar/kotak) | blast 330/792, blight 330/585, brownspot 330/1.137, healthy 330/447, hispa 264/708 | Daun Padi Normal 500/500, Hawar Daun 500/500, Narrow Brown Spot 500/500 | Brownspot 742/2.937, leafblast 935/1.871, Hispa 882/5.051 | 8 kelas (BLB 427/646, Brown_Spot 451/1.403, HealthyLeaf 433/821, Leaf_Blast 449/763, Leaf_Scald, Narrow_Brown, Neck_Blast, Hispa) |
| Format label | YOLO kotak | YOLO kotak | YOLO kotak | campuran (6.237 baris poligon); 24 gambar tanpa label |
| Median luas kotak | blast 0,81%; blight 3,0%; brownspot 0,10%; healthy 46% | 52–69% (satu kotak = seluruh daun) | ~0,07% (bercak sangat kecil) | BLB 18,4%; Leaf_Blast 3,1%; Brown_Spot 0,29% |
| Jenis foto | 79% daun petik berlatar polos | 100% daun petik berlatar polos (papan putih/biru) | 94% daun petik berlatar polos | 87% daun di tanaman/lapangan |
| Petunjuk asal | nama IMG_#### (561; seri yang sama dengan ds3) + IMG_20190419_* (996; juga ada di ds1) | nama berwaktu 20231116/20231117; judul & kelas berbahasa Indonesia | nama "*Kaggle##" (1.867) dan "*BKL##" (132) -> unggah ulang dari Kaggle | nama aug_* (3.012), IMG2020* (453) |

Acuan prov_v7 train (median luas kotak): ds4 blast 0,29%, ds4 brown spot 0,28%, ds1 BLB 20,0%, ds1 brown spot 0,35%, ds1 blast 3,15%.

## 2. Kembaran

Kriteria (sama dengan ds3): phash <=14 (8 varian dihedral) ATAU cos DINOv2 >=0,97 ATAU SIFT >=21 inlier. Pembanding foto mentah:
ds1 (8.667), ds2 Original (1.701), ds3 (240), ds4 Original (9.769, termasuk Rice Tungro). Untuk ds3 dicek dua arah
(foto ds3 -> 100 tetangga di tiap kandidat).

| Gambar kandidat yang punya kembar di | gymvi | aditia | daunpadi | Kaggle hadiur |
|---|---|---|---|---|
| **ds3** (harus 0) | **220** (blast 102 pasangan, blight 123; 304 dengan transitif) | 0 | 0 (juga 0 transitif lewat gymvi) | 0 |
| ds1 | 22 | 0 | 94 (80 kelompok) | **3.536 dari 3.567** |
| ds2 | 0 | 0 | 0 | 2.077 |
| ds4 | 0 | 0 | 0 | 0 |
| antar kandidat | daunpadi 228 | 0 | gymvi 121 | 0 |
| Kelompok internal | 1.049 | 9 (lihat catatan) | 1.783 | 710 |

Cek visual (`kembar/pasangan_teratas_*.jpg`):
- **gymvi ~ ds3:** daun yang sama (diputar, diperkecil ke 416, diberi tempelan potongan). Seri IMG_#### gymvi = foto ds3.
- **Kaggle ~ ds1:** gambar identik (cos 1,0; 402 inlier). Ada juga 34 gambar Kaggle yang kembar dengan **test prov_v7** dan
  116 yang kembar dengan val prov_v7.
- **daunpadi ~ gymvi:** daun yang sama pada pasangan berinlier tinggi (seri IMG_20190419). Pasangan berinlier rendah (25–33)
  pada daun polos meragukan.
- **aditia internal:** 9 kelompok adalah rantai embedding karena latar papan yang sama (21.661 pasangan "emb" saja). Tanpa
  pasangan emb-saja ada 68 kelompok; SIFT juga membentuk rantai besar (418), kemungkinan dari latar. Petunjuk yang lebih jujur:
  **677 cap waktu foto unik** (maks 5 gambar per cap waktu). Ukuran sampel efektif aditia <=677 foto, bukan 1.500.
  **[DIKOREKSI 2026-10-14]** Analisis lanjutan (`aditia_kelompok/`) membuktikan angka 677 terlalu tinggi dan dugaan "rantai dari latar"
  keliru: SIFT menyatukan foto dari helai daun yang SAMA. Tiap kelas = rentetan foto beruntun dari sedikit helai:
  - Normal: 500 foto dalam 4,2 menit, 6 kelompok waktu.
  - Hawar Daun: 500 foto dalam 10,4 menit, 11 kelompok waktu, 1 kelompok dengan SIFT.
  - Narrow Brown Spot: 500 foto dalam 25,4 menit, 81 kelompok waktu, 1 kelompok dengan SIFT.
  Sampel efektif sekitar 1 helai daun per kelas.
- **aditia: kelas = hari pemotretan.** Hawar Daun semuanya 2023-11-17; Normal dan Narrow Brown Spot semuanya 2023-11-16.

## 3. Rekomendasi

| Kandidat | Rekomendasi | Alasan utama |
|---|---|---|
| rf_gymvi_5class | **DITOLAK** | 220 gambar kembar dengan ds3 (blast dan blight); blast/blight-nya hampir seluruhnya seri foto ds3 (IMG_####), sehingga sisa yang "bebas kembar" pun satu sesi dengan ds3. Memakainya merusak ds3 sebagai pembanding. BLB lesion-level-nya menarik, tetapi justru berasal dari sumber ds3. |
| kg_hadiur_ricespot | **DITOLAK** | 3.536/3.567 gambar adalah unggah ulang ds1/ds2 (hanya 9 bebas), 34 kembar dengan test prov_v7, augmentasi flip bawaan, dan lisensi NC-SA. Bukan sumber baru. |
| rf_daunpadi_b1cjw | **TRAIN** (bersyarat) | 0 kembar ds3/ds2/ds4; daun petik berlatar polos dengan bercak kecil (prioritas 2 dan 3). Pakai hanya gambar satu kelas (aturan tanpa multi-penyakit) di luar kelompok kembar ds1: **Brownspot 459 gambar / 412 kelompok, leafblast 445 / 402 kelompok** (median kotak 0,07%). Hispa (534 gambar satu kelas) di luar cakupan -> tidak dipakai kecuali diputuskan lain. Bukan foto Indonesia asli (sumber Kaggle), jadi tidak cocok untuk set uji. |
| rf_aditia_4ued0 | **SET UJI SEGAR** (level citra) — **[DIKOREKSI: ~1 helai daun per kelas; tidak layak sebagai set uji kuantitatif]** | 0 kembar dengan semua sumber dan kandidat; foto Indonesia, daun petik berlatar polos. Sebagai uji: Normal -> sehat, Hawar Daun -> BLB (**perlu verifikasi label** seperti ds3; "hawar daun" belum tentu hawar daun bakteri), Narrow Brown Spot -> di luar cakupan (sebaran diagnosis). Tidak untuk TRAIN: kotak seluruh daun (gaya kotak penyebab dugaan shortcut BLB) dan kelas terkacaukan dengan hari pemotretan. |

Kekosongan yang tetap ada:
- **BLB dari sumber selain ds1 untuk TRAIN: tidak ada kandidat yang lolos.** Satu-satunya BLB bersih (aditia) diusulkan untuk uji.
- **Tungro: tidak ada di kandidat mana pun.** ds4 punya 2.244 foto Rice Tungro (lapangan, Bangladesh) yang belum dipakai di
  prov_v7 dan 0 kembar dengan ds3. Bila dipakai sebagai negatif TRAIN, set uji Tungro segar harus dari sumber lain (foto sendiri).
- **Blast dan Brown spot untuk set uji Indonesia: tidak ada.** aditia tidak punya kelas ini; perlu foto sendiri (protokol di
  KEPUTUSAN_PROYEK.md).
