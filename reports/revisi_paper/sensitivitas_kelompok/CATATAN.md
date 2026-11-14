# Catatan: sensitivitas ambang pengelompokan (ANALISIS UNTUK REVISI PAPER SOFTT)

Dijalankan 13 November 2026 dengan `scripts/paper/sensitivitas_kelompok.py` (sekitar 15 detik; tanpa training, tanpa menghitung
ulang embedding atau SIFT) dan `scripts/paper/montase_kembar_longgar.py`. Dataset beku prov_v7, split, dan ambang tidak diubah;
ds3 tidak dibuka.

## Sumber data (semua sudah tersimpan)

| Kriteria | Bawaan | Longgar | Sumber |
|---|---|---|---|
| DINOv2-small cosine (8 varian dihedral) | 0,97 | 0,95 | `data/interim/dino_emb.npz` (8959/8959 foto cocok ukuran+mtime) |
| SIFT inlier (sebaran kotak >= 2% tetap) | 21 | 15 | `reports/prov_v6/sift_pairs.csv` (kNN k=100; semua pasangan terverifikasi dengan inlier >= 8 disimpan, jadi ambang 15 lengkap) |
| Selang waktu (jendela geser, ds4+ds1) | 5 dtk | 10 dtk | `Dataset metadata.xlsx` (ds4) dan nama berkas (ds1); fungsi `dedupe_rules.time_pairs` yang sama |

prov_v7 mewarisi kelompok dan split prov_v6 tanpa perubahan (`dataset_meta.json`: grup_split_sha256 sama), jadi data prov_v6 berlaku.

## Pemeriksaan kewajaran

- Pengaturan bawaan: 0 foto val/test/test_new_session punya pasangan di train menurut ketiga kriteria. Ini sesuai dengan split beku.
- Waktu 10 dtk: 69/629 test (10,97%) dan 91/1259 val (7,23%) punya foto train dalam 10 detik. Angka ini **sama persis** dengan audit
  ±10 dtk di `reports/prov_v6/split_report.md` bagian 18.

## Temuan utama

- **SIFT 15 dan waktu 10 dtk** menangkap foto yang memang berdekatan (adegan atau tanaman yang sama, beberapa daun yang sama;
  `montase_kembar_longgar.jpg` baris 3-4). Bersama-sama: 74/629 test (11,76%) punya pasangan langsung di train; 0/299 test sesi baru.
- **DINOv2 0,95 tidak spesifik**: 117/629 test (18,6%) tetapi juga 90/299 test sesi baru (30,1%). Test sesi baru adalah sesi
  pemotretan terpisah (jeda >= 600 dtk, 0 tautan SIFT/waktu ke train). Pada ds4 saja lajunya setara (test 20,7-32,7% per kelas, sesi
  baru 20,8-38,8%). Montase baris 1-2: daun berbeda dengan komposisi serupa, termasuk pasangan lintas kelas (blas ~ bercak cokelat).
  Ini sejalan dengan kalibrasi lama (`classes_v7.yaml`: pada 0,95 terbentuk grup raksasa karena rantai kemiripan).
- Kelompok (rantai) membesar cepat: 1544 kelompok (bawaan) -> 1506 (SIFT 15), 1245 (waktu 10), 1055 (DINOv2 0,95), 841 (gabungan),
  dengan kelompok terbesar 204 -> 3191 foto. Persentase "sekelompok dengan train" (rantai) jauh lebih besar daripada kembaran
  langsung dan merupakan batas atas yang longgar.
- BLB (ds1, test 195 foto) tidak punya kembaran di train pada pengaturan mana pun (dikelompokkan juga dengan kunci K4 dan phash).

## Kalimat untuk Subbab III-B (bahasa Inggris)

> With looser grouping thresholds (SIFT >= 15 inliers and a capture gap <= 10 s instead of 21 inliers and 5 s), 74 of the 629 test
> images (11.8%) but none of the 299 new-session images would have a near-duplicate in the training set; lowering the DINOv2
> threshold to 0.95 flagged 18.6% of test and 30.1% of new-session images, but these pairs showed different leaves with a similar
> composition, indicating generic visual similarity rather than duplication.

## Batasan

- Simpul graf = 8959 foto prov_v7. Foto yang dibuang setelah pengelompokan (karantina 141, pangkasan background, 9 foto lintas sesi)
  tidak ikut sebagai jembatan rantai pada sisi longgar; rantai bawaan lewat foto itu sudah termuat di kolom `group`. Jumlah
  kembaran LANGSUNG ke train tidak terpengaruh (foto train semuanya ada di graf); jumlah kelompok bisa sedikit lebih besar.
- SIFT hanya diverifikasi pada 100 tetangga terdekat embedding (seperti pembangunan dataset); pasangan di luar itu tidak terlihat.
- Montase hanya 4 pasangan per kriteria (sampel acak seed tetap), bukan penilaian manual menyeluruh.
