# prov_v8 (BEKU 2026-10-14 13:01; manifest c175e1ad…67d3)

**DIARSIPKAN: v8 dihentikan sebelum training; tidak pernah dilatih. Model final = v7. Lihat `ARSIP_DIHENTIKAN.md`.**

Isi: prov_v7 (semua 8.959 baris identik) + daunpadi (train/val) + Tungro ds4 (background negatif, train saja). Keputusan:
KEPUTUSAN_PROYEK.md 2026-10-14 "KEPUTUSAN v8". Skrip: `scripts/bangun_prov_v8.py` (seleksi, bangun, audit, bekukan). Tidak ada
model yang dijalankan.

## Gambar / grup unik per kelas, sumber, split

| Kelas | Sumber | train | val | test | test_new_session |
|---|---|---|---|---|---|
| background | ds2 | 121 / 76 | 24 / 18 | 12 / 9 | 0 |
| background | ds4 (sehat) | 556 / 212 | 293 / 89 | 146 / 49 | 56 / 35 |
| background | **ds4_tungro (baru)** | **600 / 202** | 0 | 0 | 0 |
| BLB | ds1_provisional | 1.401 / 145 | 390 / 41 | 195 / 21 | 0 |
| Brown spot | **daunpadi (baru)** | **372 / 325** | **85 / 81** | 0 | 0 |
| Brown spot | ds1 | 1.970 / 194 | 0 | 0 | 0 |
| Brown spot | ds4 | 1.480 / 225 | 330 / 74 | 165 / 43 | 147 / 71 |
| Blast | **daunpadi (baru)** | **353 / 317** | **87 / 78** | 0 | 0 |
| Blast | ds1 | 411 / 67 | 0 | 0 | 0 |
| Blast | ds4 | 833 / 249 | 222 / 84 | 111 / 48 | 96 / 62 |
| **Total** | | **8.097 / 1.817** | **1.431 / 404** | 629 / 125 | 299 / 133 |

Mesin: `tabel_split.csv`. Satu grup bisa memuat lebih dari satu kelas, jadi jumlah grup per baris tidak dijumlahkan.

- Kotak baru: daunpadi train Brown spot 1.594, Blast 775; daunpadi val Brown spot 368, Blast 194.
- 0 kotak dikliping.
- 12 gambar daunpadi bertanda augmentasi piksel (aturan v7): seluruh kelompoknya ditaruh di train.
- Background daun sehat di train 9,03% (batas 10%). Tungro dihitung terpisah: 600 = 7,41% dari train.
- data_valsel v8 (`data/derived/prov_v8_valsel`): 1.041 gambar = 869 val bersih v7 + 172 val daunpadi.

## Aturan kelompok dan seleksi

Graf atas semua foto mentah (ds1, ds2 Original, ds3, ds4 semua folder, aditia, daunpadi).

Sisi graf:
- kelompok prov_v7 yang sama;
- jendela waktu geser 5 dtk (ds4 dan ds1);
- untuk simpul baru (Tungro, Leaf smut, Sheath blight, daunpadi): phash ≤14 (8 varian dihedral), cos DINOv2 ≥0,97, dan SIFT ≥21 inlier
  pada 100 tetangga, tanpa syarat sebaran (lebih konservatif daripada v5).

Aturan seleksi:
- **daunpadi:** satu kelas saja. Dipakai 897 dari 1.999 gambar:
  - 1.038 dibuang karena berlabel campuran, Hispa, atau tanpa label;
  - 64 dibuang karena kelompoknya memuat ds1.
  Pembagian 80/20 per kelompok per kelas, seed 42.
- **Tungro:** 1.003 dari 2.244 layak. Penyebab tidak layak (satu foto bisa punya lebih dari satu penyebab):
  - kelompoknya memuat Leaf smut/Sheath blight (987), val v7 (765), test v7 (341), atau test_new_session (31);
  - berada di rentang sesi test_new_session ±600 dtk (57).
  Dipilih acak 600 dengan seed 42. Sebanyak 284 di antaranya sekelompok dengan foto train v7 (aman, sama-sama train).
- **Luar cakupan ds4:** foto yang kelompoknya memuat train/val prov_v8 dikeluarkan.
  - Leaf smut: 278 dari 724.
  - Sheath blight: 1.431 dari 1.722.

## Pemeriksaan (semua LOLOS)

| Pemeriksaan | Hasil |
|---|---|
| Kesetaraan dengan prov_v7 (`pemeriksaan_kesetaraan.json`) | baris, gambar, dan label v7 identik; test dan test_new_session tidak berubah; 0 kelompok baru lintas split |
| Audit SIFT k=200, val/test/tns vs train (`audit.json`) | val 3/1.431 (0,21%), test 1/629 (0,16%), test_new_session 0; total 0,17% (< 1%); **0 melibatkan gambar baru** |
| Audit waktu ds4 ±5 dtk | 0 di semua split (termasuk terhadap Tungro) |
| Audit waktu ±10 dtk (hanya dilaporkan) | val 110, test 79, test_new_session 0; dari Tungro: val 23, test 18 |
| Kembar langsung dengan set uji lain | ds3 (240), aditia (1.500), luar cakupan ds4 (1.709): 0 terhadap train baru dan 0 terhadap seluruh train |
| verify_dataset (`verify_dataset.log`) | LOLOS: re-hash lintas split 0 pasangan (jarak min 24); label 0 cacat |
