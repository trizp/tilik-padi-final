# Sensitivitas ambang pengelompokan kembar (ANALISIS UNTUK REVISI PAPER; tanpa training, split beku prov_v7)

Foto: 8959 (test 629, test_new_session 299, val 1259). Kelompok beku dipakai sebagai dasar, lalu sisi dari ambang longgar ditambahkan. 'Kembar langsung' = foto punya pasangan di TRAIN menurut kriteria pengaturan itu; 'sekelompok' = kelompok (rantai) foto memuat foto train.

| Pengaturan | Kelompok | Terbesar | Test kembar langsung | Test sekelompok | TNS kembar langsung | TNS sekelompok | Val kembar langsung |
|---|---|---|---|---|---|---|---|
| bawaan (0,97 / 21 / 5 dtk) | 1544 | 204 | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| DINOv2 0,95 | 1055 | 2829 | 117 (18.6%) | 293 (46.58%) | 90 (30.1%) | 200 (66.89%) | 216 (17.16%) |
| SIFT 15 inlier | 1506 | 204 | 6 (0.95%) | 65 (10.33%) | 0 (0.0%) | 0 (0.0%) | 12 (0.95%) |
| waktu 10 dtk | 1245 | 321 | 69 (10.97%) | 191 (30.37%) | 0 (0.0%) | 0 (0.0%) | 91 (7.23%) |
| gabungan (0,95 / 15 / 10 dtk) | 841 | 3191 | 164 (26.07%) | 379 (60.25%) | 90 (30.1%) | 218 (72.91%) | 287 (22.8%) |
| SIFT 15 + waktu 10 dtk (DINOv2 0,97) | 1211 | 321 | 74 (11.76%) | 195 (31.0%) | 0 (0.0%) | 0 (0.0%) | 102 (8.1%) |

Rincian per kriteria (foto test dengan kembaran langsung di train):

- bawaan (0,97 / 21 / 5 dtk): dino_0.97=0, sift_21=0, waktu_5=0
- DINOv2 0,95: dino_0.95=117, sift_21=0, waktu_5=0
- SIFT 15 inlier: dino_0.97=0, sift_15=6, waktu_5=0
- waktu 10 dtk: dino_0.97=0, sift_21=0, waktu_10=69
- gabungan (0,95 / 15 / 10 dtk): dino_0.95=117, sift_15=6, waktu_10=69
- SIFT 15 + waktu 10 dtk (DINOv2 0,97): dino_0.97=0, sift_15=6, waktu_10=69
