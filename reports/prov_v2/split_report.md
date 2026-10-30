# Laporan split dataset `prov_v2`

- Seed: 42 | rasio split: {'train': 0.7, 'val': 0.2, 'test': 0.1} | phash 16x16, ambang 14 (8 varian dihedral: rotasi 0/90/180/270 x flip)
- Total gambar: **9068** | `healthy_mode`: background
- **PROVISIONAL**: sumber `ds1_provisional` belum final. Run diberi awalan `prov_v2_` dan **tidak boleh dipakai untuk tabel paper**.

## 1. Gambar dan box per kelas per split

| kelas | train | val | test | total gambar | grup unik |
|---|---|---|---|---|---|
| leaf_blast | 1344 (3918 box) | 262 (1099 box) | 131 (526 box) | 1737 | 723 |
| brown_spot | 3504 (21803 box) | 430 (1838 box) | 215 (869 box) | 4149 | 1032 |
| bacterial_leaf_blight | 1400 (1703 box) | 391 (434 box) | 195 (200 box) | 1986 | 213 |
| background | 694 | 335 | 167 | 1196 | 805 |
| **total** | 6942 | 1418 | 708 | 9068 | 2673 |

Porsi background: **13.2%** dari total (train 10.0%, val 23.6%, test 23.6%). Ultralytics menyarankan 0-10%.

## 2. Gambar per kelas per SUMBER per split

| kelas | sumber | train | val | test | total | porsi dari kelas |
|---|---|---|---|---|---|---|
| bacterial_leaf_blight | ds1_provisional ⚠ provisional | 1400 (149 grup) | 391 (42 grup) | 195 (22 grup) | 1986 (213 grup) | 100% |
| brown_spot | ds1 | 1972 (199 grup) | 0 (0 grup) | 0 (0 grup) | 1972 (199 grup) | 48% |
| brown_spot | ds4 | 1532 (572 grup) | 430 (170 grup) | 215 (91 grup) | 2177 (833 grup) | 52% |
| leaf_blast | ds1 | 412 (72 grup) | 0 (0 grup) | 0 (0 grup) | 412 (72 grup) | 24% |
| leaf_blast | ds4 | 932 (446 grup) | 262 (133 grup) | 131 (72 grup) | 1325 (651 grup) | 76% |
| background | ds2 | 121 (76 grup) | 24 (18 grup) | 12 (9 grup) | 157 (103 grup) | 13% |
| background | ds4 | 573 (414 grup) | 311 (191 grup) | 155 (97 grup) | 1039 (702 grup) | 87% |

**Kelas dengan satu sumber saja:** `bacterial_leaf_blight` hanya dari `ds1_provisional`.

Jumlah **box** per kelas per sumber per split ada di `split_counts.csv`.

## 3. Statistik luas box per kelas (relatif terhadap gambar, semua split)

| kelas | sumber | gambar | box | box/gambar | median luas | rata2 luas | box > 1% luas gambar | box dari poligon |
|---|---|---|---|---|---|---|---|---|
| leaf_blast | *semua sumber* | 1737 | 5543 | 3.19 | 0.35% | 1.65% | 17.6% | 12.4% |
| leaf_blast | ds1 | 412 | 699 | 1.7 | 3.15% | 9.74% | 72.1% | 98.4% |
| leaf_blast | ds4 | 1325 | 4844 | 3.66 | 0.30% | 0.48% | 9.7% | 0.0% |
| brown_spot | *semua sumber* | 4149 | 24510 | 5.91 | 0.31% | 0.64% | 14.8% | 4.0% |
| brown_spot | ds1 | 1972 | 15161 | 7.69 | 0.35% | 0.76% | 21.0% | 6.5% |
| brown_spot | ds4 | 2177 | 9349 | 4.29 | 0.28% | 0.44% | 4.7% | 0.0% |
| bacterial_leaf_blight | *semua sumber* | 1986 | 2337 | 1.18 | 21.27% | 25.97% | 97.6% | 15.6% |
| bacterial_leaf_blight | ds1_provisional | 1986 | 2337 | 1.18 | 21.27% | 25.97% | 97.6% | 15.6% |

"Box dari poligon" = bbox pembatas yang diturunkan dari label poligon (segmentasi) sumber; sisanya box asli.

Perbandingan antar kelas: median luas terbesar `bacterial_leaf_blight` (21.27%) vs terkecil `brown_spot` (0.31%) = **69x**. **Skala box antar kelas tidak seragam** (unit anotasi berbeda) -> baca metrik per kelas terpisah.

## 4. Deduplikasi (phash)

- Grup unik: 2673 dari 9068 gambar; grup berisi >1 gambar: **939** (mencakup 7334 gambar).
- Grup terbesar: 116 gambar. Distribusi ukuran grup: {1: 1734, 2: 247, 3: 113, 4: 62, 5: 64, 6: 49, 7: 37, 8: 47} ...
- Grup lintas SUMBER: **1**; grup lintas KELAS pada dataset akhir: **93** (kemiripan adegan embedding yang memuat >1 kelas sebelum penyaringan: 97; tetap dijaga dalam satu split, tidak dibuang).
- Gambar yang berada di grup duplikat per sumber: {'ds4': 2951, 'ds1_provisional': 1962, 'ds1': 2342, 'ds2': 79}
- Contoh grup lintas kelas: 16 gambar {'leaf_blast': 6, 'brown_spot': 10}; mis. blast_1000.jpg, blast_1001.jpg, blast_1002.jpg
- Contoh grup lintas kelas: 5 gambar {'leaf_blast': 2, 'brown_spot': 3}; mis. blast_1020.jpg, blast_1021.jpg, brown spot_1270.jpg
- Contoh grup lintas kelas: 6 gambar {'leaf_blast': 3, 'brown_spot': 3}; mis. blast_1112.jpg, blast_1113.jpg, blast_1114.jpg
- Contoh grup lintas kelas: 25 gambar {'leaf_blast': 3, 'brown_spot': 22}; mis. blast_1180.jpg, blast_1181.jpg, blast_1182.jpg
- Contoh grup lintas kelas: 6 gambar {'leaf_blast': 2, 'brown_spot': 4}; mis. blast_1183.jpg, blast_1184.jpg, brown spot_1716.jpg
- Pasangan yang menyatukan grup: phash <= 14 (8 varian dihedral): **3862**
- Pasangan yang menyatukan grup: embedding cosine >= 0.97 (DINOv2-small, 8 varian dihedral): **7587**
- Pasangan yang menyatukan grup: nama dasar Roboflow sama (.rf.<hash>): **3017**
- Pasangan yang menyatukan grup: kunci ID K4 (fam,jenis,id,ekstensi; lintas awalan; ds1): **5626**
- Pasangan yang menyatukan grup: SIFT >= 21 inlier pada 30 tetangga embedding (belum satu grup): **9693**
- **Verifikasi leakage** (jarak phash minimum antar split, 8 varian dihedral; ambang 14): val-train: 0 pasangan <= 14 (jarak min 48), test-train: 0 pasangan <= 14 (jarak min 60), test-val: 0 pasangan <= 14 (jarak min 66). **LOLOS (0 pasangan).**
- **Verifikasi embedding** (cosine maks antar split, 8 varian dihedral): val-train: maks 0.9698 (0 pasangan >= ambang), test-train: maks 0.9695 (0 pasangan >= ambang), test-val: maks 0.9673 (0 pasangan >= ambang). Versi Roboflow dengan nama dasar sama yang terpecah antar split: **0**.

## 5. Test multi-penyakit (gambar dengan >= 2 kelas)

- Gambar >= 2 kelas di seluruh dataset: **0**; di split test: **0**. Daftar: `data/processed/prov_v2/lists/test_multidisease.txt`.
- Tidak ada satu pun gambar multi-penyakit pada sumber yang dipakai (semua label satu kelas per gambar). Syarat test multi-penyakit pada CLAUDE.md **belum bisa dipenuhi**; perlu foto yang dianotasi manual.

## 6. Gambar/box yang dibuang

| sumber | alasan | jumlah |
|---|---|---|
| ds1 | duplikat_label_bertentangan | 79 |
| ds1 | memuat_kelas_di_luar_target | 4216 |
| ds1 | tanpa_label | 2 |
| ds2 | folder_tidak_dipakai:Bacterial Leaf Blight | 180 |
| ds2 | folder_tidak_dipakai:Brown Spot | 267 |
| ds2 | folder_tidak_dipakai:Leaf Blast | 305 |
| ds2 | folder_tidak_dipakai:Leaf scald | 189 |
| ds2 | folder_tidak_dipakai:Narrow Brown Leaf Spot | 117 |
| ds2 | folder_tidak_dipakai:Rice Hispa | 215 |
| ds2 | folder_tidak_dipakai:Sheath Blight | 271 |
| ds2 | path_dikecualikan:augmented | 5188 |
| ds4 | dipangkas_max_background_frac_train | 535 |
| ds4 | duplikat_label_bertentangan | 2 |
| ds4 | folder_tidak_dipakai:Leaf smut | 724 |
| ds4 | folder_tidak_dipakai:Rice Tungro | 2244 |
| ds4 | folder_tidak_dipakai:Sheath blight | 1722 |
| ds4 | label_kosong | 1 |

Catatan per sumber:
```json
{
  "ds4": {
    "id_label_mentah_per_folder_(diabaikan)": {
      "Blast": {
        "0": 4844
      },
      "Brown spot": {
        "1": 9352
      }
    }
  },
  "ds1": {
    "kelas_sumber_di_luar_target_(jumlah_box)": {
      "rice_leafsmut": 2000,
      "rice_leafscald": 456,
      "rice_neckblast": 453,
      "rice_narrowbrownleafspot": 443,
      "rice_healthy": 433,
      "rice_hispa": 431
    }
  },
  "ds2": {
    "background_dipakai": 157
  },
  "ds3": {
    "tidak_diproses": "role=external_test_reserved"
  },
  "grup_label_bertentangan_dibuang": [
    [
      "brown_spot <- data/raw/ds4/images_labels/RiceLeafDiseaseBD/Original images/Brown spot/brown spot_68.jpg",
      "background <- data/raw/ds4/images_labels/RiceLeafDiseaseBD/Original images/Healthy/Healthy_48.jpg"
    ],
    [
      "bacterial_leaf_blight <- data/raw/ds1/rice/images/train/rls_test_aug_0_609_jpg.rf.087dfe73c217d5c7594897efcb1eba9a.jpg",
      "brown_spot <- data/raw/ds1/rice/images/train/rls_test_aug_0_609_jpg.rf.08dcc4fb787feaf54a5f5f050cf43dd9.jpg"
    ],
    [
      "brown_spot <- data/raw/ds1/rice/images/train/rls_train_aug_0_1154_jpg.rf.1defbab0e5338d69dde1f883656d30d7.jpg",
      "leaf_blast <- data/raw/ds1/rice/images/train/rls_train_aug_0_1154_jpg.rf.315830ac7c3fed8bb0fd2aa9426f3f97.jpg",
      "brown_spot <- data/raw/ds1/rice/images/train/rls_train_aug_0_1154_jpg.rf.344a9047fe5ee0e11d14cce85200c634.jpg",
      "leaf_blast <- data/raw/ds1/rice/images/train/rls_train_aug_0_1154_jpg.rf.6d8643f9a964e4d8742c997584a287c7.jpg",
      "brown_spot <- data/raw/ds1/rice/images/train/rls_train_aug_0_1154_jpg.rf.957fffc9c2407feb7b6aa794f2463adc.jpg",
      "leaf_blast <- data/raw/ds1/rice/images/train/rls_train_aug_0_1154_jpg.rf.be1d0331f9e300498bb43eff88b7fbf4.jpg"
    ],
    [
      "bacterial_leaf_blight <- data/raw/ds1/rice/images/train/rls_train_aug_0_1280_jpg.rf.2859807f961c2b5786a7029d5efcd03e.jpg",
      "leaf_blast <- data/raw/ds1/rice/images/train/rls_train_aug_0_1280_jpg.rf.3b7d9c0d54d8d6e07115fbc030996c57.jpg",
      "leaf_blast <- data/raw/ds1/rice/images/train/rls_train_aug_0_1280_jpg.rf.b97d45b5f076fb64f4102cc05c00e81d.jpg",
      "leaf_blast <- data/raw/ds1/rice/images/train/rls_train_aug_0_1280_jpg.rf.e5c1e0d040447fdd0ea9c1976f2ba71d.jpg"
    ],
    [
      "leaf_blast <- data/raw/ds1/rice/images/train/rls_train_aug_0_1489_jpg.rf.165b58cb3f4f7373c3bfc8da027b7422.jpg",
      "leaf_blast <- data/raw/ds1/rice/images/train/rls_train_aug_0_1489_jpg.rf.6f5705f9273aa46ce094db8c68c5ffbf.jpg",
      "leaf_blast <- data/raw/ds1/rice/images/train/rls_train_aug_0_1489_jpg.rf.924d0c16af8c8957553d12e0dd8fc7f1.jpg",
      "bacterial_leaf_blight <- data/raw/ds1/rice/images/train/rls_train_aug_0_1489_jpg.rf.e6201cf374d64255d8b39a2ee672eed7.jpg"
    ],
    [
      "leaf_blast <- data/raw/ds1/rice/images/train/rls_train_aug_0_190_jpg.rf.14b09e7e5160e21ff9cad1c9c1975c72.jpg",
      "brown_spot <- data/raw/ds1/rice/images/train/rls_train_aug_0_190_jpg.rf.a8cbf52e241c971099331c010a6c59ad.jpg",
      "leaf_blast <- data/raw/ds1/rice/images/train/rls_train_aug_0_190_jpg.rf.d77ecc866e1f016d81cb2395eec7e2c3.jpg",
      "brown_spot <- data/raw/ds1/rice/images/train/rls_train_aug_0_190_jpg.rf.ea9a665d50a1b4c715936075277e6ea0.jpg"
    ],
    [
      "brown_spot <- data/raw/ds1/rice/images/train/rls_train_aug_0_222_jpg.rf.9693106a78bb126a58ac444510537599.jpg",
      "leaf_blast <- data/raw/ds1/rice/images/train/rls_train_aug_0_222_jpg.rf.acc95c2fc9dbd027085ed6380049b55c.jpg",
      "leaf_blast <- data/raw/ds1/rice/images/train/rls_train_aug_0_222_jpg.rf.cfaaae569c17ab5546191836c276bb17.jpg",
      "brown_spot <- data/raw/ds1/rice/images/val/rls_train_aug_0_222_jpg.rf.8363d7a5866a0ab37566ffba8256e897.jpg"
    ],
    [
      "leaf_blast <- data/raw/ds1/rice/images/train/rls_train_aug_0_2328_jpg.rf.020e470970a88a2d071a0798e8acd375.jpg",
      "leaf_blast <- data/raw/ds1/rice/images/train/rls_train_aug_0_2328_jpg.rf.1784551df0c2746fdfd98faa51faa78c.jpg",
      "bacterial_leaf_blight <- data/raw/ds1/rice/images/train/rls_train_aug_0_2328_jpg.rf.54e548d04030d926b1bf4324eded3a07.jpg",
      "bacterial_leaf_blight <- data/raw/ds1/rice/images/train/rls_train_aug_0_2328_jpg.rf.7436852c3f95ef6d98ef3cf3442c506a.jpg",
      "leaf_blast <- data/raw/ds1/rice/images/val/rls_train_aug_0_2328_jpg.rf.cd5965b5cf1775fee97462a5f27577db.jpg"
    ],
    [
      "leaf_blast <- data/raw/ds1/rice/images/train/rls_train_aug_0_334_jpg.rf.00888bf9df17917066bdc4dffd09589c.jpg",
      "leaf_blast <- data/raw/ds1/rice/images/train/rls_train_aug_0_334_jpg.rf.66af4a03f3804f9c211fc9dacc818cd6.jpg",
      "bacterial_leaf_blight <- data/raw/ds1/rice/images/train/rls_train_aug_0_334_jpg.rf.eea790e19eef19b2fc81ab201a1c7579.jpg",
      "leaf_blast <- data/raw/ds1/rice/images/val/rls_train_aug_0_334_jpg.rf.6b84e1457b656f9e4b1f09fd5ea8c9f5.jpg"
    ],
    [
      "bacterial_leaf_blight <- data/raw/ds1/rice/images/train/rls_train_aug_0_789_jpg.rf.4e7a10772af7dd0870bcffba171f223b.jpg",
      "bacterial_leaf_blight <- data/raw/ds1/rice/images/train/rls_train_aug_0_789_jpg.rf.5d05a16905eca2e668d064d19aae9d9f.jpg",
      "brown_spot <- data/raw/ds1/rice/images/train/rls_train_aug_0_789_jpg.rf.c9d526a701fc3fbf709d3d67069a242b.jpg",
      "brown_spot <- data/raw/ds1/rice/images/train/rls_train_aug_0_789_jpg.rf.e5a6ef42914b6192bddb5d7871efe180.jpg",
      "bacterial_leaf_blight <- data/raw/ds1/rice/images/train/rls_train_aug_0_789_jpg.rf.fabd87292f11f6b08fbc126cc97b5ff9.jpg"
    ],
    [
      "brown_spot <- data/raw/ds1/rice/images/train/rls_train_aug_0_875_jpg.rf.228f71cf7b9af282e666b9e9b8af4f53.jpg",
      "leaf_blast <- data/raw/ds1/rice/images/train/rls_train_aug_0_875_jpg.rf.4f0f84cc6001928019151e1112eeb703.jpg",
      "brown_spot <- data/raw/ds1/rice/images/train/rls_train_aug_0_875_jpg.rf.6bba819b7b751191787c183f80b6ccc8.jpg",
      "leaf_blast <- data/raw/ds1/rice/images/train/rls_train_aug_0_875_jpg.rf.91d7d8a28753ef19f9f2e259723d549a.jpg"
    ],
    [
      "brown_spot <- data/raw/ds1/rice/images/train/rls_val_aug_0_813_jpg.rf.b1e5710ed51515d8245d0f571f3ed895.jpg",
      "leaf_blast <- data/raw/ds1/rice/images/train/rls_val_aug_0_813_jpg.rf.f4678c41ac39556690408887bd93e169.jpg"
    ],
    [
      "brown_spot <- data/raw/ds1/rice/images/train/rls_train_aug_0_1004_jpg.rf.1cefcc27e9860622984788755413cb0d.jpg",
      "brown_spot <- data/raw/ds1/rice/images/train/rls_train_aug_0_1004_jpg.rf.691adbedc78bdaedea8c3a490755b5c4.jpg",
      "leaf_blast <- data/raw/ds1/rice/images/train/rls_train_aug_0_1004_jpg.rf.6965f5f75443689bad83df5f0859e194.jpg",
      "leaf_blast <- data/raw/ds1/rice/images/train/rls_train_aug_0_1004_jpg.rf.7f39766756dd8cba42d89186d94b0858.jpg",
      "brown_spot <- data/raw/ds1/rice/images/train/rls_train_aug_0_1004_jpg.rf.857ebe38bb340086cb01d8e724baebba.jpg",
      "leaf_blast <- data/raw/ds1/rice/images/val/rls_train_aug_0_1004_jpg.rf.932e7141c09a8a1880e12a425b5143ac.jpg"
    ],
    [
      "bacterial_leaf_blight <- data/raw/ds1/rice/images/train/rls_train_aug_0_1018_jpg.rf.50ac7c2873142fdfab47179e4323f805.jpg",
      "bacterial_leaf_blight <- data/raw/ds1/rice/images/train/rls_train_aug_0_1018_jpg.rf.a49a7cf6608ad0a5eeead549a1736370.jpg",
      "bacterial_leaf_blight <- data/raw/ds1/rice/images/train/rls_train_aug_0_1018_jpg.rf.a569b503d86582911bbbd940148f3412.jpg",
      "brown_spot <- data/raw/ds1/rice/images/train/rls_train_aug_0_1018_jpg.rf.d008a7a4a07f73e030038a821c8dfb32.jpg",
      "brown_spot <- data/raw/ds1/rice/images/val/rls_train_aug_0_1018_jpg.rf.a777e8fc8e7d62b50f9787123599b826.jpg"
    ],
    [
      "bacterial_leaf_blight <- data/raw/ds1/rice/images/train/rls_train_aug_0_1203_jpg.rf.3ac5b5e43440c18ceb943e39b61cb277.jpg",
      "brown_spot <- data/raw/ds1/rice/images/train/rls_train_aug_0_1203_jpg.rf.ab1ac3d9d461aa814cb68e2fdf01f734.jpg",
      "brown_spot <- data/raw/ds1/rice/images/val/rls_train_aug_0_1203_jpg.rf.833679aa16f3864567acbd66e700bfe0.jpg"
    ],
    [
      "brown_spot <- data/raw/ds1/rice/images/train/rls_train_aug_0_1396_jpg.rf.482bb1daaf6a91923351fe2abff3d7a0.jpg",
      "leaf_blast <- data/raw/ds1/rice/images/train/rls_train_aug_0_1396_jpg.rf.79c6865ab2b6ac55389a8536ad585661.jpg",
      "brown_spot <- data/raw/ds1/rice/images/train/rls_train_aug_0_1396_jpg.rf.953003fe122c2f1fc65f440893aa310a.jpg",
      "leaf_blast <- data/raw/ds1/rice/images/val/rls_train_aug_0_1396_jpg.rf.628e74230548216b77efcf355d1ae641.jpg"
    ],
    [
      "leaf_blast <- data/raw/ds1/rice/images/train/rls_train_aug_0_327_jpg.rf.03bf6adf0f62d243fec927a8dc91705a.jpg",
      "leaf_blast <- data/raw/ds1/rice/images/val/rls_train_aug_0_327_jpg.rf.40f27e3e9bb56558b4d38326141276e3.jpg",
      "bacterial_leaf_blight <- data/raw/ds1/rice/images/val/rls_train_aug_0_327_jpg.rf.d6db2569ba07481dbb8f0cdaa929adec.jpg"
    ],
    [
      "brown_spot <- data/raw/ds1/rice/images/train/rls_train_aug_0_435_jpg.rf.39faf01ff763fa0584675a33464f3447.jpg",
      "brown_spot <- data/raw/ds1/rice/images/train/rls_train_aug_0_435_jpg.rf.4ae0ec3c288b6f0f08f165fdc8407f4e.jpg",
      "leaf_blast <- data/raw/ds1/rice/images/train/rls_train_aug_0_435_jpg.rf.79e8d1cdde626258bed8b9f2b3499d24.jpg",
      "leaf_blast <- data/raw/ds1/rice/images/val/rls_train_aug_0_435_jpg.rf.b3eff67877e888af20d5a2ae0a4e0e16.jpg"
    ],
    [
      "leaf_blast <- data/raw/ds1/rice/images/train/rls_train_aug_0_647_jpg.rf.dd9bc1d9681369652778a5a50f4cfbc5.jpg",
      "leaf_blast <- data/raw/ds1/rice/images/train/rls_train_aug_0_647_jpg.rf.feddae9cf3657959979a97e801d9714a.jpg",
      "brown_spot <- data/raw/ds1/rice/images/val/rls_train_aug_0_647_jpg.rf.1f80223dce5763447cb90c54e32d3126.jpg"
    ],
    [
      "brown_spot <- data/raw/ds1/rice/images/train/rls_train_aug_0_65_jpg.rf.201885d10a36b36a5650692bdde8a006.jpg",
      "brown_spot <- data/raw/ds1/rice/images/train/rls_train_aug_0_65_jpg.rf.85b765bd0eec53f0dd9217de6bc3cb1a.jpg",
      "leaf_blast <- data/raw/ds1/rice/images/train/rls_train_aug_0_65_jpg.rf.a276230e2b5700945c7e0751702e6ca3.jpg",
      "leaf_blast <- data/raw/ds1/rice/images/val/rls_train_aug_0_65_jpg.rf.15df09300dc579e886b1b8239e0c91a0.jpg",
      "leaf_blast <- data/raw/ds1/rice/images/val/rls_train_aug_0_65_jpg.rf.c72c901020018ca27d6e8ad8351525f0.jpg"
    ],
    [
      "leaf_blast <- data/raw/ds1/rice/images/train/rls_val_aug_0_1362_jpg.rf.8dbe9c52368fd1a72d258d396e8cb3d8.jpg",
      "brown_spot <- data/raw/ds1/rice/images/val/rls_val_aug_0_1362_jpg.rf.7f499d8ba8813ff00becef5e13d6e071.jpg"
    ]
  ]
}
```

## 7. Kesesuaian rasio split per stratum (kelas|sumber)

| stratum | total | train | val | test |
|---|---|---|---|---|
| bacterial_leaf_blight|ds1_provisional | 1986 | 1400 (70%) | 391 (20%) | 195 (10%) |
| brown_spot|ds1 | 1972 | 1972 (100%) | 0 (0%) | 0 (0%) |
| brown_spot|ds4 | 2177 | 1532 (70%) | 430 (20%) | 215 (10%) |
| leaf_blast|ds1 | 412 | 412 (100%) | 0 (0%) | 0 (0%) |
| leaf_blast|ds4 | 1325 | 932 (70%) | 262 (20%) | 131 (10%) |
| background|ds2 | 157 | 121 (77%) | 24 (15%) | 12 (8%) |
| background|ds4 | 1039 | 573 (55%) | 311 (30%) | 155 (15%) |

Target 70/20/10. Stratum kecil atau berisi grup duplikat besar bisa menyimpang karena grup tidak boleh dipecah.

## 8. Deteksi augmentasi dan kelayakan val/test

Detektor: (a) pola nama (sumber ['ds1']), (b) heuristik piksel {'corner_black': 0.25, 'border_black': 0.08, 'smear': 0.25} (sumber ['ds1', 'ds4', 'ds2']), (c) embedding DINOv2-small cosine >= 0.97 (flag 'punya saudara' hanya sumber ['ds1']; juga menggabungkan grup). Kelas exempt (tetap boleh ke val/test, hanya diberi flag `aug_possible`): ['bacterial_leaf_blight']. Kalibrasi dan kontrol positif: `reports/aug_detect/`.

| kelas | sumber | gambar | flag nama | flag piksel | flag embedding | aug_possible (salah satu) | wajib train (train-only atau aug) | aug_possible di val/test |
|---|---|---|---|---|---|---|---|---|
| bacterial_leaf_blight | ds1_provisional | 1986 | 1848 | 774 | 868 | 1938 | 0 | 575 |
| brown_spot | ds1 | 1972 | 1864 | 615 | 1045 | 1940 | 1972 | 0 |
| brown_spot | ds4 | 2177 | 0 | 13 | 0 | 13 | 13 | 0 |
| leaf_blast | ds1 | 412 | 406 | 52 | 311 | 411 | 412 | 0 |
| leaf_blast | ds4 | 1325 | 0 | 6 | 0 | 6 | 6 | 0 |
| background | ds2 | 157 | 0 | 22 | 0 | 22 | 22 | 0 |
| background | ds4 | 1039 | 0 | 5 | 0 | 5 | 5 | 0 |

Gambar wajib-train menurut sebab: {'aug': 46, 'train_only': 2384}. Gambar kelas exempt yang `aug_possible` tetap boleh ke val/test; metrik pada subset itu sebaiknya dilaporkan terpisah (kolom `aug_possible` di manifest).

## 9. Pemangkasan background (hanya train)

- Batas 10.0% dari gambar train. Gambar train berpenyakit: 6248 -> batas background 694.
- Background train: 1229 -> **694** (dipangkas 535 dari sumber ['ds4'], acak dengan seed 42); dipertahankan per sumber: {'ds4': 573, 'ds2': 121}.
- Batas tercapai: **True**. Val dan test tidak diubah (porsi background val/test lebih besar daripada train; metrik background/false positive di val/test tidak sebanding langsung dengan distribusi train).

## 10. Pasangan acak pada jarak phash 12-14 (pemeriksaan visual)

- Jumlah pasangan di pita jarak 12-14 (dalam dataset akhir): **514**; ditampilkan 10 acak (seed 42): `reports/prov_v2/pairs_12_14.jpg`. Gambar kiri sudah diputar/di-flip ke varian yang cocok dengan gambar kanan.
| # | jarak | varian | gambar A | gambar B |
|---|---|---|---|---|
| 1 | 12 | 4 | data/raw/ds1/rice/images/train/r6k_train_blight_orig_002_png_jpg.rf.6a293726f190344dfd34829a5e2bfc4f.jpg | data/raw/ds1/rice/images/train/r6k_train_blight_rotated_002_png_jpg.rf.23b69471220dec2aa1bedee73dbbffc5.jpg |
| 2 | 12 | 0 | data/raw/ds4/images_labels/RiceLeafDiseaseBD/Original images/Brown spot/brown spot_1461.jpg | data/raw/ds4/images_labels/RiceLeafDiseaseBD/Original images/Brown spot/brown spot_1462.jpg |
| 3 | 12 | 0 | data/raw/ds1/rice/images/train/r6k_train_brownspot_orig_059_jpg.rf.70558ac89751764e984fbcbd4a9cc621.jpg | data/raw/ds1/rice/images/val/r6k_train_brownspot_orig_059_jpg.rf.3507574957b5ed95a18a8efb43472b7b.jpg |
| 4 | 12 | 0 | data/raw/ds1/rice/images/train/r6k_train_blight_rotated_095_jpg.rf.303502659e578e56705c0f1d8f6b4f3d.jpg | data/raw/ds1/rice/images/val/r6k_train_blight_rotated_095_jpg.rf.6475ac0a430b21b554e386928084685b.jpg |
| 5 | 12 | 0 | data/raw/ds1/rice/images/train/r6k_train_blight_rotated_031_jpg.rf.d402a47affc9caa3ad1ae2e4e4bee45d.jpg | data/raw/ds1/rice/images/val/r6k_train_blight_rotated_031_jpg.rf.8ee67d5112dd41b8d50d9234dea60875.jpg |
| 6 | 14 | 4 | data/raw/ds1/rice/images/train/r6k_train_blight_orig_021_jpg.rf.57a0f41ea0b3752272df37c98387250a.jpg | data/raw/ds1/rice/images/train/r6k_train_blight_rotated_021_jpg.rf.a246ec2ceef04bb1fcd5a735b48c9658.jpg |
| 7 | 14 | 0 | data/raw/ds1/rice/images/train/r6k_test_img_brown_68_jpeg_jpg.rf.fc5353da99b76e38d6d61c4ecef6afc9.jpg | data/raw/ds1/rice/images/train/r6k_train_img_brown_68_jpeg_jpg.rf.9eb76b36e658fada56c5b8dd496152f3.jpg |
| 8 | 12 | 0 | data/raw/ds1/rice/images/train/r6k_test_img_blb_76_jpeg_jpg.rf.c0bea949e43faad4790adb779f98eff2.jpg | data/raw/ds1/rice/images/train/r6k_train_img_blb_76_jpeg_jpg.rf.f9d104fd935a4cc5d3a4a2fc90679270.jpg |
| 9 | 12 | 0 | data/raw/ds1/rice/images/train/r6k_train_img_brown_41_jpg.rf.0e1537f158cf1aab0cce185c81532cb4.jpg | data/raw/ds1/rice/images/train/r6k_val_img_brown_41_jpg.rf.0602049e8df65d88539a31b5fcdca285.jpg |
| 10 | 14 | 0 | data/raw/ds4/images_labels/RiceLeafDiseaseBD/Original images/Brown spot/brown spot_1643.jpg | data/raw/ds4/images_labels/RiceLeafDiseaseBD/Original images/Brown spot/brown spot_1644.jpg |

## 11. Verifikasi SIFT pada kandidat kNN embedding (aturan v2)

- Kandidat unik (k=30 tetangga terdekat per gambar, tanpa ambang cosine): **229285**; sudah satu grup sebelum SIFT: 24492; diverifikasi: 204793; **terverifikasi >= 21 inlier: 9693**.
- Waktu: deskriptor 265.7 s, verifikasi 210.0 s.
- Histogram inlier pada pasangan yang diverifikasi: 0-0: 142089, 1-5: 47358, 6-10: 2455, 11-20: 3198, 21-40: 3420, 41-100: 4160, 101-+: 2113
- Pasangan terverifikasi lintas KELAS: **539**; lintas SUMBER: **8**. Pasangan lintas kelas tetap digabung dalam satu split tetapi TIDAK dibuang (belum dinilai manusia). Pasangan lengkap: `sift_pairs.csv`.
  - contoh lintas kelas: leaf_blast `blast_1000.jpg` ~ brown_spot `brown spot_1253.jpg`
  - contoh lintas kelas: leaf_blast `blast_1000.jpg` ~ brown_spot `brown spot_1254.jpg`
  - contoh lintas kelas: leaf_blast `blast_1000.jpg` ~ brown_spot `brown spot_1255.jpg`
  - contoh lintas kelas: leaf_blast `blast_1001.jpg` ~ brown_spot `brown spot_1253.jpg`
  - contoh lintas kelas: leaf_blast `blast_1001.jpg` ~ brown_spot `brown spot_1254.jpg`

## 12. Audit independen: gambar val/test yang masih punya saudara di train

Target 0 menurut aturan baru. Sinyal audit berbeda dari pembentuk grup: SIFT pada **k lebih besar** tetangga embedding di train, kunci K4, phash, embedding. Piksel test dibaca hanya untuk deteksi duplikat.

| pasangan split | gambar query | SIFT>=21 (k=50) | kunci K4 | phash<=14 | embedding>=0.97 | gabungan |
|---|---|---|---|---|---|---|
| val-train | 1418 | 2 | 0 | 0 | 0 | **2** |
| test-train | 708 | 2 | 0 | 0 | 0 | **2** |
| test-val | 708 | 0 | 0 | 0 | 0 | **0** |

- val-train: contoh ['data/raw/ds4/images_labels/RiceLeafDiseaseBD/Original images/Brown spot/brown spot_1301.jpg', 'data/raw/ds4/images_labels/RiceLeafDiseaseBD/Original images/Healthy/Healthy_354.jpg']
- test-train: contoh ['data/raw/ds4/images_labels/RiceLeafDiseaseBD/Original images/Brown spot/brown spot_735.jpg', 'data/raw/ds4/images_labels/RiceLeafDiseaseBD/Original images/Healthy/Healthy_221.jpg']

## 13. Perubahan keanggotaan split TEST dibanding `prov`

- Test lama (`data/processed/prov/manifest.csv`): **714**; test baru: **708**.
- Ringkas: {'masuk_ke_test': 633, 'keluar_dari_test': 639, 'tetap_test': 75}. Gambar yang keluar dari test pindah ke: {'train': 441, 'val': 127, 'dibuang': 71}; gambar yang masuk test berasal dari: {'train': 389, 'val': 176, '-': 68}.
- Daftar lengkap: `reports/prov_v2/test_changes.csv`.

## 14. Metadata dataset

- Versi aturan pengelompokan: `v2`. v2 = phash16<=14 (8 varian dihedral) + embedding DINOv2-small cosine>=0.97 (8 varian) + nama dasar Roboflow + kunci ID K4 (fam,jenis,id,ekstensi; lintas awalan; ds1) + SIFT>=21 inlier pada 30 tetangga terdekat embedding (tanpa ambang cosine; semua kelas dan sumber)
- `manifest_sha256`: `083868bcef6e2431bf65e9bd7df2b8b87046636b89939deb315d6a97a4634183`; `grup_split_sha256`: `ddc22b3369d45ffd46a0323a479c6e06ace46af2b4be5b6bca39ebfa3fadb4e7`; `config_sha256`: `826d6f4077779fd83423d8e3f70d6f1167d9ee3e85b50ebaa75200758550f1fd` (rinci di `dataset_meta.json`).
