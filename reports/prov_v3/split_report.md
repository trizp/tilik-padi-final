# Laporan split dataset `prov_v3`

- Seed: 42 | rasio split: {'train': 0.7, 'val': 0.2, 'test': 0.1} | phash 16x16, ambang 14 (8 varian dihedral: rotasi 0/90/180/270 x flip)
- Total gambar: **8934** | `healthy_mode`: background
- **PROVISIONAL**: sumber `ds1_provisional` belum final. Run diberi awalan `prov_v3_` dan **tidak boleh dipakai untuk tabel paper**.

## 1. Gambar dan box per kelas per split

| kelas | train | val | test | total gambar | grup unik |
|---|---|---|---|---|---|
| leaf_blast | 1299 (3979 box) | 249 (820 box) | 125 (446 box) | 1673 | 661 |
| brown_spot | 3464 (21537 box) | 419 (1877 box) | 209 (881 box) | 4092 | 916 |
| bacterial_leaf_blight | 1400 (1648 box) | 391 (469 box) | 195 (220 box) | 1986 | 219 |
| background | 684 | 333 | 166 | 1183 | 687 |
| **total** | 6847 | 1392 | 695 | 8934 | 2345 |

Porsi background: **13.2%** dari total (train 10.0%, val 23.9%, test 23.9%). Ultralytics menyarankan 0-10%.

## 2. Gambar per kelas per SUMBER per split

| kelas | sumber | train | val | test | total | porsi dari kelas |
|---|---|---|---|---|---|---|
| bacterial_leaf_blight | ds1_provisional ⚠ provisional | 1400 (154 grup) | 391 (43 grup) | 195 (22 grup) | 1986 (219 grup) | 100% |
| brown_spot | ds1 | 1970 (200 grup) | 0 (0 grup) | 0 (0 grup) | 1970 (200 grup) | 48% |
| brown_spot | ds4 | 1494 (483 grup) | 419 (150 grup) | 209 (83 grup) | 2122 (716 grup) | 52% |
| leaf_blast | ds1 | 411 (75 grup) | 0 (0 grup) | 0 (0 grup) | 411 (75 grup) | 25% |
| leaf_blast | ds4 | 888 (394 grup) | 249 (124 grup) | 125 (68 grup) | 1262 (586 grup) | 75% |
| background | ds2 | 120 (78 grup) | 25 (19 grup) | 12 (9 grup) | 157 (106 grup) | 13% |
| background | ds4 | 564 (341 grup) | 308 (156 grup) | 154 (84 grup) | 1026 (581 grup) | 87% |

**Kelas dengan satu sumber saja:** `bacterial_leaf_blight` hanya dari `ds1_provisional`.

Jumlah **box** per kelas per sumber per split ada di `split_counts.csv`.

## 3. Statistik luas box per kelas (relatif terhadap gambar, semua split)

| kelas | sumber | gambar | box | box/gambar | median luas | rata2 luas | box > 1% luas gambar | box dari poligon |
|---|---|---|---|---|---|---|---|---|
| leaf_blast | *semua sumber* | 1673 | 5245 | 3.14 | 0.36% | 1.72% | 18.4% | 13.1% |
| leaf_blast | ds1 | 411 | 697 | 1.7 | 3.15% | 9.76% | 72.2% | 98.4% |
| leaf_blast | ds4 | 1262 | 4548 | 3.6 | 0.31% | 0.49% | 10.1% | 0.0% |
| brown_spot | *semua sumber* | 4092 | 24295 | 5.94 | 0.31% | 0.64% | 14.8% | 4.0% |
| brown_spot | ds1 | 1970 | 15153 | 7.69 | 0.35% | 0.76% | 21.0% | 6.5% |
| brown_spot | ds4 | 2122 | 9142 | 4.31 | 0.28% | 0.44% | 4.7% | 0.0% |
| bacterial_leaf_blight | *semua sumber* | 1986 | 2337 | 1.18 | 21.27% | 25.97% | 97.6% | 15.6% |
| bacterial_leaf_blight | ds1_provisional | 1986 | 2337 | 1.18 | 21.27% | 25.97% | 97.6% | 15.6% |

"Box dari poligon" = bbox pembatas yang diturunkan dari label poligon (segmentasi) sumber; sisanya box asli.

Perbandingan antar kelas: median luas terbesar `bacterial_leaf_blight` (21.27%) vs terkecil `brown_spot` (0.31%) = **69x**. **Skala box antar kelas tidak seragam** (unit anotasi berbeda) -> baca metrik per kelas terpisah.

## 4. Deduplikasi (phash)

- Grup unik: 2345 dari 8934 gambar; grup berisi >1 gambar: **1090** (mencakup 7679 gambar).
- Grup terbesar: 114 gambar. Distribusi ukuran grup: {1: 1255, 2: 311, 3: 155, 4: 86, 5: 79, 6: 45, 7: 44, 8: 46} ...
- Grup lintas SUMBER: **1**; grup lintas KELAS pada dataset akhir: **128** (kemiripan adegan embedding yang memuat >1 kelas sebelum penyaringan: 135; tetap dijaga dalam satu split, tidak dibuang).
- Gambar yang berada di grup duplikat per sumber: {'ds4': 3301, 'ds1_provisional': 1961, 'ds1': 2340, 'ds2': 77}
- Contoh grup lintas kelas: 16 gambar {'leaf_blast': 6, 'brown_spot': 10}; mis. blast_1000.jpg, blast_1001.jpg, blast_1002.jpg
- Contoh grup lintas kelas: 6 gambar {'leaf_blast': 2, 'brown_spot': 4}; mis. blast_1020.jpg, blast_1021.jpg, brown spot_1269.jpg
- Contoh grup lintas kelas: 6 gambar {'leaf_blast': 3, 'brown_spot': 3}; mis. blast_1112.jpg, blast_1113.jpg, blast_1114.jpg
- Contoh grup lintas kelas: 2 gambar {'leaf_blast': 1, 'brown_spot': 1}; mis. blast_117.jpg, brown spot_24.jpg
- Contoh grup lintas kelas: 2 gambar {'leaf_blast': 1, 'brown_spot': 1}; mis. blast_1185.jpg, brown spot_1720.jpg
- Pasangan yang menyatukan grup: phash <= 14 (8 varian dihedral): **3862**
- Pasangan yang menyatukan grup: embedding cosine >= 0.97 (DINOv2-small, 8 varian dihedral): **7587**
- Pasangan yang menyatukan grup: nama dasar Roboflow sama (.rf.<hash>): **3017**
- Pasangan yang menyatukan grup: kunci ID K4 (fam,jenis,id,ekstensi; lintas awalan; ds1): **5626**
- Pasangan yang menyatukan grup: jendela waktu <= 2 s (ds4; tanggal-jam dari nama berkas asli): **769**
- Pasangan yang menyatukan grup: SIFT >= 21 inlier dan sebaran >= 15% pada 50 tetangga embedding: **8848**
- **Verifikasi leakage** (jarak phash minimum antar split, 8 varian dihedral; ambang 14): val-train: 0 pasangan <= 14 (jarak min 48), test-train: 0 pasangan <= 14 (jarak min 64), test-val: 0 pasangan <= 14 (jarak min 68). **LOLOS (0 pasangan).**
- **Verifikasi embedding** (cosine maks antar split, 8 varian dihedral): val-train: maks 0.9697 (0 pasangan >= ambang), test-train: maks 0.9694 (0 pasangan >= ambang), test-val: maks 0.9686 (0 pasangan >= ambang). Versi Roboflow dengan nama dasar sama yang terpecah antar split: **0**.

## 5. Test multi-penyakit (gambar dengan >= 2 kelas)

- Gambar >= 2 kelas di seluruh dataset: **0**; di split test: **0**. Daftar: `data/processed/prov_v3/lists/test_multidisease.txt`.
- Tidak ada satu pun gambar multi-penyakit pada sumber yang dipakai (semua label satu kelas per gambar). Syarat test multi-penyakit pada CLAUDE.md **belum bisa dipenuhi**; perlu foto yang dianotasi manual.

## 6. Gambar/box yang dibuang

| sumber | alasan | jumlah |
|---|---|---|
| ds1 | duplikat_label_bertentangan | 79 |
| ds1 | karantina_lintas_kelas_inlier>100 | 3 |
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
| ds4 | dipangkas_max_background_frac_train | 530 |
| ds4 | folder_tidak_dipakai:Leaf smut | 724 |
| ds4 | folder_tidak_dipakai:Rice Tungro | 2244 |
| ds4 | folder_tidak_dipakai:Sheath blight | 1722 |
| ds4 | karantina_lintas_kelas_inlier>100 | 138 |
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
| brown_spot|ds1 | 1970 | 1970 (100%) | 0 (0%) | 0 (0%) |
| brown_spot|ds4 | 2122 | 1494 (70%) | 419 (20%) | 209 (10%) |
| leaf_blast|ds1 | 411 | 411 (100%) | 0 (0%) | 0 (0%) |
| leaf_blast|ds4 | 1262 | 888 (70%) | 249 (20%) | 125 (10%) |
| background|ds2 | 157 | 120 (76%) | 25 (16%) | 12 (8%) |
| background|ds4 | 1026 | 564 (55%) | 308 (30%) | 154 (15%) |

Target 70/20/10. Stratum kecil atau berisi grup duplikat besar bisa menyimpang karena grup tidak boleh dipecah.

## 8. Deteksi augmentasi dan kelayakan val/test

Detektor: (a) pola nama (sumber ['ds1']), (b) heuristik piksel {'corner_black': 0.25, 'border_black': 0.08, 'smear': 0.25} (sumber ['ds1', 'ds4', 'ds2']), (c) embedding DINOv2-small cosine >= 0.97 (flag 'punya saudara' hanya sumber ['ds1']; juga menggabungkan grup). Kelas exempt (tetap boleh ke val/test, hanya diberi flag `aug_possible`): ['bacterial_leaf_blight']. Kalibrasi dan kontrol positif: `reports/aug_detect/`.

| kelas | sumber | gambar | flag nama | flag piksel | flag embedding | aug_possible (salah satu) | wajib train (train-only atau aug) | aug_possible di val/test |
|---|---|---|---|---|---|---|---|---|
| bacterial_leaf_blight | ds1_provisional | 1986 | 1848 | 774 | 868 | 1938 | 0 | 569 |
| brown_spot | ds1 | 1970 | 1862 | 615 | 1043 | 1938 | 1970 | 0 |
| brown_spot | ds4 | 2122 | 0 | 13 | 0 | 13 | 13 | 0 |
| leaf_blast | ds1 | 411 | 405 | 52 | 310 | 410 | 411 | 0 |
| leaf_blast | ds4 | 1262 | 0 | 6 | 0 | 6 | 6 | 0 |
| background | ds2 | 157 | 0 | 22 | 0 | 22 | 22 | 0 |
| background | ds4 | 1026 | 0 | 6 | 0 | 6 | 6 | 0 |

Gambar wajib-train menurut sebab: {'aug': 47, 'train_only': 2381}. Gambar kelas exempt yang `aug_possible` tetap boleh ke val/test; metrik pada subset itu sebaiknya dilaporkan terpisah (kolom `aug_possible` di manifest).

## 9. Pemangkasan background (hanya train)

- Batas 10.0% dari gambar train. Gambar train berpenyakit: 6163 -> batas background 684.
- Background train: 1214 -> **684** (dipangkas 530 dari sumber ['ds4'], acak dengan seed 42); dipertahankan per sumber: {'ds4': 564, 'ds2': 120}.
- Batas tercapai: **True**. Val dan test tidak diubah (porsi background val/test lebih besar daripada train; metrik background/false positive di val/test tidak sebanding langsung dengan distribusi train).

## 10. Pasangan acak pada jarak phash 12-14 (pemeriksaan visual)

- Jumlah pasangan di pita jarak 12-14 (dalam dataset akhir): **513**; ditampilkan 10 acak (seed 42): `reports/prov_v3/pairs_12_14.jpg`. Gambar kiri sudah diputar/di-flip ke varian yang cocok dengan gambar kanan.
| # | jarak | varian | gambar A | gambar B |
|---|---|---|---|---|
| 1 | 14 | 4 | data/raw/ds1/rice/images/train/r6k_train_blight_orig_002_png_jpg.rf.6a293726f190344dfd34829a5e2bfc4f.jpg | data/raw/ds1/rice/images/train/r6k_train_blight_rotated_002_png_jpg.rf.7d2d321a9a915ad3ea10a040f853de2a.jpg |
| 2 | 14 | 0 | data/raw/ds4/images_labels/RiceLeafDiseaseBD/Original images/Brown spot/brown spot_1530.jpg | data/raw/ds4/images_labels/RiceLeafDiseaseBD/Original images/Brown spot/brown spot_1531.jpg |
| 3 | 12 | 0 | data/raw/ds1/rice/images/train/r6k_train_brownspot_orig_060_jpg.rf.0c04202017ce37444d14781914f6c4c3.jpg | data/raw/ds1/rice/images/train/r6k_train_brownspot_orig_060_jpg.rf.a3eaa81b8cc4aed6b61208ceb44b823d.jpg |
| 4 | 12 | 0 | data/raw/ds1/rice/images/train/r6k_train_blight_rotated_095_jpg.rf.303502659e578e56705c0f1d8f6b4f3d.jpg | data/raw/ds1/rice/images/val/r6k_train_blight_rotated_095_jpg.rf.ce27bf48b8e122268ae6e11f33fe0cbe.jpg |
| 5 | 14 | 0 | data/raw/ds1/rice/images/train/r6k_train_blight_rotated_032_jpg.rf.68a6ae15e707c8a67a65fe38d6f09418.jpg | data/raw/ds1/rice/images/train/r6k_train_blight_rotated_032_jpg.rf.e4773eb34aa2499a6e8ff6ef37e4bfc6.jpg |
| 6 | 14 | 4 | data/raw/ds1/rice/images/train/r6k_train_blight_orig_021_jpg.rf.57a0f41ea0b3752272df37c98387250a.jpg | data/raw/ds1/rice/images/val/r6k_test_blight_rotated_021_jpg.rf.df2559fefaf0b02d9fca07b78a5a35c9.jpg |
| 7 | 14 | 0 | data/raw/ds1/rice/images/train/r6k_test_img_brown_68_jpeg_jpg.rf.fc5353da99b76e38d6d61c4ecef6afc9.jpg | data/raw/ds1/rice/images/train/r6k_train_img_brown_68_jpeg_jpg.rf.bdb887ea521299e12ccdce45880f790d.jpg |
| 8 | 14 | 0 | data/raw/ds1/rice/images/train/r6k_test_img_blb_76_jpeg_jpg.rf.c0bea949e43faad4790adb779f98eff2.jpg | data/raw/ds1/rice/images/val/r6k_train_img_blb_76_jpeg_jpg.rf.9520a6a5df03219354d148cca7fd451c.jpg |
| 9 | 12 | 0 | data/raw/ds1/rice/images/train/r6k_train_img_brown_41_jpg.rf.a71227afe94a1242c764b03fa7ae0cb0.jpg | data/raw/ds1/rice/images/train/r6k_train_img_brown_41_jpg.rf.ced1732afdd90be08f720deca8e610e1.jpg |
| 10 | 14 | 0 | data/raw/ds4/images_labels/RiceLeafDiseaseBD/Original images/Brown spot/brown spot_1818.jpg | data/raw/ds4/images_labels/RiceLeafDiseaseBD/Original images/Brown spot/brown spot_1820.jpg |

## 11. Verifikasi SIFT pada kandidat kNN embedding (aturan v2)

- Kandidat unik (k=50 tetangga terdekat per gambar, tanpa ambang cosine): **391296**; sudah satu grup sebelum SIFT: 26337; diverifikasi: 364959; **terverifikasi >= 21 inlier: 8848**.
- Waktu: deskriptor 238.5 s, verifikasi 356.7 s.
- Aturan sebaran: pasangan dengan inlier >= 21 tetapi kotak pembatas inlier < 15% luas gambar DITOLAK (dugaan kecocokan watermark/teks kecil): **1866** pasangan. Daftar dengan status di `sift_pairs.csv`.
- Histogram inlier pada pasangan yang diverifikasi: 0-0: 265558, 1-5: 81938, 6-10: 2917, 11-20: 3832, 21-40: 3939, 41-100: 4553, 101-+: 2222
- Pasangan terverifikasi lintas KELAS: **603**; lintas SUMBER: **1**. Pasangan lintas kelas tetap digabung dalam satu split tetapi TIDAK dibuang (belum dinilai manusia). Pasangan lengkap: `sift_pairs.csv`.
  - contoh lintas kelas: leaf_blast `blast_1001.jpg` ~ brown_spot `brown spot_1255.jpg`
  - contoh lintas kelas: leaf_blast `blast_1020.jpg` ~ brown_spot `brown spot_1272.jpg`
  - contoh lintas kelas: leaf_blast `blast_1112.jpg` ~ brown_spot `brown spot_1542.jpg`
  - contoh lintas kelas: leaf_blast `blast_1114.jpg` ~ brown_spot `brown spot_1541.jpg`
  - contoh lintas kelas: leaf_blast `blast_1180.jpg` ~ brown_spot `brown spot_1697.jpg`

## 12. Audit independen: gambar val/test yang masih punya saudara di train

Target 0 menurut aturan baru. Sinyal audit berbeda dari pembentuk grup: SIFT pada **k lebih besar** tetangga embedding di train, kunci K4, phash, embedding. Piksel test dibaca hanya untuk deteksi duplikat.

| pasangan split | gambar query | SIFT>=21 (k=100) | kunci K4 | phash<=14 | embedding>=0.97 | gabungan |
|---|---|---|---|---|---|---|
| val-train | 1392 | 8 | 0 | 0 | 0 | **8** |
| test-train | 695 | 1 | 0 | 0 | 0 | **1** |
| test-val | 695 | 2 | 0 | 0 | 0 | **2** |

- val-train: contoh ['data/raw/ds4/images_labels/RiceLeafDiseaseBD/Original images/Blast/blast_619.jpg', 'data/raw/ds4/images_labels/RiceLeafDiseaseBD/Original images/Brown spot/brown spot_1302.jpg', 'data/raw/ds4/images_labels/RiceLeafDiseaseBD/Original images/Brown spot/brown spot_1550.jpg', 'data/raw/ds4/images_labels/RiceLeafDiseaseBD/Original images/Brown spot/brown spot_1551.jpg', 'data/raw/ds4/images_labels/RiceLeafDiseaseBD/Original images/Brown spot/brown spot_1552.jpg']
- test-train: contoh ['data/raw/ds1/rice/images/train/rls_train_aug_0_8068_jpg.rf.8a2c2d7e93af71f679bad4dd33c12651.jpg']
- test-val: contoh ['data/raw/ds4/images_labels/RiceLeafDiseaseBD/Original images/Brown spot/brown spot_1465.jpg', 'data/raw/ds2/Rice Leaf Bacterial and Fungal Disease Dataset/Original/Original Images/Original Images/Healthy Rice Leaf/Healthy_rice_leaf  (49).jpg']

## 13. Perubahan keanggotaan split TEST dibanding `prov`

- Test lama (`data/processed/prov_v2/manifest.csv`): **708**; test baru: **695**.
- Ringkas: {'keluar_dari_test': 672, 'masuk_ke_test': 659, 'tetap_test': 36}. Gambar yang keluar dari test pindah ke: {'val': 190, 'train': 374, 'dibuang': 108}; gambar yang masuk test berasal dari: {'val': 131, 'train': 478, '-': 50}.
- Daftar lengkap: `reports/prov_v3/test_changes.csv`.

## 14. Metadata dataset

- Versi aturan pengelompokan: `v3`. v3 = phash16<=14 (8 varian dihedral) + embedding DINOv2-small cosine>=0.97 (8 varian) + nama dasar Roboflow + kunci ID K4 (fam,jenis,id,ekstensi; lintas awalan; ds1) + SIFT>=21 inlier DAN kotak pembatas inlier>=15% luas gambar pada 50 tetangga terdekat embedding (tanpa ambang cosine; semua kelas dan sumber) + jendela waktu <=2 s (ds4, dari tanggal-jam nama berkas asli) ; karantina pasangan lintas kelas dengan inlier>100
- `manifest_sha256`: `1738d0bd98ed8c3d75b255084f1e8cca284893820dd44ee03852ff52a3138cb6`; `grup_split_sha256`: `8fc973b458652aeaa05a614f19fe52b623db9c7deacb907c20baaafd78927637`; `config_sha256`: `1b0174755cd42dfcef46bdc9b4604d24148f44c9ea449a097db30faf6b40ef6c` (rinci di `dataset_meta.json`).

## 15. Karantina (pasangan lintas kelas dengan inlier > 100)

- **141 gambar** dikeluarkan dari train/val/test pada build ini (berkas asli dan label TIDAK diubah/dihapus); dari 126 pasangan. Daftar: `quarantine.csv`.
| kelas | sumber | gambar dikarantina |
|---|---|
| background|ds4 | 19 |
| brown_spot|ds1 | 2 |
| brown_spot|ds4 | 56 |
| leaf_blast|ds1 | 1 |
| leaf_blast|ds4 | 63 |

Per pasangan kelas: brown_spot~leaf_blast: 115, background~brown_spot: 26

## 16. Aturan jendela waktu (ds4)

- Selisih waktu pemotretan <= 2 detik; gambar ds4 bertanggal: 2147; pasangan berantai: 769.
|  | gambar ds4 | grup unik | gambar di grup > 1 | grup terbesar |
|---|---|---|---|---|
| tanpa aturan waktu | 4940 | 2423 | 3086 | 44 |
| dengan aturan waktu | 4940 | 1933 | 3738 | 61 |

Rasio split akhir per stratum ada di bagian 7.

- ds4 bertanggal per split: {'train': 1142, 'val': 450, 'test': 233}; tanggal unik: {'train': 12, 'val': 10, 'test': 11}; seluruh tanggal val/test juga ada di train: **True** (dicatat di `dataset_meta.json`).
