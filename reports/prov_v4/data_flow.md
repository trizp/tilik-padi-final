# Alur data `prov_v4`

Setiap foto mentah dicatat tepat satu hasil: masuk train/val/test, dikarantina, dibuang (dengan alasan), atau dicadangkan. Kolom `tidak_tercatat` harus 0 (artinya tidak ada foto yang hilang tanpa catatan).

## ds1: 8667 foto mentah

| kelas asli | cabang | mentah | train | val | test | karantina | dibuang | dicadangkan | tidak tercatat | alasan dibuang |
|---|---|---|---|---|---|---|---|---|---|---|
| (tanpa label) | - | 2 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | tanpa file label: 2 |
| BacterialLeafBlight | - | 2000 | 1401 | 390 | 195 | 0 | 14 | 0 | 0 | foto sama berlabel bertentangan (duplikat sungguhan): 14 |
| BrownSpot | - | 2000 | 1970 | 0 | 0 | 2 | 28 | 0 | 0 | foto sama berlabel bertentangan (duplikat sungguhan): 28 |
| Healthy | - | 433 | 0 | 0 | 0 | 0 | 433 | 0 | 0 | kelas di luar 3 kelas target: 433 |
| Hispa | - | 431 | 0 | 0 | 0 | 0 | 431 | 0 | 0 | kelas di luar 3 kelas target: 431 |
| LeafBlast | - | 449 | 411 | 0 | 0 | 1 | 37 | 0 | 0 | foto sama berlabel bertentangan (duplikat sungguhan): 37 |
| LeafScald | - | 456 | 0 | 0 | 0 | 0 | 456 | 0 | 0 | kelas di luar 3 kelas target: 456 |
| LeafSmut | - | 2000 | 0 | 0 | 0 | 0 | 2000 | 0 | 0 | kelas di luar 3 kelas target: 2000 |
| NarrowBrownLeafSpot | - | 443 | 0 | 0 | 0 | 0 | 443 | 0 | 0 | kelas di luar 3 kelas target: 443 |
| NeckBlast | - | 453 | 0 | 0 | 0 | 0 | 453 | 0 | 0 | kelas di luar 3 kelas target: 453 |
| **jumlah** | | **8667** | **3782** | **390** | **195** | **3** | **4297** | **0** | **0** | |

## ds2: 6889 foto mentah

| kelas asli | cabang | mentah | train | val | test | karantina | dibuang | dicadangkan | tidak tercatat | alasan dibuang |
|---|---|---|---|---|---|---|---|---|---|---|
| Bacterial Leaf Blight | Augmented | 536 | 0 | 0 | 0 | 0 | 536 | 0 | 0 | cabang Augmented tidak dipakai (augmentasi hanya boleh saat training): 536 |
| Bacterial Leaf Blight | Original | 180 | 0 | 0 | 0 | 0 | 180 | 0 | 0 | folder/kelas di luar cakupan (tidak dipakai): 180 |
| Brown Spot | Augmented | 810 | 0 | 0 | 0 | 0 | 810 | 0 | 0 | cabang Augmented tidak dipakai (augmentasi hanya boleh saat training): 810 |
| Brown Spot | Original | 267 | 0 | 0 | 0 | 0 | 267 | 0 | 0 | folder/kelas di luar cakupan (tidak dipakai): 267 |
| Healthy Rice Leaf | Augmented | 511 | 0 | 0 | 0 | 0 | 511 | 0 | 0 | cabang Augmented tidak dipakai (augmentasi hanya boleh saat training): 511 |
| Healthy Rice Leaf | Original | 157 | 121 | 24 | 12 | 0 | 0 | 0 | 0 |  |
| Leaf Blast | Augmented | 929 | 0 | 0 | 0 | 0 | 929 | 0 | 0 | cabang Augmented tidak dipakai (augmentasi hanya boleh saat training): 929 |
| Leaf Blast | Original | 305 | 0 | 0 | 0 | 0 | 305 | 0 | 0 | folder/kelas di luar cakupan (tidak dipakai): 305 |
| Leaf scald | Augmented | 560 | 0 | 0 | 0 | 0 | 560 | 0 | 0 | cabang Augmented tidak dipakai (augmentasi hanya boleh saat training): 560 |
| Leaf scald | Original | 189 | 0 | 0 | 0 | 0 | 189 | 0 | 0 | folder/kelas di luar cakupan (tidak dipakai): 189 |
| Narrow Brown Leaf Spot | Augmented | 353 | 0 | 0 | 0 | 0 | 353 | 0 | 0 | cabang Augmented tidak dipakai (augmentasi hanya boleh saat training): 353 |
| Narrow Brown Leaf Spot | Original | 117 | 0 | 0 | 0 | 0 | 117 | 0 | 0 | folder/kelas di luar cakupan (tidak dipakai): 117 |
| Rice Hispa | Augmented | 664 | 0 | 0 | 0 | 0 | 664 | 0 | 0 | cabang Augmented tidak dipakai (augmentasi hanya boleh saat training): 664 |
| Rice Hispa | Original | 215 | 0 | 0 | 0 | 0 | 215 | 0 | 0 | folder/kelas di luar cakupan (tidak dipakai): 215 |
| Sheath Blight | Augmented | 825 | 0 | 0 | 0 | 0 | 825 | 0 | 0 | cabang Augmented tidak dipakai (augmentasi hanya boleh saat training): 825 |
| Sheath Blight | Original | 271 | 0 | 0 | 0 | 0 | 271 | 0 | 0 | folder/kelas di luar cakupan (tidak dipakai): 271 |
| **jumlah** | | **6889** | **121** | **24** | **12** | **0** | **6732** | **0** | **0** | |

## ds3: 240 foto mentah

| kelas asli | cabang | mentah | train | val | test | karantina | dibuang | dicadangkan | tidak tercatat | alasan dibuang |
|---|---|---|---|---|---|---|---|---|---|---|
| blast | - | 80 | 0 | 0 | 0 | 0 | 0 | 80 | 0 |  |
| blight | - | 80 | 0 | 0 | 0 | 0 | 0 | 80 | 0 |  |
| tungro | - | 80 | 0 | 0 | 0 | 0 | 0 | 80 | 0 |  |
| **jumlah** | | **240** | **0** | **0** | **0** | **0** | **0** | **240** | **0** | |

## ds4: 9769 foto mentah

| kelas asli | cabang | mentah | train | val | test | karantina | dibuang | dicadangkan | tidak tercatat | alasan dibuang |
|---|---|---|---|---|---|---|---|---|---|---|
| Blast | - | 1326 | 888 | 249 | 125 | 63 | 1 | 0 | 0 | file label kosong pada folder penyakit: 1 |
| Brown spot | - | 2178 | 1494 | 419 | 209 | 56 | 0 | 0 | 0 |  |
| Healthy | - | 1575 | 563 | 308 | 154 | 19 | 531 | 0 | 0 | dipangkas agar background train 10%: 531 |
| Leaf smut | - | 724 | 0 | 0 | 0 | 0 | 724 | 0 | 0 | folder/kelas di luar cakupan (tidak dipakai): 724 |
| Rice Tungro | - | 2244 | 0 | 0 | 0 | 0 | 2244 | 0 | 0 | folder/kelas di luar cakupan (tidak dipakai): 2244 |
| Sheath blight | - | 1722 | 0 | 0 | 0 | 0 | 1722 | 0 | 0 | folder/kelas di luar cakupan (tidak dipakai): 1722 |
| **jumlah** | | **9769** | **2945** | **976** | **488** | **138** | **5222** | **0** | **0** | |

## Jumlah seluruh sumber

Mentah 25565 = train 6848 + val 1390 + test 695 + karantina 141 + dibuang 16251 + dicadangkan 240 + tidak tercatat 0.

## Di luar cakupan: kandidat test terpisah (ds4)

Foto ds4 kelas Rice Tungro (2244), Sheath blight (1722), dan Leaf smut (724) TIDAK dimasukkan ke train/val/test. Daftar lengkap (dengan jumlah box label dan foto ganda di folder lain): `out_of_scope_ds4_test_candidates.csv` (4690 foto; 13 di antaranya fotonya juga ada di folder kelas lain).
