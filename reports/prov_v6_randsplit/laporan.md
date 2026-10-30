# prov_v6_randsplit: dataset pembanding "split acak" (PROVISIONAL)

Dibuat 2026-10-08 oleh `scripts/buat_randsplit.py` (seed 42). Dataset beku: manifest `9e841c17…12ba`.
Induk: prov_v6 (manifest `2963885f…a4d7`, tidak diubah; berkas di-hardlink).

## Aturan
- Pool = semua gambar train + val + test prov_v6 (8.660; karantina memang tidak ada di prov_v6).
- Dibagi ACAK PER GAMBAR 70/20/10, tanpa memperhatikan kelompok duplikat dan tanpa aturan train_only (praktik umum).
- Background train dipangkas ke 10% dengan cara prov_v6 (hanya ds4): 785 -> 586 (199 dipangkas, tidak dipakai).
- test_new_session (299) disalin apa adanya dari prov_v6; tidak ada anggota kelompoknya di train/val/test acak.

| split | blast | brown spot | BLB | background | total |
|---|---|---|---|---|---|
| train | 1107 | 2758 | 1412 | 586 (10,0%) | 5863 |
| val | 307 | 802 | 374 | 249 (14,4%) | 1732 |
| test | 163 | 385 | 200 | 118 (13,6%) | 866 |
| test_new_session | 96 | 147 | 0 | 56 | 299 |

## Kebocoran split acak (ukuran: kelompok prov_v6)
Gambar val/test acak yang kelompok prov_v6-nya juga punya anggota di train acak:

| | gambar | punya kembar di train | % |
|---|---|---|---|
| val | 1732 | 1596 | 92,1 |
| test | 866 | 793 | 91,6 |
| val+test | 2598 | 2389 | 92,0 |

Per sumber (val+test): ds1 97,0% · ds1_provisional (BLB) 97,9% · ds4 87,9% · ds2 44,4%.
Kelompok yang terbelah lintas split: 698 dari 850 kelompok berisi >1 gambar.
Gambar val/test dari sumber yang di prov_v6 hanya boleh di train (ds1 Blast/Brown spot, hasil augmentasi): 740.

**Cara membaca:** "kembar" = satu kelompok prov_v6. Kelompok itu menggabungkan duplikat sungguhan (phash, SIFT, nama dasar
Roboflow, kunci ID ds1) DAN gambar yang berkaitan (kemiripan embedding 0,97, jepretan beruntun <=5 detik), secara berantai
(kelompok terbesar 203 gambar). Jadi 92% adalah batas atas "ada gambar terkait di train", bukan jumlah duplikat identik.

Rincian per gambar: `kebocoran_per_gambar.csv`; angka lengkap: `ringkasan.json`.

## Pemakaian
- Model randsplit melihat 433 gambar test prov_v6 saat training -> TIDAK boleh dievaluasi pada test prov_v6.
- Set yang sebanding antara model randsplit dan model final: hanya test_new_session.
