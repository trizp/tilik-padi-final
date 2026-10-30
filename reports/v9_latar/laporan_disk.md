# Laporan disk sebelum membangun prov_v9 (2026-10-16)

TIDAK ada yang dihapus. Daftar kandidat di bawah hanya usulan dan menunggu persetujuan pemilik skripsi.

## Ruang

- D: 297 GB, terpakai 285 GB, **tersisa ±12 GB** (sebagian besar pemakaian D: berada di luar folder proyek).
- C: tersisa ±31 GB.

## Pemakaian folder besar di proyek (du; berkas hardlink dihitung sekali per pemanggilan, sehingga folder dataset yang berbagi berkas tampak kecil)

| Folder | Ukuran |
|---|---|
| data/raw | 9,7 GB |
| data/interim | 7,2 GB (sift2 3,5 GB; ds4_orientasi_v7_jpegtran 1,5 GB; ds4_orientasi_v7_q95_DITOLAK 1,5 GB; sift 1,0 GB; dino_emb 0,2 GB) |
| .venv (venv proyek) | 3,8 GB |
| runs | 3,2 GB |
| tools | 1,1 GB (_venv_latar 0,48 GB; _model_latar 0,49 GB; libjpeg-turbo) |
| app | 0,54 GB |
| data/processed/prov_v7 | 0,32 GB (sisanya hardlink ke versi lain) |
| data/derived (prov_v6_valsel, prov_v7_valsel, prov_v8_valsel masing-masing ±0,28 GB salinan; v9_latar 0,28 GB) | ±1,1 GB |
| reports | 0,14 GB |

## Kebutuhan v9 (perkiraan)

| Butir | Perkiraan |
|---|---|
| prov_v9 (foto tanpa latar JPEG q95 ±44% ukuran asli; foto cadangan = hardlink) | ±1 GB |
| prov_v9_valsel (salinan val bersih) | ±0,1 GB |
| 6 run training (bobot best/last, plot, log) | ±1 GB |
| Masker (PNG) | ±0,03 GB |
| **Sisa D: setelah semuanya** | **±10 GB (> 5 GB, training boleh dimulai)** |

## Kandidat yang aman dihapus (BELUM dihapus)

| Kandidat | Ukuran | Alasan aman | Yang hilang bila dihapus |
|---|---|---|---|
| data/interim/ds4_orientasi_v7_q95_DITOLAK | 1,5 GB | metode putar ulang q95 yang DITOLAK; prov_v7 memakai jpegtran (lossless) | kemampuan memeriksa ulang perbandingan q95 vs jpegtran (laporannya tetap di reports/) |
| data/derived/prov_v8_valsel | 0,29 GB | salinan val bersih prov_v8; v8 DIHENTIKAN (2026-10-14) | tidak ada; bisa dibuat ulang dengan buat_valsel.py dari prov_v8 beku |
| data/derived/prov_v6_valsel | 0,28 GB | salinan val bersih prov_v6; v6 digantikan v7 (bug orientasi) | tidak ada; bisa dibuat ulang dari prov_v6 beku |
| tools/_model_latar/u2net (bobot rembg u2net dan isnet) | ±0,35 GB | rembg tidak dipilih (tahap 0) | tahap 0 rembg tidak bisa diulang tanpa mengunduh ulang |
| data/derived/v9_latar/val_tanpa_latar_hitam | ±0,1 GB | hanya uji kepekaan isian hitam (tahap 1, 2 model) | gambar uji isian hitam (hasil JSON tetap ada) |
| data/interim/sift dan sift2 | 4,5 GB | cache fitur SIFT untuk pengelompokan; prov_v7 sudah beku | membangun ulang pengelompokan dari awal butuh berjam-jam komputasi ulang; HANYA bila ruang benar-benar kurang |

BUKAN kandidat: data/interim/ds4_orientasi_v7_jpegtran (berkasnya di-hardlink oleh prov_v7, menghapus tidak membebaskan ruang dan
memutus jejak asal), data/raw, runs, venv, dataset beku.
