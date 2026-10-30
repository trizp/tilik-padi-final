# Tahap 0: uji coba pemisahan daun (VAL v7, ±100 foto, tanpa training)

Masker dibuat TANPA label; kotak bercak hanya dipakai untuk mengukur. Latar diganti abu-abu (114,114,114); kotak tidak diubah.
Terpotong parah (proksi): foto berkotak dengan < 50% total luas kotak di dalam masker, atau masker < 5% foto.

| Metode | Kelompok | Foto | Kotak | Kotak >= 50% di daun | Foto terpotong parah | Median luas masker | Median waktu/foto |
|---|---|---|---|---|---|---|---|
| rembg_u2net | semua | 100 | 233 | 56.7% | 47.0% | 14.6% | 0.313 s (CPU (onnxruntime)) |
| rembg_u2net | ds4 Blast | 30 | 87 | 89.7% | 23.3% | 22.2% | 0.276 s (CPU (onnxruntime)) |
| rembg_u2net | ds4 Brown spot | 30 | 125 | 38.4% | 56.7% | 12.9% | 0.317 s (CPU (onnxruntime)) |
| rembg_u2net | ds4 sehat | 20 | 0 | None% | 45.0% | 13.6% | 0.327 s (CPU (onnxruntime)) |
| rembg_u2net | ds1 BLB | 20 | 21 | 28.6% | 70.0% | 13.0% | 0.323 s (CPU (onnxruntime)) |
| rembg_isnet | semua | 100 | 233 | 26.2% | 78.0% | 0.8% | 0.924 s (CPU (onnxruntime)) |
| rembg_isnet | ds4 Blast | 30 | 87 | 39.1% | 66.7% | 0.6% | 0.99 s (CPU (onnxruntime)) |
| rembg_isnet | ds4 Brown spot | 30 | 125 | 20.0% | 80.0% | 0.4% | 0.922 s (CPU (onnxruntime)) |
| rembg_isnet | ds4 sehat | 20 | 0 | None% | 80.0% | 1.9% | 0.94 s (CPU (onnxruntime)) |
| rembg_isnet | ds1 BLB | 20 | 21 | 9.5% | 90.0% | 0.6% | 0.836 s (CPU (onnxruntime)) |
| sam2 | semua | 100 | 233 | 87.1% | 15.0% | 25.7% | 1.442 s (GPU RTX 3050 Ti (SAM2)) |
| sam2 | ds4 Blast | 30 | 87 | 83.9% | 10.0% | 25.8% | 1.438 s (GPU RTX 3050 Ti (SAM2)) |
| sam2 | ds4 Brown spot | 30 | 125 | 97.6% | 0.0% | 28.6% | 1.452 s (GPU RTX 3050 Ti (SAM2)) |
| sam2 | ds4 sehat | 20 | 0 | None% | 0.0% | 29.8% | 1.457 s (GPU RTX 3050 Ti (SAM2)) |
| sam2 | ds1 BLB | 20 | 21 | 38.1% | 60.0% | 9.9% | 1.389 s (GPU RTX 3050 Ti (SAM2)) |
| sam2_v2 | semua | 100 | 233 | 89.3% | 9.0% | 28.4% | 1.389 s (GPU RTX 3050 Ti (SAM2)) |
| sam2_v2 | ds4 Blast | 30 | 87 | 86.2% | 6.7% | 25.8% | 1.384 s (GPU RTX 3050 Ti (SAM2)) |
| sam2_v2 | ds4 Brown spot | 30 | 125 | 95.2% | 3.3% | 28.6% | 1.398 s (GPU RTX 3050 Ti (SAM2)) |
| sam2_v2 | ds4 sehat | 20 | 0 | None% | 0.0% | 29.8% | 1.397 s (GPU RTX 3050 Ti (SAM2)) |
| sam2_v2 | ds1 BLB | 20 | 21 | 66.7% | 30.0% | 65.1% | 1.344 s (GPU RTX 3050 Ti (SAM2)) |
| warna | semua | 100 | 233 | 97.0% | 5.0% | 96.9% | 0.007 s (CPU) |
| warna | ds4 Blast | 30 | 87 | 98.9% | 0.0% | 99.2% | 0.007 s (CPU) |
| warna | ds4 Brown spot | 30 | 125 | 100.0% | 0.0% | 96.3% | 0.007 s (CPU) |
| warna | ds4 sehat | 20 | 0 | None% | 0.0% | 96.4% | 0.007 s (CPU) |
| warna | ds1 BLB | 20 | 21 | 71.4% | 25.0% | 90.0% | 0.002 s (CPU) |

## Catatan metode dan penilaian visual 20 contoh montase

- rembg_u2net / rembg_isnet: rembg 2.0.85 di venv terpisah `tools/_venv_latar` (onnxruntime CPU).
- sam2: SAM2.1-b mode otomatis (grid 16x16, batch 16) lewat ultralytics venv proyek tanpa memasang paket (YOLO_AUTOINSTALL=false;
  `pip freeze` sama sebelum/sesudah, `_freeze_venv_proyek_sebelum.txt`), aturan v1 = luas x pusat x ketajaman.
- sam2_v2: aturan v2, disusun SETELAH melihat montase v1 (ketajaman memilih latar bergulma; foto tanpa kandidat jadi abu-abu penuh):
  skor = sqrt(luas) x (0,2 + cakupan kotak tengah), dan cadangan "tidak dipisah" bila tidak ada kandidat. Karena disusun dengan melihat
  sampel ini, angka sam2_v2 di tabel sedikit optimistis; tahap 1 (1.159 foto VAL lain) menjadi pemeriksaan di luar sampel.
- warna: ambang HSV vegetasi; menyisakan ±97% foto (latar sawah juga hijau), jadi praktis tidak menghapus latar. Hanya pembanding.

Penilaian visual saya atas 20 contoh yang sama (bukan pengukuran otomatis):

| Metode | Daun utama terpisah baik | Salah objek / terpotong | Tidak dipisah (cadangan) |
|---|---|---|---|
| rembg_u2net | 8 | 12 (tangan, kaki, malai, potongan kecil) | 0 |
| sam2 (v1) | 17 | 3 (1 latar terpilih, 2 foto BLB abu-abu penuh) | 0 |
| sam2_v2 | 17 | 1 (latar terpilih pada 1 foto Blast) | 2 (BLB) |

## Rekomendasi

- **Metode: SAM2.1-b mode otomatis + aturan v2 + cadangan.** Paling banyak mempertahankan bercak (89,3% kotak >= 50% di daun; ds4 Brown spot
  95%, ds4 Blast 86%) sambil benar-benar membuang latar (median masker 28% foto). Kelemahan: lambat (1,4-2,9 detik per foto di GPU) dan
  lemah pada ds1 BLB (foto dipegang tangan atau sangat dekat): sepertiga foto BLB jatuh ke cadangan.
- **Aturan buang-kotak:** kotak dengan < 50% luasnya di dalam masker daun dibuang dari label (VAL penuh: Blast 7,3%, Brown spot 2,3%,
  BLB 29,4%; `buang_kotak_val.json`).
