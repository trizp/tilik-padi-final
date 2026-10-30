# Kebocoran split acak: batas atas vs batas bawah

Dibuat 2026-10-09T09:20:17. Kembaran langsung = phash<=14 (8 varian dihedral) ATAU nama dasar Roboflow/K4 sama ATAU SIFT>=21 inlier ke satu gambar train (kandidat: k=200 tetangga embedding); TANPA kelompok berantai dan jendela waktu.

| himpunan | gambar | kelompok berantai (batas atas) | kembaran langsung (batas bawah) |
|---|---|---|---|
| split val | 1732 | 92.1% | 77.3% |
| split test | 866 | 91.6% | 76.1% |
| val+test | 2598 | 92.0% | 76.9% |
| sumber ds1 | 740 | 97.0% | 96.4% |
| sumber ds1_provisional | 574 | 97.9% | 97.9% |
| sumber ds2 | 45 | 44.4% | 31.1% |
| sumber ds4 | 1239 | 87.9% | 57.1% |
