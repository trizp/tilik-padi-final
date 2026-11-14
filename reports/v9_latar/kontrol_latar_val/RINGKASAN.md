# Kontrol latar alternatif (ANALISIS VALIDASI, bukan data uji)

Foto: 861 (cadangan tanpa masker: 16); runs: v7_final_prov_yolov8s_s0, v7_final_prov_yolov8s_s1, v7_final_prov_yolov8s_s2, v7_final_prov_yolo11s_s0, v7_final_prov_yolo11s_s1, v7_final_prov_yolo11s_s2
Rata-rata selisih abu-abu dari masker vs foto v9: 0.14 (0-255; kecil = masker cocok)

Kolom contoh_kondisi.jpg: original, gray, black, blur, swap

| Model | Kondisi | Akurasi | Blast dR | Brown dR | Sehat benar |
|---|---|---|---|---|---|
| yolov8s | original | 0.874 ± 0.006 | 0.855 ± 0.038 | 0.881 ± 0.005 | 0.879 ± 0.008 |
| yolov8s | gray | 0.815 ± 0.008 | 0.555 ± 0.016 | 0.874 ± 0.015 | 0.930 ± 0.007 |
| yolov8s | black | 0.747 ± 0.009 | 0.282 ± 0.053 | 0.914 ± 0.017 | 0.886 ± 0.028 |
| yolov8s | blur | 0.744 ± 0.006 | 0.358 ± 0.075 | 0.876 ± 0.013 | 0.869 ± 0.045 |
| yolov8s | swap | 0.838 ± 0.005 | 0.791 ± 0.034 | 0.835 ± 0.015 | 0.872 ± 0.024 |
| yolo11s | original | 0.879 ± 0.007 | 0.852 ± 0.019 | 0.885 ± 0.021 | 0.891 ± 0.029 |
| yolo11s | gray | 0.703 ± 0.048 | 0.170 ± 0.027 | 0.829 ± 0.155 | 0.932 ± 0.045 |
| yolo11s | black | 0.645 ± 0.042 | 0.047 ± 0.040 | 0.780 ± 0.178 | 0.909 ± 0.048 |
| yolo11s | blur | 0.643 ± 0.024 | 0.107 ± 0.032 | 0.783 ± 0.097 | 0.859 ± 0.020 |
| yolo11s | swap | 0.837 ± 0.011 | 0.780 ± 0.019 | 0.820 ± 0.035 | 0.892 ± 0.043 |

## Pemeriksaan terhadap hasil lama (latar hitam, seed 0)
- yolov8s: sekarang 0.7573, sebelumnya 0.7491 -> COCOK
- yolo11s: sekarang 0.6690, sebelumnya 0.6617 -> COCOK

## Baris tabel LaTeX (Akurasi, Blast dR; rata-rata 3 seed)
Original & 0.874 & 0.855 & 0.879 & 0.852 \\
Gray & 0.815 & 0.555 & 0.703 & 0.170 \\
Black & 0.747 & 0.282 & 0.645 & 0.047 \\
Blurred & 0.744 & 0.358 & 0.643 & 0.107 \\
Swapped & 0.838 & 0.791 & 0.837 & 0.780 \\
