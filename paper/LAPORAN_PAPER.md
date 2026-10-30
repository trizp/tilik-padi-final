# Laporan gambar dan pengecekan angka paper

Ditulis 2026-10-26. Semua angka dibaca dari hasil yang sudah ada. Tidak ada evaluasi ulang, tidak ada inferensi model, dan model, ambang,
dataset, serta `main.tex` tidak diubah (sha256 `main.tex` 8bc3ebf3…caceb, waktu ubah 14:22, sebelum pekerjaan ini).

Skrip: `scripts/paper/{fig_background, fig_leaf_examples, tabel_data_prov_v7}.py`. Keluaran: `paper/figures/`.

## Gambar (butir 1-2)

| Berkas | Isi | Ukuran | Data |
|---|---|---|---|
| `fig_background.pdf` / `.png` | 3 panel (mAP50-95, diagnosis accuracy, blast recall) × 4 kondisi train/test, batang YOLOv8s (biru, polos) dan YOLO11s (jingga, arsir), galat IK 95% | 3,5 × 1,9 in | `fig_background.csv` |
| `fig_leaf_examples.pdf` / `.png` | 2 × 3 (Blast, Brown spot, Healthy); atas foto asli + kotak label tipis, bawah tanpa latar | 3,5 × 2,4 in | `fig_leaf_examples.csv` |

Keduanya memakai Times New Roman 8 pt (Nimbus Roman tidak terpasang di mesin ini), font tertanam sebagai TrueType (fonttype 42; diperiksa:
`/FontFile2` ada, tanpa Type 3), label bahasa Inggris, tanpa judul gambar, palet Okabe-Ito.

**fig_background.**
- Rata-rata 3 seed diambil dari hasil yang sama dengan `tabel1_2x2_utama.csv` dan `tabel4_sel_silang.csv` (test bersih irisan,
  432 foto, 104 kelompok).
- IK sel diagonal (Orig/Orig, Leaf/Leaf) ada di `tabel1`. IK sel silang (Orig/Leaf, Leaf/Orig) belum ada di tabel mana pun, jadi
  dihitung dari keluaran evaluasi per foto yang sudah tersimpan (`runs/*/eval_*`), dengan fungsi dan urutan resample yang sama dengan
  `scripts/analisis_akhir_v9.py` (B = 1000, seed 20261017).
- Bukti kesamaan cara hitung: rata-rata ke-24 sel identik dengan `tabel4`, dan IK ke-12 sel diagonal identik dengan `tabel1`. Skrip
  berhenti bila ada yang tidak sama.
- IK sel silang (baru):

| Model | Train/test | mAP50-95 | Accuracy | Blast recall |
|---|---|---|---|---|
| YOLOv8s | Orig/Leaf | 0.197 [0.163, 0.228] | 0.754 [0.695, 0.816] | 0.514 [0.403, 0.637] |
| YOLOv8s | Leaf/Orig | 0.226 [0.178, 0.263] | 0.755 [0.693, 0.809] | 0.781 [0.670, 0.866] |
| YOLO11s | Orig/Leaf | 0.116 [0.093, 0.145] | 0.665 [0.585, 0.749] | 0.108 [0.058, 0.167] |
| YOLO11s | Leaf/Orig | 0.224 [0.163, 0.269] | 0.745 [0.686, 0.794] | 0.787 [0.640, 0.892] |

**fig_leaf_examples.**
- Kerangka sampel: foto VAL prov_v7 bersumber ds4 per kelas, yang ada di VAL prov_v9 dengan latar benar-benar dihapus.
- Foto cadangan (tidak dipisah) dikeluarkan dari kerangka: Blast 2, Brown spot 1, Healthy 9. Alasannya, gambar ini memperlihatkan hasil
  penghapusan latar.
- Satu foto per kelas dipilih dengan `random.Random(20261026)`:

| Kolom | Foto terpilih | Asal di RiceLeafDiseaseBD | Kotak label |
|---|---|---|---|
| Blast | `ds4_blast-1134_57bcead0.jpg` | `Blast/blast_1134.jpg` | 3 |
| Brown spot | `ds4_brown-spot-1675_52edef0c.jpg` | `Brown spot/brown spot_1675.jpg` | 8 |
| Healthy | `ds4_Healthy-1187_669033b2.jpg` | `Healthy/Healthy_1187.jpg` | 0 |

- Semua dari ds4; tidak ada ds1, ds2, atau ds3.
- Foto ditanam pada 450 px (±400 dpi di sel ±1,1 in); PDF 1,3 MB.

## Butir 3 — Angka Tabel I (prov_v7)

Sumber: `data/processed/prov_v7/manifest.csv` (kolom `split`, `class`, `group`). Sel = foto/kelompok.

| Kelas | Train | Val | Test | Test sesi baru |
|---|---|---|---|---|
| Blast | 1244/316 | 222/84 | 111/48 | 96/62 |
| Brown spot | 3450/419 | 330/74 | 165/43 | 147/71 |
| BLB (provisional) | 1401/145 | 390/41 | 195/21 | 0/0 |
| Healthy (background) | 677/288 | 317/107 | 158/58 | 56/35 |
| Total (kelompok berbeda) | 6772/1041 | 1259/245 | 629/125 | 299/133 |

- Tidak ada kelompok yang melintasi split.
- Jumlah kelompok per kelas tidak dijumlahkan menjadi total. Contohnya train: 316 + 419 + 145 + 288 = 1.168, sedangkan kelompok
  berbeda 1.041, karena satu kelompok bisa memuat lebih dari satu kelas.
- Total foto per split sama dengan Tabel I di `main.tex`. Total kelompok sama dengan ringkasan v7 (train 1.041, test 125, sesi baru
  133).

LaTeX (juga di `paper/figures/tabel_data_prov_v7.tex`):

```latex
\begin{table}[htbp]
\caption{Dataset Splits: Images/Groups per Class}
\begin{center}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{|l|c|c|c|c|}
\hline
\textbf{Class} & \textbf{Train} & \textbf{Val} & \textbf{Test} & \textbf{New session} \\
\hline
Blast & 1244/316 & 222/84 & 111/48 & 96/62 \\
Brown spot & 3450/419 & 330/74 & 165/43 & 147/71 \\
BLB$^{\mathrm{a}}$ & 1401/145 & 390/41 & 195/21 & 0/0 \\
Healthy$^{\mathrm{b}}$ & 677/288 & 317/107 & 158/58 & 56/35 \\
\hline
Total & 6772/1041 & 1259/245 & 629/125 & 299/133 \\
\hline
\end{tabular}
\par\vspace{2pt}{\scriptsize Cells: images/groups. Groups contain near-duplicate and burst images and never cross splits. $^{\mathrm{a}}$Provisional, single source. $^{\mathrm{b}}$Background images without boxes. A group may contain images of more than one class, so class groups do not sum to the total, which counts distinct groups.}
\label{tab:data}
\end{center}
\end{table}
```

Tabel ini tidak memuat baris External (ds3, 70 foto: 35 BLB/35 kelompok, 35 Blast/33 kelompok; `reports/final_eval_v7/ds3/RINGKASAN_DS3.md`).
Bila baris itu ingin dipertahankan, tambahkan sebagai catatan atau baris terpisah.

## Butir 4 — Verifikasi Tabel V (test bersih irisan, 432 foto, rata-rata 3 seed)

Nilai mutlak dihitung ulang dari keluaran evaluasi tersimpan (`paper/figures/tabel5_nilai_mutlak.csv`, 4 desimal) dan identik dengan
`reports/final_eval_v9/tabel/tabel4_sel_silang.csv` setelah dibulatkan.

| Model | Train/Test | mAP50-95 (hitung / main.tex) | Accuracy (hitung / main.tex) | Blast R (hitung / main.tex) |
|---|---|---|---|---|
| YOLOv8s | Orig/Orig | 0.2485 → 0.249 / 0.249 ✓ | 0.7508 → 0.751 / 0.751 ✓ | 0.7868 → 0.787 / 0.787 ✓ |
| YOLOv8s | Orig/Leaf | 0.1968 → 0.197 / 0.197 ✓ | 0.7539 → 0.754 / 0.754 ✓ | 0.5135 → 0.514 / 0.514 ✓ |
| YOLOv8s | Leaf/Leaf | 0.2525 → 0.253 / 0.253 ✓ | 0.7670 → 0.767 / 0.767 ✓ | 0.8018 → 0.802 / 0.802 ✓ |
| YOLOv8s | Leaf/Orig | 0.2263 → **0.226** / 0.227 ✗ | 0.7554 → 0.755 / 0.755 ✓ | 0.7808 → 0.781 / 0.781 ✓ |
| YOLO11s | Orig/Orig | 0.2728 → 0.273 / 0.273 ✓ | 0.7824 → 0.782 / 0.782 ✓ | 0.8468 → 0.847 / 0.847 ✓ |
| YOLO11s | Orig/Leaf | 0.1158 → 0.116 / 0.116 ✓ | 0.6651 → 0.665 / 0.665 ✓ | 0.1081 → 0.108 / 0.108 ✓ |
| YOLO11s | Leaf/Leaf | 0.2774 → 0.277 / 0.277 ✓ | 0.7886 → 0.789 / 0.789 ✓ | 0.7958 → 0.796 / 0.796 ✓ |
| YOLO11s | Leaf/Orig | 0.2235 → **0.224** / 0.223 ✗ | 0.7454 → **0.745** / 0.746 ✗ | 0.7868 → 0.787 / 0.787 ✓ |

Hasil: 21 dari 24 angka cocok; 3 berbeda di desimal ketiga, semuanya pada baris Leaf/Orig (baris 4 dan 8 yang ditandai `\todo`).
Nilai benar: YOLOv8s Leaf/Orig mAP50-95 **0.226**; YOLO11s Leaf/Orig mAP50-95 **0.224** dan accuracy **0.745**. Baris 2 dan 6 cocok.

## Butir 5 — Audit semua angka di main.tex

Sumber utama:
- `reports/final_eval_v7/tabel/` (T1 = tabel1_utama_per_model, T4 = tabel4_kecepatan, T5b = tabel5b_kebocoran_split_acak,
  T8 = tabel8_v8_vs_v11_berpasangan, T10 = tabel10_ketepatan_perkiraan);
- `reports/final_eval_v9/tabel/` (V1 = tabel1_2x2_utama, V2 = tabel2_selisih, V4 = tabel4_sel_silang, V5 = tabel5_kontras_latar,
  V7 = tabel7_ds3, V8 = tabel8_audit_pemisahan);
- manifest prov_v7/prov_v9, KEPUTUSAN_PROYEK.md, dan ringkasan laporan yang disebut per baris.

### Berbeda (dengan nilai yang benar)

| Lokasi di main.tex | Tertulis | Benar | Sumber |
|---|---|---|---|
| Tabel V, YOLOv8s Leaf-only/Original, mAP50-95 | 0.227 | 0.226 | V4 (0.226 ± 0.006); hitung ulang 0.2263 |
| Tabel V, YOLO11s Leaf-only/Original, mAP50-95 | 0.223 | 0.224 | V4 (0.224 ± 0.012); hitung ulang 0.2235 |
| Tabel V, YOLO11s Leaf-only/Original, Acc. | 0.746 | 0.745 | V4 (0.745 ± 0.047); hitung ulang 0.7454 |
| III-A: "930 images ... had their pixels stored in a rotated orientation" | 930 ditemukan | 931 foto ds4 kotaknya tidak cocok; 930 diputar dengan jpegtran, 1 diperbaiki dengan menerapkan tag EXIF | `reports/prov_v7_prep/jpegtran_ringkasan.json`; KEPUTUSAN 2026-10-11 ("930 dari 931 ... tidak bertag orientasi"); manifest prov_v7 (931 baris `perbaikan_orientasi`) |

Catatan isi (bukan angka, tetapi tidak sesuai data): III-A menyebut ds1 hanya sebagai sumber BLB. Menurut manifest prov_v7, ds1 juga
menyumbang **1.970 foto Brown spot dan 411 foto Blast** ke train (salinan augmentasi; hanya train). Kalimat ini sebaiknya diperbaiki,
misalnya "BLB images, and additional training images of blast and brown spot, were taken from a Kaggle dataset [ds1]".

### Tidak ditemukan sumbernya

| Lokasi | Klaim | Yang ada di berkas |
|---|---|---|
| III-A | jpegtran "raised mAP50-95 by roughly 65% in a preliminary comparison" | Tidak ada perbandingan pendahuluan dengan angka 65%. Yang ada adalah perbandingan run final v6 (sebelum perbaikan) vs v7 (`final_eval_v7/tabel/lampiran_v6_vs_v7.csv`): test bersih mAP50-95 naik +52% sampai +70% per model (rata-rata 4 model 0.162 → 0.263, ±+62%); test semua kelas +41% sampai +62%; test sesi baru 0.006–0.011 → 0.228–0.269. Itu bukan "preliminary comparison". |

### Cocok

Abstrak dan Pendahuluan:
- inflasi split acak "about 0.11" (T10: 0.106 dan 0.119);
- YOLO11s 78.2% dan YOLOv8s 75.0% (T1 test bersih 0.782, 0.750);
- penurunan recall Blast 0.74 dan 0.27 (V5: −0.739 dan −0.273, irisan 432 foto);
- 4 varian × 3 seed.

III-A dan III-B:
- empat sumber;
- kriteria kelompok (`reports/prov_v6/split_report.md`, aturan v5, diwarisi prov_v7): phash pada 8 varian dihedral, DINOv2 cos ≥ 0.97,
  SIFT ≥ 21 inlier pada 100 tetangga, nama dasar sama, jendela waktu ≤ 5 s. Catatan: main.tex tidak menyebut ambang phash ≤ 14, syarat
  sebaran inlier ≥ 2%, dan kunci ID K4 ds1.

III-C dan Tabel I:
- 70:20:10;
- satu sesi dalam satu jam (sesi 13: 17:13:54–18:05:31, `prov_v6/split_report.md` §17);
- 6772 / 1259 / 629 (434 + 195) / 299 (96 / 147 / 56) (manifest);
- External 70 = 35 BLB + 35 Blast; kappa 0.43 (0.431); 70 foto diterima kedua penilai (`ds3/RINGKASAN_DS3.md`).

IV:
- 3 seed (0, 1, 2), 100 epoch, imgsz 640, batch 8, patience 0, model COCO (`runs/*/args.yaml`);
- RTX 3050 Ti 4 GB (T4 caption; nvidia-smi 4096 MiB);
- val bersih 869 (222 + 330 + 317; KEPUTUSAN);
- bootstrap 1000 per kelompok;
- split acak 70:20:10 untuk YOLOv8n dan YOLO11n (KEPUTUSAN, 1 seed);
- SAM 2.1, abu 114, kotak < 50% dibuang;
- 626 foto dikeluarkan, "mostly BLB" (476 dari 626 = 76% BLB; KEPUTUSAN 2026-10-17);
- 50 background train dipangkas acak, background 10%;
- audit 50 foto VAL acak di luar sampel perancangan aturan: 37/38 (97.4%, Wilson 86.5–99.5%) dan BLB 4/12 (V8; `v9_latar/audit_masker`);
- irisan 432 dan 292 (`final_eval_v9/angka_kunci_v9.json`).

V-A dan Tabel II:
- "76–92%" (T5b test: 76.1% kembaran langsung, 91.6% kelompok berantai);
- 0.106–0.119 dan 0.007–0.025 (T10);
- "about 40%" (0.119/0.271 = 44%, 0.106/0.283 = 37%; ringkasan v7: "sekitar 40%");
- Tabel II, keenam baris × 2 kolom (T10).

V-B, Tabel III, dan Tabel IV:
- Tabel III, semua sel P/R/F1/mAP50/mAP50-95/accuracy untuk 8 baris (T1);
- Params / GFLOPs / CPU ms (T4);
- "434" dan "299 images";
- teks 0.782 / 0.750 / 0.764;
- Tabel IV, kedelapan selisih dan IK (T8, baris "gabungan");
- "seven of eight";
- "0.02 to 0.04" (nilai sebenarnya 0.017–0.041);
- "3 to 7 pp" (0.031–0.071);
- YOLO11 lebih sedikit parameter dan lebih cepat di CPU untuk varian s (83.4 vs 95.1 ms).

Kesalahan diagnosis (dihitung dari CSV yang sudah ada di `runs/v7_final_prov_*/eval_test/`, 12 run):
- BLB presisi 0.977–1.000 ("close to 1.0") dan recall 0.605–0.774 ("about 0.7");
- daun sehat didiagnosis Brown spot 21–32% per run (rata-rata per model 0.236–0.304; ringkasan v7 "sekitar 24% pada model terbaik tiap
  keluarga"), jadi "about one in four" cocok secara kasar; YOLO11n mendekati satu dari tiga.

V-C dan Tabel V:
- −0.052, 0.787 → 0.514, −0.157, −0.117, 0.847 → 0.108 (V5, V4);
- "most of the lost blast images were diagnosed as brown spot" (YOLO11s 277 dari 297 Blast yang salah = 93%; YOLOv8s 78%;
  `eval_v9tl_test` confusion, 3 seed);
- −0.465 [−0.543, −0.371] dan −0.447 [−0.528, −0.375] (V5);
- Tabel V: 21/24 cocok (lihat butir 4);
- −0.051 [−0.108, −0.003]; +0.025 [0.010, 0.046]; +0.022 [−0.004, 0.050]; −0.006 [−0.039, 0.025] (V5, V2);
- "all differences ... within their CIs" (V5 baris "v9 tanpa − v7 dengan", test bersih dan sesi baru, kecuali recall Blast YOLO11s di
  test bersih).

V-D dan Tabel VI:
- 0.18–0.34 dan Blast hampir tidak dikenali (ringkasan v7);
- 0.224 → 0.433, 0.048 → 0.486, 0.262 → 0.305 dengan IK memuat nol ([−0.073, 0.157]), BLB recall turun (V7);
- Tabel VI, semua 12 angka (V7);
- 2.9% cadangan (V7).

VI dan VII: "about 0.11", "roughly 40%", dan pernyataan kualitatif sesuai V2/V5/V7.

Catatan kecil (cocok, tetapi perlu disadari):
- Tabel III YOLOv8s test "0.249 ± 0.003" sama dengan T1. T10 menulis ± 0.004 (simpangan baku 0.0035 tepat di batas pembulatan).
- V-C "models trained on leaf-only images also lost some performance when tested on original photographs, although considerably less":
  benar untuk mAP50-95 dan recall Blast (V5). Untuk accuracy YOLOv8s, arah sebaliknya tidak turun (+0.003), sedangkan Leaf→Orig turun
  0.012; keduanya memuat nol.
