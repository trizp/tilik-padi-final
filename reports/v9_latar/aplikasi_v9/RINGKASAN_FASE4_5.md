# Aplikasi v9: hasil fase 4-5 (evaluasi alur aplikasi)

Ditulis 2026-10-20. Rencana, kriteria, dan ambang aplikasi dicatat SEBELUM fase 5 (KEPUTUSAN_PROYEK.md 2026-10-20). `app/` dan `tilik-padi`
tidak diubah. Setiap set dievaluasi SEKALI. Alur aplikasi = pemisah daun YOLOv8n-seg (tau 0,35) -> foto cadangan diminta foto ulang ->
detektor YOLOv8s v9.

## Fase 4 - Model aplikasi v9 dan ambang

- `v9_deploy_yolov8s_s2`: train+val+test prov_v9 (7.654 foto, background 9,99%, tanpa test sesi baru), 50 epoch (median epoch terbaik
  YOLOv8s v9: 31/59/50), seed 2, last.pt; 1,84 jam GPU, tanpa resume. ONNX 42,7 MB (metadata bersih); ONNX vs .pt pada 20 foto val: kelas
  identik 20/20, selisih keyakinan 0 (`fase4_onnx_model_aplikasi.json`).
- Ambang aplikasi (VAL, alur aplikasi, model final s2, 72 foto cadangan tidak dihitung): **terdeteksi 0,36**, **t_rendah 0,30** (tiga
  tingkat; "kemungkinan" 0,30-0,36 tidak kosong). **Bercak berkeyakinan rendah tetap nonaktif** (tidak ada batas dengan proporsi sehat
  <= 25%; terendah 36,9%). (`fase4_ambang_aplikasi.json`)

## Fase 5 butir 3 - Alur aplikasi vs alur SAM2 vs v7 (model final YOLOv8s, 3 seed; irisan foto)

Foto cadangan pada alur aplikasi = diminta foto ulang: tidak didiagnosis (dihitung salah pada akurasi) dan tanpa prediksi kotak pada mAP.
Rata-rata ± std 3 seed; [IK 95% bootstrap per kelompok]. (`fase5_alur_aplikasi_final.csv`, `fase5_selisih_alur.csv`)

| Set | Alur | mAP50-95 | P / R / F1 | Akurasi diagnosis | Akurasi pada foto yang didiagnosis | Recall Blast | % foto ulang |
|---|---|---|---|---|---|---|---|
| Test bersih (432) | aplikasi | 0,253 [0,201; 0,289] | 0,571 / 0,629 / 0,599 | 0,731 [0,656; 0,800] | 0,759 | 0,805 | 3,7% |
| | SAM2 (v9) | 0,253 | 0,577 / 0,619 / 0,597 | 0,767 [0,697; 0,838] | 0,767 | 0,802 | 0% |
| | v7 (foto asli) | 0,249 | 0,558 / 0,631 / 0,592 | 0,751 [0,675; 0,821] | 0,751 | 0,787 | 0% |
| Test sesi baru (292) | aplikasi | 0,235 [0,210; 0,271] | 0,491 / 0,609 / 0,543 | 0,689 [0,620; 0,759] | 0,709 | 0,904 | 2,7% |
| | SAM2 (v9) | 0,238 | 0,487 / 0,622 / 0,545 | 0,711 | 0,711 | 0,926 | 0% |
| | v7 (foto asli) | 0,244 | 0,501 / 0,600 / 0,546 | 0,692 | 0,692 | 0,926 | 0% |

Selisih berpasangan (alur aplikasi − pembanding):
- Test bersih: vs SAM2 mAP50-95 +0,000 [-0,004; 0,004], **akurasi -0,036 [-0,069; -0,013]**, recall Blast +0,003; vs v7 mAP50-95 +0,004,
  akurasi -0,020 [-0,064; 0,019], recall Blast +0,018 (semua memuat nol).
- Test sesi baru: vs SAM2 mAP50-95 -0,002, **akurasi -0,022 [-0,046; -0,003]**, recall Blast -0,021 [-0,061; 0,000]; vs v7 mAP50-95 -0,008,
  akurasi -0,002, recall Blast -0,021 (semua memuat nol).
- Penurunan akurasi vs SAM2 terutama dari foto yang diminta foto ulang (dihitung salah). Pada foto yang tetap didiagnosis: test bersih
  0,759 vs 0,767; test sesi baru 0,709 vs 0,711. Catatan: di VAL (fase 2) foto cadangan dikirim ke model v9 sebagai foto asli, sehingga
  kriteria fase 2 tidak memperhitungkan aturan foto ulang; di test, foto ulang 3,7% (test bersih) lebih banyak daripada di VAL bersih (1%).
- BLB (terpisah; 150 foto, 18 kelompok): foto ulang 29,3%; akurasi 0,398 vs SAM2 0,636 vs v7 0,687 (pada yang didiagnosis 0,563).

## Fase 5 butir 4 - Model aplikasi v9 vs model aplikasi v7 (PEMBANDING, bukan hasil utama)

(`fase5_model_aplikasi_v9_vs_v7.json`; v9 = alur aplikasi, ambang 0,36; v7 = foto asli, ambang 0,37)

| Set | Ukuran | Model aplikasi v7 | Model aplikasi v9 |
|---|---|---|---|
| Test sesi baru (292 irisan) | mAP50-95 | 0,279 | 0,273 |
| | akurasi diagnosis (foto ulang dihitung salah) | 0,801 | 0,764 (8 foto ulang) |
| | akurasi pada foto yang didiagnosis | 0,801 | 0,785 |
| | recall Blast | 0,872 | 0,883 |
| ds3 uji utama (70; sudah terbuka) | akurasi (foto ulang dihitung salah) | 0,471 | 0,529 (12 foto ulang = 17%) |
| | akurasi pada foto yang didiagnosis | 0,471 | 0,638 |
| | recall Blast / recall BLB | 0,000 / 0,943 | 0,343 / 0,714 |
| Tungro ds3 (80, di luar cakupan) | disebut BLB | 75/80 (93,8%) | 55/80 (68,8%); 85,9% dari 64 yang didiagnosis; 16 foto ulang |

## Fase 5 butir 5 - RAM dan waktu (prototipe di `_prototipe_app_v9/`, venv terpisah; Streamlit + pemisah daun + detektor ONNX, CPU)

| Ukuran | Aplikasi v7 sekarang | Prototipe v9 |
|---|---|---|
| RAM diam (model dimuat) | 150 MB | 152 MB |
| RAM puncak (8 foto VAL + 3 foto 4000x3000) | 241 MB | **273 MB** (target < ±600 MB: terpenuhi) |
| RAM diam setelah 11 foto | 164 MB | 192 MB |
| Waktu inti per foto VAL (median) | 119 ms | 177 ms |
| Waktu ujung ke ujung di browser (median) | 223 ms | 287 ms |
| Foto 4000x3000 ujung ke ujung | 391 ms | 512 ms |

- Temuan teknis: dua sesi ONNX Runtime yang dijalankan bergantian saling memperlambat bila "spinning" thread aktif (pengukuran pertama: inti
  400 ms). Setelah `session.intra_op.allow_spinning = 0` pada kedua sesi (prototipe saja), inti turun ke 177 ms. Pengaturan ini perlu ikut
  bila aplikasi diubah. Aplikasi v9 juga memerlukan paket tambahan `opencv-python-headless` (pasca-proses masker).
- Prototipe hanya untuk pengukuran: foto cadangan memakai pesan galat bawaan ("Foto belum bisa diperiksa ... Foto ulang ..."); tampilan
  khusus belum dibuat.

## Berkas

`fase4_ambang_aplikasi.json`, `fase4_ambang_per_citra.csv`, `fase4_onnx_model_aplikasi.json`, `fase4_onnx_vs_pt_20_val.csv`,
`fase5_cadangan.json`, `fase5_alur_aplikasi_final.csv`, `fase5_selisih_alur.csv`, `fase5_model_aplikasi_v9_vs_v7.json`,
`fase5_sumber_daya_prototipe.{json,md}`; run `runs/v9_deploy_yolov8s_s2/` dan `runs/*/eval_alurapp_*`; dataset `data/derived/prov_v9_deploy`,
`data/derived/v9_alur_aplikasi`; skrip `scripts/bangun_deploy_v9.py`, `scripts/v9_latar/{kunci_ambang_aplikasi_v9, fase5_alur_aplikasi,
analisis_alur_aplikasi_v9}.py`; prototipe `_prototipe_app_v9/`.
