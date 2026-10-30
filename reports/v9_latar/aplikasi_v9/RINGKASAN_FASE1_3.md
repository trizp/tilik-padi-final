# Rencana aplikasi v9: hasil fase 1-3 (VAL saja)

Ditulis 2026-10-20. Rencana dan kriteria dicatat SEBELUM dijalankan (KEPUTUSAN_PROYEK.md 2026-10-20). Test, test sesi baru, dan ds3 tidak
disentuh. Aplikasi v7 (`app/`) dan `tilik-padi` tidak diubah. Tidak ada paket yang dipasang; venv proyek sama dengan sebelumnya.

## Fase 1 - Pemisah daun ringan (YOLOv8n-seg, distilasi SAM2)

- Data: 5.101 foto train dan 980 foto val prov_v9 yang benar-benar dipisah SAM2 (foto asli berlatar, label = masker SAM2 aturan v2).
  Foto cadangan tidak dipakai. Training 100 epoch (±2 jam GPU); terbaik epoch 70: mAP50-95 masker 0,909, mAP50 0,962 (vs masker SAM2).
- Aturan pemilih daun = tiruan aturan v2; ambang keyakinan dipilih di VAL: **tau = 0,35** (rata-rata IoU alur tertinggi 0,900).
- ONNX: **12,7 MB**; CPU (ONNX Runtime + pasca-proses numpy + aturan): **69 ms/foto** (semua inti), **99 ms/foto** (2 thread); RAM proses
  naik ±41 MB setelah model dimuat (puncak ±90 MB termasuk numpy/OpenCV). ONNX vs .pt pada 20 foto VAL: IoU 0,995, status cadangan sama 20/20.
  Pembanding: SAM2 1,4-2,9 detik/foto di GPU.

Ukuran terhadap SAM2 pada VAL prov_v9 (`fase1_ringkasan.json`):

| Kelompok | Foto | IoU alur rata-rata / median | IoU bila keduanya dipisah (rata-rata / median) | Kotak >= 50% di daun: seg / SAM2 | Cadangan: SAM2 / seg |
|---|---|---|---|---|---|
| Semua | 1.138 | 0,900 / 0,990 | 0,955 / 0,991 | 98,0% / 97,7% | 158 / 72 |
| ds4 (Blast, Brown spot, sehat) | 837 | 0,956 / 0,991 | 0,964 / 0,991 | 98,9% / 97,7% | 12 / 3 |
| ds1 BLB | 277 | 0,734 / 0,947 | 0,902 / 0,984 | 91,5% / 97,6% | 142 / 63 |
| ds2 sehat | 24 | 0,872 / 0,983 | 0,919 / 0,980 | - | 4 / 6 |

- Status cadangan cocok pada 1.020 dari 1.138 foto. Ketidakcocokan terbanyak: SAM2 cadangan tetapi seg memisahkan (102 foto, 93 di antaranya BLB).
- Kecocokan dengan penilaian manual (50 foto): dari 41 foto "baik", seg mirip SAM2 (IoU >= 0,8) pada 39; ketiga foto "salah objek" (SAM2
  salah) diberi masker yang berbeda oleh seg; dari 6 foto "cadangan", seg juga cadangan pada 3.
- Montase 20 contoh: `fase1_montase_20.jpg`. Pada satu foto ds4, SAM2 memilih latar sedangkan seg memilih daun yang benar (IoU 0,00).

## Fase 2 - Kelayakan alur (VAL bersih, 861 foto; model final YOLOv8s v9 seed 0-2, ambang terkunci)

| Alur | mAP50-95 | Akurasi diagnosis | Recall diagnosis Blast |
|---|---|---|---|
| SAM2 (hasil yang ada; dihitung ulang, cocok persis) | 0,2267 | 0,8657 | 0,8474 |
| YOLOv8n-seg (baru) | 0,2228 | 0,8618 | 0,8380 |
| Penurunan | 0,0039 | **0,0039 (0,4 poin; batas 2)** | **0,0094 (0,9 poin; batas 3)** |

Per seed (akurasi SAM2 / seg): 0,860 / 0,858; 0,873 / 0,863; 0,864 / 0,864. Recall Blast: 0,911 / 0,907; 0,832 / 0,808; 0,799 / 0,799.
BLB (terpisah) mAP50-95: SAM2 0,154, seg 0,156. **Kesimpulan: LAYAK** menurut kriteria yang dicatat sebelum menjalankan.

## Fase 3 - Foto cadangan (VAL bersih)

- Alur seg menghasilkan **9 foto cadangan dari 861 (1,0%)**: 8 daun sehat, 1 Blast. (Pada BLB ds1, yang tidak termasuk subset bersih,
  cadangan seg 63 dari 277 = 23%.)

| Aturan untuk foto cadangan | Akurasi pada 9 foto cadangan | Akurasi seluruh VAL bersih | Cakupan |
|---|---|---|---|
| (a) dialihkan ke model v7 YOLOv8s (foto asli) | 1,000 | 0,862 | 100% |
| (b) tidak didiagnosis, pengguna memotret ulang | - | 0,852 (cadangan dihitung salah); 0,861 pada foto yang didiagnosis | 98,9% |
| (pembanding) model v9 pada foto asli, perilaku fase 2 | 0,963 | 0,862 | 100% |

RAM proses ONNX Runtime CPU (tanpa Streamlit): 1 detektor (aplikasi v7) 102 MB; pemisah daun + 1 detektor 121 MB; pemisah daun + 2 detektor
169 MB (+48 MB). Ukuran berkas: detektor YOLOv8s 42,7 MB; pemisah daun 12,7 MB.

**Rekomendasi: aturan (b), foto ulang.** Aturan (a) hanya menyelamatkan sekitar 1 poin akurasi pada 1% foto, tetapi menambah satu detektor
(+43 MB berkas, +48 MB RAM, dua alur model yang harus dipelihara dan dijelaskan di skripsi). Cadangan berarti sistem tidak menemukan satu
helai daun yang jelas; meminta foto ulang sesuai petunjuk memotret lebih jujur dan sejalan dengan alur tanpa latar. Catatan: pada foto
BLB dekat/dipegang tangan (ds1), cadangan jauh lebih sering (23%), sehingga pengguna seperti itu akan lebih sering diminta foto ulang.

## Berkas

- Fase 1: `fase1_pilih_tau.json`, `fase1_ringkasan.json`, `fase1_per_foto.csv`, `fase1_onnx.json`, `fase1_montase_20.jpg`;
  run `runs/v9_pemisah_daun_yolov8n_seg/` (best.pt, best.onnx); dataset `data/derived/v9_daun_seg/`.
- Fase 2: `fase2_ringkasan.json`; Fase 3: `fase3_ringkasan.json`.
- Skrip: `scripts/v9_latar/{bangun_data_daun_seg, latih_pemisah_daun, pemisah_daun_seg, fase1_pemisah_daun, fase23_alur}.py`.
