# Model aplikasi web (PEMBANDING, bukan hasil utama paper)

Tanggal: 13 Oktober 2026. Disetujui dospem 13 Okt. Tabel utama paper tetap dari run final v7 (split bersih).

## Model

- `runs/v7_deploy_yolo11s_s2/weights/last.pt` (sha256 `8d380436…d22f`): YOLO11s seed 2, dilatih dengan train+val+test prov_v7
  (8.342 gambar; background dipangkas ke 10%; tanpa test_new_session), 63 epoch, hyperparameter lain identik dengan run final,
  tanpa validasi per epoch, checkpoint = last.pt. 2,49 jam GPU, tanpa resume.
- Ambang diagnosis 0,35 (DITETAPKAN, tidak dipilih ulang). Tiga tingkat jawaban: terdeteksi >= 0,35; kemungkinan 0,29–0,35;
  tidak terdeteksi < 0,29 (dipilih di val dengan model final YOLO11s s2; `tiga_tingkat_val.json`).
- ONNX: `runs/v7_deploy_yolo11s_s2/weights/last.onnx` (sha256 `ed986219…35ff`, 36,2 MB; .pt 18,3 MB).

## Pembanding (dievaluasi sekali, ambang 0,35)

| Model | Sesi baru mAP50-95 | Sesi baru akurasi diagnosis | ds3 uji utama akurasi | ds3 recall BLB | ds3 recall Blast |
|---|---|---|---|---|---|
| final YOLO11s s2 (split bersih, best.pt, 100 epoch) | 0,247 | 0,726 | 0,000 | 0,000 | 0,000 |
| model web (semua data, last.pt, 63 epoch) | 0,289 | 0,829 | 0,586 | 1,000 | 0,171 |

Cara membaca: satu model, satu run; tidak ada IK antar-seed. Di ds3, model final YOLO11s s2 mendiagnosis semua 35 foto BLB sebagai
"tidak terdeteksi" (kotak teratas BLB di bawah 0,35) dan semua foto blast sebagai brown spot/tidak terdeteksi. Model web mengenali
semua foto BLB ds3, tetapi tungro ds3 tidak diuji untuk model ini, sehingga belum diketahui apakah BLB "dikenali" atau model hanya
lebih mudah menyebut BLB pada daun petik yang menguning (pola ini terlihat pada model lain). Blast di ds3 tetap lemah (6 dari 35).

## ONNX vs PyTorch (50 gambar val prov_v7, CPU, batch 1)

- Kelas kotak teratas identik 50/50; selisih confidence maksimum < 0,00001 -> LOLOS (syarat < 0,01).
- Kecepatan CPU (median, ujung ke ujung termasuk baca berkas): .pt 88,9 ms; ONNX 61,2 ms (ONNX Runtime 1.31.0, CPUExecutionProvider).

Berkas: `model_web_vs_final.csv`, `onnx_bukti.json`, `onnx_vs_pt_50_val.csv`, `tiga_tingkat_val.json`.
