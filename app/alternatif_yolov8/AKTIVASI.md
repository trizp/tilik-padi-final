> **SUDAH DILAKUKAN 2026-10-16** (keputusan dospem, pilihan B). Berkas ini disimpan sebagai riwayat; lihat KEPUTUSAN_PROYEK.md.

# Mengaktifkan model YOLOv8s di aplikasi (BELUM dilakukan)

Disiapkan 2026-10-15 karena judul dan rumusan masalah proposal menyebut YOLOv8 (KEPUTUSAN_PROYEK.md). Alasannya kesesuaian judul,
BUKAN kinerja: di test, YOLO11s sedikit lebih baik. Aktifkan hanya bila dospem memilih YOLOv8.

## Mencoba tanpa mengaktifkan

```powershell
cd <folder app>
$env:TILIK_CONFIG = "config_yolov8.yaml"
.venv\Scripts\streamlit run streamlit_app.py
```

## Mengaktifkan

1. Arsipkan konfigurasi YOLO11s: salin `config.yaml` menjadi `config_yolo11s_arsip.yaml`.
2. Salin `config_yolov8.yaml` menjadi `config.yaml`.
3. Salin `alternatif_yolov8/README.md` ke `README.md` dan `alternatif_yolov8/docs/*.md` ke `docs/`.
4. Salin hasil uji YOLOv8s menjadi nama aktif: `docs/uji_kesetaraan_yolov8s.md` dan `docs/uji/sumber_daya_yolov8s.md` (atau sesuaikan tautan).
5. Ambil ulang tangkapan layar (`tools/tangkap_layar.py`). Tingkat "kemungkinan" tidak ada, dan contoh "tidak terdeteksi" perlu foto
   lain, karena foto `Healthy_1002` didiagnosis Brown spot 49% oleh YOLOv8s.
6. Ubah test `test_config_aktif_tetap_yolo11s_dan_tiga_tingkat` di `tests/test_inti.py` agar sesuai model aktif. Jalankan `pytest`.
7. Ganti `models/v7_deploy_yolo11s_s2.onnx` dengan `models/v7_deploy_yolov8s_s2.onnx` di daftar izin `tools/siapkan_unggah.py`, lalu
   bangun ulang folder unggah. Model YOLO11s tetap disimpan di `app/models` sebagai arsip.

Ambang: terdeteksi = 0,37, rendah = 0,37 (`reports/final_eval_v7/aplikasi/tiga_tingkat_val_yolov8s.json`).
