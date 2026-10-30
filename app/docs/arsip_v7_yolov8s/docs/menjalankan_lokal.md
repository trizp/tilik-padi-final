# Menjalankan aplikasi di komputer sendiri

Butuh Python 3.12 (Windows, macOS, atau Linux). Tidak butuh GPU, torch, atau Ultralytics.

## Pertama kali

Windows (PowerShell), dari folder `app/`:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

macOS/Linux:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Menjalankan

Selalu dari folder `app/`, supaya `.streamlit/config.toml` (tema, batas unggah 10 MB, menu tersembunyi) terbaca:

```powershell
cd <folder aplikasi>        # folder yang berisi streamlit_app.py
.venv\Scripts\streamlit run streamlit_app.py
```

Buka **http://localhost:8501** di browser. Untuk mencoba dari HP di jaringan Wi-Fi yang sama, buka alamat "Network URL" yang tercetak
di terminal (misalnya `http://192.168.1.12:8501`). Kamera browser HP biasanya hanya aktif lewat HTTPS atau localhost, jadi di jaringan
lokal tombol "Ambil foto daun" mungkin ditolak browser. Tombol "Unggah foto" tetap bisa memakai kamera HP lewat pemilih berkas.

## Pengujian (opsional, butuh paket pengembang)

```powershell
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m pytest -q tests          # 67 test; test yang butuh dataset proyek dilewati bila dataset tidak ada
```

Hasil uji yang sudah dijalankan di folder proyek skripsi tercatat di `docs/uji_kesetaraan.md`, `docs/uji/blackbox.md`, dan
`docs/uji/sumber_daya.md`. Alat uji kesetaraan, pengukuran RAM/waktu, dan tangkapan layar otomatis memerlukan dataset proyek dan
Chrome, sehingga hanya ada di folder proyek dan tidak ikut repositori ini.

## Tangkapan layar

Tangkapan layar di `docs/tangkapan_layar/` dibuat otomatis dengan Chrome headless di folder proyek skripsi. Untuk mengambil sendiri:
1. Buka aplikasi di Chrome dan tekan F12.
2. Pilih "Toggle device toolbar" (Ctrl+Shift+M) dan set lebar 390, lalu ulangi dengan lebar 1280.
3. Unggah foto, lalu pilih menu titik tiga, "Capture full size screenshot".
