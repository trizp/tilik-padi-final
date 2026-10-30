# Memasang ke Streamlit Community Cloud (BELUM dilakukan)

Streamlit Community Cloud gratis, mengambil kode dari repositori GitHub, dan menyediakan memori sekitar 690 MB–2,7 GB. Aplikasi ini
diukur memakai RAM puncak 273 MB (`docs/uji/sumber_daya.md`, alur v9: pemisah daun YOLOv8n-seg + YOLOv8s). Ikuti langkah di bawah secara berurutan.

## 0. Periksa sebelum memasang

- [ ] Lisensi ds2 (Kaggle soni535) tercatat **"unknown"** di metadata Kaggle, dan sudah ditulis "lisensi tidak dinyatakan oleh
      penyusun" beserta tautannya. Foto ds2 hanya dipakai untuk melatih model (tidak ditampilkan di aplikasi). Putuskan bersama
      dospem apakah hal ini cukup untuk publikasi.
- [ ] `rbs/rules.yaml` R001–R003 sudah sama dengan Tabel 3.1 proposal. BAB 3 skripsi masih memuat Tabel 3.2 versi 2022 (lihat README).
- [ ] Lisensi: repositori memakai **AGPL-3.0** (`LICENSE`), mengikuti model YOLOv8 dari Ultralytics. Repositori publik dengan kode
      lengkap memenuhi kewajiban membuka kode.
- [ ] Sembilan tangkapan layar di `docs/tangkapan_layar/` hanya memuat foto dari RiceLeafDiseaseBD (CC BY 4.0), dan atribusinya ada di README.
      Aplikasi sendiri hanya memakai ilustrasi SVG buatan sendiri.

## 1. Siapkan folder siap-unggah

Langkah ini dijalankan di folder proyek skripsi (`rice_yolo`). Skrip `siapkan_unggah.py` hanya ada di sana dan tidak ikut
repositori.

```powershell
app\.venv\Scripts\python.exe app\tools\siapkan_unggah.py --timpa
```

Skrip ini menyusun folder **`tilik-padi`** di samping folder proyek (satu tingkat di atas `rice_yolo`). Isinya:
- kode aplikasi, model ONNX, `rbs/rules.yaml`, `config.yaml`, `requirements.txt`, `.streamlit/`, README, `LICENSE` AGPL-3.0, `.gitignore`;
- dokumentasi inti dan test.

Yang tidak ikut: `.venv`, dataset, `runs/`, foto uji, dan berkas kerja (catatan kritik, tangkapan layar lama, alat pengembangan).

Buka laporan `rice_yolo/app/docs/siap_unggah.md` (tidak ikut repositori) dan pastikan **Hasil: LOLOS**. Laporan itu memuat:
- ukuran total dan berkas terbesar (harus < 100 MB);
- hasil pemindaian jalur pribadi dan kunci API (harus 0 temuan);
- hasil uji `.gitignore` dengan berkas umpan (tidak ada yang ikut terlacak).

## 2. Siapkan akun dan alat

1. Akun GitHub (https://github.com).
2. Git for Windows (https://git-scm.com/download/win). Periksa dengan `git --version` di PowerShell.
3. Agar alamat surel pribadi tidak tercantum di riwayat commit publik:
   - di GitHub buka **Settings → Emails**, centang "Keep my email addresses private";
   - salin alamat `…@users.noreply.github.com` yang tertera.

## 3. Repositori publik di GitHub

Repositori **`tilik-padi`** sudah dibuat (alamat `https://github.com/USERNAME/tilik-padi`; ganti `USERNAME` dengan nama akun
GitHub Anda). Pastikan repositori itu **Public**.

Bila repositori dibuat ulang: GitHub → **New**, beri nama `tilik-padi`, pilih **Public**, dan **jangan** centang "Add a README",
".gitignore", atau "license", karena semuanya sudah ada di folder unggah.

## 4. Unggah isi folder

Unggah lewat git (unggah lewat browser tidak bisa: model detektor 42,7 MB melebihi batas 25 MB per berkas di web GitHub; pemisah daun 12,7 MB).

```powershell
cd <folder tilik-padi>        # folder hasil langkah 1, di samping folder proyek
git init -b main
git config user.name "Nama Anda"
git config user.email "<id>+<akun>@users.noreply.github.com"
git add .
git status
```

Periksa hasil `git status`: harus berisi berkas aplikasi, tanpa `.venv`, `data`, `runs`, atau foto `.jpg`. Lalu:

```powershell
git commit -m "Aplikasi Tilik Padi (Streamlit)"
git remote add origin https://github.com/USERNAME/tilik-padi.git
git push -u origin main
```

**Catatan untuk repositori yang sudah dibuat:** bila repositori `tilik-padi` di GitHub sudah berisi README, `.gitignore`, atau
`LICENSE` bawaan (misalnya karena kotak centang itu dipilih saat membuat repositori), `git push -u origin main` biasa akan ditolak
karena riwayatnya berbeda. Untuk push pertama saja, pakai:

```powershell
git push -u origin main --force
```

Perintah ini mengganti seluruh isi repositori dengan isi folder unggah. Setelah itu, periksa di GitHub:
- `LICENSE` harus berisi **GNU AFFERO GENERAL PUBLIC LICENSE Version 3** (bukan lisensi bawaan GitHub seperti MIT);
- README harus berjudul "Tilik Padi".

Jangan memakai `--force` lagi untuk push berikutnya.

Saat `push` pertama, jendela masuk GitHub muncul (Git Credential Manager). Setelah selesai, muat ulang halaman repositori;
README tampil di bawah daftar berkas.

## 5. Hubungkan ke Streamlit Community Cloud

1. Buka https://share.streamlit.io dan pilih **Continue with GitHub**. Izinkan Streamlit membaca repositori Anda.
2. **Create app**, lalu pilih **Deploy a public app from GitHub**.
3. Isi isian berikut:
   - **Repository**: `USERNAME/tilik-padi`;
   - **Branch**: `main`;
   - **Main file path**: `streamlit_app.py`;
   - **App URL**: `tilik-padi` (alamat menjadi `https://tilik-padi.streamlit.app`). Bila nama itu sudah dipakai orang lain, pilih
     nama lain dan sesuaikan alamat di langkah 6 dan 7.
4. **Advanced settings**: Python version **3.12** (versi yang diuji). Secrets dikosongkan, karena aplikasi tidak memakai kunci apa pun.
5. **Deploy**. Pemasangan pertama memasang paket dari `requirements.txt` dan memakan beberapa menit. Log tampil di panel kanan bawah.
6. Bila gagal, baca log. Penyebab umum:
   - versi Python bukan 3.12;
   - berkas model belum ikut terunggah (periksa `models/` di GitHub).

## 6. Uji di komputer (setelah aktif)

1. Buka `https://tilik-padi.streamlit.app` dan periksa halaman awal:
   - tombol "Ambil foto daun" dan "Unggah foto";
   - petunjuk memotret bergambar;
   - tiga penyakit.
2. Unggah foto daun padi (JPG atau PNG) dan periksa:
   - foto bertanda kotak dan potongan diperbesar;
   - pita keyakinan;
   - rekomendasi "Tindakan sekarang" dan "Pencegahan musim berikutnya";
   - panel "Mengapa sistem menyimpulkan ini".
3. Unggah berkas yang bukan foto. Pemilih berkas hanya menawarkan JPG/PNG; bila berkas lain diseret ke tombol, Streamlit menolak
   dengan pesan bawaan berbahasa Inggris.
4. Tekan "Periksa foto lain". Halaman kembali ke awal.

## 7. Uji kamera di HP

Kamera browser hanya aktif lewat HTTPS. Alamat `tilik-padi.streamlit.app` sudah HTTPS.

1. Buka `https://tilik-padi.streamlit.app` di **Chrome (Android)** atau **Safari (iPhone)**.
2. Tekan **Ambil foto daun**. Saat browser meminta izin kamera, pilih **Izinkan**.
   - Bila izin pernah ditolak:
     - Chrome Android: ketuk ikon di kiri alamat → **Izin** → **Kamera** → Izinkan;
     - Safari iPhone: ketuk **aA** di bilah alamat → **Pengaturan Situs Web** → **Kamera** → Izinkan.
3. Periksa:
   - tombol di bawah gambar kamera berbunyi **"Ambil foto"** (bukan "Take Photo");
   - setelah memotret, tombolnya berbunyi **"Hapus foto dan ambil ulang"**.
4. Arahkan ke satu helai daun dari jarak 15–30 cm, lalu tekan **Ambil foto**. Hasil muncul dan halaman langsung bergulir ke hasil.
5. Catat kamera mana yang terbuka. Di sebagian HP, `st.camera_input` membuka kamera depan. Bila begitu:
   - tekan **Tutup kamera**;
   - pakai **Unggah foto**, lalu pilih **Kamera** di pemilih berkas HP.
   Cara ini memakai aplikasi kamera bawaan (kamera belakang, resolusi penuh).
6. Coba juga di bawah sinar matahari. Periksa:
   - teks dan label tingkat terbaca;
   - tombol mudah disentuh.
7. Tekan **Periksa foto lain**, lalu tekan **Ambil foto daun** sekali lagi untuk memastikan kamera bisa dipakai ulang.

Catat hasilnya, misalnya:

| HP / browser | Izin kamera | Kamera yang terbuka | Label "Ambil foto" | Hasil tampil | Catatan |
|---|---|---|---|---|---|
| … / Chrome | ya / tidak | depan / belakang | ya / tidak | ya / tidak | |

## 8. Memperbarui dan mengelola

- **Mengubah teks atau aturan:** ubah `config.yaml` atau `rbs/rules.yaml` di folder `tilik-padi` (atau jalankan ulang
  `siapkan_unggah.py --timpa` setelah mengubah `app/`). Jalankan test, lalu `git add .`, `git commit -m "..."`, dan `git push`.
  Cloud memasang ulang otomatis.
- **Memulai ulang aplikasi:** di https://share.streamlit.io, menu ⋮ pada aplikasi → **Reboot**. Untuk menghapusnya, pilih **Delete**.
- **Aplikasi tidur:** aplikasi yang tidak dikunjungi beberapa hari akan "tidur". Kunjungan berikutnya membangunkannya dalam waktu
  sekitar satu menit.
- **Privasi:** foto tidak disimpan oleh aplikasi, dan log Streamlit Cloud tidak memuat isi foto.
