# Tangkapan layar dan kritik desain

Diambil otomatis dengan `tools/tangkap_layar.py`:
- Chrome headless + DevTools Protocol;
- foto diunggah sungguhan lewat input berkas;
- kamera memakai perangkat palsu Chrome, yang menampilkan pola uji hijau;
- lebar 390 px dan 1280 px.

Model aktif: **YOLOv8s** (sejak 2026-10-16). Foto ds4 VAL yang dipakai:
- terdeteksi: `ds4_blast-104` (Blast, 68%; "Cara membedakan" terlipat);
- kemungkinan lain: `ds4_blast-321` (Blast 73%, kemungkinan lain Brown spot 62%; "Cara membedakan" terbuka);
- tidak terdeteksi: `ds4_Healthy-1014` (daun sehat, 0%);
- bercak sangat kecil: `ds4_brown-spot-1148` (Brown spot 76%).

Foto kemungkinan lain dan tidak terdeteksi dipilih dari `docs/uji/pindai_val.csv` (pemindaian YOLOv8s)
karena memicu keadaannya (bukan acak). `kemungkinan_*.png`, `bercak_rendah_*.png` (fitur dinonaktifkan 16 Okt), dan `*_yolov8s.png` di folder ini adalah sisa versi sebelumnya (tidak dipakai
README); tangkapan layar YOLO11s lengkap ada di `arsip_yolo11s/tangkapan_layar/`.

Folder:
- `tangkapan_layar/` = versi akhir (YOLOv8s, 16 Okt);
- `arsip_yolo11s/tangkapan_layar/` = versi YOLO11s 15 Okt (dengan penyampaian tambahan);
- `sebelum_penyampaian/` = versi 14 Okt sebelum penyampaian tambahan;
- `sebelum_revisi/` = versi sebelum revisi;
- `putaran_awal/` = versi paling awal;
- `banding/` = sebelum dan sesudah revisi berdampingan.

## Versi akhir

| Keadaan | 390 px | 1280 px |
|---|---|---|
| Halaman awal | ![](tangkapan_layar/awal_390.png) | ![](tangkapan_layar/awal_1280.png) |
| Kamera terbuka | ![](tangkapan_layar/kamera_390.png) | ![](tangkapan_layar/kamera_1280.png) |
| Terdeteksi | ![](tangkapan_layar/terdeteksi_390.png) | ![](tangkapan_layar/terdeteksi_1280.png) |
| Tidak terdeteksi | ![](tangkapan_layar/tidak_terdeteksi_390.png) | ![](tangkapan_layar/tidak_terdeteksi_1280.png) |
| Bercak sangat kecil (potongan diperbesar) | ![](tangkapan_layar/bercak_kecil_390.png) | ![](tangkapan_layar/bercak_kecil_1280.png) |
| Kemungkinan lain + cara membedakan | ![](tangkapan_layar/kemungkinan_lain_390.png) | ![](tangkapan_layar/kemungkinan_lain_1280.png) |
| Tentang | ![](tangkapan_layar/tentang_390.png) | ![](tangkapan_layar/tentang_1280.png) |

## Sebelum dan sesudah revisi

| Keadaan | 390 px | 1280 px (diperkecil 50%) |
|---|---|---|
| Halaman awal | ![](tangkapan_layar/banding/awal_390.png) | ![](tangkapan_layar/banding/awal_1280.png) |
| Terdeteksi | ![](tangkapan_layar/banding/terdeteksi_390.png) | ![](tangkapan_layar/banding/terdeteksi_1280.png) |
| Kemungkinan | ![](tangkapan_layar/banding/kemungkinan_390.png) | ![](tangkapan_layar/banding/kemungkinan_1280.png) |
| Tidak terdeteksi | ![](tangkapan_layar/banding/tidak_terdeteksi_390.png) | ![](tangkapan_layar/banding/tidak_terdeteksi_1280.png) |

## Kritik putaran revisi dan perbaikannya

| Temuan (putaran revisi 1–2) | Terhadap | Perbaikan |
|---|---|---|
| Setelah foto diunggah muncul tiga "Unggah foto": aturan CSS juga mengubah tombol hapus berkas dan tombol "Add files" | poin 4 | gaya tombol hanya untuk tombol unggah bawaan; chip nama berkas disembunyikan (foto sudah tampil di hasil); setelah ada foto tombolnya berbunyi "Unggah foto lain" |
| Catatan "JPG atau PNG, maksimal 10 MB" berada di bawah tombol kamera | poin 4 | dipindah ke bawah tombol unggah |
| Ikon tanda kata terbaca seperti huruf "Y" | poin 5 | ikon padi satu garis digambar ulang: batang, malai merunduk dengan bulir, dan satu helai daun |
| Di HP ada celah kosong besar antara petunjuk dan "Tiga penyakit" | tata letak | jarak antar-kolom yang bertumpuk dihapus |
| Dalam mode kamera, "Tutup kamera" menempel pada "Unggah foto" | target sentuh | jarak 12 px antar-tombol yang bertumpuk dikembalikan |
| Batas menguning pada ilustrasi hawar terlalu halus | poin 6 (bab 2.2: batas bergelombang) | batas daerah kuning dibuat bergelombang |
| Ilustrasi penyakit tampil dua kali berturut-turut di HP (samping potongan dan bawah nama) | pengulangan | di bawah 1024 px, ilustrasi di bawah nama disembunyikan bila potongan diperbesar ada. Di desktop keduanya tetap ada (poin 10 dan 13), di kolom yang berbeda |

## Pemeriksaan versi akhir terhadap poin 4–13

| Poin | Versi akhir |
|---|---|
| 4 Satu aksi unggah | satu tombol "Unggah foto" (uploader Streamlit bergaya garis tepi Daun padi). Teks bahasa Inggris ("Upload", "Browse files", "Drag and drop", "10MB per file") tersembunyi, diganti "JPG atau PNG, maksimal 10 MB". Tombol kamera membuka dan menutup `st.camera_input`. Di HP kedua tombol bertumpuk penuh lebar |
| 5 Kepala | ikon padi SVG + nama aplikasi (sekarang "Tilik Padi") + "Tentang"; judul tunggal "Periksa daun padi dari foto" dan subjudul |
| 6 Tiga penyakit | tiga potongan helai daun memanjang (bukan kartu), masing-masing dengan nama, patogen, dan satu kalimat ciri. Dipakai ulang di hasil |
| 7 Petunjuk memotret | dua ilustrasi berdampingan, "Seperti ini" dan "Bukan seperti ini" (ditandai teks, centang, dan silang, bukan warna saja), plus satu kalimat jarak dan cahaya dari config |
| 8 Desktop awal | dua kolom: kiri judul, aksi, petunjuk; kanan tiga penyakit |
| 9 Hasil | hasil langsung di bawah aksi; judul, petunjuk, dan tiga penyakit dilipat ke "Cara memotret"; gulir otomatis instan ke hasil; tombol "Periksa foto lain" |
| 10 Potongan diperbesar | potongan persegi di sekitar kotak teratas (±3× untuk bercak kecil), berlabel "Bercak yang paling diyakini sistem", berdampingan dengan ilustrasi penyakit |
| 11 Pita | hanya label zona dan angka keyakinan; batas 29% dan 35% ada di panel "Mengapa sistem menyimpulkan ini" |
| 12 Rekomendasi | "Tindakan sekarang" lebih dulu, lalu "Pencegahan musim berikutnya"; kata-kata persis Tabel 3.1 (diuji). Di bawahnya catatan tetap dan sumber aturan |
| 13 Desktop hasil | dua kolom: kiri foto dan potongan; kanan tingkat, nama dan ilustrasi, pita, rekomendasi, jejak, peringatan |

## Pemeriksaan terhadap daftar HINDARI

| HINDARI | Versi akhir |
|---|---|
| Gradien dekoratif | tidak ada (ilustrasi memakai isian rata) |
| Ungu/biru generik | tidak ada; Langit hanya untuk "tidak terdeteksi" dan garis peringatan |
| Emoji | tidak ada; ikon dan tanda centang/silang digambar sebagai SVG |
| Kata promosi | tidak ada |
| Kartu bulat seragam berbayang | tidak ada; tiga penyakit berupa baris berpemisah garis tipis, tanpa bayangan |
| Label kapital di atas judul | tidak ada |
| Panah "→" di tombol | tidak ada |
| Pemisah titik tengah | tidak ada |
| Monospace untuk label | tidak ada (angka pada pola uji kamera palsu berasal dari Chrome, bukan aplikasi) |
| Latar krem + terakota | tidak ada; latar Air sawah. Kertas abu-krem hanya ada di dalam ilustrasi "Bukan seperti ini" |
| Latar gelap + neon | tidak ada |
| Animasi muncul | tidak ada; gulir ke hasil berlangsung instan (`scroll-behavior: auto`) |

## Perbaikan sebelum pemasangan online

- **Filter jenis berkas dikembalikan** (jpg, jpeg, png), sehingga di HP pemilih berkas langsung membuka galeri. Validasi di backend tetap
  memberi pesan Indonesia.
- **Label tombol kamera tidak lagi bergantung pada CSS `:has()`.** Teksnya diganti skrip kecil (`ui.skrip_label`) yang hanya mengubah
  nilai teks dan `aria-label`. Bila skrip gagal, teks bawaan tetap terlihat.
  - Diuji dengan `tools/uji_kamera_tanpa_has.py`: semua aturan `:has()` di halaman dihapus.
  - Hasilnya: "Ambil foto", lalu "Hapus foto dan ambil ulang". Tanpa skrip: "Take Photo"/"Clear photo", tidak pernah kosong.
  - Rincian: `docs/uji/kamera_tanpa_has.md` dan `tangkapan_layar/kamera_tanpa_has_390.png`.
  - CSS aplikasi kini tidak memakai `:has()` sama sekali (diuji).
- **Desktop:** ilustrasi penyakit hanya tampil sekali, di kolom kanan di samping nama penyakit. Kolom kiri hanya foto dan potongan
  diperbesar. Di HP, potongan tetap berdampingan dengan ilustrasi, dan ilustrasi di bawah nama disembunyikan (penanda kelas dari
  server, bukan `:has()`).

## Sisa kekurangan yang disadari

- **Pesan galat bawaan Streamlit berbahasa Inggris:** bila berkas selain JPG/PNG diseret ke tombol unggah, pesannya bawaan Streamlit.
  Ini sudah diterima.
- **Pesan izin kamera** dari Streamlit tetap berbahasa Inggris bila browser belum memberi izin.
- **Tombol hapus foto** di dalam uploader tersembunyi. Nama aksesibelnya ("Remove …") tidak diubah.
