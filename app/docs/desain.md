# Desain aplikasi "Tilik Padi"

Pengguna: petani dan penyuluh di Indonesia, memakai HP di sawah di bawah cahaya terang. Sebagian tidak terbiasa dengan teknologi.
Tugas utama: **foto daun → tahu penyakitnya → tahu tindakannya.** Semua keputusan desain di bawah melayani urutan itu.

## 1. Palet (wajib, dipakai persis)

| Nama | Hex | Peran | Kontras (WCAG) |
|---|---|---|---|
| Daun padi | `#2F6B3B` | warna utama: tombol, tautan, judul bagian | putih di atasnya 6,39; di atas Air sawah 5,86 |
| Air sawah | `#F3F6F2` | latar halaman | — |
| Tinta daun | `#1F2D24` | semua teks | di atas Air sawah 13,21 |
| Karat bercak | `#A4442A` | tingkat **terdeteksi** | putih di atasnya 6,11; sebagai teks di Air sawah 5,61 |
| Jerami | `#C9A227` | tingkat **kemungkinan** | **hanya sebagai isian dengan teks Tinta (5,95)**; tidak pernah sebagai warna teks (2,22) atau dengan teks putih (2,42) |
| Langit | `#4F6D8A` | tingkat **tidak terdeteksi** | putih di atasnya 5,40; sebagai teks di Air sawah 4,96 |

Warna turunan (bukan warna baru, hanya campuran palet dengan putih/latar):
- garis pemisah `#D5DED3`;
- permukaan foto `#FFFFFF`;
- isian zona pita yang tidak aktif = warna zona dengan opasitas 30%.

Warna tidak pernah menjadi satu-satunya penanda. Setiap tingkat selalu tertulis: "Terdeteksi", "Kemungkinan", "Tidak terdeteksi".

## 2. Huruf

- **Plus Jakarta Sans** (Google Fonts, bobot 400/600/700) untuk semua teks, termasuk angka. Fallback:
  `system-ui, -apple-system, "Segoe UI", Roboto, sans-serif`.
- Angka tingkat keyakinan memakai `font-variant-numeric: tabular-nums`, supaya "47%" dan "35%" sejajar.
- Nama latin patogen ditulis miring (kaidah penulisan ilmiah), ukuran kecil, warna Tinta.

| Peran | Ukuran / tinggi baris | Bobot |
|---|---|---|
| Judul halaman | 28 / 34 px | 700 |
| Nama penyakit (hasil) | 30 / 36 px | 700 |
| Judul bagian ("Yang perlu dilakukan") | 20 / 28 px | 700 |
| Teks isi | 17 / 26 px | 400 |
| Teks kecil (patogen, keterangan pita) | 14 / 20 px | 400 |
| Angka keyakinan di pita | 17 px | 700, tabular |

Teks isi 17 px (bukan 16 px), karena dibaca di HP di luar ruangan.

## 3. Elemen khas: pita keyakinan

Pita horizontal tiga zona, tepat di bawah nama penyakit:

```
  tidak terdeteksi      kemungkinan    terdeteksi
 |######## Langit #####|## Jerami ##|########## Karat ##########|
 0%                   29%          35%                       100%
                                        ▲
                               Tingkat keyakinan 47%
```

- Zona aktif berwarna penuh; dua zona lainnya pudar (opasitas 30%). Nama zona aktif ditebalkan.
- **Skala bertingkat, bukan linear**: 0–29% mengisi 36% lebar, 29–35% mengisi 28%, dan 35–100% mengisi 36%.
  - Alasannya: pada skala linear, zona "kemungkinan" hanya 6% lebar pita (±20 px di HP), sehingga label dan penandanya tidak terbaca.
  - Agar tetap jujur, batas 29% dan 35% ditulis di bawah pita dan persen keyakinan selalu tertulis sebagai angka.
  - Pembagian awal 40/20/40 diubah ke 36/28/36 setelah dihitung: lebar isi di HP 390 px ≈ 358 px, sehingga zona tengah 20% = 72 px,
    sedangkan kata "kemungkinan" pada 14 px ≈ 83 px. Dengan 28% = 100 px, kata itu muat.
- Penanda: segitiga Tinta di bawah pita, dengan teks persen di bawahnya. Di atas pita ada kalimat "Tingkat keyakinan 47%".
  Tidak ada animasi.
- Dibangun dengan HTML/CSS (bukan SVG ber-viewBox), supaya ukuran huruf tetap 14–17 px di layar sempit dan tidak ikut mengecil.
  Wadahnya memakai `role="img"` dan `aria-label` berisi kalimat lengkap (contoh: "Tingkat keyakinan 47 persen, zona terdeteksi").

## 4. Urutan halaman (satu kolom, mobile-first)

### HP 390 px

```
┌──────────────────────────────────────┐
│ Tilik Padi                 Tentang   │  ← navigasi atas, teks biasa
├──────────────────────────────────────┤
│ Cek penyakit daun padi               │  28 px
│ Foto daun padi yang sakit. Sistem    │
│ akan menebak penyakitnya dan memberi │
│ saran tindakan.                      │
│                                      │
│ ┌──────────────────────────────────┐ │
│ │        Ambil foto daun           │ │  tombol utama, tinggi 56 px, Daun padi
│ └──────────────────────────────────┘ │
│ ┌──────────────────────────────────┐ │
│ │          Unggah foto             │ │  tombol kedua, garis Daun padi
│ └──────────────────────────────────┘ │
│ Cara memotret                        │
│ Foto daun langsung di tanamannya,    │  teks dari config.yaml
│ satu helai daun memenuhi ...         │
├──────────────────────────────────────┤  (setelah foto dikirim)
│ ┌──────────────────────────────────┐ │
│ │  foto dengan kotak bercak        │ │  garis tebal, warna sesuai tingkat
│ └──────────────────────────────────┘ │
│ [Terdeteksi]                         │  label tingkat: teks + warna
│ Blas daun                            │  30 px
│ Magnaporthe oryzae                   │  14 px miring
│ <pita keyakinan>                     │
│                                      │
│ Yang perlu dilakukan                 │  20 px
│ 1. Semprotkan fungisida ...          │
│ 2. ...                               │
│                                      │
│ ▸ Mengapa sistem menyimpulkan ini    │  expander (tertutup)
│                                      │
│ Sistem ini hanya mengenali Blast,    │  peringatan tenang, garis kiri Langit
│ Brown spot, dan BLB ... Baca selengkapnya │
└──────────────────────────────────────┘
```

### Desktop 1280 px

Tetap satu kolom (lebar isi maksimal 720 px, rata tengah), supaya urutan baca sama di semua perangkat. Ruang kosong di
kiri-kanan sengaja dibiarkan polos.

```
┌────────────────────────────────────────────────────────────────────────────┐
│ Tilik Padi                                                       Tentang   │
│                 ┌──────────────────────────────────────────┐               │
│                 │ Cek penyakit daun padi                   │               │
│                 │ kalimat pembuka                          │               │
│                 │ [ Ambil foto daun ] [ Unggah foto ]      │ ← dua tombol berdampingan
│                 │ Cara memotret ...                        │   bila lebar ≥ 640 px
│                 │ ┌──────────────────────────────────────┐ │               │
│                 │ │ foto (maks. tinggi 520 px)           │ │               │
│                 │ └──────────────────────────────────────┘ │               │
│                 │ [Terdeteksi] Blas daun / patogen          │               │
│                 │ pita keyakinan                           │               │
│                 │ Yang perlu dilakukan ...                 │               │
│                 └──────────────────────────────────────────┘               │
└────────────────────────────────────────────────────────────────────────────┘
```

## 5. Aksesibilitas

- Kontras teks ≥ 4,5:1 (tabel palet). Jerami tidak pernah dipakai sebagai warna teks.
- Target sentuh ≥ 44 px; tombol utama 56 px.
- Tingkat selalu berupa teks. Kotak di foto diberi warna tingkat, dan tingkatnya juga tertulis di bawah foto.
- Tidak ada animasi otomatis. Animasi bawaan Streamlit (misalnya fade) dimatikan lewat CSS
  `transition: none; animation: none`.
- Bahasa sederhana dan kalimat aktif. Istilah yang dipakai adalah "tingkat keyakinan", bukan "confidence".

## 6. Tinjauan terhadap daftar HINDARI (sebelum membangun)

Draf pertama (di kepala, sebelum kode) lalu diperiksa terhadap daftar HINDARI. Yang generik diganti:

| Draf pertama | Masalah (HINDARI) | Diganti dengan |
|---|---|---|
| Spanduk judul hijau bergradasi ke hijau muda | gradien dekoratif | judul teks biasa di atas Air sawah; tanpa spanduk |
| Tiap bagian (foto, hasil, tindakan, jejak) di kartu putih bersudut bulat dengan bayangan | kartu bulat seragam dengan bayangan abu-abu | bagian dipisah jarak dan satu garis tipis `#D5DED3`; hanya foto yang punya bingkai putih tipis tanpa bayangan |
| Label kecil "HASIL DIAGNOSIS" huruf kapital di atas nama penyakit | label kapital di atas judul | label tingkat ("Terdeteksi") ditulis biasa, sebagai penanda berwarna yang memuat informasi, bukan hiasan |
| Tombol "📷 Ambil foto →" | emoji dan panah di tombol | "Ambil foto daun", "Unggah foto" |
| Subjudul "Diagnosis berbasis AI canggih" | kata promosi | kalimat pembuka yang menjelaskan apa yang terjadi: "Sistem akan menebak penyakitnya dan memberi saran tindakan." |
| Baris info "YOLOv8s · ONNX · 69 ms" | pemisah titik tengah; detail teknis untuk petani | dipindah ke halaman Tentang, ditulis sebagai kalimat |
| Label pita "TIDAK TERDETEKSI / KEMUNGKINAN / TERDETEKSI" dengan huruf monospace | kapital semua; monospace untuk label | label huruf kecil di awal kalimat, dalam Plus Jakarta Sans |
| Penanda pita bergeser beranimasi saat hasil muncul | animasi muncul | penanda statis |
| Kotak peringatan kuning bawaan `st.warning` dengan ikon | terlalu keras dan generik; Jerami sudah berarti "kemungkinan" | paragraf tenang dengan garis kiri Langit dan tautan "Baca selengkapnya di halaman Tentang" |
| Latar krem `#FAF6EE` dengan aksen terakota | latar krem + terakota | Air sawah `#F3F6F2`; Karat hanya untuk tingkat terdeteksi |

Elemen bawaan Streamlit yang disembunyikan: menu utama, tombol deploy, toolbar, footer "Made with Streamlit", dan dekorasi garis
pelangi di atas halaman.

## 7. Kritik setelah tangkapan layar

Tangkapan layar 390 px dan 1280 px dikritik terhadap daftar HINDARI dalam tiga putaran. Catatan kritik dan gambar sebelum/sesudah
disimpan di folder proyek skripsi (tidak ikut repositori). Perbaikan utama:
- kontras teks tombol;
- ikon bawaan yang rusak karena aturan font;
- penanda pita yang menabrak label;
- kotak bercak yang terlalu tipis, ditambah garis luar Tinta;
- hierarki judul;
- tombol unggah ganda;
- ilustrasi yang tampil dua kali;
- ketergantungan pada CSS `:has()`.

Hasil akhirnya ada di `docs/tangkapan_layar/`.

## 8. Revisi (14 Okt): halaman awal dan hasil

Masalah yang dilaporkan: halaman awal terlalu biasa, aksi unggah ganda, judul ganda; dari tangkapan layar HP, hasil terlalu jauh di
bawah. Keputusan revisi:

- **Satu aksi unggah.** File uploader Streamlit sendiri digayakan sebagai tombol besar "Unggah foto" (garis tepi Daun padi). Semua
  teks bawaan berbahasa Inggris disembunyikan. Batas unggah server dinaikkan ke 25 MB, sedangkan batas 10 MB diperiksa aplikasi supaya
  pesannya berbahasa Indonesia. Jenis berkas tidak lagi difilter oleh widget; pemeriksaannya juga dilakukan aplikasi dengan pesan
  Indonesia.
- **Kepala.** Tanda kata dengan ikon padi satu garis (SVG). Judul tunggal "Periksa daun padi dari foto".
- **Elemen khas kedua: helai daun bergejala.** Tiga ilustrasi potongan helai daun memanjang, mengikuti ciri bab 2.2, dengan warna
  alami redup di dalam gambar saja. Bentuk memanjang dipilih agar terbaca sebagai daun padi dan menghindari kartu seragam. Ilustrasi
  yang sama dipakai di hasil sebagai pembanding dengan potongan foto.
- **Petunjuk bergambar.** "Seperti ini" dan "Bukan seperti ini" ditandai teks, centang, dan silang (Tinta), bukan warna saja. Karat tidak
  dipakai untuk "salah" karena Karat sudah berarti "terdeteksi".
- **Hasil di atas.** Setelah foto diproses, judul, petunjuk, dan tiga penyakit dilipat ke "Cara memotret", lalu halaman langsung
  bergulir (instan) ke hasil.
- **Potongan diperbesar.** Bercak berkeyakinan tertinggi ditampilkan ±3× di samping ilustrasi, supaya petani bisa membandingkan bercak
  di fotonya dengan ciri penyakitnya.
- **Pita.** Hanya label zona dan angka keyakinan. Angka batas pindah ke panel jejak, karena bagi petani angka batas tidak membantu
  keputusan.
- **Rekomendasi berkelompok.** "Tindakan sekarang" lebih dulu, lalu "Pencegahan musim berikutnya". Kata-kata persis Tabel 3.1,
  ditutup catatan tetap tentang dosis pestisida dan sumber aturan.
- **Desktop ≥ 1024 px dua kolom** (awal: aksi | tiga penyakit; hasil: foto | penjelasan). Di bawah 1024 px satu kolom maksimal 720 px.

## 9. Penyampaian hasil (15 Okt): kemungkinan lain, cara membedakan, bercak berkeyakinan rendah

Latar: di VAL, model sering menukar Blast dan Brown spot, dan pada sebagian kasus kelas benarnya tetap muncul sebagai kotak kedua
(`reports/final_eval_v7/aplikasi/analisis_val_kemungkinan_kedua/RINGKASAN.md`, analisis val, bukan hasil utama). Model, ambang, dan
saran tidak diubah; yang ditambah hanya cara menyampaikan.

- **Kemungkinan lain** berupa satu baris teks biasa di bawah nama penyakit ("Kemungkinan lain: Bercak cokelat (38%)"), bukan label
  berwarna, supaya tidak bersaing dengan diagnosis utama. Hanya satu penyakit, keyakinan tertinggi.
- **Cara membedakan** memakai ilustrasi helai daun yang sama, diperbesar ke bagian tengah, berdampingan. Ditaruh di antara pita dan
  saran: petani melihat cara memastikan sebelum membaca saran fungisida. Kalimat penutup mengakui bahwa sistem juga bisa tertukar.
- **Bercak berkeyakinan rendah** tidak memakai warna tingkat (Karat/Jerami/Langit) karena bukan diagnosis: kotaknya netral (garis
  terang putus-putus di atas garis Tinta) berlabel "keyakinan rendah", dan kotak sangat kecil diperlebar ke sisi minimum 1/10 foto
  agar terlihat di HP. Kalimat di kepala hasil diberi garis kiri putus-putus Tinta, sepadan dengan kotaknya.

## 10. Model YOLOv8s aktif (16 Okt): dua tingkat, batas tampilan, cara membedakan terlipat

- **Dua tingkat.** Dengan YOLOv8s, prosedur yang sama di val menghasilkan batas bawah = ambang terdeteksi (0,37), sehingga pita hanya
  punya dua zona 50/50 (tidak terdeteksi | terdeteksi). Skala bertingkat 36/28/36 di bagian 3 hanya berlaku untuk tiga tingkat
  (arsip YOLO11s).
- **Bercak berkeyakinan rendah dinonaktifkan.** Batas yang memenuhi syarat foto sehat <= 25% (VAL v7, model final YOLOv8s seed 2)
  hanya 0,35; tanda itu tampil pada 2 foto val dengan model aplikasi dan sensitif terhadap beberapa foto. Syarat tidak dilonggarkan,
  jadi tampilan ini dimatikan lewat config (kodenya tetap ada).
- **Cara membedakan** terbuka hanya untuk pasangan Blast dan Brown spot (diagnosis utama + kemungkinan lain), karena hanya di situ
  sistem sendiri ragu di antara keduanya. Pada hasil Blast atau Brown spot lain, bagian ini terlipat dengan judul yang sama, supaya
  halaman hasil di HP tidak memanjang.

## 11. Alur v9 aktif (20 Okt): pemisah daun, tiga tingkat, foto ulang

- **Tiga tingkat lagi.** Dengan alur v9, prosedur tiga tingkat di val menghasilkan batas bawah 0,30 di bawah ambang terdeteksi 0,36,
  sehingga pita kembali punya zona "kemungkinan" (skala bertingkat bagian 3). Bercak berkeyakinan rendah tetap nonaktif (tidak ada
  batas yang memenuhi syarat foto sehat <= 25%).
- **Foto ulang memakai pesan galat yang sudah ada** ("Foto belum bisa diperiksa." + kalimat cara memperbaiki) di halaman awal, di atas
  petunjuk memotret. Alasannya: tindakan yang diminta sama dengan foto yang ditolak (memotret ulang), dan petunjuk memotret langsung
  terlihat di bawahnya. Teksnya di `config.yaml` (`teks.pesan_foto_ulang`).
- **Foto yang ditampilkan tetap foto asli.** Latar abu hanya untuk model; pengguna melihat fotonya sendiri dengan kotak bercak, supaya
  hasil mudah dicocokkan dengan daun di tangannya.
- **Halaman Tentang** menjelaskan langkah pemisah daun, memperingatkan bahwa foto BLB dan daun yang dipegang tangan lebih sering diminta
  foto ulang, dan mencantumkan angka evaluasi alur aplikasi apa adanya (termasuk persentase foto ulang; `config.yaml`, `hasil_evaluasi`).
