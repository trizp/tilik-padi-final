# Arsitektur

Tampilan (Streamlit) dipisahkan dari logika. `core/` dan `rbs/` tidak mengimpor Streamlit, sehingga bisa diuji dan dipakai ulang tanpa
antarmuka. `ui/komponen.py` hanya membuat string HTML.

```mermaid
flowchart TD
    U[Pengguna: kamera HP atau unggah JPG/PNG maks. 10 MB] --> S[streamlit_app.py<br/>tampilan saja]
    S -->|bytes foto| A[core/alur.py<br/>periksa_foto]
    A --> G[core/gambar.py<br/>periksa berkas, exif_transpose,<br/>RGB, perkecil > 2048 px]
    G --> I[core/inferensi.py<br/>letterbox 640, ONNX Runtime CPU,<br/>NMS per kelas, skala balik]
    M[(models/v7_deploy_yolov8s_s2.onnx)] --> I
    C[(config.yaml<br/>ambang 0,37,<br/>teks)] --> A
    I --> D[core/diagnosis.py<br/>kotak teratas, dua tingkat,<br/>kemungkinan lain]
    D --> R[rbs/mesin.py<br/>forward chaining]
    RY[(rbs/rules.yaml)] --> R
    D --> V[core/visual.py<br/>kotak tebal berwarna tingkat,<br/>kotak putus-putus keyakinan rendah (nonaktif),<br/>potongan diperbesar, JPEG]
    R --> S
    V --> S
    S --> K[ui/komponen.py + ui/ilustrasi.py<br/>kepala hasil, kemungkinan lain, cara membedakan,<br/>ilustrasi SVG, pita keyakinan,<br/>saran berkelompok, jejak, peringatan]
```

## Penyampaian tambahan (tidak mengubah diagnosis, tingkat, atau saran)

| Fitur | Syarat (`core/diagnosis.py`) | Tampilan | Jejak |
|---|---|---|---|
| Kemungkinan lain | tingkat bukan "tidak terdeteksi"; ada kotak penyakit LAIN dengan keyakinan ≥ ambang rendah; satu, tertinggi | "Kemungkinan lain: <penyakit> (<keyakinan>%)" di bawah nama penyakit | fakta `kemungkinan_lain`, `keyakinan_kemungkinan_lain` + catatan tampilan |
| Cara membedakan | **terbuka** bila diagnosis utama dan kemungkinan lain adalah pasangan Blast dan Brown spot; **terlipat** bila Blast atau Brown spot muncul tanpa pasangan itu; tidak ada untuk BLB saja atau tidak terdeteksi | ilustrasi blas daun dan bercak cokelat berdampingan (ciri bab 2.2) + saran konfirmasi ke penyuluh; versi terlipat berjudul "Cara membedakan blas daun dan bercak cokelat" | - |

Tampilan "bercak berkeyakinan rendah" (kotak putus-putus pada hasil tidak terdeteksi) masih ada di kode, tetapi dinonaktifkan di
`config.yaml` (`tampilan.bercak_rendah_aktif: false`; KEPUTUSAN_PROYEK.md 2026-10-16).

Fakta tambahan ikut masuk memori kerja, tetapi tidak ada aturan di `rules.yaml` yang memakainya, sehingga rantai aturan dan saran sama
persis dengan tanpa fitur ini (diuji di `tests/test_inti.py`).

## Alur forward chaining

```mermaid
flowchart TD
    F0["Fakta awal:<br/>penyakit, tingkat, keyakinan"] --> L{"Ada aturan yang belum aktif<br/>dan semua syaratnya benar?"}
    L -- tidak --> E["Berhenti.<br/>Tampilkan saran dan jejak"]
    L -- ya --> P["Aktifkan aturan pertama (urutan rules.yaml):<br/>tambah fakta baru, tambah saran, catat jejak"]
    P --> L
```

Rantai yang terjadi (dua putaran, lalu berhenti):

| Tingkat | Putaran 1 | Fakta baru | Putaran 2 | Saran |
|---|---|---|---|---|
| terdeteksi | D01 Diagnosis diterima | `diagnosis = <penyakit>` | R001 / R002 / R003 | butir Tabel 3.1: tindakan sekarang [S] + pencegahan musim berikutnya [P] |
| kemungkinan | (tidak terjadi dengan YOLOv8s: batas bawah = ambang terdeteksi) | - | - | - |
| tidak terdeteksi | N01 Tidak ada penyakit yang dikenali | `perlu_foto_ulang = ya` | N02 | foto ulang + tanya penyuluh + pencegahan umum |

## Memori dan kecepatan

Diukur di folder proyek skripsi dengan Chrome headless (hasil: `docs/uji/sumber_daya.md`):
- RAM diam 150 MB, puncak 240 MB, dan 164 MB setelah 11 foto;
- waktu inti ±0,1 detik per foto di CPU laptop.

Penghemat memori:
- model dimuat sekali (`st.cache_resource`);
- arena memori ONNX Runtime dimatikan;
- foto > 2048 px didekode kecil;
- foto hasil dikirim sebagai JPEG.
