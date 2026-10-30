# prov_v8 — DIARSIPKAN (v8 dihentikan sebelum training, 14 Okt 2026)

- **Status:**
  - `data/processed/prov_v8` BEKU (manifest `c175e1adc923fdba74dc89d226cf853295b29497d05121aa6af3cf98aced67d3`, dibekukan 2026-10-14 13:01).
  - **Tidak pernah dipakai untuk training.** Tidak ada run `v8_*`.
- **Alasan penghentian:** tidak ada set uji Indonesia yang segar.
  - aditia = rentetan foto dari ~1 helai daun per kelas (`reports/kandidat_v8/aditia_kelompok/`).
  - gymvi memuat foto ds3.
  - Foto lapangan sendiri tidak memungkinkan.
  - Tanpa set uji segar, perbaikan v8 tidak bisa diukur tanpa memakai set yang sudah terbuka.
- **Model final penelitian ini = v7** (KEPUTUSAN_PROYEK.md 2026-10-14).
- **Untuk penelitian lanjutan:**
  - prov_v8 siap dilatih dengan resep v7 (100 epoch, imgsz 640, batch 8, workers 2, patience 0; best.pt dari
    `data/derived/prov_v8_valsel`; nama run `v8_`), SETELAH set uji Indonesia segar dikumpulkan dan dibekukan.
  - Set uji segar itu harus dicek kembar terhadap prov_v8 (skrip `scripts/bangun_prov_v8.py audit` dapat diadaptasi).
- **Isi dan hasil pemeriksaan:** `LAPORAN_PROV_V8.md`. Daftar uji luar cakupan: `data/processed/prov_v8/luar_cakupan/`.
