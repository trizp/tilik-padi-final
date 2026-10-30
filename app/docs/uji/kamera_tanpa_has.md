# Uji tombol kamera tanpa CSS :has()

**Hasil: LOLOS**

| Keadaan | Aturan :has() dihapus | Teks sebelum memotret | Teks sesudah memotret | Ukuran huruf |
|---|---|---|---|---|
| A_normal | 0 | Ambil foto | Hapus foto dan ambil ulang | 17px |
| B_tanpa_has | 20 | Ambil foto | Hapus foto dan ambil ulang | 17px |
| C_tanpa_has_tanpa_skrip | 20 | Take Photo | Clear photo | 17px |

C = skrip label dimatikan: teks bawaan Streamlit (bahasa Inggris) tetap terlihat, tidak pernah tombol kosong.
