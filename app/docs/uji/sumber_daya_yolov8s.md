# Pengukuran RAM dan waktu respons

Mesin: Intel64 Family 6 Model 154 Stepping 3, GenuineIntel (16 inti logis), Windows-10-10.0.19045-SP0, Python 3.12.3. CPU saja.

| Ukuran | Nilai |
|---|---|
| RAM server siap (model belum dimuat) | 66.1 MB |
| RAM diam (model sudah dimuat) | 151.8 MB |
| RAM puncak saat memproses (8 foto VAL + 3 foto 4000x3000) | 253.8 MB |
| RAM diam setelah 11 foto | 164.2 MB |
| Waktu inti per foto VAL (median / maks, 20 foto) | 107.0 / 120.7 ms |
| Waktu inti foto 4000x3000 (median 5x) | 259.7 ms |
| Waktu ujung ke ujung di browser, foto VAL (median 8) | 232.5 ms |
| Waktu ujung ke ujung di browser, foto 4000x3000 (median 3) | 390.1 ms |

RSS = memori fisik proses streamlit + anaknya. Waktu inti = baca foto + inferensi ONNX + diagnosis + forward chaining + gambar kotak.
