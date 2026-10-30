# Pengukuran RAM dan waktu respons

Mesin: Intel64 Family 6 Model 154 Stepping 3, GenuineIntel (16 inti logis), Windows-10-10.0.19045-SP0, Python 3.12.3. CPU saja.

| Ukuran | Nilai |
|---|---|
| RAM server siap (model belum dimuat) | 66.4 MB |
| RAM diam (model sudah dimuat) | 155.1 MB |
| RAM puncak saat memproses (8 foto VAL + 3 foto 4000x3000) | 254.1 MB |
| RAM diam setelah 11 foto | 168.3 MB |
| Waktu inti per foto VAL (median / maks, 20 foto) | 125.3 / 139.9 ms |
| Waktu inti foto 4000x3000 (median 5x) | 272.5 ms |
| Waktu ujung ke ujung di browser, foto VAL (median 8) | 237.4 ms |
| Waktu ujung ke ujung di browser, foto 4000x3000 (median 3) | 394.1 ms |

RSS = memori fisik proses streamlit + anaknya. Waktu inti = baca foto + inferensi ONNX + diagnosis + forward chaining + gambar kotak.
