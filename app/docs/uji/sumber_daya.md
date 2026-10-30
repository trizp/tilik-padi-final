# Pengukuran RAM dan waktu respons

Mesin: Intel64 Family 6 Model 154 Stepping 3, GenuineIntel (16 inti logis), Windows-10-10.0.19045-SP0, Python 3.12.3. CPU saja.

| Ukuran | Nilai |
|---|---|
| RAM server siap (model belum dimuat) | 66.4 MB |
| RAM diam (model sudah dimuat) | 172.2 MB |
| RAM puncak saat memproses (8 foto VAL + 3 foto 4000x3000) | 272.5 MB |
| RAM diam setelah 11 foto | 191.4 MB |
| Waktu inti per foto VAL (median / maks, 20 foto) | 215.4 / 241.7 ms |
| Waktu inti foto 4000x3000 (median 5x) | 462.2 ms |
| Waktu ujung ke ujung di browser, foto VAL (median 8) | 332.0 ms |
| Waktu ujung ke ujung di browser, foto 4000x3000 (median 3) | 595.9 ms |

RSS = memori fisik proses streamlit + anaknya. Waktu inti = baca foto + pemisah daun YOLOv8n-seg + inferensi ONNX detektor + diagnosis + forward chaining + gambar kotak.
