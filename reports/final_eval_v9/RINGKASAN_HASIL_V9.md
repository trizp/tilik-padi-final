# Ringkasan hasil v9: model yang dilatih dengan foto tanpa latar

Ditulis 2026-10-17. Semua angka dibaca dari hasil evaluasi di folder ini; tidak ada model, ambang, atau dataset yang diubah setelah evaluasi.
Rencana evaluasi dicatat SEBELUM training (KEPUTUSAN_PROYEK.md 2026-10-16) dan dijalankan persis, setiap set satu kali.
Dataset tetap PROVISIONAL (BLB dari ds1).

## Apa yang diuji

- **v7** = model yang dilatih dengan foto asli (ada latar sawah, tanah, tangan, dan lain-lain).
- **v9** = model yang dilatih dengan foto yang latarnya dihapus (hanya daun; latar abu-abu). Daun dipisahkan otomatis dengan SAM2
  (tanpa melihat label).
- Dua model: YOLOv8s dan YOLO11s, masing-masing 3 seed. Ambang diagnosis tiap model dikunci di val masing-masing sebelum test dibuka.
- **Dengan latar** = model v7 pada foto asli. **Tanpa latar** = model v9 pada foto tanpa latar.
  **Sel silang** (tambahan) = model v7 pada foto tanpa latar dan model v9 pada foto asli.
- Semua perbandingan memakai **irisan foto** yang ada di v7 dan v9: test bersih 432 foto (104 kelompok; 2 foto test v7 tidak ada di v9),
  test sesi baru 292 foto (129 kelompok; 7 foto tidak ada di v9). BLB dilaporkan terpisah (150 foto, 18 kelompok).
- Rentang keyakinan 95% (IK) dihitung dengan bootstrap per kelompok foto (1.000 kali). Selisih dianggap meyakinkan bila IK-nya tidak
  memuat nol.

## Kualitas pemisahan daun

- Audit manusia (pemilik skripsi; 50 foto VAL acak di luar sampel uji coba): **ds4 baik 37 dari 38 (97,4%; IK 86,5-99,5%)**, BLB ds1
  baik 4 dari 12 (3 salah objek dari satu foto asal, 5 tidak dipisah). Semua: 41 dari 50 baik (82%).
- ds3 (daun di atas kertas putih), diperiksa SEBELUM model dijalankan: foto yang tidak dapat dipisah (cadangan) **2,9%** pada uji utama
  (2 dari 70), 1,2% pada set tambahan, 5,0% pada Tungro; 20 dari 20 contoh montase terpisah baik (`ds3_pemisahan/`).

## Hasil utama: dengan vs tanpa latar (rata-rata 3 seed)

Test bersih (432 foto, tanpa BLB):

| Model | Kondisi | mAP50 | mAP50-95 [IK 95%] | Akurasi diagnosis [IK 95%] | Recall diagnosis Blast |
|---|---|---|---|---|---|
| YOLOv8s | dengan latar (v7) | 0,574 | 0,249 [0,194; 0,290] | 0,751 [0,680; 0,824] | 0,787 |
| YOLOv8s | tanpa latar (v9) | 0,588 | 0,253 [0,201; 0,292] | 0,767 [0,697; 0,839] | 0,802 |
| YOLO11s | dengan latar (v7) | 0,627 | 0,273 [0,221; 0,312] | 0,782 [0,717; 0,847] | 0,847 |
| YOLO11s | tanpa latar (v9) | 0,636 | 0,277 [0,231; 0,310] | 0,789 [0,736; 0,849] | 0,796 |

Test sesi baru (292 foto):

| Model | Kondisi | mAP50 | mAP50-95 [IK 95%] | Akurasi diagnosis [IK 95%] | Recall diagnosis Blast |
|---|---|---|---|---|---|
| YOLOv8s | dengan latar (v7) | 0,546 | 0,244 [0,218; 0,279] | 0,692 [0,632; 0,755] | 0,926 |
| YOLOv8s | tanpa latar (v9) | 0,540 | 0,238 [0,212; 0,274] | 0,711 [0,648; 0,774] | 0,926 |
| YOLO11s | dengan latar (v7) | 0,587 | 0,260 [0,230; 0,300] | 0,753 [0,691; 0,813] | 0,915 |
| YOLO11s | tanpa latar (v9) | 0,603 | 0,266 [0,239; 0,304] | 0,739 [0,683; 0,791] | 0,918 |

Selisih tanpa latar - dengan latar (v9 - v7), test bersih: mAP50-95 YOLOv8s +0,004 [-0,014; 0,020], YOLO11s +0,005 [-0,013; 0,028];
akurasi YOLOv8s +0,016 [-0,013; 0,047], YOLO11s +0,006 [-0,023; 0,041]; recall Blast YOLOv8s +0,015 [-0,053; 0,065],
YOLO11s -0,051 [-0,108; -0,003]. Pada test sesi baru semua selisih v9 - v7 memuat nol (`tabel5_kontras_latar.csv`).

## Kepekaan terhadap latar (sel silang)

Test bersih, rata-rata 3 seed (`tabel4_sel_silang.csv`, `tabel5_kontras_latar.csv`):

| Model | v7 foto asli | v7 foto tanpa latar | Penurunan v7 | v9 foto tanpa latar | v9 foto asli | Penurunan v9 |
|---|---|---|---|---|---|---|
| mAP50-95 YOLOv8s | 0,249 | 0,197 | -0,052 | 0,253 | 0,226 | -0,026 |
| mAP50-95 YOLO11s | 0,273 | 0,116 | **-0,157** | 0,277 | 0,224 | -0,054 |
| Akurasi YOLOv8s | 0,751 | 0,754 | +0,003 | 0,767 | 0,755 | -0,012 |
| Akurasi YOLO11s | 0,782 | 0,665 | **-0,117** | 0,789 | 0,745 | -0,043 |
| Recall Blast YOLOv8s | 0,787 | 0,514 | -0,273 | 0,802 | 0,781 | -0,021 |
| Recall Blast YOLO11s | 0,847 | 0,108 | **-0,739** | 0,796 | 0,787 | -0,009 |

Selisih penurunan YOLO11s - YOLOv8s untuk model v7 (latar dihapus), test bersih: mAP50-95 -0,105 [-0,121; -0,084], akurasi -0,120
[-0,185; -0,056], recall Blast -0,465 [-0,543; -0,371]; test sesi baru: mAP50-95 -0,093 [-0,122; -0,069], akurasi -0,130
[-0,203; -0,065]. Semua IK tidak memuat nol.

## YOLO11s - YOLOv8s di setiap sel (berpasangan; `tabel2_selisih_yolo11s_yolov8s.csv`)

| Set | Kondisi | mAP50-95 | Akurasi | Recall Blast |
|---|---|---|---|---|
| Test bersih | dengan latar (v7) | +0,024 [0,009; 0,037] | +0,032 [0,012; 0,050] | +0,060 [0,005; 0,106] |
| Test bersih | tanpa latar (v9) | +0,025 [0,010; 0,046] | +0,022 [-0,004; 0,050] | -0,006 [-0,039; 0,025] |
| Test sesi baru | dengan latar (v7) | +0,016 [0,004; 0,029] | +0,062 [0,030; 0,095] | -0,011 [-0,045; 0,018] |
| Test sesi baru | tanpa latar (v9) | +0,028 [0,017; 0,041] | +0,027 [-0,001; 0,056] | -0,007 [-0,037; 0,024] |

## ds3 (daun petik di atas kertas putih; PEMBANDING, ds3 sudah terbuka)

Uji utama 70 foto (35 BLB, 35 Blast); v7 pada foto asli vs v9 pada foto tanpa latar (cadangan v9 2,9%):

| Model | Kondisi | Akurasi [IK 95%] | Recall BLB | Recall Blast |
|---|---|---|---|---|
| YOLOv8s | v7, ds3 asli | 0,224 [0,155; 0,309] | 0,400 | 0,048 |
| YOLOv8s | v9, ds3 tanpa latar | 0,433 [0,348; 0,523] | 0,381 | 0,486 |
| YOLO11s | v7, ds3 asli | 0,262 [0,192; 0,338] | 0,524 | 0,000 |
| YOLO11s | v9, ds3 tanpa latar | 0,305 [0,222; 0,385] | 0,248 | 0,362 |

- Selisih v9 - v7: YOLOv8s akurasi +0,210 [0,095; 0,329], recall Blast +0,438 [0,284; 0,602]; YOLO11s akurasi +0,043 [-0,073; 0,157],
  recall Blast +0,362 [0,228; 0,500], tetapi recall BLB turun -0,276.
- Pada v7, foto Blast ds3 hampir selalu disebut Brown spot (YOLO11s 89 dari 105, gabungan 3 seed); pada v9, 36-49% Blast dikenali benar.
- Set tambahan (160 foto) searah: YOLOv8s akurasi +0,150 [0,077; 0,225]; YOLO11s +0,033 [-0,039; 0,103].
- Tungro (di luar cakupan) disebut BLB: YOLOv8s 49,6% -> 35,0%; YOLO11s 47,9% -> 45,4%.
- Angka ds3 sangat bervariasi antar seed (mis. recall BLB YOLOv8s v9 0,381 ± 0,313).

## BLB (terpisah; tidak dipakai untuk kesimpulan tentang latar)

Test BLB provisional (150 foto irisan, 18 kelompok): mAP50-95 YOLOv8s 0,103 -> 0,136, YOLO11s 0,116 -> 0,167; akurasi diagnosis
YOLOv8s 0,687 -> 0,636, YOLO11s 0,764 -> 0,724. IK sangat lebar karena hanya 18 kelompok (`tabel6_blb_terpisah.csv`).

## Jawaban

**(a) Apakah model yang dilatih tanpa latar setara dengan v7 pada foto lapangan?**
Ya. Pada test bersih dan test sesi baru, v9 (tanpa latar) dan v7 (dengan latar) praktis sama: semua selisih mAP dan akurasi kecil
(paling besar ±0,02) dan IK-nya memuat nol. Satu pengecualian kecil: recall Blast YOLO11s pada test bersih sedikit lebih rendah di v9
(-0,051; IK -0,108 sampai -0,003). Artinya, bercak di daun saja sudah cukup untuk dipelajari; latar tidak diperlukan untuk mencapai
kinerja v7.

**(b) Apakah YOLOv8 dan YOLO11 berbeda dalam kepekaan terhadap latar?**
Ya, dan jelas. Bila latar dihapus dari foto uji, model v7 YOLO11s turun jauh lebih tajam daripada YOLOv8s: mAP50-95 -0,157 vs -0,052,
akurasi -0,117 vs +0,003, recall Blast -0,739 vs -0,273 (selisih penurunan YOLO11s - YOLOv8s semuanya meyakinkan, pada test bersih
dan test sesi baru). Jadi sebagian keunggulan YOLO11s di v7 bergantung pada petunjuk dari latar, terutama untuk Blast. Setelah dilatih
tanpa latar (v9), YOLO11s masih unggul pada deteksi bercak (mAP50-95 +0,025, meyakinkan), tetapi keunggulannya pada akurasi diagnosis
mengecil dan tidak lagi meyakinkan (+0,022; IK memuat nol), dan pada recall Blast hilang (-0,006).

**(c) Apakah v9 lebih baik dari v7 pada ds3?**
Ya untuk YOLOv8s, sebagian untuk YOLO11s, dengan **bukti lemah** karena ds3 sudah terbuka (pernah dilihat saat analisis v7) dan
hanya 70 foto. YOLOv8s v9 hampir menggandakan akurasi (0,224 -> 0,433, meyakinkan) dan mulai mengenali Blast (0,048 -> 0,486). YOLO11s v9
juga mulai mengenali Blast (0 -> 0,362), tetapi kehilangan sebagian BLB sehingga akurasinya hanya naik sedikit (+0,043, tidak
meyakinkan). Pemisahan daun di ds3 hampir sempurna (cadangan 2,9%), jadi perbedaan ini bukan akibat masker yang gagal. Kinerja di ds3
tetap jauh di bawah test (akurasi 0,31-0,43 vs 0,77-0,79), sehingga pergeseran domain belum teratasi.

## Keterbatasan

- Dataset provisional; BLB dari satu sumber dengan pemisahan daun yang lemah.
- Data uji "tanpa latar" juga diproses dengan SAM2; yang diukur adalah efek seluruh alur (pemisahan + model), bukan model saja.
  Label kondisi tanpa latar sedikit berbeda (kotak < 50% di daun dibuang: Blast 7%, Brown spot 2% di VAL).
- ds3 sudah terbuka dan kecil (70 foto, 68 kelompok); hasilnya hanya pembanding.
- Sel silang adalah analisis tambahan; foto tanpa latar disimpan ulang sebagai JPEG q95, foto asli tidak.
- Hanya varian s yang dilatih ulang (pilihan B).

## Berkas

- Tabel CSV dan LaTeX (format IEEE): `tabel/` - tabel1 (2x2 utama), tabel2 (selisih YOLO11s - YOLOv8s), tabel3 (per kelas, deteksi
  dan diagnosis), tabel4 (sel silang), tabel5 (kontras dan kepekaan), tabel6 (BLB terpisah), tabel7 (ds3), tabel8 (audit pemisahan).
- Grafik dan confusion matrix: `grafik/`.
- Angka mesin: `angka_kunci_v9.json`; ambang terkunci: `kunci_model.csv`; log: `evaluasi_akhir.log`, `analisis_akhir_v9.log`.
- Kualitas pemisahan ds3: `ds3_pemisahan/`; audit manusia: `../v9_latar/audit_masker/`.
- Skrip: `scripts/evaluasi_akhir_v9.py` (kunci, uji), `scripts/analisis_akhir_v9.py`, `scripts/v9_latar/ds3_tanpa_latar.py`.
