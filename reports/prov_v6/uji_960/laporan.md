# Tahap B lanjutan: cek memori dan uji ukuran input 960 (PROVISIONAL, tidak untuk paper)

Tanggal: 2026-10-04. Dataset prov_v6 beku (manifest 2963885f...a4d7). Split test dan test_new_session tidak disentuh.

## Pemilihan checkpoint (val bersih)
- `data/derived/prov_v6_valsel/data_valsel.yaml` dibuat oleh `scripts/buat_valsel.py`: salinan val tanpa ds1_provisional
  (869 dari 1259 gambar: 222 blast, 330 brown spot, 317 background). Label BLB di val bersih: 0.
- Ultralytics 8.4.171: mAP dihitung hanya untuk kelas berlabel (np.unique(target_cls)); fitness = mAP50-95 saja.
- `evaluate.py` kini juga melapor subset `bersih` dan `blb_provisional` secara terpisah.

## Memori (pagefile C 14000 MB + D 16000 MB, RAM 15.73 GB, batas commit 45.03 GB)
| run | train s | val s | s/iterasi | GPU util rata2 | kenaikan commit puncak | sisa terendah |
|---|---|---|---|---|---|---|
| Tahap A yolov8s 640 b8 workers 4 | 121.3 | 23.9 | 0.1342 | (median 92%) | tidak diukur | - |
| cekmemori_w2 yolov8s 640 b8 workers 2 | 117.8 | 17.5 | 0.1336 | 84.7% | 14.95 GB (val) | 20.78 GB |
| yolov8n 960 b8 (100 epoch) | 124.6/epoch | 6.7/epoch | 0.147 | 93.2% | 19.30 GB | 16.56 GB |

Workers 2 tidak melambat (train -2.9%). Puncak memori terjadi saat validasi (worker val = 2x workers).
Ambang `configs/ambang_memori.json` = 19.3 GB; train.py menolak bila tersedia < 24.3 GB.

## VRAM imgsz 960 (1 epoch, close_mosaic 0)
| model | batch | GPU puncak (dari 4 GB) | reserved | s/iterasi | status |
|---|---|---|---|---|---|
| yolov8n | 8 | 3.08 | 2.50 | 0.147 | muat |
| yolov8s | 8 | 3.93 | 4.11 | 0.830 | TIDAK muat (tumpah ke RAM bersama, 5.6x lebih lambat) |
| yolov8s | 4 | 2.60 | 2.08 | 0.150 | muat |

## Perbandingan kasar yolov8n 640 vs 960 (val bersih, seed 0, satu run per ukuran)
Lihat `perbandingan_640_960.csv`. Ringkas (best.pt masing-masing dihitung ulang pada val bersih):

| kelas | mAP50 640 | mAP50 960 | mAP50-95 640 | mAP50-95 960 |
|---|---|---|---|---|
| leaf_blast | 0.2710 | 0.2613 | 0.0912 | 0.0912 |
| brown_spot | 0.4223 | 0.3957 | 0.1821 | 0.1661 |
| rata2 2 kelas | 0.3467 | 0.3285 | 0.1366 | 0.1286 |

PERBANDINGAN KASAR, tidak sebanding penuh:
- 640 = run 150 epoch (jadwal LR 150 epoch), workers 4, best.pt dipilih dari val PENUH (termasuk BLB) -> epoch 66.
- 960 = run 100 epoch, workers 2, best.pt dipilih dari val bersih -> epoch 64.
- 640 tidak punya bobot per epoch, jadi "puncak 100 epoch pertama tanpa BLB" yang persis tidak bisa dihitung.
  Perkiraan dari log per kelas (masih memuat FP pada gambar BLB): rata2 2 kelas mAP50-95 0.1372 (ep 35).
- Satu seed; variasi antar-seed belum diketahui, jadi selisih ~0.01 belum tentu nyata.
- BLB provisional terpisah: 640 0.2513 vs 960 0.1252 (mAP50-95), tetapi 640 dipilih dengan BLB dan 960 tanpa BLB.
