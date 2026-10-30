# Kurva latihan Tahap B (VAL saja)

| run | kelas | n_epoch | mAP50_95_maks | epoch_maks | epoch_capai_95 | epoch_stabil_95 | epoch_capai_99 | epoch_stabil_99 | mAP50_95_ep100 | mAP50_95_maks_s.d._ep100 | mAP50_95_ep150 | mAP50_ep100 | mAP50_ep150 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| prov_v6_yolov8n_s0 | SEMUA | 150 | 0.1723 | 66 | 48 |  | 66 |  | 0.157 | 0.1723 | 0.1572 | 0.3826 | 0.3843 |
| prov_v6_yolov8n_s0 | bacterial_leaf_blight (provisional) | 150 | 0.2495 | 66 | 60 |  | 66 |  | 0.2176 | 0.2495 | 0.2127 | 0.5044 | 0.5044 |
| prov_v6_yolov8n_s0 | brown_spot | 150 | 0.1797 | 35 | 22 |  | 22 |  | 0.1664 | 0.1797 | 0.1703 | 0.3936 | 0.4034 |
| prov_v6_yolov8n_s0 | leaf_blast | 150 | 0.0964 | 53 | 25 |  | 49 |  | 0.087 | 0.0964 | 0.0885 | 0.2497 | 0.2451 |
| prov_v6_yolo11n_s0 | SEMUA | 150 | 0.175 | 61 | 56 |  | 57 |  | 0.162 | 0.175 | 0.1562 | 0.4044 | 0.3852 |
| prov_v6_yolo11n_s0 | bacterial_leaf_blight (provisional) | 150 | 0.2385 | 58 | 57 |  | 58 |  | 0.2076 | 0.2385 | 0.2006 | 0.513 | 0.4712 |
| prov_v6_yolo11n_s0 | brown_spot | 150 | 0.1848 | 40 | 28 |  | 40 |  | 0.173 | 0.1848 | 0.1687 | 0.4259 | 0.4177 |
| prov_v6_yolo11n_s0 | leaf_blast | 150 | 0.1133 | 80 | 49 |  | 80 |  | 0.1053 | 0.1133 | 0.0993 | 0.2743 | 0.2667 |

## Tanda overfitting (loss dihaluskan 5 epoch)

### prov_v6_yolov8n_s0

| index | epoch_val_loss_min | val_loss_min_halus | val_loss_akhir_halus | val_naik_dari_min_persen | train_berubah_di_rentang_sama_persen | tanda_overfitting |
|---|---|---|---|---|---|---|
| box | 29 | 1.6733 | 1.7737 | 6.0 | -30.6 | True |
| cls | 150 | 4.5555 | 4.5555 | 0.0 | 0.0 | False |
| dfl | 28 | 1.4696 | 1.5983 | 8.8 | -19.1 | True |
| jumlah | 150 | 7.9275 | 7.9275 | 0.0 | 0.0 | False |

### prov_v6_yolo11n_s0

| index | epoch_val_loss_min | val_loss_min_halus | val_loss_akhir_halus | val_naik_dari_min_persen | train_berubah_di_rentang_sama_persen | tanda_overfitting |
|---|---|---|---|---|---|---|
| box | 26 | 1.6605 | 1.7693 | 6.6 | -32.8 | True |
| cls | 150 | 3.6807 | 3.6807 | 0.0 | 0.0 | False |
| dfl | 26 | 1.4475 | 1.5877 | 9.7 | -20.6 | True |
| jumlah | 150 | 7.0376 | 7.0376 | 0.0 | 0.0 | False |

## Waktu

```json
{
 "prov_v6_yolov8n_s0": {
  "epoch_tercatat": 150,
  "jam_gpu": 2.97,
  "detik_per_epoch_rata2": 71.3,
  "detik_per_epoch_mosaic_mati_rata2": 69.3,
  "segmen": [
   {
    "jenis": "baru",
    "mulai": "2026-10-03T18:13:14",
    "epoch_awal": 1,
    "gpu_terpakai_sebelum_run_gb": 0.703
   },
   {
    "jenis": "lanjutan (resume)",
    "mulai": "2026-10-03T19:25:39",
    "epoch_awal": 62,
    "gpu_terpakai_sebelum_run_gb": 0.801
   },
   {
    "jenis": "lanjutan (resume)",
    "mulai": "2026-10-03T19:42:54",
    "epoch_awal": 74,
    "gpu_terpakai_sebelum_run_gb": 0.717
   }
  ]
 },
 "prov_v6_yolo11n_s0": {
  "epoch_tercatat": 150,
  "jam_gpu": 3.28,
  "detik_per_epoch_rata2": 78.6,
  "detik_per_epoch_mosaic_mati_rata2": 80.0,
  "segmen": [
   {
    "jenis": "baru",
    "mulai": "2026-10-03T21:14:24",
    "epoch_awal": 1,
    "gpu_terpakai_sebelum_run_gb": 0.437
   }
  ]
 }
}
```
## Proyeksi jam GPU run final (3 seed x 4 model, batch 8)

```json
{
 "detik_per_epoch": {
  "yolov8n": 71.3,
  "yolo11n": 78.6,
  "yolov8s": 116.0,
  "yolo11s": 116.5
 },
 "rasio_koreksi_tahap_A": 0.843,
 "jam_gpu_3seed_4model_100_epoch": 31.9,
 "jam_gpu_3seed_4model_150_epoch": 47.8
}
```