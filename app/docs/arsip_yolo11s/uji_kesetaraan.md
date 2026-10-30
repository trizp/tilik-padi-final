# Uji kesetaraan inferensi app/core vs hasil evaluasi tersimpan

Acuan: `reports\final_eval_v7\aplikasi\onnx_vs_pt_50_val.csv` (Ultralytics 8.4.171, conf 0,001, iou 0,7). 20 foto val pertama di berkas acuan.

**Hasil: LOLOS**: kelas teratas identik 20/20; selisih keyakinan maksimum 0.00198 (vs ONNX Ultralytics) dan 0.00198 (vs .pt).

| Foto | Kelas acuan | Keyakinan acuan (ONNX) | Kelas app | Keyakinan app | Selisih |
|---|---|---|---|---|---|
| ds4_brown-spot-2173_3657de07.jpg | 1 | 0.63379 | 1 | 0.63305 | 0.00074 |
| ds4_brown-spot-516_43ab9f72.jpg | 1 | 0.73617 | 1 | 0.73632 | 0.00015 |
| ds1_provisional_r6k-train-blight-rotated-003-jpg-rf-14a2_a7fb66f7.jpg | 2 | 0.88163 | 2 | 0.88146 | 0.00017 |
| ds4_blast-558_ba273edb.jpg | 0 | 0.45980 | 0 | 0.45961 | 0.00019 |
| ds4_Healthy-130_e39d3e87.jpg | 2 | 0.00724 | 2 | 0.00724 | 0.00000 |
| ds4_Healthy-1029_6d6a316a.jpg | 1 | 0.05572 | 1 | 0.05439 | 0.00133 |
| ds4_brown-spot-383_0ed30d7c.jpg | 1 | 0.59312 | 1 | 0.59510 | 0.00198 |
| ds4_blast-94_9f932d59.jpg | 0 | 0.72668 | 0 | 0.72679 | 0.00011 |
| ds4_Healthy-1010_be7c3073.jpg | 1 | 0.11877 | 1 | 0.11760 | 0.00117 |
| ds4_brown-spot-1460_6865e2a4.jpg | 1 | 0.71933 | 1 | 0.71924 | 0.00009 |
| ds4_Healthy-470_ce3289e3.jpg | 1 | 0.00454 | 1 | 0.00439 | 0.00015 |
| ds4_blast-125_c17df59b.jpg | 0 | 0.75659 | 0 | 0.75630 | 0.00029 |
| ds4_Healthy-1180_2a19d576.jpg | 1 | 0.03562 | 1 | 0.03464 | 0.00098 |
| ds1_provisional_r6k-train-img-blb-86-jpg-rf-a7c3045efecb_39e344cf.jpg | 2 | 0.62859 | 2 | 0.62815 | 0.00044 |
| ds4_blast-682_a02d5aa8.jpg | 0 | 0.71181 | 0 | 0.71147 | 0.00034 |
| ds1_provisional_r6k-train-img-blb-86-jpg-rf-ed30ea3aa394_b9bee907.jpg | 2 | 0.67898 | 2 | 0.67839 | 0.00059 |
| ds1_provisional_r6k-train-img-blb-133-jpeg-jpg-rf-007032_11a178c3.jpg | 2 | 0.83202 | 2 | 0.83210 | 0.00008 |
| ds4_blast-442_52869e25.jpg | 0 | 0.83896 | 0 | 0.83885 | 0.00011 |
| ds4_Healthy-1545_82309143.jpg | -1 | 0.00000 | -1 | 0.00000 | 0.00000 |
| ds4_Healthy-692_f90c5c0c.jpg | 1 | 0.15285 | 1 | 0.15371 | 0.00086 |
