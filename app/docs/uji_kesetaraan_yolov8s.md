# Uji kesetaraan inferensi app/core vs hasil evaluasi tersimpan

Acuan: `reports\final_eval_v7\aplikasi\onnx_vs_pt_20_val_yolov8s.csv` (Ultralytics 8.4.171, conf 0,001, iou 0,7). 20 foto val pertama di berkas acuan.

**Hasil: LOLOS**: kelas teratas identik 20/20; selisih keyakinan maksimum 0.00706 (vs ONNX Ultralytics) dan 0.00706 (vs .pt).

| Foto | Kelas acuan | Keyakinan acuan (ONNX) | Kelas app | Keyakinan app | Selisih |
|---|---|---|---|---|---|
| ds4_brown-spot-2173_3657de07.jpg | 1 | 0.36914 | 1 | 0.36208 | 0.00706 |
| ds4_brown-spot-516_43ab9f72.jpg | 1 | 0.64658 | 1 | 0.64676 | 0.00018 |
| ds1_provisional_r6k-train-blight-rotated-003-jpg-rf-14a2_a7fb66f7.jpg | 2 | 0.93875 | 2 | 0.93858 | 0.00017 |
| ds4_blast-558_ba273edb.jpg | 0 | 0.50568 | 0 | 0.50617 | 0.00049 |
| ds4_Healthy-130_e39d3e87.jpg | 1 | 0.00406 | 1 | 0.00407 | 0.00001 |
| ds4_Healthy-1029_6d6a316a.jpg | 1 | 0.01141 | 1 | 0.01075 | 0.00066 |
| ds4_brown-spot-383_0ed30d7c.jpg | 1 | 0.61219 | 1 | 0.61110 | 0.00109 |
| ds4_blast-94_9f932d59.jpg | 0 | 0.72435 | 0 | 0.72413 | 0.00022 |
| ds4_Healthy-1010_be7c3073.jpg | 1 | 0.11545 | 1 | 0.11596 | 0.00051 |
| ds4_brown-spot-1460_6865e2a4.jpg | 1 | 0.79972 | 1 | 0.79940 | 0.00032 |
| ds4_Healthy-470_ce3289e3.jpg | 1 | 0.00212 | 1 | 0.00205 | 0.00007 |
| ds4_blast-125_c17df59b.jpg | 0 | 0.77033 | 0 | 0.76974 | 0.00059 |
| ds4_Healthy-1180_2a19d576.jpg | 1 | 0.14046 | 1 | 0.13372 | 0.00674 |
| ds1_provisional_r6k-train-img-blb-86-jpg-rf-a7c3045efecb_39e344cf.jpg | 2 | 0.83001 | 2 | 0.83028 | 0.00027 |
| ds4_blast-682_a02d5aa8.jpg | 0 | 0.76946 | 0 | 0.76948 | 0.00002 |
| ds1_provisional_r6k-train-img-blb-86-jpg-rf-ed30ea3aa394_b9bee907.jpg | 2 | 0.81687 | 2 | 0.81641 | 0.00046 |
| ds1_provisional_r6k-train-img-blb-133-jpeg-jpg-rf-007032_11a178c3.jpg | 2 | 0.86392 | 2 | 0.86387 | 0.00005 |
| ds4_blast-442_52869e25.jpg | 0 | 0.79530 | 0 | 0.79545 | 0.00015 |
| ds4_Healthy-1545_82309143.jpg | 2 | 0.02266 | 2 | 0.02430 | 0.00164 |
| ds4_Healthy-692_f90c5c0c.jpg | 1 | 0.15505 | 1 | 0.15381 | 0.00124 |
