# Uji kesetaraan inferensi app/core vs hasil evaluasi tersimpan

Acuan: `reports\v9_latar\aplikasi_v9\fase4_onnx_vs_pt_20_val.csv` (Ultralytics 8.4.171, conf 0,001, iou 0,7). 20 foto val pertama di berkas acuan.

Syarat lulus: kelas teratas identik (foto dengan keyakinan acuan >= 0,01), tingkat jawaban sama, dan selisih keyakinan < 0,01.

**Hasil: LOLOS**: kelas teratas identik 18/20 (pada foto dengan keyakinan acuan >= 0,01: 12/12); tingkat jawaban sama 20/20; selisih keyakinan maksimum 0.00161 (vs ONNX Ultralytics) dan 0.00161 (vs .pt).

| Foto | Kelas acuan | Keyakinan acuan (ONNX) | Kelas app | Keyakinan app | Selisih |
|---|---|---|---|---|---|
| ds4_brown-spot-690_93aab3f3.jpg | 1 | 0.65840 | 1 | 0.65913 | 0.00073 |
| ds4_Healthy-1016_ecf273a6.jpg | 0 | 0.00315 | 1 | 0.00318 | 0.00003 |
| ds1_provisional_r6k-train-blight-rotated-003-jpg-rf-f64b_505acae2.jpg | 2 | 0.89167 | 2 | 0.89157 | 0.00010 |
| ds4_brown-spot-1110_58b7ca74.jpg | 1 | 0.72870 | 1 | 0.72831 | 0.00039 |
| ds4_Healthy-372_be9ae65f.jpg | -1 | 0.00000 | -1 | 0.00000 | 0.00000 |
| ds4_Healthy-21_06de02e9.jpg | -1 | 0.00000 | -1 | 0.00000 | 0.00000 |
| ds4_brown-spot-960_7b814cd6.jpg | 1 | 0.67398 | 1 | 0.67393 | 0.00005 |
| ds4_brown-spot-1520_967ad566.jpg | 1 | 0.68711 | 1 | 0.68677 | 0.00034 |
| ds4_Healthy-1575_1e883ab4.jpg | 1 | 0.00365 | 0 | 0.00366 | 0.00001 |
| ds4_brown-spot-483_8f6f4af4.jpg | 1 | 0.58643 | 1 | 0.58607 | 0.00036 |
| ds4_blast-621_568f1e53.jpg | 0 | 0.47018 | 0 | 0.47179 | 0.00161 |
| ds4_Healthy-333_cdcb6f84.jpg | 1 | 0.32716 | 1 | 0.32821 | 0.00105 |
| ds2_Healthy-rice-leaf-2_559d3866.jpg | -1 | 0.00000 | -1 | 0.00000 | 0.00000 |
| ds4_brown-spot-1375_94f274b7.jpg | 1 | 0.67523 | 1 | 0.67561 | 0.00038 |
| ds2_Healthy-rice-leaf-31_09a59d72.jpg | -1 | 0.00000 | -1 | 0.00000 | 0.00000 |
| ds1_provisional_r6k-train-img-blb-32-JPG-jpg-rf-cba33b8e_48db280a.jpg | 2 | 0.85064 | 2 | 0.85056 | 0.00008 |
| ds4_blast-967_21b44414.jpg | 0 | 0.72574 | 0 | 0.72572 | 0.00002 |
| ds4_Healthy-586_2dcdcd08.jpg | -1 | 0.00000 | -1 | 0.00000 | 0.00000 |
| ds2_Healthy-rice-leaf-96_999b4bac.jpg | -1 | 0.00000 | -1 | 0.00000 | 0.00000 |
| ds4_brown-spot-1679_db1141f6.jpg | 1 | 0.75296 | 1 | 0.75320 | 0.00024 |
