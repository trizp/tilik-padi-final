# YOLOv8 vs YOLO11 for rice leaf disease detection

Research code for the paper **"YOLOv8 Versus YOLO11 for Rice Leaf Disease Detection: Leakage-Free Evaluation and Background
Dependence"** and the accompanying undergraduate thesis at Telkom University. The study compares YOLOv8 and YOLO11 (n and s, three seeds
each) for leaf blast, brown spot, and bacterial leaf blight (BLB), under a leakage-free protocol. Near-duplicate and burst images are
grouped before splitting, and one capture session is held out. Results use group-level bootstrap confidence intervals. A background-removal
experiment (SAM 2.1 leaf-only images) tests how much each detector relies on context outside the leaf. The `app/` folder contains the web
application *Tilik Padi* (Streamlit), also published separately.

> Ringkasan (Bahasa Indonesia): repositori ini berisi kode, konfigurasi, manifest split, tabel hasil, dan sumber paper untuk skripsi
> "Implementasi Deep Learning YOLOv8 dan Rule-Based System untuk Diagnosis dan Rekomendasi Penanganan Penyakit Tanaman Padi Berbasis Web".
> Dataset, bobot model, dan hasil training tidak disertakan. Catatan keputusan penelitian (bahasa Indonesia) ada di `KEPUTUSAN_PROYEK.md`.

## What is (and is not) in this repository

| Folder | Content |
|---|---|
| `scripts/` | dataset building, grouping and leakage checks, training (`train.py`), evaluation (`evaluate.py`, `evaluasi_akhir_v7.py`, `evaluasi_akhir_v9.py`), analysis and bootstrap (`analisis_akhir_v7.py`, `analisis_akhir_v9.py`), background removal (`v9_latar/`), paper figures (`paper/`), presentation figures (`presentasi/`) |
| `configs/` | dataset configurations (`classes_v*.yaml`; class mapping by name, raw data paths, grouping parameters) |
| `data/processed/prov_v7*`, `prov_v9`, `data/derived/*` | **manifests and split lists only** (file name, split, class, source, duplicate group, original path); no images or labels |
| `reports/` | result tables (CSV/LaTeX/JSON), summaries, and charts for every dataset version and evaluation |
| `paper/` | LaTeX source (`main.tex`, `references.bib`), figures, number audit |
| `app/` | Tilik Padi web app (code, tests, docs); models and screenshots are not included here |
| `KEPUTUSAN_PROYEK.md` | dated log of every research decision (newest first, in Indonesian) |

Not included: datasets (`data/raw`, images and labels), training runs (`runs/`), model weights (`*.pt`, `*.onnx`), virtual environments,
and dataset photos or photo montages.

## Environment

Tested on Windows 10, Python 3.12.3, NVIDIA RTX 3050 Ti Laptop GPU (4 GB), CUDA 13.0 build of PyTorch, Ultralytics 8.4.171.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.lock --extra-index-url https://download.pytorch.org/whl/cu130
$env:YOLO_AUTOINSTALL = "false"     # keep the environment identical to the locked versions
```

`requirements-ekspor.lock` lists the extra packages used only for ONNX export and checks. The app has its own `app/requirements.txt`.

## Datasets (download from the original sources)

The images are not redistributed. Download each source and place it as follows (paths used by `configs/classes_v6.yaml`):

| ID | Dataset | Source | Place in |
|---|---|---|---|
| ds4 | RiceLeafDiseaseBD, **version 3** (main source; blast, brown spot, healthy) | Mendeley Data, doi:10.17632/86s4jzj2m4.3, CC BY 4.0 | `data/raw/ds4/images_labels/RiceLeafDiseaseBD/` (zip content) and `data/raw/ds4/docs/` |
| ds1 | RICE Leaf Diseases (YOLO format; all BLB images, extra blast and brown spot for training) | Kaggle `yusufmurtaza01/rice-leaf-diseases`, CC0 | `data/raw/ds1/` |
| ds2 | Rice leaf bacterial and fungal disease dataset (healthy images only, as background) | Mendeley Data, doi:10.17632/hx6f852hw4.2 (also on Kaggle `soni535/rice-leaf-bacterial-and-fungal-disease`) | `data/raw/ds2/` |
| ds3 | Leaf rice disease, Southeast Sulawesi, Indonesia (external test only) | Kaggle `tedisetiady/leaf-rice-disease-indonesia` | `data/raw/ds3/{blast,blight,tungro}/` |

```powershell
kaggle datasets download -d yusufmurtaza01/rice-leaf-diseases -p data/raw/ds1 --unzip
kaggle datasets download -d tedisetiady/leaf-rice-disease-indonesia -p data/raw/ds3 --unzip
```

Check each dataset page for its license before use. Note for ds4: the class-ID table in the dataset README swaps blast and brown spot
relative to the label files; this project maps classes by folder name (see `paper/CEK_VERSI_DS4.md`).

## Rebuilding the datasets

The committed manifests (`data/processed/prov_v7/manifest.csv`, `prov_v7_randsplit`, `prov_v9`) are the reference. Every image is listed
with its split, duplicate group, and original path, so a rebuilt dataset can be checked file by file. The datasets were built in stages.
Details and checks are in each script's docstring and in `KEPUTUSAN_PROYEK.md`.

1. `scripts/prepare_dataset.py --config configs/classes_v6.yaml`: collect sources, map classes by name, group near-duplicates (pHash,
   DINOv2, SIFT, burst window), split 70:20:10 by group plus the held-out session.
2. `scripts/perbaiki_orientasi_ds4_jpegtran.py`, then `scripts/bangun_prov_v7.py bangun` / `bekukan`: lossless rotation of 931 ds4
   images (requires libjpeg-turbo `jpegtran`).
3. `scripts/bangun_randsplit_v7.py`: conventional random split for the leakage comparison.
4. `scripts/buat_valsel.py`: clean validation subset (869 images, no provisional BLB) for checkpoint selection.
5. `scripts/v9_latar/` (SAM 2.1 masks), then `scripts/bangun_prov_v9.py`: leaf-only dataset.

The `data.yaml` files contain the absolute path of the original machine (`path:` line); change it to your own location.

## Training and evaluation

```powershell
# all 12 final runs (YOLOv8n/s, YOLO11n/s x seeds 0-2) and the 2 random-split runs, 100 epochs, imgsz 640, batch 8
.venv\Scripts\python scripts\antrean_v7.py
# or a single run
.venv\Scripts\python scripts\train.py --data data/derived/prov_v7_valsel/data_valsel.yaml --model yolov8s.pt --epochs 100 --seed 0 --batch 8 --patience 0 --nama v7_final_prov_yolov8s_s0

# lock diagnosis thresholds on validation, then evaluate each test set once, then aggregate with group bootstrap
.venv\Scripts\python scripts\evaluasi_akhir_v7.py kunci
.venv\Scripts\python scripts\evaluasi_akhir_v7.py uji
.venv\Scripts\python scripts\analisis_akhir_v7.py

# background experiment (leaf-only models): same pattern
.venv\Scripts\python scripts\antrean_v9.py
.venv\Scripts\python scripts\evaluasi_akhir_v9.py kunci
.venv\Scripts\python scripts\evaluasi_akhir_v9.py uji
.venv\Scripts\python scripts\analisis_akhir_v9.py

# paper figures and Table I
.venv\Scripts\python scripts\paper\fig_background.py
.venv\Scripts\python scripts\paper\fig_leaf_examples.py
.venv\Scripts\python scripts\paper\tabel_data_prov_v7.py
```

`evaluate.py` refuses to evaluate a test split without `--izinkan-test` and does not overwrite existing results. Thresholds are locked on
validation before any test set is opened. The reported numbers are in `reports/final_eval_v7/` and `reports/final_eval_v9/`.

## Citation

```bibtex
@unpublished{hermawan2026yolo,
  author = {Hermawan, Tristan Satria Perdana and Gunawan, Putu Harry},
  title  = {{YOLOv8} Versus {YOLO11} for Rice Leaf Disease Detection: Leakage-Free Evaluation and Background Dependence},
  note   = {Manuscript, School of Computing, Telkom University},
  year   = {2026}
}
```

Please also cite the datasets above (BibTeX entries in `paper/references.bib`).

## License

The code in this repository is released under the **GNU Affero General Public License v3.0** (`LICENSE`), following Ultralytics YOLOv8 and
YOLO11 (AGPL-3.0), on which the models are based. Datasets keep their own licenses and are not part of this repository.
