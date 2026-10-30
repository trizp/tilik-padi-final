#!/usr/bin/env python3
"""
pseudo_label.py - Pra-label (pseudo-label) gambar BLB dari model awal agar koreksi manual lebih cepat.

Keluaran di --out (default data/annotation/blb_ds2/prelabel):
  labels/<id>.txt        : YOLO satu kelas (ID 0 = bacterial_leaf_blight), hanya gambar yang punya prediksi
  prelabel_scores.csv    : per gambar: jumlah prediksi, conf maksimum, conf rata-rata
  no_prediction.txt      : gambar tanpa prediksi (tidak dibuatkan file label)
  roboflow_prelabeled.zip: format YOLOv8 (data.yaml + train/images + train/labels)
  cvat_prelabeled.zip    : format YOLO 1.1 (obj.names, obj.data, train.txt, obj_train_data/)

PENTING:
  * Pra-label hanyalah titik awal; SELURUH box harus diperiksa manusia. Model awal yang dilatih dari
    BLB ds1 (anotasi tingkat wilayah) cenderung menghasilkan box besar -> perkecil ke tingkat lesi.
  * Gambar tanpa prediksi: di ZIP Roboflow TIDAK dibuatkan file label (supaya tidak terbaca sebagai "tanpa objek";
    perilaku Roboflow untuk label kosong belum diuji); di ZIP CVAT dibuatkan file .txt KOSONG karena importer
    Datumaro menolak gambar tanpa .txt (ditemukan saat uji).
  * --any-class hanya untuk uji mekanik (mis. bobot COCO); jangan dipakai untuk anotasi sungguhan.

Pemakaian:
  python scripts/pseudo_label.py --weights runs/prov_.../weights/best.pt
  python scripts/pseudo_label.py --weights best.pt --conf 0.15 --imgsz 1280
"""
import argparse
import csv
import shutil
import sys
import zipfile
from pathlib import Path

import yaml
from jalur import jalur_keluaran  # noqa: E402  (pengaman jalur keluaran)

sys.path.insert(0, str(Path(__file__).parent))
from prepare_dataset import ROOT  # noqa: E402

CLASS = "bacterial_leaf_blight"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", default="data/annotation/blb_ds2/images")
    ap.add_argument("--out", default="data/annotation/blb_ds2/prelabel")
    ap.add_argument("--class-name", default=CLASS, help="nama kelas pada model yang diambil")
    ap.add_argument("--any-class", action="store_true", help="UJI MEKANIK: terima semua kelas prediksi")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--max-det", type=int, default=100)
    ap.add_argument("--device", default="0")
    ap.add_argument("--limit", type=int, default=0, help="batasi jumlah gambar (uji)")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)
    names = model.names
    if args.any_class:
        want = set(names)
    else:
        want = {i for i, n in names.items() if n == args.class_name}
        if not want:
            raise SystemExit(f"Kelas '{args.class_name}' tidak ada pada model. Kelas model: {names}")

    img_dir, out = ROOT / args.images, jalur_keluaran(args.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "labels").mkdir(parents=True)
    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if args.limit:
        imgs = imgs[:args.limit]

    rows, none = [], []
    for r in model.predict([str(p) for p in imgs], conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                           max_det=args.max_det, device=args.device, stream=True, verbose=False):
        p = Path(r.path)
        sel = [(b.xywhn[0].tolist(), float(b.conf)) for b in r.boxes if int(b.cls) in want]
        if not sel:
            none.append(p.name)
            rows.append([p.name, 0, "", ""])
            continue
        (out / "labels" / (p.stem + ".txt")).write_text(
            "\n".join(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for (x, y, w, h), _ in sel) + "\n")
        confs = [c for _, c in sel]
        rows.append([p.name, len(sel), f"{max(confs):.3f}", f"{sum(confs) / len(confs):.3f}"])

    with open(out / "prelabel_scores.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "n_pred", "conf_max", "conf_mean"])
        w.writerows(rows)
    (out / "no_prediction.txt").write_text("\n".join(none) + ("\n" if none else ""))

    # --- Roboflow (YOLOv8): data.yaml + train/images + train/labels
    with zipfile.ZipFile(out / "roboflow_prelabeled.zip", "w", zipfile.ZIP_STORED) as z:
        z.writestr("data.yaml", yaml.safe_dump({"train": "train/images", "val": "train/images",
                                                "nc": 1, "names": [CLASS]}))
        for p in imgs:
            z.write(p, f"train/images/{p.name}")
            lb = out / "labels" / (p.stem + ".txt")
            if lb.exists():
                z.write(lb, f"train/labels/{p.stem}.txt")
    # --- CVAT (YOLO 1.1): obj.names, obj.data, train.txt, obj_train_data/
    with zipfile.ZipFile(out / "cvat_prelabeled.zip", "w", zipfile.ZIP_STORED) as z:
        z.writestr("obj.names", CLASS + "\n")
        z.writestr("obj.data", "classes = 1\ntrain = train.txt\nnames = obj.names\nbackup = backup/\n")
        z.writestr("train.txt", "".join(f"data/obj_train_data/{p.name}\n" for p in imgs))
        for p in imgs:
            z.write(p, f"obj_train_data/{p.name}")
            lb = out / "labels" / (p.stem + ".txt")
            # Datumaro (importer YOLO 1.1 yang dipakai CVAT) GAGAL jika sebuah gambar tidak punya .txt
            # -> untuk gambar tanpa prediksi tulis file kosong (artinya: belum ada anotasi di CVAT).
            if lb.exists():
                z.write(lb, f"obj_train_data/{p.stem}.txt")
            else:
                z.writestr(f"obj_train_data/{p.stem}.txt", "")

    n_pred = sum(r[1] for r in rows)
    print(f"{len(imgs)} gambar | dengan prediksi: {len(imgs) - len(none)} | tanpa prediksi: {len(none)} | total box: {n_pred}")
    print(f"keluaran: {args.out}")


if __name__ == "__main__":
    main()
