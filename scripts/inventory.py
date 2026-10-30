#!/usr/bin/env python3
"""
inventory.py - Audit dataset penyakit padi sebelum dipakai untuk YOLO.

Fungsi:
  - Mendeteksi format tiap dataset: YOLO (txt), Pascal VOC (xml), COCO (json),
    atau KLASIFIKASI (folder per kelas, tanpa bounding box).
  - Menghitung jumlah gambar, jumlah box per kelas, gambar per kelas,
    gambar tanpa label (background), dan gambar dengan >= 2 kelas (multi-penyakit).
  - Menyimpan laporan ke inventory_report.json

Pemakaian:
  python scripts/inventory.py data/raw/ds1 data/raw/ds2 data/raw/ds3
  python scripts/inventory.py data/raw/ds1 --check-images
"""
import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from jalur import jalur_keluaran  # noqa: E402  (pengaman jalur keluaran)

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_NAMES = {"train", "training", "valid", "val", "validation", "test", "testing"}


def load_yaml_names(root: Path):
    """Baca daftar nama kelas dari data.yaml (jika ada)."""
    for yml in list(root.rglob("data.yaml")) + list(root.rglob("*.yaml")):
        try:
            text = yml.read_text(encoding="utf-8", errors="ignore")
            try:
                import yaml
                cfg = yaml.safe_load(text)
                names = cfg.get("names")
            except Exception:
                names = None
                m = re.search(r"names:\s*\[(.*?)\]", text, re.S)
                if m:
                    names = [n.strip().strip("'\"") for n in m.group(1).split(",")]
            if isinstance(names, dict):
                names = [names[k] for k in sorted(names)]
            if names:
                return list(names), yml
        except Exception:
            continue
    return None, None


def split_of(path: Path):
    for part in path.parts:
        if part.lower() in SPLIT_NAMES:
            return part.lower()
    return "-"


def find_yolo_label(img: Path):
    """Cari file label YOLO untuk sebuah gambar."""
    cands = [img.with_suffix(".txt")]
    parts = list(img.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() == "images":
            p = parts.copy()
            p[i] = "labels"
            cands.append(Path(*p).with_suffix(".txt"))
            break
    for c in cands:
        if c.exists():
            return c
    return None


def parse_yolo(label: Path):
    ids = []
    for line in label.read_text(errors="ignore").splitlines():
        tok = line.split()
        if len(tok) >= 5:
            try:
                ids.append(int(float(tok[0])))
            except ValueError:
                pass
    return ids


def summarize(per_image_classes, class_names=None):
    """per_image_classes: list of list nama/ID kelas (satu list per gambar)."""
    box_count, img_count = Counter(), Counter()
    multi = background = 0
    combos = Counter()
    for cls_list in per_image_classes:
        if not cls_list:
            background += 1
            continue
        box_count.update(cls_list)
        uniq = sorted(set(cls_list), key=str)
        img_count.update(uniq)
        if len(uniq) >= 2:
            multi += 1
            combos[" + ".join(map(str, uniq))] += 1
    def nm(k):
        if class_names and isinstance(k, int) and 0 <= k < len(class_names):
            return f"{k}:{class_names[k]}"
        return str(k)
    return {
        "boxes_per_class": {nm(k): v for k, v in sorted(box_count.items(), key=lambda x: str(x[0]))},
        "images_per_class": {nm(k): v for k, v in sorted(img_count.items(), key=lambda x: str(x[0]))},
        "images_without_label_background": background,
        "images_with_2plus_classes": multi,
        "multi_class_combinations": {k: v for k, v in combos.most_common()},
    }


def audit(root: Path, check_images=False):
    imgs = [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT]
    report = {"root": str(root), "n_images": len(imgs)}
    if not imgs:
        report["format"] = "TIDAK ADA GAMBAR"
        return report

    names, yml = load_yaml_names(root)
    coco_files = [j for j in root.rglob("*.json") if "coco" in j.name.lower() or "annotation" in j.name.lower()]
    xml_files = list(root.rglob("*.xml"))

    # --- YOLO ---
    labeled = {i: find_yolo_label(i) for i in imgs}
    n_with_txt = sum(1 for v in labeled.values() if v is not None)
    if n_with_txt > 0.5 * len(imgs):
        report["format"] = "YOLO (txt) - object detection"
        report["class_names_from_yaml"] = names
        report["yaml_file"] = str(yml) if yml else None
        by_split = defaultdict(list)
        for img, lab in labeled.items():
            by_split[split_of(img.relative_to(root))].append(parse_yolo(lab) if lab else [])
        report["splits"] = {s: len(v) for s, v in by_split.items()}
        report["overall"] = summarize([c for v in by_split.values() for c in v], names)
        report["per_split"] = {s: summarize(v, names) for s, v in by_split.items()}

    # --- COCO ---
    elif coco_files:
        report["format"] = "COCO (json) - object detection"
        per_img = defaultdict(list)
        cats = {}
        for jf in coco_files:
            try:
                d = json.loads(jf.read_text(errors="ignore"))
                cats.update({c["id"]: c["name"] for c in d.get("categories", [])})
                for a in d.get("annotations", []):
                    per_img[(str(jf), a["image_id"])].append(cats.get(a["category_id"], a["category_id"]))
                for im in d.get("images", []):
                    per_img.setdefault((str(jf), im["id"]), [])
            except Exception:
                continue
        report["overall"] = summarize(list(per_img.values()))

    # --- Pascal VOC ---
    elif xml_files:
        report["format"] = "Pascal VOC (xml) - object detection"
        per_img = []
        for x in xml_files:
            try:
                tree = ET.parse(x)
                per_img.append([o.findtext("name") for o in tree.findall("object")])
            except Exception:
                continue
        report["overall"] = summarize(per_img)

    # --- Klasifikasi ---
    else:
        report["format"] = "KLASIFIKASI (folder per kelas) - TIDAK ADA bounding box"
        counts = defaultdict(Counter)
        for i in imgs:
            rel = i.relative_to(root)
            parent = i.parent.name
            counts[split_of(rel)][parent] += 1
        report["images_per_folder_per_split"] = {s: dict(c) for s, c in counts.items()}
        report["catatan"] = ("Dataset ini tidak bisa langsung dipakai untuk YOLO detection; "
                             "perlu anotasi bounding box, atau dipakai sebagai gambar Healthy/background.")

    if check_images:
        try:
            from PIL import Image
            bad, sizes = [], Counter()
            for i in imgs:
                try:
                    with Image.open(i) as im:
                        im.verify()
                    with Image.open(i) as im:
                        sizes[im.size] += 1
                except Exception:
                    bad.append(str(i))
            report["corrupt_images"] = bad[:50]
            report["n_corrupt"] = len(bad)
            report["top_image_sizes"] = {f"{w}x{h}": c for (w, h), c in sizes.most_common(5)}
        except ImportError:
            report["check_images"] = "Pillow belum terpasang (pip install pillow)"
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", help="folder dataset hasil unzip")
    ap.add_argument("--check-images", action="store_true", help="cek gambar rusak & ukuran")
    ap.add_argument("--out", default="inventory_report.json")
    args = ap.parse_args()

    reports = []
    for r in args.roots:
        root = Path(r)
        if not root.exists():
            print(f"[!] Folder tidak ditemukan: {root}", file=sys.stderr)
            continue
        rep = audit(root, args.check_images)
        reports.append(rep)
        print("=" * 70)
        print(json.dumps(rep, indent=2, ensure_ascii=False))
    jalur_keluaran(args.out).write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nLaporan disimpan ke {args.out}")


if __name__ == "__main__":
    main()
