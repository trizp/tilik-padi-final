#!/usr/bin/env python3
"""
blb_audit.py - Menyiapkan bahan audit label BLB ds1 untuk DINILAI MANUSIA (skrip ini tidak menyimpulkan apa pun).

Mengambil N gambar BLB acak dari dataset hasil prepare_dataset.py, menggambar ulang label DI ATAS gambar asli
ds1 dan menyimpannya ke reports/blb_audit/ beserta index.csv dan index.html (galeri).

Legenda gambar:
  kotak merah   = bbox yang dipakai di dataset (box asli 5 kolom, atau bbox pembatas dari poligon)
  garis biru    = poligon asli ds1 (jika baris label berformat poligon)
Setiap gambar diberi nomor di pojok kiri atas; nomor yang sama ada di index.csv.

Pemakaian:  python scripts/blb_audit.py [--n 50] [--seed 123]
"""
import argparse
import csv
import html
import random
import sys
from pathlib import Path

import yaml
from PIL import Image, ImageDraw
from jalur import jalur_keluaran  # noqa: E402  (pengaman jalur keluaran)

sys.path.insert(0, str(Path(__file__).parent))
from inventory import find_yolo_label, load_yaml_names  # noqa: E402
from prepare_dataset import ROOT  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/classes.yaml")
    ap.add_argument("--class-name", default="bacterial_leaf_blight")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", default="reports/blb_audit")
    ap.add_argument("--max-side", type=int, default=900)
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    D = ROOT / cfg["output_dir"]
    man = [r for r in csv.DictReader(open(D / "manifest.csv", encoding="utf-8")) if r["class"] == args.class_name]
    sel = random.Random(args.seed).sample(man, args.n)
    names, _ = load_yaml_names(ROOT / "data/raw/ds1")
    out = jalur_keluaran(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.jpg"):
        old.unlink()

    rows = []
    for k, r in enumerate(sel, 1):
        src = ROOT / r["orig_path"]
        lab = find_yolo_label(src)
        im = Image.open(src).convert("RGB")
        W, H = im.size
        d = ImageDraw.Draw(im)
        lw = max(2, round(max(W, H) / 250))
        n_box = n_poly = 0
        areas = []
        for ln in lab.read_text().splitlines():
            t = ln.split()
            if not t or names[int(float(t[0]))].replace("Rice__", "").lower() != "bacterialleafblight":
                continue
            v = list(map(float, t[1:]))
            if len(t) == 5:
                x1, y1, x2, y2 = v[0] - v[2] / 2, v[1] - v[3] / 2, v[0] + v[2] / 2, v[1] + v[3] / 2
                n_box += 1
            else:
                xs, ys = v[0::2], v[1::2]
                pts = [(x * W, y * H) for x, y in zip(xs, ys)]
                d.line(pts + [pts[0]], fill=(0, 120, 255), width=lw)
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                n_poly += 1
            d.rectangle([x1 * W, y1 * H, x2 * W, y2 * H], outline=(255, 0, 0), width=lw)
            areas.append((x2 - x1) * (y2 - y1) * 100)
        s = args.max_side / max(W, H)
        if s < 1:
            im = im.resize((round(W * s), round(H * s)), Image.LANCZOS)
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 38, 22], fill=(255, 255, 255))
        d.text((4, 5), f"#{k:02d}", fill=(0, 0, 0))
        fn = f"{k:02d}_{r['split']}.jpg"
        im.save(out / fn, quality=88)
        rows.append(dict(no=k, file=fn, split=r["split"], n_box_asli=n_box, n_box_dari_poligon=n_poly,
                         luas_box_pct="/".join(f"{a:.1f}" for a in areas), aug_possible=r["aug_possible"],
                         aug_reasons=r["aug_reasons"], orig_path=r["orig_path"]))
    with open(out / "index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    cells = "".join(
        f'<figure><img src="{x["file"]}" loading="lazy"><figcaption>#{x["no"]:02d} · {x["split"]} · box {x["n_box_asli"]}'
        f' / poligon {x["n_box_dari_poligon"]} · luas % {html.escape(x["luas_box_pct"])}<br>'
        f'<small>{html.escape(x["orig_path"].split("/")[-1])}</small></figcaption></figure>' for x in rows)
    (out / "index.html").write_text(f"""<!doctype html><meta charset="utf-8"><title>Audit label BLB ds1</title>
<style>body{{font:14px system-ui;margin:16px;background:#fafafa}}main{{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:12px}}
figure{{margin:0;background:#fff;border:1px solid #ddd;padding:6px}}img{{width:100%}}figcaption{{padding:4px 2px}}</style>
<h2>Audit label BLB ds1 ({args.n} gambar acak, seed {args.seed})</h2>
<p>Kotak merah = bbox di dataset. Garis biru = poligon asli ds1. Penilaian dilakukan manual; berkas ini tidak memuat kesimpulan.</p>
<main>{cells}</main>""", encoding="utf-8")
    print(f"{args.n} gambar -> {args.out}/ (index.html, index.csv)")


if __name__ == "__main__":
    main()
