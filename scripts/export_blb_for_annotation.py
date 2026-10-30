#!/usr/bin/env python3
"""
export_blb_for_annotation.py - Ekspor gambar Bacterial Leaf Blight (BLB) dari ds2 untuk dianotasi ulang
secara lesion-level di Roboflow atau CVAT.

Yang dilakukan:
  1. Ambil gambar folder "Bacterial Leaf Blight" ds2. Default hanya cabang Original (180 gambar);
     cabang Augmented adalah salinan hasil augmentasi sehingga tidak diikutkan (pakai --branch all
     hanya jika Anda sadar risikonya: saudara augmentasi bisa terpisah antar split).
  2. Buang duplikat dekat di dalam himpunan ini (phash 8 varian dihedral, ambang sama dengan config).
  3. Perkecil sisi terpanjang ke --max-side (default 1600) dan simpan sebagai blb_ds2_0001.jpg dst.
     Koordinat box YOLO ternormalisasi, jadi anotasi di resolusi ini tetap valid untuk ukuran lain.
  4. Tulis manifest.csv (id <-> file asli), classes.txt, panduan_anotasi.md, dan images.zip
     (zip berisi gambar saja; dapat diunggah langsung ke Roboflow maupun CVAT).

Pemakaian:
  python scripts/export_blb_for_annotation.py
  python scripts/export_blb_for_annotation.py --branch all --max-side 2048
"""
import argparse
import csv
import hashlib
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from jalur import jalur_keluaran  # noqa: E402  (pengaman jalur keluaran)

sys.path.insert(0, str(Path(__file__).parent))
from inventory import IMG_EXT  # noqa: E402
from prepare_dataset import ROOT, _hash_one, norm, phash_pairs, union_groups  # noqa: E402

GUIDE = """# Panduan anotasi BLB (DRAF USULAN - WAJIB disetujui pakar penyakit padi dan pembimbing)

**Unit anotasi: area gejala per segmen daun** (konsisten dengan label BLB ds1). Ini BERBEDA dari unit bercak
(lesion-level) pada Blast dan Brown spot di ds4. Akibatnya kelas BLB punya skala box jauh lebih besar
(di ds1: median sekitar 21% luas gambar), sehingga metrik per kelas harus dibaca terpisah.
Protokol ds4 (Annotation_Protocol.pdf) tidak memuat aturan BLB; aturan di bawah adalah USULAN, bukan salinan protokol.

## Aturan
1. Kelas tunggal: `bacterial_leaf_blight` (ID 0 di berkas ini; ID final diatur saat penggabungan dataset).
2. Satu box per **segmen daun yang bergejala BLB**: guratan/area kuning-oranye sampai cokelat keputihan yang
   merambat dari tepi atau ujung daun, tepi bergelombang, kadang tampak basah. Box mencakup seluruh bagian
   bergejala pada segmen itu (termasuk bagian yang menguning atau mengering).
3. **Box harus berlabuh pada jaringan daun bergejala.** Tepi box ditentukan oleh segmen daun bergejala,
   BUKAN oleh objek latar. Tidak boleh ada box yang berisi rumput, tanah, tangan, pakaian, atau latar tanpa
   gejala BLB. (Untuk daun diagonal, persegi pembatas pasti memuat sebagian latar; itu wajar. Yang dilarang
   adalah box yang letaknya meleset dari daun bergejala.)
4. Dua helai daun terpisah = dua box. Gejala pada satu helai yang terputus jelas oleh jaringan sehat yang
   panjang: usulan satu box per bagian (ambang "panjang" perlu ditetapkan pakar; catat keputusannya).
5. Jangan menandai gejala penyakit lain (bercak cokelat bulat = brown spot; bercak belah ketupat = blast).
   Jika ragu, tulis nama berkas di `catatan_ragu.txt` (satu per baris) dan jangan buat box.
6. Gambar tanpa gejala BLB yang jelas: tanpa box, catat di `catatan_ragu.txt`.
7. QC: satu orang untuk lintasan pertama, 10% sampel ditinjau orang kedua (ikuti alur QC protokol ds4);
   simpan IoU antar-penganotasi untuk bagian metodologi paper.

## Mengunggah
- **Roboflow**: proyek Object Detection, kelas `bacterial_leaf_blight`, unggah `images.zip` (atau folder `images/`).
  Untuk pra-label (scripts/pseudo_label.py), unggah `roboflow_prelabeled.zip` (format YOLOv8).
- **CVAT**: buat task, label `bacterial_leaf_blight` (Rectangle), unggah `images.zip`. Untuk pra-label:
  Upload annotations -> format "YOLO 1.1" -> `cvat_prelabeled.zip`.
- Setelah selesai, ekspor YOLOv8 (Roboflow) / YOLO 1.1 (CVAT) ke `data/annotation/blb_ds2/annotated/`.
- Impor ke Roboflow/CVAT BELUM diuji dari sisi kami (lihat `import_test/README.md`); cek jumlah gambar dan box
  setelah mengunggah sebelum mulai mengoreksi.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="data/raw/ds2")
    ap.add_argument("--folder", default="bacterial_leaf_blight", help="nama folder kelas (dinormalisasi)")
    ap.add_argument("--branch", choices=["original", "augmented", "all"], default="original")
    ap.add_argument("--max-side", type=int, default=1600)
    ap.add_argument("--out", default="data/annotation/blb_ds2")
    ap.add_argument("--config", default="configs/classes.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    thr, hs = cfg["dup_threshold"], cfg["hash_size"]
    out = jalur_keluaran(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    imgs = []
    for p in sorted((ROOT / args.src).rglob("*")):
        if p.suffix.lower() not in IMG_EXT or norm(p.parent.name) != args.folder:
            continue
        posix = p.as_posix().lower()
        branch = "augmented" if "/augmented" in posix else ("original" if "/original" in posix else "?")
        if args.branch in ("all", branch):
            imgs.append((p, branch))
    if not imgs:
        raise SystemExit("Tidak ada gambar yang cocok.")
    print(f"ditemukan {len(imgs)} gambar ({args.branch})")

    # duplikat dekat (8 varian dihedral)
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(_hash_one, [(str(p), hs) for p, _ in imgs], chunksize=8))
    keep = [i for i, r in enumerate(res) if r is not None]
    var = np.array([res[i] for i in keep])
    groups = union_groups(len(var), [phash_pairs(var, thr)])
    seen, dup_of = {}, {}
    for k, g in zip(keep, groups):
        if g in seen:
            dup_of[k] = seen[g]
        else:
            seen[g] = k
    kept = [i for i in keep if i not in dup_of]
    print(f"duplikat dekat dibuang: {len(dup_of)} | tidak terbaca: {len(imgs) - len(keep)} | diekspor: {len(kept)}")

    rows = []
    for n, i in enumerate(kept, 1):
        p, branch = imgs[i]
        eid = f"blb_ds2_{n:04d}"
        with Image.open(p) as im:
            im = im.convert("RGB")
            w0, h0 = im.size
            s = args.max_side / max(im.size)
            if s < 1:
                im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
            im.save(out / "images" / f"{eid}.jpg", quality=95)
            rows.append([eid, p.relative_to(ROOT).as_posix(), branch, f"{w0}x{h0}", f"{im.width}x{im.height}"])
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "orig_path", "branch", "orig_size", "exported_size"])
        w.writerows(rows)
    if dup_of:
        with open(out / "dibuang_duplikat.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["dibuang", "duplikat_dari"])
            for k, v in dup_of.items():
                w.writerow([imgs[k][0].relative_to(ROOT).as_posix(), imgs[v][0].relative_to(ROOT).as_posix()])
    (out / "classes.txt").write_text("bacterial_leaf_blight\n", encoding="utf-8")
    (out / "panduan_anotasi.md").write_text(GUIDE, encoding="utf-8")
    with zipfile.ZipFile(out / "images.zip", "w", zipfile.ZIP_STORED) as z:    # JPEG sudah terkompres
        for r in rows:
            z.write(out / "images" / f"{r[0]}.jpg", f"{r[0]}.jpg")
    mb = (out / "images.zip").stat().st_size / 1e6
    print(f"selesai -> {args.out}  (images/, images.zip {mb:.0f} MB, manifest.csv, classes.txt, panduan_anotasi.md)")


if __name__ == "__main__":
    main()
