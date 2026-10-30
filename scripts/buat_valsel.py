#!/usr/bin/env python3
"""
buat_valsel.py - Buat berkas turunan data_valsel.yaml: val untuk PEMILIHAN CHECKPOINT (best.pt) tanpa gambar BLB provisional.

Aturan (keputusan pemilik skripsi, 2026-10-04):
  * best.pt dipilih HANYA dari skor val bersih = val tanpa gambar bersumber 'ds1_provisional' (BLB provisional).
  * Dataset beku TIDAK diubah: gambar+label val bersih DISALIN (bukan hardlink) ke data/derived/<dataset>_valsel/,
    sehingga cache label Ultralytics (labels/val.cache) juga tertulis di folder turunan, bukan di dataset beku.
  * train tetap menunjuk ke images/train dataset beku (jalur absolut).
  * mAP Ultralytics hanya dihitung untuk kelas yang punya label di val (np.unique(target_cls)); skrip ini memastikan
    val bersih tidak memuat label kelas provisional, sehingga mAP pemilihan = rata-rata kelas berlabel saja.
  * BLB val provisional tetap dilaporkan terpisah oleh evaluate.py (subset 'blb_provisional').

Pemakaian:
  python scripts/buat_valsel.py --data data/processed/prov_v6/data.yaml
Keluaran: data/derived/prov_v6_valsel/{data_valsel.yaml, images/val, labels/val, daftar_val_bersih.txt, valsel_meta.json}
"""
import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import yaml
from jalur import jalur_keluaran  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ATURAN = ("best.pt dipilih HANYA dari skor val bersih (fitness Ultralytics = mAP50-95 rata-rata kelas berlabel) pada val "
          "tanpa gambar sumber provisional {src}; dataset beku tidak diubah (salinan di folder turunan); BLB val provisional "
          "dilaporkan terpisah oleh evaluate.py.")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/processed/prov_v6/data.yaml")
    ap.add_argument("--out", default=None, help="baku: data/derived/<dataset>_valsel")
    a = ap.parse_args()

    data_p = (ROOT / a.data).resolve()
    dy = yaml.safe_load(data_p.read_text(encoding="utf-8"))
    droot = Path(dy["path"])
    sha = sha256(droot / "manifest.csv")
    if sha != dy.get("manifest_sha256"):
        sys.exit(f"[GAGAL] manifest dataset berubah: {sha} != {dy.get('manifest_sha256')}")
    prov_src = set(dy.get("provisional_sources") or [])
    if not prov_src:
        sys.exit("[GAGAL] data.yaml tidak punya provisional_sources; val bersih = val asli, tidak perlu berkas turunan.")

    rows = [r for r in csv.DictReader(open(droot / "manifest.csv", encoding="utf-8")) if r["split"] == "val"]
    bersih = [r for r in rows if r["source"] not in prov_src]
    dibuang = [r for r in rows if r["source"] in prov_src]
    names = {int(k): v for k, v in dy["names"].items()}
    kelas_prov = {r["class"] for r in dibuang}

    out = jalur_keluaran(a.out or f"data/derived/{dy['dataset_name']}_valsel")
    if droot.resolve() == out or droot.resolve() in out.parents:
        sys.exit("[GAGAL] folder turunan tidak boleh di dalam dataset beku")
    for sub in ("images/val", "labels/val"):
        if (out / sub).exists():
            shutil.rmtree(out / sub)
        (out / sub).mkdir(parents=True)
    for c in out.glob("labels/*.cache"):
        c.unlink()

    inst, nimg, beda = Counter(), Counter(), []
    for r in bersih:
        img = droot / "images" / "val" / r["file"]
        lab = droot / "labels" / "val" / (Path(r["file"]).stem + ".txt")
        di, dl = out / "images" / "val" / img.name, out / "labels" / "val" / lab.name
        shutil.copy2(img, di)
        if lab.exists():
            shutil.copy2(lab, dl)
            cls = [int(float(ln.split()[0])) for ln in lab.read_text().splitlines() if ln.split()]
            inst.update(cls)
            nimg.update(set(cls))
            if sha256(dl) != sha256(lab):
                beda.append(lab.name)
        if sha256(di) != sha256(img):
            beda.append(img.name)
    if beda:
        sys.exit(f"[GAGAL] salinan tidak identik: {beda[:5]}")
    kelas_berlabel = sorted(names[c] for c in inst)
    bocor = sorted(set(kelas_berlabel) & kelas_prov)
    if bocor:
        sys.exit(f"[GAGAL] val bersih masih memuat label kelas provisional {bocor}")

    (out / "daftar_val_bersih.txt").write_text("\n".join(sorted(r["file"] for r in bersih)) + "\n", encoding="utf-8")
    aturan = ATURAN.format(src=sorted(prov_src))
    yml = {"path": str(out).replace("\\", "/"), "train": str(droot / "images" / "train").replace("\\", "/"), "val": "images/val",
           "names": dy["names"],
           # metadata disalin dari induk agar train.py tetap mencatat dataset beku yang sama
           "dataset_name": dy["dataset_name"], "run_prefix": dy.get("run_prefix", ""), "provisional": dy.get("provisional"),
           "provisional_sources": sorted(prov_src), "manifest_sha256": dy["manifest_sha256"], "frozen": dy.get("frozen"),
           "induk_data": str(data_p).replace("\\", "/"), "aturan_pemilihan_checkpoint": aturan,
           "val_dikecualikan_sumber": sorted(prov_src), "val_kelas_berlabel": kelas_berlabel}
    (out / "data_valsel.yaml").write_text(yaml.safe_dump(yml, allow_unicode=True, sort_keys=False), encoding="utf-8")
    meta = {"dibuat": datetime.now().isoformat(timespec="seconds"), "induk_data": str(data_p), "manifest_sha256_induk": sha,
            "aturan": aturan, "n_val_induk": len(rows), "n_val_bersih": len(bersih), "n_val_dikecualikan": len(dibuang),
            "dikecualikan_per_sumber_kelas": Counter(f"{r['source']}|{r['class']}" for r in dibuang),
            "val_bersih_per_sumber_kelas": Counter(f"{r['source']}|{r['class']}" for r in bersih),
            "val_bersih_instance_per_kelas": {names[c]: n for c, n in sorted(inst.items())},
            "val_bersih_gambar_per_kelas": {names[c]: n for c, n in sorted(nimg.items())},
            "val_bersih_gambar_background": sum(1 for r in bersih if r["class"] == "background"),
            "kelas_tanpa_label_di_val_bersih": sorted(set(names.values()) - set(kelas_berlabel)),
            "sha256_daftar": sha256(out / "daftar_val_bersih.txt")}
    (out / "valsel_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=1, ensure_ascii=False))
    print(f"[OK] {out / 'data_valsel.yaml'}")


if __name__ == "__main__":
    main()
