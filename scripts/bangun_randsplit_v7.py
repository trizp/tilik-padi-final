#!/usr/bin/env python3
"""
bangun_randsplit_v7.py - prov_v7_randsplit = pembagian prov_v6_randsplit (SAMA PERSIS, per berkas) dengan berkas dari prov_v7
(931 foto ds4 orientasi diperbaiki). Pemeriksaan: ID+split identik dengan prov_v6_randsplit, label identik, hash berubah hanya
pada foto ds4 hasil jpegtran, setiap berkas ds4 byte-identik dengan prov_v7 (yang lolos audit orientasi), tanpa tag EXIF.
Dibekukan otomatis bila semua lolos. (verify_dataset TIDAK berlaku: split acak sengaja bocor.)
Keluaran: data/processed/prov_v7_randsplit/, reports/prov_v7_randsplit/pemeriksaan.json
"""
import csv
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
R6 = ROOT / "data/processed/prov_v6_randsplit"
V7 = ROOT / "data/processed/prov_v7"
OUT = ROOT / "data/processed/prov_v7_randsplit"
REP = ROOT / "reports/prov_v7_randsplit"
SPL = ("train", "val", "test", "test_new_session")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tautkan(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    if OUT.exists():
        sys.exit(f"[DITOLAK] {OUT} sudah ada")
    dy6 = yaml.safe_load((R6 / "data.yaml").read_text(encoding="utf-8"))
    if sha(R6 / "manifest.csv") != dy6["manifest_sha256"]:
        sys.exit("[GAGAL] manifest prov_v6_randsplit berubah")
    dv7 = yaml.safe_load((V7 / "data.yaml").read_text(encoding="utf-8"))
    if not dv7.get("frozen") or sha(V7 / "manifest.csv") != dv7["manifest_sha256"]:
        sys.exit("[GAGAL] prov_v7 belum beku / manifest berubah")
    perb = {r["file"]: r["perbaikan_orientasi"] for r in csv.DictReader(open(V7 / "manifest.csv", encoding="utf-8"))}
    man = list(csv.DictReader(open(R6 / "manifest.csv", encoding="utf-8")))
    kol = list(man[0].keys()) + ["perbaikan_orientasi"]
    for sp in SPL:
        (OUT / "images" / sp).mkdir(parents=True)
        (OUT / "labels" / sp).mkdir(parents=True)
    for r in man:
        asal = r["split_prov_v6"]
        tautkan(V7 / "images" / asal / r["file"], OUT / "images" / r["split"] / r["file"])
        lab = V7 / "labels" / asal / (Path(r["file"]).stem + ".txt")
        if lab.exists():
            tautkan(lab, OUT / "labels" / r["split"] / lab.name)
        r["perbaikan_orientasi"] = perb[r["file"]]
    with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=kol)
        w.writeheader()
        w.writerows(man)

    # ---- pemeriksaan
    m7 = list(csv.DictReader(open(OUT / "manifest.csv", encoding="utf-8")))
    id6 = {(r["split"], r["file"], r["split_prov_v6"], r["group"], r["class"]) for r in csv.DictReader(open(R6 / "manifest.csv", encoding="utf-8"))}
    id7 = {(r["split"], r["file"], r["split_prov_v6"], r["group"], r["class"]) for r in m7}
    berubah, luar, label_beda, beda_v7, tag = [], [], 0, 0, 0
    for r in m7:
        sp, f = r["split"], r["file"]
        a, b = R6 / "images" / sp / f, OUT / "images" / sp / f
        if sha(a) != sha(b):
            berubah.append(f)
            if perb[f] == "-":
                luar.append(f)
        if r["source"] == "ds4":
            beda_v7 += sha(b) != sha(V7 / "images" / r["split_prov_v6"] / f)
            tag += Image.open(b).getexif().get(274) not in (None, 1)
        l6, l7 = R6 / "labels" / sp / (Path(f).stem + ".txt"), OUT / "labels" / sp / (Path(f).stem + ".txt")
        label_beda += l6.exists() != l7.exists() or (l6.exists() and sha(l6) != sha(l7))
    n_jpegtran = sum(1 for r in m7 if perb[r["file"]] != "-")
    disk = {sp: {p.name for p in (OUT / "images" / sp).iterdir()} for sp in SPL}
    hasil = {"id_split_identik_dengan_prov_v6_randsplit": id6 == id7, "selisih_id_split": len(id6 ^ id7),
             "jumlah_per_split": {sp: len(disk[sp]) for sp in SPL},
             "disk_sama_dengan_manifest": all(disk[sp] == {r["file"] for r in m7 if r["split"] == sp} for sp in SPL),
             "foto_jpegtran_dalam_randsplit": n_jpegtran, "gambar_hash_berubah": len(berubah), "berubah_di_luar_jpegtran": len(luar),
             "label_berbeda": label_beda, "ds4_tidak_identik_dengan_prov_v7": beda_v7, "ds4_bertag_exif_orientasi": tag}
    hasil["LOLOS"] = (hasil["id_split_identik_dengan_prov_v6_randsplit"] and hasil["selisih_id_split"] == 0 and hasil["disk_sama_dengan_manifest"]
                      and len(berubah) == n_jpegtran == 931 and not luar and label_beda == 0 and beda_v7 == 0 and tag == 0)
    REP.mkdir(parents=True, exist_ok=True)
    (REP / "pemeriksaan.json").write_text(json.dumps(hasil, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(hasil, indent=1, ensure_ascii=False))
    if not hasil["LOLOS"]:
        sys.exit(1)
    now = datetime.now().isoformat(timespec="seconds")
    msha = sha(OUT / "manifest.csv")
    dy = dict(dy6, path=str(OUT).replace("\\", "/"), dataset_name="prov_v7_randsplit", run_prefix="v7_randsplit_", induk_dataset="prov_v7",
              induk_manifest_sha256=dv7["manifest_sha256"], pembagian_dari="prov_v6_randsplit (manifest " + dy6["manifest_sha256"][:8] + "...)",
              manifest_sha256=msha, frozen=True)
    (OUT / "data.yaml").write_text(yaml.safe_dump(dy, allow_unicode=True, sort_keys=False), encoding="utf-8")
    meta6 = json.loads((R6 / "dataset_meta.json").read_text(encoding="utf-8"))
    meta = dict(meta6, dataset_name="prov_v7_randsplit", dibuat=now, manifest_sha256=msha, frozen=True,
                induk={"dataset": "prov_v7", "manifest_sha256": dv7["manifest_sha256"]}, pembagian_dari="prov_v6_randsplit",
                catatan="Dataset PEMBANDING. Pembagian identik dengan prov_v6_randsplit; berkas dari prov_v7 (orientasi ds4 diperbaiki). "
                        "Model randsplit TIDAK boleh dievaluasi pada test prov_v7.", pemeriksaan="reports/prov_v7_randsplit/pemeriksaan.json")
    (OUT / "dataset_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "FROZEN.txt").write_text(f"DIBEKUKAN {now}\nmanifest_sha256={msha}\nPembanding split acak (pembagian = prov_v6_randsplit; berkas = prov_v7).\n",
                                    encoding="utf-8")
    print(f"[OK] prov_v7_randsplit dibekukan; manifest {msha}")


if __name__ == "__main__":
    main()
