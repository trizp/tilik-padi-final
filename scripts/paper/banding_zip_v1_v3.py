#!/usr/bin/env python3
"""banding_zip_v1_v3.py - Berkas mana yang berubah antara zip V1/V2 (sama persis, sha256 ce1362d6...) dan zip V3 RiceLeafDiseaseBD,
dari central directory saja (HTTP Range; tidak mengunduh foto). Khusus diperiksa: 931 foto ds4 yang orientasinya kita perbaiki di prov_v7.
Hanya membaca. Keluaran: reports/ds4_versi/{banding_v1_v3.json, banding_v1_v3_per_berkas.csv}"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cek_versi_ds4 as C  # noqa: E402

ROOT = C.ROOT
OUT = C.OUT


def kategori(n):
    if "/labels/" in n:
        return "label"
    if "Original images/" in n:
        return "foto asli"
    if "/visuals/" in n:
        return "foto bertanda (visual)"
    return "lain"


def main():
    f1 = C.get_json(f"{C.API}/files?folder_id=root&version=1")
    z1 = next(x for x in f1 if x["filename"].endswith(".zip"))
    v1 = {e["nama"]: e for e in C.central_directory(z1["content_details"]["download_url"], z1["content_details"]["size"])
          if not e["nama"].endswith("/")}
    v3 = {r["nama"]: {"crc32": int(r["crc32"]), "ukuran": int(r["ukuran"])}
          for r in csv.DictReader(open(OUT / "zip_v3_isi.csv", encoding="utf-8")) if not r["nama"].endswith("/")}
    hasil = Counter()
    rows = []
    for n in sorted(set(v1) | set(v3)):
        k = kategori(n)
        if n not in v1:
            st = "hanya V3"
        elif n not in v3:
            st = "hanya V1"
        elif (v1[n]["crc32"], v1[n]["ukuran"]) == (v3[n]["crc32"], v3[n]["ukuran"]):
            st = "sama"
        else:
            st = "berubah"
        hasil[(k, st)] += 1
        if st != "sama":
            rows.append({"berkas": n, "kategori": k, "status": st})
    with open(OUT / "banding_v1_v3_per_berkas.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["berkas", "kategori", "status"])
        w.writeheader()
        w.writerows(rows)
    # 931 foto yang diputar di prov_v7: apakah foto aslinya berubah di V3?
    man = [r for r in csv.DictReader(open(ROOT / "data/processed/prov_v7/manifest.csv", encoding="utf-8"))
           if r["source"] == "ds4" and r["perbaikan_orientasi"] != "-"]
    diputar = Counter()
    for r in man:
        n = r["orig_path"].replace("data/raw/ds4/images_labels/", "")
        lab = n.replace("Original images/", "Annotated images ( visual with labels)/")
        lab = lab.rsplit("/", 1)[0] + "/labels/" + Path(n).stem + ".txt"
        lab = lab.replace("( visual with labels)/" + Path(n).parent.name + "/labels", "( visual with labels)/" + Path(n).parent.name + "/labels")
        sf = "sama" if n in v1 and n in v3 and v1[n]["crc32"] == v3[n]["crc32"] else ("tidak ada di V1" if n not in v1 else "berubah")
        sl = ("sama" if lab in v1 and lab in v3 and v1[lab]["crc32"] == v3[lab]["crc32"]
              else ("tidak ada" if lab not in v1 or lab not in v3 else "berubah"))
        diputar[(f"foto {sf}", f"label {sl}")] += 1
    ring = {"zip_v1_sha256": z1["content_details"]["sha256_hash"], "berkas_v1": len(v1), "berkas_v3": len(v3),
            "per_kategori": {f"{k} | {s}": n for (k, s), n in sorted(hasil.items())},
            "foto_diputar_prov_v7": {"jumlah": len(man), "status_v1_ke_v3": {f"{a}; {b}": n for (a, b), n in diputar.items()}}}
    (OUT / "banding_v1_v3.json").write_text(json.dumps(ring, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(ring, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
