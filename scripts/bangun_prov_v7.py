#!/usr/bin/env python3
"""
bangun_prov_v7.py - prov_v7 = prov_v6 dengan orientasi foto ds4 diperbaiki (jpegtran, lossless). HANYA piksel 931 foto ds4
berubah; ID foto, split, kelompok, karantina, test_new_session, label SAMA PERSIS dengan prov_v6.

  python scripts/bangun_prov_v7.py bangun     # tulis data/processed/prov_v7 (belum beku) + pemeriksaan kesetaraan
  python scripts/bangun_prov_v7.py bekukan    # setelah SEMUA pemeriksaan lolos (termasuk verify_dataset)

Pemeriksaan (bangun): ID+split identik dengan prov_v6 (selisih 0), label identik byte, hash gambar berubah HANYA pada 931 foto
ds4 yang diputar jpegtran, tidak ada tag EXIF orientasi pada foto ds4. (Audit orientasi dan verify_dataset dijalankan terpisah.)
Keluaran pemeriksaan: reports/prov_v7/pemeriksaan_kesetaraan.json
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
V6 = ROOT / "data/processed/prov_v6"
V7 = ROOT / "data/processed/prov_v7"
FIX = ROOT / "data/interim/ds4_orientasi_v7_jpegtran"
REP = ROOT / "reports/prov_v7"
SPL = ("train", "val", "test", "test_new_session")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tautkan(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def labels_sha(D):
    h = hashlib.sha256()
    for sp in SPL:
        for lf in sorted((D / "labels" / sp).iterdir()):
            h.update(f"{sp}/{lf.name}\n".encode())
            h.update(lf.read_bytes())
    return h.hexdigest()


def bangun():
    if V7.exists():
        sys.exit(f"[DITOLAK] {V7} sudah ada")
    if sha(V6 / "manifest.csv") != yaml.safe_load((V6 / "data.yaml").read_text(encoding="utf-8"))["manifest_sha256"]:
        sys.exit("[GAGAL] manifest prov_v6 berubah")
    jt = {r["file"]: r for r in csv.DictReader(open(ROOT / "reports/prov_v7_prep/jpegtran_per_foto.csv", encoding="utf-8"))}
    diubah = {f for f, r in jt.items() if r["aksi"] != "salin_byte_identik"}
    man = list(csv.DictReader(open(V6 / "manifest.csv", encoding="utf-8")))
    kol = list(man[0].keys()) + ["perbaikan_orientasi"]
    for sp in SPL:
        (V7 / "images" / sp).mkdir(parents=True)
        (V7 / "labels" / sp).mkdir(parents=True)
    for r in man:
        sp, f = r["split"], r["file"]
        src = FIX / f if f in diubah else V6 / "images" / sp / f
        tautkan(src, V7 / "images" / sp / f)
        lab = V6 / "labels" / sp / (Path(f).stem + ".txt")
        if lab.exists():
            tautkan(lab, V7 / "labels" / sp / lab.name)
        r["perbaikan_orientasi"] = (f"jpegtran_rotate_cw_{jt[f]['rotasi_cw_jpegtran']}" if f in diubah else "-")
    with open(V7 / "manifest.csv", "w", newline="", encoding="utf-8") as fo:
        w = csv.DictWriter(fo, fieldnames=kol)
        w.writeheader()
        w.writerows(man)
    shutil.copytree(V6 / "lists", V7 / "lists")
    # data.yaml dan meta (belum beku)
    dy6 = yaml.safe_load((V6 / "data.yaml").read_text(encoding="utf-8"))
    dy7 = dict(dy6, path=str(V7).replace("\\", "/"), dataset_name="prov_v7", run_prefix="v7_", manifest_sha256=sha(V7 / "manifest.csv"),
               frozen=False, induk_dataset="prov_v6", induk_manifest_sha256=dy6["manifest_sha256"],
               perbaikan="orientasi 931 foto ds4 diperbaiki dengan jpegtran (lossless); label, split, kelompok sama dengan prov_v6")
    (V7 / "data.yaml").write_text(yaml.safe_dump(dy7, allow_unicode=True, sort_keys=False), encoding="utf-8")
    m6 = json.loads((V6 / "dataset_meta.json").read_text(encoding="utf-8"))
    m7 = dict(m6, dataset_name="prov_v7", dibuat=datetime.now().isoformat(timespec="seconds"), manifest_sha256=dy7["manifest_sha256"],
              frozen=False, freeze={"frozen": False, "alasan_tidak_dibekukan": "menunggu pemeriksaan"},
              induk={"dataset": "prov_v6", "manifest_sha256": dy6["manifest_sha256"], "grup_split_sha256": m6.get("grup_split_sha256")},
              perbaikan_orientasi={"alat": "jpegtran libjpeg-turbo 3.2.0 (tools/libjpeg-turbo)", "foto_diubah": len(diubah),
                                   "rincian": "reports/prov_v7_prep/jpegtran_per_foto.csv",
                                   "bukti_lossless": "reports/prov_v7_prep/jpegtran_bukti_lossless.json (bolak-balik identik bit 931/931)"})
    m7["catatan"] = list(m6.get("catatan", [])) + ["prov_v7: hanya piksel 931 foto ds4 diputar (lossless) agar cocok dengan label penulis ds4; "
                                                   "selain itu identik dengan prov_v6 (ERRATA prov_v6: metrik kotak foto ds4 tidak valid)."]
    (V7 / "dataset_meta.json").write_text(json.dumps(m7, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- pemeriksaan kesetaraan
    m7r = list(csv.DictReader(open(V7 / "manifest.csv", encoding="utf-8")))
    id6 = {(r["split"], r["file"], r["group"], r["class"], r["source"]) for r in man}
    id7 = {(r["split"], r["file"], r["group"], r["class"], r["source"]) for r in m7r}
    berubah, label_beda, lain_berubah = [], 0, []
    for r in m7r:
        sp, f = r["split"], r["file"]
        if sha(V6 / "images" / sp / f) != sha(V7 / "images" / sp / f):
            berubah.append(f)
            if f not in diubah:
                lain_berubah.append(f)
        lab6, lab7 = V6 / "labels" / sp / (Path(f).stem + ".txt"), V7 / "labels" / sp / (Path(f).stem + ".txt")
        if lab6.exists() != lab7.exists() or (lab6.exists() and sha(lab6) != sha(lab7)):
            label_beda += 1
    tag = sum(1 for r in m7r if r["source"] == "ds4" and Image.open(V7 / "images" / r["split"] / r["file"]).getexif().get(274) not in (None, 1))
    disk = {sp: {p.name for p in (V7 / "images" / sp).iterdir()} for sp in SPL}
    hasil = {"id_split_kelompok_kelas_sumber_identik": id6 == id7, "selisih_id_split": len(id6 ^ id7),
             "jumlah_per_split_v6": {sp: sum(1 for r in man if r["split"] == sp) for sp in SPL},
             "jumlah_per_split_v7": {sp: len(disk[sp]) for sp in SPL},
             "gambar_hash_berubah": len(berubah), "berubah_semua_ds4_jpegtran": not lain_berubah and set(berubah) == diubah,
             "berubah_di_luar_daftar_jpegtran": len(lain_berubah), "label_berbeda": label_beda,
             "ds4_masih_bertag_exif_orientasi": tag, "labels_sha256_v6": labels_sha(V6), "labels_sha256_v7": labels_sha(V7)}
    hasil["LOLOS"] = (hasil["id_split_kelompok_kelas_sumber_identik"] and hasil["selisih_id_split"] == 0 and hasil["gambar_hash_berubah"] == 931
                      and hasil["berubah_semua_ds4_jpegtran"] and label_beda == 0 and tag == 0 and hasil["labels_sha256_v6"] == hasil["labels_sha256_v7"])
    REP.mkdir(parents=True, exist_ok=True)
    (REP / "pemeriksaan_kesetaraan.json").write_text(json.dumps(hasil, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(hasil, indent=1, ensure_ascii=False))
    sys.exit(0 if hasil["LOLOS"] else 1)


def bekukan():
    pk = json.loads((REP / "pemeriksaan_kesetaraan.json").read_text(encoding="utf-8"))
    au = json.loads((REP / "audit_orientasi/ringkasan.json").read_text(encoding="utf-8"))
    vd = (REP / "verify_dataset.log").read_text(encoding="utf-8", errors="replace")
    au_ok = all(v["label_tidak_cocok"] == 0 for v in au["per_split"].values())
    if not (pk["LOLOS"] and au_ok and "HASIL: LOLOS" in vd):
        sys.exit(f"[DITOLAK] pemeriksaan belum lolos: kesetaraan={pk['LOLOS']} audit={au_ok} verify={'HASIL: LOLOS' in vd}")
    now = datetime.now().isoformat(timespec="seconds")
    dy = yaml.safe_load((V7 / "data.yaml").read_text(encoding="utf-8"))
    dy["frozen"] = True
    (V7 / "data.yaml").write_text(yaml.safe_dump(dy, allow_unicode=True, sort_keys=False), encoding="utf-8")
    m = json.loads((V7 / "dataset_meta.json").read_text(encoding="utf-8"))
    m["frozen"] = True
    m["freeze"] = {"frozen": True, "frozen_at": now, "alasan_tidak_dibekukan": None, "manifest_sha256": sha(V7 / "manifest.csv"),
                   "data_yaml_sha256": sha(V7 / "data.yaml"), "labels_sha256": labels_sha(V7),
                   "pemeriksaan": {"kesetaraan": "reports/prov_v7/pemeriksaan_kesetaraan.json", "audit_orientasi": "reports/prov_v7/audit_orientasi/ringkasan.json",
                                   "verify_dataset": "reports/prov_v7/verify_dataset.log"}}
    (V7 / "dataset_meta.json").write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    (V7 / "FROZEN.txt").write_text(f"DIBEKUKAN {now}\nmanifest_sha256={m['freeze']['manifest_sha256']}\n"
                                   "Perubahan lanjutan harus menjadi versi baru.\n", encoding="utf-8")
    print(f"[OK] prov_v7 dibekukan {now}; manifest {m['freeze']['manifest_sha256']}")


if __name__ == "__main__":
    {"bangun": bangun, "bekukan": bekukan}[sys.argv[1]]()
