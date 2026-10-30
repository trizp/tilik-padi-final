#!/usr/bin/env python3
"""
buat_randsplit.py - Dataset turunan pembanding "split acak" (prov_v6_randsplit), meniru praktik umum.

* Sumber: gambar train + val + test prov_v6 (karantina sudah tidak ada di prov_v6). test_new_session TIDAK ikut dibagi;
  disalin apa adanya sebagai split test_new_session (satu-satunya set yang sebanding antar kedua model).
* Bagi ulang ACAK PER GAMBAR 70/20/10 dengan seed tetap, TANPA memperhatikan kelompok duplikat dan tanpa aturan
  train_only (praktik umum). prov_v6 yang beku tidak diubah: berkas di-hardlink dari prov_v6.
* Batas background 10% di train dipertahankan dengan cara yang sama dengan prov_v6 (prepare_dataset.py):
  cap = int(n_train_penyakit * 0.1 / 0.9); hanya background ds4 yang dipangkas (acak, seed tetap); val/test tidak diubah.
  Gambar yang dipangkas tidak dipakai di split mana pun.
* Ukuran kebocoran: gambar val/test acak yang kelompoknya (kolom group manifest prov_v6) juga punya anggota di train.

Keluaran: data/processed/prov_v6_randsplit/ (images, labels, manifest.csv, data.yaml, dataset_meta.json, FROZEN.txt)
          reports/prov_v6_randsplit/ (laporan.md, kebocoran_per_gambar.csv, ringkasan.json)
"""
import csv
import hashlib
import json
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
INDUK = ROOT / "data/processed/prov_v6"
OUT = ROOT / "data/processed/prov_v6_randsplit"
REP = ROOT / "reports/prov_v6_randsplit"
SEED = 42
RASIO = (0.70, 0.20, 0.10)
MAX_BG = 0.10
PRUNABLE = {"ds4"}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def tautkan(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    dy = yaml.safe_load((INDUK / "data.yaml").read_text(encoding="utf-8"))
    sha_induk = sha256(INDUK / "manifest.csv")
    if sha_induk != dy["manifest_sha256"]:
        sys.exit(f"[GAGAL] manifest prov_v6 berubah: {sha_induk}")
    if OUT.exists():
        sys.exit(f"[DITOLAK] {OUT} sudah ada (dataset turunan beku). Hapus manual bila memang ingin membuat ulang.")
    rows = list(csv.DictReader(open(INDUK / "manifest.csv", encoding="utf-8")))
    kolom = list(rows[0].keys())
    pool = [r for r in rows if r["split"] in ("train", "val", "test")]
    tns = [r for r in rows if r["split"] == "test_new_session"]

    # ---- bagi acak per gambar
    rng = random.Random(SEED)
    urut = sorted(pool, key=lambda r: r["file"])        # urutan awal deterministik
    rng.shuffle(urut)
    n = len(urut)
    n_tr, n_va = round(n * RASIO[0]), round(n * RASIO[1])
    for i, r in enumerate(urut):
        r["split_prov_v6"] = r["split"]
        r["split"] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")

    # ---- batas background 10% di train (cara prov_v6)
    tr = [r for r in urut if r["split"] == "train"]
    n_dis = sum(1 for r in tr if r["class"] != "background")
    cap = int(n_dis * MAX_BG / (1 - MAX_BG))
    bg_all = [r for r in tr if r["class"] == "background"]
    bg_prun = sorted((r for r in bg_all if r["source"] in PRUNABLE), key=lambda r: r["file"])
    n_remove = max(0, min(len(bg_all) - cap, len(bg_prun)))
    drop = {r["file"] for r in rng.sample(bg_prun, n_remove)} if n_remove else set()
    bg_info = {"max_frac": MAX_BG, "train_penyakit": n_dis, "cap": cap, "bg_train_sebelum": len(bg_all), "dipangkas": len(drop),
               "bg_train_sesudah": len(bg_all) - len(drop), "sumber_prunable": sorted(PRUNABLE),
               "dipertahankan_per_sumber": dict(Counter(r["source"] for r in bg_all if r["file"] not in drop))}
    akhir = [r for r in urut if r["file"] not in drop]
    for r in tns:
        r["split_prov_v6"] = "test_new_session"

    # ---- tulis dataset (hardlink dari prov_v6)
    for sp in ("train", "val", "test", "test_new_session"):
        (OUT / "images" / sp).mkdir(parents=True)
        (OUT / "labels" / sp).mkdir(parents=True)
    for r in akhir + tns:
        src_sp = r["split_prov_v6"]
        tautkan(INDUK / "images" / src_sp / r["file"], OUT / "images" / r["split"] / r["file"])
        lab = INDUK / "labels" / src_sp / (Path(r["file"]).stem + ".txt")
        if lab.exists():
            tautkan(lab, OUT / "labels" / r["split"] / lab.name)
    with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=kolom + ["split_prov_v6"])
        w.writeheader()
        w.writerows(sorted(akhir + tns, key=lambda r: (r["split"], r["file"])))
    sha = sha256(OUT / "manifest.csv")

    # ---- kebocoran: kelompok prov_v6 yang punya anggota di train acak
    g_train = defaultdict(int)
    for r in akhir:
        if r["split"] == "train":
            g_train[r["group"]] += 1
    bocor_rows = []
    for r in akhir:
        if r["split"] in ("val", "test"):
            k = g_train.get(r["group"], 0)
            bocor_rows.append({"file": r["file"], "split_acak": r["split"], "split_prov_v6": r["split_prov_v6"], "class": r["class"],
                               "source": r["source"], "group_prov_v6": r["group"], "anggota_grup_di_train_acak": k, "bocor": k > 0,
                               "train_only_di_prov_v6": r["train_only"], "aug_possible": r["aug_possible"]})
    REP.mkdir(parents=True, exist_ok=True)
    with open(REP / "kebocoran_per_gambar.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(bocor_rows[0]))
        w.writeheader()
        w.writerows(bocor_rows)

    def ringkas(rs):
        b = sum(1 for x in rs if x["bocor"])
        return {"gambar": len(rs), "punya_kembar_di_train": b, "persen": round(100 * b / len(rs), 1) if rs else None}
    kb = {sp: ringkas([x for x in bocor_rows if x["split_acak"] == sp]) for sp in ("val", "test")}
    kb["val+test"] = ringkas(bocor_rows)
    kb_kelas = {sp: {c: ringkas([x for x in bocor_rows if x["split_acak"] == sp and x["class"] == c])
                     for c in ("leaf_blast", "brown_spot", "bacterial_leaf_blight", "background")} for sp in ("val", "test")}
    kb_sumber = {s: ringkas([x for x in bocor_rows if x["source"] == s]) for s in sorted({x["source"] for x in bocor_rows})}
    grup_split = defaultdict(set)
    for r in akhir:
        grup_split[r["group"]].add(r["split"])
    g_multi = sum(1 for v in grup_split.values() if len(v) > 1)
    g_besar = sum(1 for g in grup_split if sum(1 for r in akhir if r["group"] == g) > 1)
    tns_g = {r["group"] for r in tns}
    tns_bocor = sum(1 for r in akhir if r["group"] in tns_g)
    train_only_vt = sum(1 for x in bocor_rows if x["train_only_di_prov_v6"] == "True")
    jumlah = {sp: dict(Counter(r["class"] for r in akhir + tns if r["split"] == sp)) for sp in ("train", "val", "test", "test_new_session")}
    for sp in jumlah:
        jumlah[sp]["TOTAL"] = sum(jumlah[sp].values())
    asal = {sp: dict(Counter(r["split_prov_v6"] for r in akhir if r["split"] == sp)) for sp in ("train", "val", "test")}

    summ = {"dataset_name": "prov_v6_randsplit", "dibuat": datetime.now().isoformat(timespec="seconds"),
            "induk": {"dataset": "prov_v6", "manifest_sha256": sha_induk},
            "aturan": f"gambar train+val+test prov_v6 dibagi ACAK PER GAMBAR {RASIO} (seed {SEED}), tanpa memperhatikan kelompok duplikat "
                      f"dan tanpa aturan train_only; background train dipangkas ke {int(MAX_BG * 100)}% (hanya ds4) seperti prov_v6; "
                      "test_new_session disalin apa adanya dari prov_v6",
            "seed": SEED, "pool_sebelum_pangkas": n, "background_train": bg_info, "jumlah": jumlah, "asal_split_prov_v6": asal,
            "kebocoran": {"definisi": "gambar val/test acak yang kelompok prov_v6-nya (phash, embedding, nama dasar Roboflow, K4, SIFT, "
                                      "jendela waktu 5 dtk; kolom group) juga punya anggota di train acak",
                          "per_split": kb, "per_kelas": kb_kelas, "per_sumber": kb_sumber,
                          "kelompok_terbelah_lintas_split": g_multi, "kelompok_berisi_>1_gambar": g_besar,
                          "val_test_dari_sumber_train_only_prov_v6": train_only_vt,
                          "anggota_kelompok_test_new_session_di_train/val/test_acak": tns_bocor},
            "manifest_sha256": sha, "frozen": True}
    (REP / "ringkasan.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False), encoding="utf-8")
    meta = dict(summ, catatan="Dataset PEMBANDING (bukan untuk model utama). Model randsplit melihat sebagian gambar test prov_v6 "
                              "saat training, jadi TIDAK boleh dievaluasi pada test prov_v6; hanya pada test acaknya sendiri dan test_new_session.")
    (OUT / "dataset_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    yml = {"path": str(OUT).replace("\\", "/"), "train": "images/train", "val": "images/val", "test": "images/test",
           "test_new_session": "images/test_new_session", "names": dy["names"], "dataset_name": "prov_v6_randsplit",
           "run_prefix": "randsplit_prov_v6_", "provisional": dy.get("provisional"), "provisional_sources": dy.get("provisional_sources"),
           "seed": SEED, "induk_dataset": "prov_v6", "induk_manifest_sha256": sha_induk, "manifest_sha256": sha, "frozen": True}
    (OUT / "data.yaml").write_text(yaml.safe_dump(yml, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (OUT / "FROZEN.txt").write_text(f"DIBEKUKAN {datetime.now():%Y-%m-%dT%H:%M:%S}\nmanifest_sha256={sha}\n"
                                    "Dataset turunan pembanding split acak dari prov_v6; jangan diubah.\n", encoding="utf-8")
    print(json.dumps({k: summ[k] for k in ("pool_sebelum_pangkas", "background_train", "jumlah", "asal_split_prov_v6", "kebocoran")},
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
