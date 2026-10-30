#!/usr/bin/env python3
"""
bangun_deploy_v9.py - Dataset latih MODEL APLIKASI v9 (KEPUTUSAN_PROYEK.md 2026-10-20; resep = bangun_deploy_v7.py): train + val + test prov_v9 (TANPA test_new_session),
background dipangkas ke 10% dengan aturan yang sama dengan prov_v6/v7/v9 (cap = int(n_penyakit * 0,1 / 0,9); hanya background ds4
yang dipangkas, acak dengan random.Random(42) pada daftar terurut; background ds2 dipertahankan). Berkas di-hardlink dari prov_v9 (foto tanpa latar).

val_cek: 100 gambar acak DARI TRAIN (hardlink) hanya karena Ultralytics tetap memvalidasi epoch terakhir; angkanya tidak bermakna
dan tidak dipakai memilih apa pun (checkpoint = last.pt).
Keluaran: data/derived/prov_v9_deploy/{images,labels}/{train,val_cek}, manifest.csv, data.yaml, dataset_meta.json
"""
import csv
import hashlib
import json
import os
import random
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
V7 = ROOT / "data/processed/prov_v9"     # nama variabel dipertahankan dari bangun_deploy_v7.py
OUT = ROOT / "data/derived/prov_v9_deploy"
SEED, MAX_BG, PRUNABLE = 42, 0.10, {"ds4"}


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
    dy = yaml.safe_load((V7 / "data.yaml").read_text(encoding="utf-8"))
    if not dy.get("frozen") or sha(V7 / "manifest.csv") != dy["manifest_sha256"]:
        sys.exit("[GAGAL] prov_v9 belum beku / manifest berubah")
    man = list(csv.DictReader(open(V7 / "manifest.csv", encoding="utf-8")))
    pool = [r for r in man if r["split"] in ("train", "val", "test")]
    n_dis = sum(1 for r in pool if r["class"] != "background")
    cap = int(n_dis * MAX_BG / (1 - MAX_BG))
    bg_all = [r for r in pool if r["class"] == "background"]
    bg_prun = sorted((r for r in bg_all if r["source"] in PRUNABLE), key=lambda r: r["file"])
    rng = random.Random(SEED)
    n_remove = max(0, min(len(bg_all) - cap, len(bg_prun)))
    drop = {r["file"] for r in rng.sample(bg_prun, n_remove)}
    akhir = [r for r in pool if r["file"] not in drop]
    for sub in ("train", "val_cek"):
        (OUT / "images" / sub).mkdir(parents=True)
        (OUT / "labels" / sub).mkdir(parents=True)
    for r in akhir:
        tautkan(V7 / "images" / r["split"] / r["file"], OUT / "images/train" / r["file"])
        lab = V7 / "labels" / r["split"] / (Path(r["file"]).stem + ".txt")
        if lab.exists():
            tautkan(lab, OUT / "labels/train" / lab.name)
    cek = random.Random(SEED + 1).sample(sorted(akhir, key=lambda r: r["file"]), 100)
    for r in cek:
        tautkan(V7 / "images" / r["split"] / r["file"], OUT / "images/val_cek" / r["file"])
        lab = V7 / "labels" / r["split"] / (Path(r["file"]).stem + ".txt")
        if lab.exists():
            tautkan(lab, OUT / "labels/val_cek" / lab.name)
    kol = list(man[0].keys()) + ["split_prov_v9"]
    with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=kol)
        w.writeheader()
        for r in sorted(akhir, key=lambda r: r["file"]):
            w.writerow(dict(r, split_prov_v9=r["split"], split="train"))
    msha = sha(OUT / "manifest.csv")
    info = {"pool_train_val_test": len(pool), "penyakit": n_dis, "background_sebelum": len(bg_all), "cap_background": cap,
            "background_dipangkas": len(drop), "background_sesudah": len(bg_all) - len(drop),
            "background_dipertahankan_per_sumber": dict(Counter(r["source"] for r in akhir if r["class"] == "background")),
            "gambar_latih": len(akhir), "persen_background": round(100 * (len(bg_all) - len(drop)) / len(akhir), 2),
            "per_sumber_kelas": {f"{k[0]}|{k[1]}": v for k, v in sorted(Counter((r["source"], r["class"]) for r in akhir).items())},
            "asal_split_prov_v9": dict(Counter(r["split"] for r in akhir)), "val_cek": "100 gambar dari train (hanya agar validasi epoch terakhir Ultralytics berjalan)"}
    yml = {"path": str(OUT).replace("\\", "/"), "train": "images/train", "val": "images/val_cek", "names": dy["names"],
           "dataset_name": "prov_v9_deploy", "run_prefix": "v9_", "provisional": dy.get("provisional"), "provisional_sources": dy.get("provisional_sources"),
           "induk_dataset": "prov_v9", "induk_manifest_sha256": dy["manifest_sha256"], "manifest_sha256": msha, "frozen": True,
           "catatan": "data latih model aplikasi v9: train+val+test prov_v9 (tanpa latar), background 10%; tanpa test_new_session; val_cek = subset train (tidak bermakna)"}
    (OUT / "data.yaml").write_text(yaml.safe_dump(yml, allow_unicode=True, sort_keys=False), encoding="utf-8")
    meta = {"dataset_name": "prov_v9_deploy", "dibuat": datetime.now().isoformat(timespec="seconds"), "frozen": True, "manifest_sha256": msha,
            "aturan": "KEPUTUSAN_PROYEK.md 2026-10-20 (fase 4 aplikasi v9; resep model aplikasi v7)", **info}
    (OUT / "dataset_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    tns = {r["file"] for r in man if r["split"] == "test_new_session"}
    print(json.dumps(info, indent=1, ensure_ascii=False), "\ntest_new_session ikut dilatih:", len(tns & {r["file"] for r in akhir}))


if __name__ == "__main__":
    main()
