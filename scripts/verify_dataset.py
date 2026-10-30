#!/usr/bin/env python3
"""
verify_dataset.py - Verifikasi MANDIRI dataset hasil prepare_dataset.py langsung dari file di disk.

Memeriksa: gambar disk == manifest, nama unik lintas split, setiap gambar punya label,
format/rentang label, kelas label == kelas manifest, dan HASH ULANG gambar akhir untuk memastikan
tidak ada pasangan lintas split dengan jarak phash <= ambang (8 varian dihedral).

Pemakaian:  python scripts/verify_dataset.py [--config configs/classes.yaml]
"""
import argparse
import csv
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from prepare_dataset import ROOT, SPLITS, _hash_one, min_dist_dihedral  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/classes.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    D, classes = ROOT / cfg["output_dir"], cfg["disease_classes"]
    thr, hs = cfg["dup_threshold"], cfg["hash_size"]
    ok = True

    man = list(csv.DictReader(open(D / "manifest.csv", encoding="utf-8")))
    SPL = [s for s in ("train", "val", "test", "test_new_session") if any(r["split"] == s for r in man)]
    disk = {s: {p.name for p in (D / "images" / s).iterdir()} for s in SPL}
    mm = {s: {r["file"] for r in man if r["split"] == s} for s in SPL}
    c1 = all(disk[s] == mm[s] for s in SPL)
    names = [n for s in SPL for n in disk[s]]
    c2 = len(names) == len(set(names))
    print(f"manifest {len(man)} baris | gambar disk == manifest: {c1} | nama unik lintas split: {c2}")
    ok &= c1 and c2

    # aturan kebijakan split: train-only dan flag augmentasi (kecuali kelas exempt) tidak boleh di val/test
    aug = cfg.get("augmentation", {}) or {}
    exempt = set(aug.get("exempt_classes", []))
    viol = [r["file"] for r in man if r["split"] != "train" and
            (r["train_only"] == "True" or (r["aug_possible"] == "True" and r["class"] not in exempt))]
    # background_cap_exclude_sources (opsional, prov_v8): sumber background negatif (mis. Tungro) TIDAK dihitung dalam batas
    # background daun sehat; dilaporkan terpisah. Tanpa kunci ini perilakunya sama dengan sebelumnya.
    neg = set(cfg.get("background_cap_exclude_sources") or [])
    tr_all = [r for r in man if r["split"] == "train"]
    tr = [r for r in tr_all if r["source"] not in neg]
    bg_frac = sum(r["class"] == "background" for r in tr) / len(tr)
    mbf = cfg.get("max_background_frac_train")
    okbg = mbf is None or bg_frac <= mbf + 1e-9
    print(f"pelanggaran aturan val/test (train-only / aug): {len(viol)} | background train: {bg_frac * 100:.2f}% (batas {mbf}) -> {okbg}")
    if neg:
        nn = sum(r["source"] in neg for r in tr_all)
        print(f"background negatif {sorted(neg)} (di luar batas): {nn} gambar = {100 * nn / len(tr_all):.2f}% dari train; "
              f"di luar train: {sum(r['source'] in neg for r in man if r['split'] != 'train')}")
        ok &= all(r["split"] == "train" and r["class"] == "background" for r in man if r["source"] in neg)
    ok &= not viol and okbg

    # dataset beku: hash manifest dan label harus sama dengan yang tercatat saat dibekukan
    mp = D / "dataset_meta.json"
    if mp.exists():
        import hashlib
        import json
        meta = json.loads(mp.read_text(encoding="utf-8"))
        fz = meta.get("freeze")
        if fz and fz.get("frozen"):
            hm = hashlib.sha256((D / "manifest.csv").read_bytes()).hexdigest()
            hl = hashlib.sha256()
            for sp_ in SPL:
                for lf in sorted((D / "labels" / sp_).iterdir()):
                    hl.update(f"{sp_}/{lf.name}\n".encode())
                    hl.update(lf.read_bytes())
            okf = hm == fz["manifest_sha256"] and hl.hexdigest() == fz["labels_sha256"]
            print(f"dataset BEKU ({fz['frozen_at']}): hash manifest dan label sama dengan saat dibekukan: {okf}")
            ok &= okf
        else:
            print("dataset belum dibekukan (frozen=false)")

    cls_of = {r["file"]: r["class"] for r in man}
    bad, mism, cnt = 0, 0, Counter()
    for s in SPL:
        for n in disk[s]:
            lb = D / "labels" / s / (Path(n).stem + ".txt")
            if not lb.exists():
                bad += 1
                continue
            ids = set()
            for ln in lb.read_text().splitlines():
                t = ln.split()
                if not t:
                    continue
                if len(t) != 5:
                    bad += 1
                    continue
                c, v = int(t[0]), list(map(float, t[1:]))
                if not (0 <= c < len(classes)) or v[2] <= 0 or v[3] <= 0 \
                        or v[0] - v[2] / 2 < -1e-5 or v[0] + v[2] / 2 > 1 + 1e-5 \
                        or v[1] - v[3] / 2 < -1e-5 or v[1] + v[3] / 2 > 1 + 1e-5:
                    bad += 1
                    continue
                ids.add(classes[c])
                cnt[(s, classes[c])] += 1
            exp = cls_of[n]
            mism += (exp == "background" and bool(ids)) or (exp != "background" and ids != {exp})
    print(f"label cacat/hilang: {bad} | kelas label != manifest: {mism}")
    ok &= bad == 0 and mism == 0
    print("box per split:kelas:", {f"{k[0]}:{k[1]}": v for k, v in sorted(cnt.items())})

    H = {}
    paths = {s: sorted((D / "images" / s).iterdir()) for s in SPL}
    with ProcessPoolExecutor() as ex:
        for s in SPL:
            r = list(ex.map(_hash_one, [(str(p), hs) for p in paths[s]], chunksize=32))
            assert all(x is not None for x in r), "ada gambar tak terbaca"
            H[s] = np.array(r)                       # (n, 8 varian dihedral, bits)
    for a, b in [("val", "train"), ("test", "train"), ("test", "val")] + ([("test_new_session", x) for x in ("train", "val", "test")] if "test_new_session" in SPL else []):
        d = min_dist_dihedral(H[a], H[b][:, 0])
        n = int((d <= thr).sum())
        print(f"re-hash {a} vs {b}: pasangan <= {thr}: {n} | jarak min: {d.min():.0f}")
        ok &= n == 0
    print("\nHASIL:", "LOLOS" if ok else "GAGAL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
