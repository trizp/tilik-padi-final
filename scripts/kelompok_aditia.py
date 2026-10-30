#!/usr/bin/env python3
"""
kelompok_aditia.py - group_id aditia (Langkah A v8): jendela waktu 5 dtk (cap waktu nama berkas) + phash<=14 (8 varian dihedral)
+ SIFT>=21 inlier (konservatif; kandidat pasangan = 50 tetangga embedding, embedding BUKAN kriteria). TIDAK menjalankan model.

Keluaran: reports/kandidat_v8/aditia_kelompok/
  ringkasan.json         - jumlah kelompok per kelas untuk tiap aturan, rentang/jeda waktu pemotretan per kelas
  per_gambar.csv         - berkas, kelas, cap waktu, group_id (waktu5+phash) dan group_id (waktu5+phash+SIFT)
  rep_<kelas>.jpg        - satu foto per kelompok waktu5+phash (untuk menghitung helai daun secara visual)
  sift_antar_kelompok.jpg - contoh pasangan SIFT yang menyatukan kelompok waktu5+phash
"""
import csv
import json
import random
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
import aug_detect as ad  # noqa: E402
import dedupe_rules as dr  # noqa: E402
import prepare_dataset as pd_  # noqa: E402
from inventory import IMG_EXT  # noqa: E402

ROOT = dr.ROOT
D = ROOT / "data/raw/kandidat_v8/rf_aditia_4ued0"
OUT = ROOT / "reports/kandidat_v8/aditia_kelompok"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    P = sorted(p for p in D.rglob("*") if p.suffix.lower() in IMG_EXT)
    names = yaml.safe_load((D / "data.yaml").read_text(encoding="utf-8"))["names"]
    lab = [names[int((p.parent.parent / "labels" / (p.stem + ".txt")).read_text().split()[0])] for p in P]
    ts = [datetime.strptime(re.match(r"(\d{8}_\d{6})", p.name).group(1), "%Y%m%d_%H%M%S") for p in P]
    n = len(P)
    o = sorted(range(n), key=lambda i: ts[i])
    tp = []
    for a in range(n):
        b = a + 1
        while b < n and (ts[o[b]] - ts[o[a]]).total_seconds() <= 5:
            tp.append((o[a], o[b]))
            b += 1
    var, _ = pd_.compute_hashes([{"path": p} for p in P], 16)
    ph = [tuple(map(int, x)) for x in pd_.phash_pairs(var, 14)]
    E = ad.embeddings(P)
    dr.ensure_descriptors(P)
    Pk = dr.knn_pairs(E, 50)
    sift = [(int(i), int(j), r) for (i, j), r in zip(Pk, dr.verify_pairs([(P[i], P[j]) for i, j in Pk])) if r[0] >= dr.MIN_INLIERS]

    def grup(pairs):
        return pd_.union_groups(n, [np.array(pairs, int).reshape(-1, 2)])
    aturan = {"waktu5": tp, "waktu5+phash": tp + ph, "waktu5+phash+sift": tp + ph + [(i, j) for i, j, _ in sift]}
    G = {k: grup(v) for k, v in aturan.items()}
    ring = {"dibuat": datetime.now().isoformat(timespec="seconds"), "model_dijalankan": False, "gambar": n, "per_aturan": {}, "waktu_per_kelas": {}}
    for k, g in G.items():
        ring["per_aturan"][k] = {kel: len({g[i] for i in range(n) if lab[i] == kel}) for kel in sorted(set(lab))}
        ring["per_aturan"][k]["kelompok_lintas_kelas"] = sum(1 for x in set(g) if len({lab[i] for i in range(n) if g[i] == x}) > 1)
    for kel in sorted(set(lab)):
        t = sorted(ts[i] for i in range(n) if lab[i] == kel)
        d = [(t[i + 1] - t[i]).total_seconds() for i in range(len(t) - 1)]
        ring["waktu_per_kelas"][kel] = {"gambar": len(t), "tanggal": sorted({x.date().isoformat() for x in t}), "mulai": t[0].isoformat(),
                                        "selesai": t[-1].isoformat(), "durasi_menit": round((t[-1] - t[0]).total_seconds() / 60, 1),
                                        "cap_waktu_unik": len(set(t)), "jeda_median_dtk": float(np.median(d))}
    g1, g2 = G["waktu5+phash"], G["waktu5+phash+sift"]
    with open(OUT / "per_gambar.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["berkas", "kelas", "cap_waktu", "group_waktu5_phash", "group_waktu5_phash_sift"])
        for i, p in enumerate(P):
            w.writerow([p.resolve().relative_to(ROOT).as_posix(), lab[i], ts[i].isoformat(), f"AW{int(g1[i]):04d}", f"AS{int(g2[i]):04d}"])
    T = 200
    rng = random.Random(5)
    for kel in sorted(set(lab)):
        gs = sorted({g1[i] for i in range(n) if lab[i] == kel}, key=lambda x: min(ts[i] for i in range(n) if g1[i] == x))
        rep = [(x, min((i for i in range(n) if g1[i] == x), key=lambda i: ts[i])) for x in gs]
        if len(rep) > 30:
            rep = sorted(rng.sample(rep, 30), key=lambda r: ts[r[1]])
        cv = Image.new("RGB", (6 * T, ((len(rep) + 5) // 6) * (T + 16) + 20), "white")
        dd = ImageDraw.Draw(cv)
        dd.text((4, 4), f"aditia {kel}: 1 foto per kelompok waktu5+phash ({len(gs)} kelompok; tampil {len(rep)})", fill="black")
        for m, (x, i) in enumerate(rep):
            X, Y = (m % 6) * T, 20 + (m // 6) * (T + 16)
            dd.text((X + 2, Y + 2), f"{ts[i].strftime('%H:%M:%S')} n={int((g1 == x).sum())}", fill="black")
            cv.paste(Image.open(P[i]).convert("RGB").resize((T, T)), (X, Y + 16))
        cv.save(OUT / f"rep_{kel.replace(' ', '_')}.jpg", quality=85)
    lintas = [(i, j, r) for i, j, r in sift if g1[i] != g1[j]]
    sm = random.Random(1).sample(lintas, min(12, len(lintas)))
    T = 240
    cv = Image.new("RGB", (4 * T, 6 * (T + 18)), "white")
    dd = ImageDraw.Draw(cv)
    for k, (i, j, r) in enumerate(sm):
        x0, y0 = (k % 2) * 2 * T, (k // 2) * (T + 18)
        dd.text((x0 + 3, y0 + 3), f"{ts[i].strftime('%H:%M:%S')} ~ {ts[j].strftime('%H:%M:%S')} inl={r[0]}", fill="black")
        for q, pp in enumerate((P[i], P[j])):
            cv.paste(Image.open(pp).convert("RGB").resize((T, T)), (x0 + q * T, y0 + 18))
    cv.save(OUT / "sift_antar_kelompok.jpg", quality=85)
    ring["pasangan_sift_antar_kelompok_waktu5_phash"] = len(lintas)
    (OUT / "ringkasan.json").write_text(json.dumps(ring, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(ring, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
