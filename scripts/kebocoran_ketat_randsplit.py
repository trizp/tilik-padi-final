#!/usr/bin/env python3
"""
kebocoran_ketat_randsplit.py - Ukuran kebocoran KETAT split acak (prov_v6_randsplit): untuk tiap gambar val/test acak,
apakah ada kembaran LANGSUNG di train acak. TIDAK memakai keanggotaan kelompok berantai maupun jendela waktu.

Kembaran langsung (salah satu cukup), selalu terhadap SATU gambar train tertentu:
  * phash 16x16 jarak Hamming <= 14 pada salah satu dari 8 varian dihedral (ambang dataset)
  * nama dasar Roboflow sama (".rf.<hash>") atau kunci ID K4 ds1 sama (fam, jenis, id, ekstensi)
  * SIFT+RANSAC >= 21 inlier (tanpa syarat sebaran) ke satu gambar train; kandidat = k tetangga embedding DINOv2
    terdekat di train (embedding HANYA untuk mencari kandidat, bukan kriteria)
Fitur dihitung dari berkas gambar dataset (yang dilihat model).

Dilaporkan per split dan per sumber: % kelompok berantai prov_v6 (batas atas) dan % kembaran langsung (batas bawah).
JADWAL: dijalankan oleh scripts/antrean_lanjutan.py SETELAH antrean run final selesai (memakai GPU untuk embedding).
--uji-kecil: uji logika pada sebagian kecil gambar, CPU saja, kandidat SIFT = semua pasangan (tanpa embedding).

Keluaran: reports/prov_v6_randsplit/kebocoran_ketat.{csv,json,md}
"""
import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import dedupe_rules as dr  # noqa: E402
import prepare_dataset as pd_  # noqa: E402

ROOT = dr.ROOT
DS = ROOT / "data/processed/prov_v6_randsplit"
INDUK = ROOT / "data/processed/prov_v6"
REP = ROOT / "reports/prov_v6_randsplit"
PHASH_THR = 14


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--knn", type=int, default=200, help="kandidat SIFT per gambar val/test (tetangga embedding di train)")
    ap.add_argument("--uji-kecil", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()

    m = list(csv.DictReader(open(DS / "manifest.csv", encoding="utf-8")))
    m = [r for r in m if r["split"] in ("train", "val", "test")]
    if a.uji_kecil:                     # ambil sedikit gambar dari beberapa kelompok agar ada kembaran yang bisa ditemukan
        rng = random.Random(0)
        g = defaultdict(list)
        for r in m:
            g[r["group"]].append(r)
        multi = [v for v in g.values() if len({x["split"] for x in v}) > 1]
        rng.shuffle(multi)
        m = [x for v in multi[:12] for x in v[:4]] + rng.sample(m, 20)
        m = list({r["file"]: r for r in m}.values())
        out_suffix = "_uji_kecil"
    else:
        out_suffix = ""
    for r in m:                         # berkas yang dilihat model (hardlink dari prov_v6)
        r["path"] = INDUK / "images" / r["split_prov_v6"] / r["file"]
    tr = [i for i, r in enumerate(m) if r["split"] == "train"]
    vt = [i for i, r in enumerate(m) if r["split"] in ("val", "test")]
    print(f"train {len(tr)}, val+test {len(vt)}", flush=True)

    hits = defaultdict(dict)            # qi -> {kriteria: (indeks train, nilai)}

    # ---- 1. phash 8 varian dihedral
    if a.uji_kecil:                     # serial, tanpa pool proses dan tanpa menulis cache (aman saat run final berjalan)
        hs = [pd_._hash_one((str(r["path"]), 16)) for r in m]
        ok = np.array([h is not None for h in hs])
        var = np.stack([h if h is not None else np.zeros((8, 256), bool) for h in hs])
    else:
        var, ok = pd_.compute_hashes([{"path": r["path"]} for r in m], 16)
    tr_ok = [i for i in tr if ok[i]]
    orig_tr = var[tr_ok, 0]
    for v in range(8):
        q_ok = [i for i in vt if ok[i]]
        for a_, b_ in pd_.near_pairs(var[q_ok, v], orig_tr, PHASH_THR, False):
            qi, ti = q_ok[int(a_)], tr_ok[int(b_)]
            d = int(min(pd_.min_dist(var[qi:qi + 1, w], var[ti:ti + 1, 0])[0] for w in range(8)))
            if "phash" not in hits[qi] or d < hits[qi]["phash"][1]:
                hits[qi]["phash"] = (ti, d)
    print(f"  phash: {sum(1 for q in hits if 'phash' in hits[q])} gambar val/test", flush=True)

    # ---- 2. nama dasar Roboflow / K4
    cands = [{"src": "ds1" if r["source"].startswith("ds1") else r["source"], "orig_path": Path(r["orig_path"]), "path": r["path"]} for r in m]
    trs = set(tr)
    for nama, P in (("roboflow", pd_.basename_pairs(cands)), ("k4", dr.k4_pairs(cands))):
        for i, j in P:
            i, j = int(i), int(j)
            for q, t in ((i, j), (j, i)):
                if q in vt and t in trs and nama not in hits[q]:
                    hits[q][nama] = (t, 1)
        print(f"  {nama}: {sum(1 for q in hits if nama in hits[q])} gambar val/test", flush=True)

    # ---- 3. SIFT >= 21 ke satu gambar train (kandidat dari embedding)
    paths = [r["path"] for r in m]
    if a.uji_kecil:
        pairs = [(q, t) for q in vt for t in tr]
    else:
        import aug_detect as ad
        E = ad.embeddings(paths)
        P = dr.knn_pairs_cross(E[vt], E[tr], a.knn)
        pairs = [(vt[int(q)], tr[int(t)]) for q, t in P]
    dr.ensure_descriptors(paths, workers=a.workers)
    res = dr.verify_pairs([(paths[q], paths[t]) for q, t in pairs], workers=a.workers)
    for (q, t), r_ in zip(pairs, res):
        if r_[0] >= dr.MIN_INLIERS and ("sift" not in hits[q] or r_[0] > hits[q]["sift"][1]):
            hits[q]["sift"] = (t, int(r_[0]))
    print(f"  SIFT: {len(pairs)} pasangan diverifikasi; {sum(1 for q in hits if 'sift' in hits[q])} gambar val/test", flush=True)

    # ---- kelompok berantai (batas atas)
    g_train = {m[i]["group"] for i in tr}
    rows = []
    for q in vt:
        r = m[q]
        h = hits.get(q, {})
        best = h.get("sift") or h.get("phash") or h.get("roboflow") or h.get("k4")
        rows.append({"file": r["file"], "split_acak": r["split"], "source": r["source"], "class": r["class"],
                     "kelompok_berantai_di_train": r["group"] in g_train, "kembar_langsung": bool(h),
                     "phash_jarak": h["phash"][1] if "phash" in h else "", "roboflow": "roboflow" in h, "k4": "k4" in h,
                     "sift_inlier": h["sift"][1] if "sift" in h else "",
                     "kembar_train_contoh": m[best[0]]["file"] if best else ""})
    REP.mkdir(parents=True, exist_ok=True)
    with open(REP / f"kebocoran_ketat{out_suffix}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def ringkas(rs):
        n = len(rs)
        if not n:
            return None
        return {"gambar": n,
                "batas_atas_kelompok_berantai_persen": round(100 * sum(x["kelompok_berantai_di_train"] for x in rs) / n, 1),
                "batas_bawah_kembar_langsung_persen": round(100 * sum(x["kembar_langsung"] for x in rs) / n, 1),
                "kembar_langsung_n": sum(x["kembar_langsung"] for x in rs),
                "per_kriteria_n": {k: sum(1 for x in rs if (x[k] not in ("", False))) for k in ("phash_jarak", "roboflow", "k4", "sift_inlier")}}
    summ = {"dibuat": datetime.now().isoformat(timespec="seconds"), "uji_kecil": a.uji_kecil,
            "kriteria_kembar_langsung": f"phash<={PHASH_THR} (8 varian dihedral) ATAU nama dasar Roboflow/K4 sama ATAU SIFT>={dr.MIN_INLIERS} inlier "
                                        f"ke satu gambar train (kandidat: {'semua pasangan' if a.uji_kecil else f'k={a.knn} tetangga embedding'}); "
                                        "TANPA kelompok berantai dan jendela waktu",
            "per_split": {sp: ringkas([x for x in rows if x["split_acak"] == sp]) for sp in ("val", "test")},
            "val+test": ringkas(rows),
            "per_sumber": {s: ringkas([x for x in rows if x["source"] == s]) for s in sorted({x["source"] for x in rows})},
            "per_kelas": {c: ringkas([x for x in rows if x["class"] == c]) for c in sorted({x["class"] for x in rows})}}
    (REP / f"kebocoran_ketat{out_suffix}.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False), encoding="utf-8")
    md = ["# Kebocoran split acak: batas atas vs batas bawah" + (" (UJI KECIL, bukan hasil)" if a.uji_kecil else ""), "",
          f"Dibuat {summ['dibuat']}. Kembaran langsung = {summ['kriteria_kembar_langsung']}.", "",
          "| himpunan | gambar | kelompok berantai (batas atas) | kembaran langsung (batas bawah) |", "|---|---|---|---|"]
    for nama, s in [(f"split {k}", v) for k, v in summ["per_split"].items()] + [("val+test", summ["val+test"])] + \
                   [(f"sumber {k}", v) for k, v in summ["per_sumber"].items()]:
        if s:
            md.append(f"| {nama} | {s['gambar']} | {s['batas_atas_kelompok_berantai_persen']}% | {s['batas_bawah_kembar_langsung_persen']}% |")
    (REP / f"kebocoran_ketat{out_suffix}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
