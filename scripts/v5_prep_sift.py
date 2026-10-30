#!/usr/bin/env python3
"""
v5_prep_sift.py - Persiapan prov_v5 (TIDAK membangun dataset): audit SIFT k=200 TANPA aturan sebaran pada prov_v4
(val/test -> train, dan test -> val), dampak ambang sebaran 1/2/3/5%, dan dua montase sebaran.
Piksel test dibaca hanya untuk deteksi duplikat.
Keluaran: reports/prov_v5_prep/ (sift_audit_pairs.csv, sift_audit_thresholds.csv, frac_below_2pct.jpg,
          frac_2_5pct_cross_split.jpg, sift_prep.json)
"""
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
import aug_detect as ad  # noqa: E402
import dedupe_rules as dr  # noqa: E402

ROOT = dr.ROOT
OUT = ROOT / "reports/prov_v5_prep"
V4 = ROOT / "data/processed/prov_v4"


def real_path(r):
    if r["source"] == "ds2":
        return ROOT / "data/interim/resized/ds2" / (hashlib.sha1(r["orig_path"].encode()).hexdigest()[:16] + ".jpg")
    return ROOT / r["orig_path"]


def montage(pairs, captions, out, per_row=2, tile=300):
    W = 30
    rows = (len(pairs) + per_row - 1) // per_row
    cv = Image.new("RGB", (per_row * 2 * tile, rows * (tile + W)), "white")
    d = ImageDraw.Draw(cv)
    for k, (a, b) in enumerate(pairs):
        x0, y0 = (k % per_row) * 2 * tile, (k // per_row) * (tile + W)
        d.text((x0 + 3, y0 + 3), captions[k][0], fill="black")
        d.text((x0 + 3, y0 + 16), captions[k][1], fill="black")
        for t, p in enumerate((a, b)):
            cv.paste(Image.open(ROOT / p).convert("RGB").resize((tile, tile)), (x0 + t * tile, y0 + W))
    cv.save(out, quality=88)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {}
    man = list(csv.DictReader(open(V4 / "manifest.csv", encoding="utf-8")))
    paths = [real_path(r) for r in man]
    spl = np.array([r["split"] for r in man])
    recs = []
    reuse = "--reuse" in sys.argv and (OUT / "sift_audit_pairs.csv").exists()
    if not reuse:
        E = ad.embeddings(paths)
        dr.ensure_descriptors(paths)
    for q_sp, r_sp in ([] if reuse else (("val", "train"), ("test", "train"), ("test", "val"))):
        Q, R = np.where(spl == q_sp)[0], np.where(spl == r_sp)[0]
        P = dr.knn_pairs_cross(E[Q], E[R], 200)
        res = dr.verify_pairs([(paths[Q[i]], paths[R[j]]) for i, j in P])
        for (i, j), r in zip(P, res):
            if r[0] >= dr.MIN_INLIERS:
                a, b = man[Q[i]], man[R[j]]
                recs.append({"pasangan_split": f"{q_sp}-{r_sp}", "query": a["orig_path"], "partner": b["orig_path"], "q_split": q_sp, "p_split": r_sp,
                             "q_kelas": a["class"], "p_kelas": b["class"], "q_sumber": a["source"], "p_sumber": b["source"],
                             "inlier": int(r[0]), "sebaran": float(r[3])})
        print(q_sp, r_sp, "selesai", flush=True)
    if reuse:
        df = pd.read_csv(OUT / "sift_audit_pairs.csv")
    else:
        df = pd.DataFrame(recs)
        df.to_csv(OUT / "sift_audit_pairs.csv", index=False, encoding="utf-8")
    n_vt = int((spl != "train").sum())
    # ---- dampak ambang: gambar val/test yang punya kembar di train (inlier>=21) tetapi SEMUA pasangannya < ambang (tidak tertangkap)
    vt = df[df.p_split == "train"]
    mx = vt.groupby("query")["sebaran"].max()
    rows = [{"ambang_sebaran": "tanpa aturan (inlier>=21 saja)", "val_test_dengan_kembar_di_train": int(len(mx)),
             "persen_val_test": round(100 * len(mx) / n_vt, 3), "tidak_tertangkap_karena_ambang": 0}]
    for X in (0.01, 0.02, 0.03, 0.05):
        miss = int((mx < X).sum())
        rows.append({"ambang_sebaran": f"{X:.0%}", "val_test_dengan_kembar_di_train": int(len(mx)), "persen_val_test": round(100 * len(mx) / n_vt, 3),
                     "tidak_tertangkap_karena_ambang": miss, "persen_tidak_tertangkap": round(100 * miss / n_vt, 3)})
    pd.DataFrame(rows).to_csv(OUT / "sift_audit_thresholds.csv", index=False, encoding="utf-8")
    rep["audit_k200_tanpa_sebaran"] = {"n_val_test": n_vt, "per_pasangan_split": {k: int(v["query"].nunique()) for k, v in df.groupby("pasangan_split")},
                                       "val_test_dengan_kembar_di_train": int(len(mx)), "persen": round(100 * len(mx) / n_vt, 3),
                                       "per_sumber": {k: int(v["query"].nunique()) for k, v in vt.groupby("q_sumber")}, "dampak_ambang": rows}
    # ---- montase: sebaran < 2% (gabungan pasangan audit + pasangan yang ditolak saat build prov_v4)
    sp = pd.read_csv(ROOT / "reports/prov_v4/sift_pairs.csv")
    split_of = {r["orig_path"]: r["split"] for r in man}
    rj = sp[(sp.inlier >= 21) & (sp.frac_kotak_inlier < 0.05)]
    pool = [(r.a, r.b, int(r.inlier), float(r.frac_kotak_inlier), split_of.get(r.a, "-"), split_of.get(r.b, "-"), f"{r.kelas_a}/{r.sumber_a}", f"{r.kelas_b}/{r.sumber_b}") for r in rj.itertuples()]
    pool += [(r.query, r.partner, r.inlier, r.sebaran, r.q_split, r.p_split, f"{r.q_kelas}/{r.q_sumber}", f"{r.p_kelas}/{r.p_sumber}") for r in df[df.sebaran < 0.05].itertuples()]
    seen, uniq = set(), []
    for x in pool:
        k = tuple(sorted((x[0], x[1])))
        if k not in seen:
            seen.add(k)
            uniq.append(x)
    lt2 = [x for x in uniq if x[3] < 0.02]
    m25 = [x for x in uniq if 0.02 <= x[3] < 0.05 and x[4] != x[5]]
    rnd = random.Random(5)
    for name, lst, tag in (("frac_below_2pct.jpg", lt2, "sebaran<2%"), ("frac_2_5pct_cross_split.jpg", m25, "sebaran 2-5%, lintas split")):
        s = rnd.sample(lst, min(20, len(lst)))
        montage([(x[0], x[1]) for x in s], [(f"{tag} inlier={x[2]} sebaran={x[3]:.4f}", f"{x[6]} [{x[4]}] ~ {x[7]} [{x[5]}]") for x in s], OUT / name)
        rep[name] = {"n_tersedia": len(lst), "n_ditampilkan": len(s),
                     "rincian": [{"no": n + 1, "a": Path(x[0]).name, "b": Path(x[1]).name, "inlier": x[2], "sebaran": round(x[3], 4), "split": f"{x[4]}~{x[5]}"} for n, x in enumerate(s)]}
    (OUT / "sift_prep.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if not k.endswith(".jpg")}, indent=1, ensure_ascii=False, default=str))
    for k in ("frac_below_2pct.jpg", "frac_2_5pct_cross_split.jpg"):
        print(k, rep[k]["n_tersedia"], "tersedia")


if __name__ == "__main__":
    main()
