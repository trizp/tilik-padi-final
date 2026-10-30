#!/usr/bin/env python3
"""
clean_candidates_check.py - Cek kebocoran calon val/test BLB BERSIH: ds2 BLB (180 Original) dan ds3 (blight)
terhadap ds1 BLB dan ds4 (dan antar ds2/ds3), dengan SIFT pada 30 tetangga terdekat embedding.

Piksel dibaca hanya untuk deteksi duplikat. Keluaran: reports/prov_v2/clean_candidates_leak.json (+ .jpg bila ada pasangan)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
import aug_detect as ad  # noqa: E402
import dedupe_rules as dr  # noqa: E402
from inventory import IMG_EXT  # noqa: E402

ROOT = dr.ROOT


def imgs(root, cond):
    return sorted(p for p in (ROOT / root).rglob("*") if p.suffix.lower() in IMG_EXT and cond(p.as_posix().lower()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/classes_v2.yaml")
    ap.add_argument("--knn", type=int, default=30)
    a = ap.parse_args()
    cfg, classes, cands = dr.build_pool(a.config)
    ci = classes.index("bacterial_leaf_blight")
    pool_blb = [c for c in cands if c["src"] == "ds1" and False] or [c for c in cands if c["tag"] == "ds1_provisional"]
    pool_ds4 = [c for c in cands if c["src"] == "ds4"]
    ds2 = imgs("data/raw/ds2", lambda s: "/original/" in s and "bacterial leaf blight" in s)
    ds3 = imgs("data/raw/ds3", lambda s: "/blight/" in s)
    print(f"ds2 BLB Original: {len(ds2)} | ds3 blight: {len(ds3)} | ds1 BLB pool: {len(pool_blb)} | ds4 pool: {len(pool_ds4)}", flush=True)
    sets = {"ds2_BLB": ds2, "ds3_blight": ds3, "ds1_BLB": [c["path"] for c in pool_blb], "ds4": [c["path"] for c in pool_ds4]}
    names = {"ds1_BLB": [c.get("orig_path", c["path"]) for c in pool_blb], "ds4": [c.get("orig_path", c["path"]) for c in pool_ds4],
             "ds2_BLB": ds2, "ds3_blight": ds3}
    E = {k: ad.embeddings(v) for k, v in sets.items()}
    dr.ensure_descriptors([p for v in sets.values() for p in v])
    combos = [("ds2_BLB", "ds1_BLB"), ("ds2_BLB", "ds4"), ("ds3_blight", "ds1_BLB"), ("ds3_blight", "ds4"),
              ("ds2_BLB", "ds3_blight"), ("ds3_blight", "ds2_BLB")]
    out, hits_all = {}, []
    for q, t in combos:
        P = dr.knn_pairs_cross(E[q], E[t], a.knn)
        res = dr.verify_pairs([(sets[q][i], sets[t][j]) for i, j in P])
        inl = np.array([r[0] for r in res])
        ok = inl >= dr.MIN_INLIERS
        qs = {int(i) for (i, _), o in zip(P, ok) if o}
        ts = {int(j) for (_, j), o in zip(P, ok) if o}
        out[f"{q} -> {t}"] = {"kandidat": int(len(P)), "pasangan>=21": int(ok.sum()), "gambar_query_terdampak": len(qs),
                               "gambar_target_terlibat": len(ts), "inlier_maks": int(inl.max()) if len(inl) else None,
                               "inlier_p99": float(np.percentile(inl, 99)) if len(inl) else None}
        for (i, j), r, o in zip(P, res, ok):
            if o:
                hits_all.append((q, t, int(i), int(j), r))
        print(q, "->", t, json.dumps(out[f"{q} -> {t}"]), flush=True)
    out["ringkas"] = {"ds2_BLB_terdampak_oleh_ds1/ds4": len({i for q, t, i, j, r in hits_all if q == "ds2_BLB" and t in ("ds1_BLB", "ds4")}),
                      "ds3_blight_terdampak_oleh_ds1/ds4": len({i for q, t, i, j, r in hits_all if q == "ds3_blight" and t in ("ds1_BLB", "ds4")}),
                      "total_pasangan_terdeteksi": len(hits_all)}
    out["contoh"] = [{"query": f"{q}:{Path(str(names[q][i])).name}", "target": f"{t}:{Path(str(names[t][j])).name}", "inlier": r[0], "skala": r[1], "rot": r[2]}
                     for q, t, i, j, r in sorted(hits_all, key=lambda h: -h[4][0])[:15]]
    if hits_all:
        top = sorted(hits_all, key=lambda h: -h[4][0])[:6]
        T, W = 300, 18
        cv = Image.new("RGB", (2 * 2 * T, ((len(top) + 1) // 2) * (T + W)), "white")
        dd = ImageDraw.Draw(cv)
        for n, (q, t, i, j, r) in enumerate(top):
            x0, y0 = (n % 2) * 2 * T, (n // 2) * (T + W)
            dd.text((x0 + 3, y0 + 3), f"{q}~{t} inlier={r[0]} sk={r[1]} rot={r[2]}", fill="black")
            for k_, p in enumerate((sets[q][i], sets[t][j])):
                cv.paste(Image.open(p).convert("RGB").resize((T, T)), (x0 + k_ * T, y0 + W))
        cv.save(ROOT / "reports/prov_v2/clean_candidates_leak.jpg", quality=88)
    (ROOT / "reports/prov_v2").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports/prov_v2/clean_candidates_leak.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out["ringkas"], ensure_ascii=False))


if __name__ == "__main__":
    main()
