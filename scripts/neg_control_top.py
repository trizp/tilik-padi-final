#!/usr/bin/env python3
"""
neg_control_top.py - Kalibrasi SIFT: kontrol NEGATIF (pasangan acak BLB ds1 dengan ID foto berbeda) dan tampilan
10 pasangan dengan inlier tertinggi (> 20) berdampingan, agar dinilai apakah ada yang tampak duplikat sungguhan.

Pasangan negatif = dua gambar BLB ds1 ber-ID dengan (keluarga, ID) berbeda. Ini BUKAN jaminan bukan-duplikat
(hanya definisi kontrol); karena itu pasangan teratas diperiksa secara visual.
Keluaran: reports/prov/neg_control_top.jpg, reports/prov/neg_control_top.csv, reports/prov/neg_control_stats.json
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
import dedupe_rules as dr  # noqa: E402

ROOT = dr.ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/classes_v2.yaml")
    ap.add_argument("--n-pairs", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()
    cfg, classes, cands = dr.build_pool(a.config)
    ci = classes.index("bacterial_leaf_blight")
    blb = [c for c in cands if c["src"] == "ds1" and ci in c["cls_ids"] or c["tag"] == "ds1_provisional"]
    for c in blb:
        c["_p"] = dr.parse_ds1_name(c.get("orig_path", c["path"]))
    blb = [c for c in blb if c["_p"]]
    rng = random.Random(a.seed)
    seen, pairs = set(), []
    while len(pairs) < a.n_pairs:
        x, y = rng.sample(range(len(blb)), 2)
        px, py = blb[x]["_p"], blb[y]["_p"]
        if (px["fam"], px["id"]) == (py["fam"], py["id"]) or (min(x, y), max(x, y)) in seen:
            continue
        seen.add((min(x, y), max(x, y)))
        pairs.append((x, y))
    dr.ensure_descriptors([c["path"] for c in blb])
    res = dr.verify_pairs([(blb[x]["path"], blb[y]["path"]) for x, y in pairs])
    inl = np.array([r[0] for r in res])
    order = np.argsort(-inl)
    top = [(int(k), int(inl[k])) for k in order if inl[k] > 20][:10]
    stats = {"n_pasangan": len(pairs), "n_gambar_BLB_ds1_berID": len(blb),
             "inlier_median/p99/p99.9/maks": [float(np.median(inl)), float(np.percentile(inl, 99)), float(np.percentile(inl, 99.9)), int(inl.max())],
             "jumlah_>20": int((inl > 20).sum()), "jumlah_>=21 (ambang)": int((inl >= dr.MIN_INLIERS).sum()),
             "fraksi_>=21": round(float((inl >= dr.MIN_INLIERS).mean()), 6)}
    print(json.dumps(stats, ensure_ascii=False))
    with open(ROOT / "reports/prov/neg_control_top.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["peringkat", "inlier", "skala", "rotasi", "a", "b", "fam_id_a", "fam_id_b"])
        for r, k in enumerate(order[:50], 1):
            x, y = pairs[int(k)]
            px, py = blb[x]["_p"], blb[y]["_p"]
            w.writerow([r, res[int(k)][0], res[int(k)][1], res[int(k)][2], Path(blb[x].get("orig_path", blb[x]["path"])).name,
                        Path(blb[y].get("orig_path", blb[y]["path"])).name, f"{px['fam']}/{px['kind']}/{px['id']}/{px['ext']}",
                        f"{py['fam']}/{py['kind']}/{py['id']}/{py['ext']}"])
    T, W = 300, 30
    cv = Image.new("RGB", (2 * 2 * T, 5 * (T + W)), "white")
    dd = ImageDraw.Draw(cv)
    for n, (k, v) in enumerate(top):
        x, y = pairs[k]
        px, py = blb[x]["_p"], blb[y]["_p"]
        x0, y0 = (n % 2) * 2 * T, (n // 2) * (T + W)
        dd.text((x0 + 3, y0 + 3), f"#{n + 1} inlier={v} skala={res[k][1]} rot={res[k][2]}", fill="black")
        dd.text((x0 + 3, y0 + 16), f"{px['fam']}/{px['kind']}/{px['id']}/{px['ext']}  vs  {py['fam']}/{py['kind']}/{py['id']}/{py['ext']}", fill="black")
        for t_, c in enumerate((blb[x], blb[y])):
            cv.paste(Image.open(c["path"]).convert("RGB").resize((T, T)), (x0 + t_ * T, y0 + W))
    cv.save(ROOT / "reports/prov/neg_control_top.jpg", quality=88)
    stats["top10_inlier"] = [v for _, v in top]
    stats["histogram_inlier_>=21"] = {"21-35": int(((inl >= 21) & (inl <= 35)).sum()), "36-60": int(((inl >= 36) & (inl <= 60)).sum()),
                                      "61-100": int(((inl >= 61) & (inl <= 100)).sum()), ">100": int((inl > 100).sum())}
    # pasangan di dekat ambang (21-35): sampel acak 10 untuk penilaian visual risiko false positive
    border = [int(k) for k in np.where((inl >= 21) & (inl <= 35))[0]]
    sel = random.Random(a.seed).sample(border, min(10, len(border)))
    cv2_ = Image.new("RGB", (2 * 2 * T, 5 * (T + W)), "white")
    d2 = ImageDraw.Draw(cv2_)
    for n, k in enumerate(sel):
        x, y = pairs[k]
        px, py = blb[x]["_p"], blb[y]["_p"]
        x0, y0 = (n % 2) * 2 * T, (n // 2) * (T + W)
        d2.text((x0 + 3, y0 + 3), f"#{n + 1} inlier={res[k][0]} skala={res[k][1]} rot={res[k][2]}", fill="black")
        d2.text((x0 + 3, y0 + 16), f"{px['fam']}/{px['kind']}/{px['id']}/{px['ext']}  vs  {py['fam']}/{py['kind']}/{py['id']}/{py['ext']}", fill="black")
        for t_, c in enumerate((blb[x], blb[y])):
            cv2_.paste(Image.open(c["path"]).convert("RGB").resize((T, T)), (x0 + t_ * T, y0 + W))
    cv2_.save(ROOT / "reports/prov/neg_control_borderline.jpg", quality=88)
    (ROOT / "reports/prov/neg_control_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
