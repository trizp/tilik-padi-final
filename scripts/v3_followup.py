#!/usr/bin/env python3
"""
v3_followup.py - Bahan pemeriksaan visual dan pengecekan grup untuk prov_v3 (tidak mengubah dataset):
  A. cross_class_pairs.jpg : 30 pasangan lintas kelas (15 inlier > 100, 15 inlier 21-100; campur ds1/ds4)
  B. watermark_rejected_pairs.jpg : 10 pasangan yang DITOLAK aturan sebaran inlier (< 15% luas gambar)
  C. nasib grup 116 gambar dari prov_v2: bagaimana ia terpecah di prov_v3
Keluaran: reports/prov_v3/v3_followup.json
"""
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path(__file__).resolve().parent.parent
RD = ROOT / "reports/prov_v3"


def montage(pairs, captions, out, per_row, tile):
    W = 30
    rows = (len(pairs) + per_row - 1) // per_row
    cv = Image.new("RGB", (per_row * 2 * tile, rows * (tile + W)), "white")
    dd = ImageDraw.Draw(cv)
    for k, (a, b) in enumerate(pairs):
        x0, y0 = (k % per_row) * 2 * tile, (k // per_row) * (tile + W)
        dd.text((x0 + 3, y0 + 3), captions[k][0], fill="black")
        dd.text((x0 + 3, y0 + 16), captions[k][1], fill="black")
        for t, p in enumerate((a, b)):
            cv.paste(Image.open(ROOT / p).convert("RGB").resize((tile, tile)), (x0 + t * tile, y0 + W))
    cv.save(out, quality=88)


def pick(df, n, rnd):
    """Pilih n pasangan: campur ds1~ds1 dan ds4~ds4 (setengah-setengah bila bisa), tiap gambar dipakai sekali."""
    used, out = set(), []
    for src_pair in (("ds1", "ds1"), ("ds4", "ds4"), None):
        want = n - len(out) if src_pair is None else (n + 1) // 2 if src_pair[0] == "ds1" else n - len(out)
        sub = df if src_pair is None else df[(df.sumber_a == src_pair[0]) & (df.sumber_b == src_pair[1])]
        idx = list(sub.index)
        rnd.shuffle(idx)
        got = 0
        for i in idx:
            r = df.loc[i]
            if r.a in used or r.b in used or i in {o for o in out}:
                continue
            out.append(i)
            used |= {r.a, r.b}
            got += 1
            if got >= want or len(out) >= n:
                break
        if len(out) >= n:
            break
    return out


def main():
    rep = {}
    sp = pd.read_csv(RD / "sift_pairs.csv")
    rnd = random.Random(7)
    cc = sp[(sp.status == "diterima") & (sp.kelas_a != sp.kelas_b)]
    hi = cc[cc.inlier > 100]
    lo = cc[(cc.inlier >= 21) & (cc.inlier <= 100)]
    sel = pick(hi, 15, rnd) + pick(lo, 15, rnd)
    pairs = [(cc.loc[i].a, cc.loc[i].b) for i in sel]
    caps = [(f"{'KARANTINA ' if cc.loc[i].inlier > 100 else ''}inlier={cc.loc[i].inlier} frac={cc.loc[i].frac_kotak_inlier}",
             f"{cc.loc[i].kelas_a}/{cc.loc[i].sumber_a} ~ {cc.loc[i].kelas_b}/{cc.loc[i].sumber_b}") for i in sel]
    montage(pairs, caps, RD / "cross_class_pairs.jpg", 3, 220)
    rep["A_lintas_kelas"] = {"n_ditampilkan": len(sel), "inlier>100": int((cc.loc[sel].inlier > 100).sum()),
                             "per_sumber": dict(Counter(f"{cc.loc[i].sumber_a}~{cc.loc[i].sumber_b}" for i in sel)),
                             "total_lintas_kelas_diterima": int(len(cc)), "total_inlier>100": int(len(hi)), "total_21-100": int(len(lo))}

    rj = sp[sp.status == "ditolak_sebaran_kecil"]
    idx = list(rj.index)
    rnd.shuffle(idx)
    idx = idx[:10]
    montage([(rj.loc[i].a, rj.loc[i].b) for i in idx],
            [(f"DITOLAK inlier={rj.loc[i].inlier} frac_kotak={rj.loc[i].frac_kotak_inlier}", f"{rj.loc[i].kelas_a}/{rj.loc[i].sumber_a} ~ {rj.loc[i].kelas_b}/{rj.loc[i].sumber_b}")
             for i in idx], RD / "watermark_rejected_pairs.jpg", 2, 300)
    rep["B_ditolak"] = {"total_ditolak": int(len(rj)), "inlier_median/p90/maks": [float(rj.inlier.median()), float(rj.inlier.quantile(.9)), int(rj.inlier.max())],
                        "frac_median": float(rj.frac_kotak_inlier.median()),
                        "per_sumber": {f"{a}~{b}": int(n) for (a, b), n in rj.groupby(["sumber_a", "sumber_b"]).size().items()},
                        "ditampilkan": [{"a": Path(rj.loc[i].a).name, "b": Path(rj.loc[i].b).name, "inlier": int(rj.loc[i].inlier), "frac": float(rj.loc[i].frac_kotak_inlier)} for i in idx]}

    # ---- C. grup 116 dari prov_v2
    v2 = list(csv.DictReader(open(ROOT / "data/processed/prov_v2/manifest.csv", encoding="utf-8")))
    v3 = {r["orig_path"]: r for r in csv.DictReader(open(ROOT / "data/processed/prov_v3/manifest.csv", encoding="utf-8"))}
    g2 = defaultdict(list)
    for r in v2:
        g2[r["group"]].append(r)
    biggest = sorted(g2.values(), key=len, reverse=True)[0]
    here = [v3.get(r["orig_path"]) for r in biggest]
    present = [x for x in here if x]
    part = Counter(x["group"] for x in present)
    rep["C_grup116_prov_v2"] = {"ukuran_di_v2": len(biggest), "komposisi_v2": dict(Counter(f"{r['class']}|{r['source']}" for r in biggest).most_common()),
                                "masih_ada_di_v3": len(present), "dikarantina_atau_dibuang": len(biggest) - len(present),
                                "terpecah_menjadi_grup_v3": len(part), "potongan_terbesar": part.most_common(3),
                                "ukuran_potongan": sorted(part.values(), reverse=True)[:15]}
    g3 = defaultdict(list)
    for r in v3.values():
        g3[r["group"]].append(r)
    rep["C_grup_terbesar_v3"] = [{"ukuran": len(v), "komposisi": dict(Counter(f"{r['class']}|{r['source']}" for r in v).most_common()),
                                 "split": dict(Counter(r["split"] for r in v))} for v in sorted(g3.values(), key=len, reverse=True)[:5]]
    # ---- D. rincian sisa audit (SIFT pada k=100 tetangga embedding di train/val), termasuk gambar ds2 (path diperkecil)
    import hashlib
    import numpy as np
    import aug_detect as ad
    import dedupe_rules as dr
    allm = list(csv.DictReader(open(ROOT / "data/processed/prov_v3/manifest.csv", encoding="utf-8")))

    def real_path(r):
        if r["source"] == "ds2":
            return ROOT / "data/interim/resized/ds2" / (hashlib.sha1(r["orig_path"].encode()).hexdigest()[:16] + ".jpg")
        return ROOT / r["orig_path"]
    paths = [real_path(r) for r in allm]
    E = ad.embeddings(paths)
    spl = np.array([r["split"] for r in allm])
    dr.ensure_descriptors(paths)
    resid = []
    for q_sp, r_sp in (("val", "train"), ("test", "train"), ("test", "val")):
        Q, R = np.where(spl == q_sp)[0], np.where(spl == r_sp)[0]
        P = dr.knn_pairs_cross(E[Q], E[R], 100)
        res = dr.verify_pairs([(paths[Q[i]], paths[R[j]]) for i, j in P])
        for (i, j), r in zip(P, res):
            if r[0] >= dr.MIN_INLIERS and r[3] >= dr.MIN_FRAC:
                resid.append({"pasangan_split": f"{q_sp}-{r_sp}", "query": allm[Q[i]]["orig_path"], "q_kelas": allm[Q[i]]["class"], "q_sumber": allm[Q[i]]["source"],
                              "partner": allm[R[j]]["orig_path"], "p_kelas": allm[R[j]]["class"], "p_sumber": allm[R[j]]["source"],
                              "inlier": int(r[0]), "skala": r[1], "rot": r[2], "frac": r[3]})
    resid.sort(key=lambda x: -x["inlier"])
    rep["D_sisa_audit"] = {"n_pasangan": len(resid), "n_gambar_query": len({x["query"] for x in resid}),
                           "inlier_median/maks": [float(np.median([x["inlier"] for x in resid])) if resid else None, max((x["inlier"] for x in resid), default=None)],
                           "detail": [{k: v for k, v in x.items()} for x in resid[:30]]}
    with open(RD / "audit_residual.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(resid[0]) if resid else ["query"])
        w.writeheader()
        w.writerows(resid)
    if resid:
        top = resid[:12]
        montage([(x["query"], x["partner"]) for x in top], [(f"{x['pasangan_split']} inlier={x['inlier']} frac={x['frac']} sk={x['skala']}", f"{x['q_kelas']}/{x['q_sumber']} ~ {x['p_kelas']}/{x['p_sumber']}") for x in top],
                RD / "audit_residual.jpg", 2, 300)
    (RD / "v3_followup.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(rep, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
