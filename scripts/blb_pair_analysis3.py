#!/usr/bin/env python3
"""
blb_pair_analysis3.py - (a) mengapa pasangan #01/#09 lolos dari phash, (b) presisi kunci ID yang diperbaiki
(keluarga, jenis, ID, ekstensi), (c) dampaknya pada pengelompokan, (d) poin 6 dengan verifikasi geometri (val saja).

HANYA MEMBACA; test tidak dibuka (hanya metadata nama file). Keluaran: reports/prov/blb_pair_analysis3.json
"""
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).parent))
import aug_detect as ad  # noqa: E402
import blb_pair_analysis as b  # noqa: E402
import blb_pair_analysis2 as g2  # noqa: E402
from prepare_dataset import ROOT  # noqa: E402


def main():
    rep = {}
    blb = [r for r in b.MAN if r["class"] == "bacterial_leaf_blight"]
    for r in blb:
        r["_p"] = b.parse_name(r["orig_path"])
    tv = [r for r in blb if r["split"] in ("train", "val")]
    test_meta = [r for r in blb if r["split"] == "test"]            # HANYA metadata
    tv_idx = {r["orig_path"]: i for i, r in enumerate(tv)}
    print("descriptor SIFT ...", flush=True)
    with ProcessPoolExecutor() as ex:
        D = list(ex.map(g2._desc, [str(ROOT / r["orig_path"]) for r in tv], chunksize=8))
    T = 21

    def inl(a, c):
        return g2.match(D[tv_idx[a["orig_path"]]], D[tv_idx[c["orig_path"]]])

    # ---------- (a) mengapa #01/#09 lolos
    audit = list(csv.DictReader(open(ROOT / "reports/blb_audit/index.csv", encoding="utf-8")))
    pa, pb = audit[0]["orig_path"], audit[8]["orig_path"]
    A, B = b.load(ROOT / pa), b.load(ROOT / pb)
    n, sc, rot = inl(next(r for r in tv if r["orig_path"] == pa), next(r for r in tv if r["orig_path"] == pb))
    # transformasi similaritas dari SIFT untuk menyejajarkan A ke B
    da = g2._desc(str(ROOT / pa))[0]
    db = g2._desc(str(ROOT / pb))[0]
    bf = cv2.BFMatcher(cv2.NORM_L2)
    m = bf.knnMatch(da[1].astype(np.float32), db[1].astype(np.float32), k=2)
    good = [x[0] for x in m if len(x) == 2 and x[0].distance < 0.75 * x[1].distance]
    src = da[0][[g.queryIdx for g in good]]
    dst = db[0][[g.trainIdx for g in good]]
    M, inlm = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0, maxIters=3000, confidence=0.995)
    s = 512 / max(A.size)                                           # koordinat SIFT berada pada skala <= 512 px
    Mf = M.copy()
    Mf[:, 2] = M[:, 2]
    # sejajarkan A ke bingkai B pada resolusi 512
    a512 = cv2.resize(np.asarray(A), None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    b512 = cv2.resize(np.asarray(B), None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    a_al = cv2.warpAffine(a512, Mf, (b512.shape[1], b512.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    PA, PB = Image.fromarray(a512), Image.fromarray(b512)
    PAal = Image.fromarray(a_al)

    def d16(x, y):
        return b.dmin(b.dih_hash(x, 16), b.dih_hash(y, 16))[0]
    def eq(x):
        return ImageOps.equalize(x)
    def gray_eq(x):
        return ImageOps.equalize(x.convert("L")).convert("RGB")
    def mean_std(x):
        a = np.asarray(x, np.float32)
        return [round(float(a.mean()), 1), round(float(a.std()), 1)]
    rep["a_mengapa_lolos"] = {
        "sift_inlier": n, "transformasi": {"skala": round(float(np.hypot(M[0, 0], M[1, 0])), 3),
                                          "rotasi_derajat": round(float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))), 2),
                                          "geser_px": [round(float(M[0, 2]), 1), round(float(M[1, 2]), 1)]},
        "kecerahan_rata2_std_A(#01)": mean_std(PA), "kecerahan_rata2_std_B(#09)": mean_std(PB),
        "phash16_dihedral": {"asli": d16(PA, PB), "setelah_equalize(histogram)": d16(eq(PA), eq(PB)),
                             "setelah_grayscale+equalize": d16(gray_eq(PA), gray_eq(PB)),
                             "setelah_median5+equalize": d16(eq(b.preprocess(PA, "median5")), eq(b.preprocess(PB, "median5"))),
                             "A_disejajarkan_ke_B(asli)": d16(PAal, PB),
                             "A_disejajarkan+median5+equalize": d16(eq(b.preprocess(PAal, "median5")), eq(b.preprocess(PB, "median5")))},
        "ambang_phash16": 14}
    print("(a)", json.dumps(rep["a_mengapa_lolos"], ensure_ascii=False), flush=True)

    # ---------- (b) presisi kunci yang diperbaiki
    def key4(r):
        p = r["_p"]
        return (p["fam"], p["kind"], p["id"], p["ext"]) if p else None
    g4 = defaultdict(list)
    for r in tv:
        if key4(r):
            g4[key4(r)].append(r)
    rows = []
    for v in g4.values():
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                a, c = v[i], v[j]
                if a["_p"]["prefix"] != c["_p"]["prefix"]:                 # lintas awalan
                    rows.append((a, c))
    res = []
    for a, c in rows:
        nn = inl(a, c)[0]
        res.append({"sama": nn >= T, "lintas_split": a["split"] != c["split"], "fam": a["_p"]["fam"],
                    "awalan": "~".join(sorted((a["_p"]["prefix"], c["_p"]["prefix"])))})

    def rate(sub):
        return {"n": len(sub), "foto_sama": int(sum(x["sama"] for x in sub)), "proporsi": round(sum(x["sama"] for x in sub) / len(sub), 4) if sub else None}
    rep["b_kunci_K4"] = {"definisi": "sama (fam,kind,id,ext) DAN awalan berbeda; pasangan train/val",
                         "semua": rate(res), "lintas_split": rate([x for x in res if x["lintas_split"]]),
                         "per_keluarga": {f: rate([x for x in res if x["fam"] == f]) for f in sorted({x["fam"] for x in res})},
                         "per_awalan": {k: rate([x for x in res if x["awalan"] == k]) for k in sorted({x["awalan"] for x in res})}}
    print("(b)", json.dumps(rep["b_kunci_K4"], ensure_ascii=False), flush=True)

    # ---------- (c) dampak; test hanya lewat metadata K4
    Eall = ad.embeddings([ROOT / r["orig_path"] for r in tv])
    S = np.max([Eall[:, v] @ Eall[:, 0].T for v in range(8)], axis=0)
    np.fill_diagonal(S, 0)
    ii, jj = np.where(np.triu(S, 1) >= 0.88)
    cand_ok = [(tv[i], tv[j]) for i, j in zip(ii.tolist(), jj.tolist()) if inl(tv[i], tv[j])[0] >= T]
    k4_pairs = [(a, c) for a, c in rows]

    def impact(label, extra):
        parent = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        first = {}
        for r in tv:
            find(id(r))
            if r["group"] in first:
                parent[find(id(r))] = find(first[r["group"]])
            else:
                first[r["group"]] = id(r)
        for a, c in extra:
            ra_, rc_ = find(id(a)), find(id(c))
            if ra_ != rc_:
                parent[ra_] = rc_
        comp = defaultdict(list)
        for r in tv:
            comp[find(id(r))].append(r)
        gs = list(comp.values())
        return {"aturan": label, "n_grup": len(gs), "grup_terbesar": max(len(x) for x in gs),
                "val_dengan_saudara_di_train": sum(1 for x in gs for r in x if r["split"] == "val" and any(y["split"] == "train" for y in x))}
    rep["c_dampak_train_val"] = [impact("R0 grup saat ini", []),
                                 impact("R10 + kunci K4 (fam,kind,id,ext; lintas awalan), tanpa verifikasi", k4_pairs),
                                 impact("R11 + pasangan embedding>=0.88 terverifikasi SIFT (tanpa ID)", cand_ok),
                                 impact("R12 = R10 + R11", k4_pairs + cand_ok)]
    # test: metadata K4 saja (tanpa membuka piksel)
    k4_train = defaultdict(int)
    for r in tv:
        if r["split"] == "train" and key4(r):
            k4_train[key4(r)] += 1
    test_with_k4 = [r for r in test_meta if key4(r) and k4_train.get(key4(r), 0) > 0]
    # hanya hitung yang awalannya berbeda dari saudara trainnya (aturan K4 = lintas awalan)
    prefs = defaultdict(set)
    for r in tv:
        if r["split"] == "train" and key4(r):
            prefs[key4(r)].add(r["_p"]["prefix"])
    test_k4_cross = [r for r in test_with_k4 if prefs[key4(r)] - {r["_p"]["prefix"]}]
    rep["c_test_metadata"] = {"BLB_test": len(test_meta), "dengan_ID_terparse": sum(1 for r in test_meta if r["_p"]),
                              "test_dengan_kembar_K4_di_train_(metadata)": len(test_with_k4),
                              "...dan_lintas_awalan(aturan_K4)": len(test_k4_cross),
                              "catatan": "hanya nama file; piksel test tidak dibuka. Presisi K4 pada train/val: lihat b_kunci_K4"}
    print("(c)", json.dumps(rep["c_dampak_train_val"], ensure_ascii=False), json.dumps(rep["c_test_metadata"], ensure_ascii=False), flush=True)

    # ---------- (d) poin 6: ds1 Brown spot (train-only) vs ds4 val Brown spot: ID angka sama, uji geometri (val saja)
    def num(name):
        m_ = re.findall(r"(\d+)", Path(name).stem)
        return int(m_[-1]) if m_ else None
    ds1b = [r for r in b.MAN if r["source"] == "ds1" and r["class"] == "brown_spot"]
    for r in ds1b:
        r["_p"] = b.parse_name(r["orig_path"])
    ds4v = [r for r in b.MAN if r["source"] == "ds4" and r["split"] == "val" and r["class"] == "brown_spot"]
    byn = defaultdict(list)
    for r in ds4v:
        byn[num(r["orig_path"])].append(r)
    pairs = [(r, q) for r in ds1b if r["_p"] for q in byn.get(r["_p"]["id"], [])]
    need = {r["orig_path"] for r, q in pairs} | {q["orig_path"] for r, q in pairs}
    pl = sorted(need)
    with ProcessPoolExecutor() as ex:
        Dm = dict(zip(pl, ex.map(g2._desc, [str(ROOT / p) for p in pl], chunksize=4)))
    ninl = [g2.match(Dm[r["orig_path"]], Dm[q["orig_path"]])[0] for r, q in pairs]
    rep["d_poin6_ds1_brown_vs_ds4_val"] = {"pasangan_ID_angka_sama": len(pairs), "inlier>=21": int(sum(x >= T for x in ninl)),
                                           "inlier_maks": int(max(ninl)) if ninl else None, "inlier_median": float(np.median(ninl)) if ninl else None}
    (ROOT / "reports/prov/blb_pair_analysis3.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print("(d)", json.dumps(rep["d_poin6_ds1_brown_vs_ds4_val"], ensure_ascii=False))


if __name__ == "__main__":
    main()
