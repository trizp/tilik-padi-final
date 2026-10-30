#!/usr/bin/env python3
"""
blb_pair_analysis2.py - Lanjutan blb_pair_analysis.py: verifikasi GEOMETRIS (SIFT + RANSAC) untuk menilai apakah
pasangan "ID kembar" benar-benar foto yang sama. phash dan embedding global gagal pada rotasi bebas/zoom, jadi
dipakai pencocokan fitur lokal yang invarian terhadap rotasi, skala, dan kecerahan.

HANYA MEMBACA; tidak membangun ulang dataset; gambar split TEST tidak dibuka (hanya train/val).
Keluaran: reports/prov/blb_pair_analysis2.json, blb_id_pairs.jpg (ditimpa dengan metrik tambahan)
"""
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
import aug_detect as ad  # noqa: E402
import blb_pair_analysis as b  # noqa: E402
from prepare_dataset import ROOT  # noqa: E402

MAXPX = 512
NFEAT = 400
_BF = None


def _desc(path):
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    s = MAXPX / max(g.shape)
    if s < 1:
        g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    g = cv2.medianBlur(g, 3)                       # meredam noise garam-merica
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)   # menormalkan kecerahan/kontras
    sift = cv2.SIFT_create(nfeatures=NFEAT)
    out = []
    for gg in (g, cv2.flip(g, 1)):                 # asli dan flip horizontal
        k, d = sift.detectAndCompute(gg, None)
        if d is None:
            out.append((np.zeros((0, 2), np.float32), np.zeros((0, 128), np.uint8)))
        else:
            out.append((np.float32([p.pt for p in k]), d.astype(np.uint8)))
    return out


def match(da, db):
    """Jumlah inlier RANSAC (transformasi similaritas) antara dua gambar; maks atas {asli, flip}. Kembalikan (inlier, skala, rotasi)."""
    global _BF
    if _BF is None:
        _BF = cv2.BFMatcher(cv2.NORM_L2)
    best = (0, 1.0, 0.0)
    for fa in (0, 1):
        pa, xa = da[fa]
        pb, xb = db[0]
        if len(xa) < 8 or len(xb) < 8:
            continue
        m = _BF.knnMatch(xa.astype(np.float32), xb.astype(np.float32), k=2)
        good = [x[0] for x in m if len(x) == 2 and x[0].distance < 0.75 * x[1].distance]
        if len(good) < 6:
            continue
        src = pa[[g.queryIdx for g in good]]
        dst = pb[[g.trainIdx for g in good]]
        M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0, maxIters=3000, confidence=0.995)
        if M is None:
            continue
        n = int(inl.sum())
        sc = math.hypot(M[0, 0], M[1, 0])
        if 0.3 <= sc <= 3.3 and n > best[0]:
            best = (n, round(sc, 3), round(math.degrees(math.atan2(M[1, 0], M[0, 0])), 1))
    return best


def main():
    rep = {}
    blb = [r for r in b.MAN if r["class"] == "bacterial_leaf_blight"]
    for r in blb:
        r["_p"] = b.parse_name(r["orig_path"])
    withid = [r for r in blb if r["_p"]]
    tv = [r for r in blb if r["split"] in ("train", "val")]
    tv_idx = {r["orig_path"]: i for i, r in enumerate(tv)}

    print("descriptor SIFT (train+val) ...", flush=True)
    with ProcessPoolExecutor() as ex:
        D = list(ex.map(_desc, [str(ROOT / r["orig_path"]) for r in tv], chunksize=8))

    def inl(a, c):
        return match(D[tv_idx[a["orig_path"]]], D[tv_idx[c["orig_path"]]])

    # ---------- kalibrasi: positif = versi Roboflow bernama dasar sama; negatif = ID berbeda
    base = defaultdict(list)
    for r in tv:
        st = Path(r["orig_path"]).stem
        if ".rf." in st:
            base[re.sub(r"\.rf\.[0-9a-f]+$", "", st)].append(r)
    pos_pairs = [(v[i], v[j]) for v in base.values() for i in range(len(v)) for j in range(i + 1, len(v))]
    rnd = random.Random(5)
    pos = [inl(a, c) for a, c in pos_pairs]
    neg_pairs = []
    while len(neg_pairs) < 2000:
        a, c = rnd.choice(tv), rnd.choice(tv)
        if a is c or not a["_p"] or not c["_p"]:
            continue
        if (a["_p"]["fam"], a["_p"]["id"]) != (c["_p"]["fam"], c["_p"]["id"]) and a["group"] != c["group"]:
            neg_pairs.append((a, c))
    neg = [inl(a, c)[0] for a, c in neg_pairs]
    pn = np.array([p[0] for p in pos])
    T = max(12, int(np.percentile(neg, 99.9)) + 1)
    rep["kalibrasi"] = {
        "positif_pasangan_nama_dasar_sama": {"n": len(pos), "inlier_p5/p25/median": [float(np.percentile(pn, q)) for q in (5, 25, 50)],
                                              "recall_pada_ambang": round(float((pn >= T).mean()), 3)},
        "negatif_ID_berbeda": {"n": len(neg), "inlier_median/p99/p99.9/maks": [float(np.median(neg)), float(np.percentile(neg, 99)),
                                                                           float(np.percentile(neg, 99.9)), float(max(neg))]},
        "ambang_inlier": T,
        "aturan_ambang": "max(12, p99.9 negatif + 1)"}
    print("kalibrasi:", json.dumps(rep["kalibrasi"], ensure_ascii=False), flush=True)

    # ---------- pasangan #01 / #09
    audit = list(__import__("csv").DictReader(open(ROOT / "reports/blb_audit/index.csv", encoding="utf-8")))
    ra = next(r for r in tv if r["orig_path"] == audit[0]["orig_path"])
    rb = next(r for r in tv if r["orig_path"] == audit[8]["orig_path"])
    n0, sc0, rot0 = inl(ra, rb)
    rep["pasangan_01_09"] = {"inlier": n0, "skala": sc0, "rotasi_derajat": rot0, "ambang": T, "terdeteksi": n0 >= T}

    # ---------- reproduksi 30 pasangan acak lintas split (seed sama dengan blb_pair_analysis.py)
    g = defaultdict(list)
    for r in withid:
        g[(r["_p"]["fam"], r["_p"]["id"])].append(r)
    pool = []
    for k, v in g.items():
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                a, c = v[i], v[j]
                if a["split"] != c["split"] and {a["split"], c["split"]} <= {"train", "val"}:
                    pool.append((a, c))
    prng = random.Random(21)
    sel = prng.sample(pool, min(30, len(pool)))
    pool_same = [(v[i], v[j]) for v in g.values() for i in range(len(v)) for j in range(i + 1, len(v))
                 if v[i]["split"] == v[j]["split"] == "train" and v[i]["_p"]["prefix"] != v[j]["_p"]["prefix"]]
    sel_same = prng.sample(pool_same, min(30, len(pool_same)))
    pool_neg = []
    while len(pool_neg) < 30:
        a, c = prng.choice(withid), prng.choice(withid)
        if a["split"] in ("train", "val") and c["split"] in ("train", "val") and (a["_p"]["fam"], a["_p"]["id"]) != (c["_p"]["fam"], c["_p"]["id"]):
            pool_neg.append((a, c))
    prev = json.load(open(ROOT / "reports/prov/blb_pair_analysis.json", encoding="utf-8"))["4_hipotesis"]
    assert [Path(a["orig_path"]).name[:60] for a, _ in sel] == [m["a"] for m in prev["30_pasangan"]], "seleksi 30 pasangan tidak sama dengan analisis sebelumnya"
    old = prev["30_pasangan"]

    def classify(m_old, n):
        if m_old["d16_raw"] <= 14 or m_old["d16_median5"] <= 14 or m_old["cos_raw"] >= 0.97 or m_old["cos_median3"] >= 0.97 or n >= T:
            return "jelas sama"
        if n <= int(np.percentile(neg, 95)) and min(m_old["d16_raw"], m_old["d16_median5"]) > 60 and max(m_old["cos_raw"], m_old["cos_median3"]) < 0.85:
            return "jelas berbeda"
        return "meragukan"
    rows30 = []
    for (a, c), mo in zip(sel, old):
        n, sc, rot = inl(a, c)
        rows30.append({**mo, "inlier": n, "skala": sc, "rot": rot, "kategori_baru": classify(mo, n)})
    cnt = Counter(r["kategori_baru"] for r in rows30)
    rep["30_pasangan_lintas_split"] = {"ringkas_kategori_baru": dict(cnt), "ringkas_kategori_lama": dict(Counter(r["kategori"] for r in rows30)),
                                       "inlier_geometri>=ambang": int(sum(r["inlier"] >= T for r in rows30)), "rinci": rows30}
    # pembanding dan kontrol negatif pada 30-an pasangan lain
    def geo_only(pairs):
        ns = [inl(a, c)[0] for a, c in pairs]
        return {"n": len(ns), "inlier>=ambang": int(sum(x >= T for x in ns)), "inlier_median": float(np.median(ns))}
    rep["pembanding_lintas_awalan_train_train"] = geo_only(sel_same)
    rep["kontrol_negatif_ID_berbeda_30"] = geo_only(pool_neg)

    # ---------- SEMUA pasangan ID kembar lintas split train-val (n=780): proporsi foto sama menurut geometri
    all_ids = [(a, c) for (a, c) in pool]
    res_all = []
    for a, c in all_ids:
        n, sc, rot = inl(a, c)
        pa, pc = a["_p"], c["_p"]
        res_all.append({"inlier": n, "sama": n >= T, "fam": pa["fam"], "kinds": f"{pa['kind']}~{pc['kind']}",
                        "ext_sama": pa["ext"] == pc["ext"], "awalan": "~".join(sorted((pa["prefix"], pc["prefix"]))),
                        "prefix_sama": pa["prefix"] == pc["prefix"]})
    def rate(sub):
        return {"n": len(sub), "foto_sama(inlier>=ambang)": int(sum(x["sama"] for x in sub)),
                "proporsi": round(sum(x["sama"] for x in sub) / len(sub), 3) if sub else None}
    rep["semua_ID_kembar_train_val"] = {
        "total": rate(res_all),
        "per_keluarga": {f: rate([x for x in res_all if x["fam"] == f]) for f in sorted({x["fam"] for x in res_all})},
        "per_ext": {"ext_sama": rate([x for x in res_all if x["ext_sama"]]), "ext_beda": rate([x for x in res_all if not x["ext_sama"]])},
        "per_awalan": {k: rate([x for x in res_all if x["awalan"] == k]) for k in sorted({x["awalan"] for x in res_all})},
        "per_jenis": {k: rate([x for x in res_all if x["kinds"] == k]) for k in sorted({x["kinds"] for x in res_all})},
        "kombinasi_fam_ext_awalan_terbaik": sorted(
            [{"kunci": f"{f}|ext_sama={e}|{p}", **rate([x for x in res_all if x["fam"] == f and x["ext_sama"] == e and x["awalan"] == p])}
             for f in {x["fam"] for x in res_all} for e in (True, False) for p in {x["awalan"] for x in res_all}
             if any(x["fam"] == f and x["ext_sama"] == e and x["awalan"] == p for x in res_all)], key=lambda z: -z["n"])[:12]}

    # ---------- dampak aturan pengelompokan (train+val; tanpa membangun ulang)
    Eall = ad.embeddings([ROOT / r["orig_path"] for r in tv])
    O = Eall[:, 0]
    S = np.max([Eall[:, v] @ O.T for v in range(8)], axis=0)
    np.fill_diagonal(S, 0)
    ii, jj = np.where(np.triu(S, 1) >= 0.88)
    cand_pairs = list(zip(ii.tolist(), jj.tolist()))
    rep["kandidat_embedding>=0.88"] = len(cand_pairs)
    cand_ok = []
    for i, j in cand_pairs:
        if inl(tv[i], tv[j])[0] >= T:
            cand_ok.append((tv[i], tv[j]))
    id_pairs_tv = [(v[i], v[j]) for v in g.values() for i in range(len(v)) for j in range(i + 1, len(v))
                   if v[i]["split"] in ("train", "val") and v[j]["split"] in ("train", "val")]
    id_ok = [(a, c) for a, c in id_pairs_tv if inl(a, c)[0] >= T]

    def uf_impact(label, extra):
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
        return {"aturan": label, "n_gambar_train_val": len(tv), "n_grup": len(gs), "grup_terbesar": max(len(x) for x in gs),
                "grup_>1_gambar": sum(len(x) > 1 for x in gs),
                "val_dengan_saudara_di_train": sum(1 for x in gs for r in x if r["split"] == "val" and any(y["split"] == "train" for y in x))}
    rep["dampak"] = [
        uf_impact("R0 grup saat ini", []),
        uf_impact("R6 + ID kembar (fam,id) terverifikasi geometri", id_ok),
        uf_impact("R7 + pasangan embedding>=0.88 terverifikasi geometri (tanpa ID)", cand_ok),
        uf_impact("R8 = R6 + R7", id_ok + cand_ok),
        uf_impact("R9 + SEMUA ID kembar (fam,id) tanpa verifikasi [pembanding]", id_pairs_tv)]
    rep["n_pasangan"] = {"ID_kembar_train_val": len(id_pairs_tv), "ID_kembar_terverifikasi": len(id_ok), "kandidat_embedding_terverifikasi": len(cand_ok)}
    # irisan: pasangan yang ditemukan geometri tetapi TIDAK oleh ID
    idset = {(id(a), id(c)) for a, c in id_ok} | {(id(c), id(a)) for a, c in id_ok}
    rep["n_pasangan"]["geometri_tanpa_ID_sama"] = sum(1 for a, c in cand_ok if (id(a), id(c)) not in idset)

    # ---------- montase 10 pasangan (sama seperti sebelumnya) dengan metrik geometri
    show = prng.sample(rows30, min(10, len(rows30)))
    srcs = {Path(a["orig_path"]).name[:60]: (a, c) for a, c in sel}
    T_, W_ = 300, 34
    cv_ = Image.new("RGB", (2 * 2 * T_, 5 * (T_ + W_)), "white")
    dr = ImageDraw.Draw(cv_)
    for k, m in enumerate(show):
        a, c = srcs[m["a"]]
        x0, y0 = (k % 2) * 2 * T_, (k // 2) * (T_ + W_)
        dr.text((x0 + 3, y0 + 3), f"#{k + 1} {m['tipe']} | {m['a_split']}~{m['b_split']} | SIFT inlier={m['inlier']} (T={T}) skala={m['skala']} rot={m['rot']}", fill="black")
        dr.text((x0 + 3, y0 + 17), f"d16={m['d16_raw']}/{m['d16_median5']} cos={m['cos_raw']}/{m['cos_median3']} -> {m['kategori_baru']}", fill="black")
        cv_.paste(b.load(ROOT / a["orig_path"]).resize((T_, T_)), (x0, y0 + W_))
        cv_.paste(b.load(ROOT / c["orig_path"]).resize((T_, T_)), (x0 + T_, y0 + W_))
    cv_.save(ROOT / "reports/prov/blb_id_pairs.jpg", quality=88)
    rep["montase_10"] = [{"no": k + 1, "tipe": m["tipe"], "inlier": m["inlier"], "kategori_baru": m["kategori_baru"]} for k, m in enumerate(show)]

    (ROOT / "reports/prov/blb_pair_analysis2.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    keep = {k: v for k, v in rep.items() if k != "30_pasangan_lintas_split"}
    print(json.dumps(keep, indent=1, ensure_ascii=False, default=str))
    print("\n== 30 pasangan ==")
    print(json.dumps({k: v for k, v in rep["30_pasangan_lintas_split"].items() if k != "rinci"}, ensure_ascii=False))
    for r in rows30:
        print(f"  {r['tipe']:32s} inl={r['inlier']:3d} sk={r['skala']:<5} rot={r['rot']:<6} d16={r['d16_raw']:3d}/{r['d16_median5']:3d} cos={r['cos_raw']:.2f} lama={r['kategori']:13s} baru={r['kategori_baru']}")


if __name__ == "__main__":
    main()
