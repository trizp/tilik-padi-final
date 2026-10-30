#!/usr/bin/env python3
"""
blb_pair_analysis.py - Analisis pasangan duplikat BLB ds1 yang lolos dari dedupe (temuan audit manual #01/#09).

HANYA MEMBACA. Tidak membangun ulang dataset dan tidak mengubah apa pun di data/. Gambar split TEST tidak
dibuka/diukur pikselnya; keberadaan di test hanya dihitung dari metadata (nama file + manifest).

Keluaran: reports/prov/blb_pair_analysis.json, blb_pair_analysis.md, blb_id_pairs.jpg
Pemakaian: python scripts/blb_pair_analysis.py
"""
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).parent))
import aug_detect as ad  # noqa: E402
from prepare_dataset import ROOT  # noqa: E402

THR16, THR8 = 14, 3            # 14/256 = 5.5% ; padanan 8x8 (64 bit): 3.5 -> 3
MAN = list(csv.DictReader(open(ROOT / "data/processed/prov/manifest.csv", encoding="utf-8")))
RE_ID = re.compile(r"^(?P<origin>r6k|rls)_(?P<sub>train|test|val|valid)_(?P<fam>img_blb|blight|img_brown|brownspot)_"
                   r"(?:(?P<kind>orig|rotated)_)?(?P<id>\d+)(?:_(?P<ext>[A-Za-z]+))?(?:_jpg)?$")


def parse_name(path):
    stem = re.sub(r"\.rf\.[0-9a-f]+$", "", Path(path).stem)
    m = RE_ID.match(stem)
    if not m:
        return None
    d = m.groupdict()
    d["id"] = int(d["id"])
    d["kind"] = d["kind"] or "-"
    d["ext"] = d["ext"] or "-"
    d["prefix"] = f"{d['origin']}_{d['sub']}"
    return d


# ------------------------------------------------------------------ hash / embedding helpers
def preprocess(im, mode):
    if "median" in mode:
        im = im.filter(ImageFilter.MedianFilter(3 if "median3" in mode else 5))
    if "resize50" in mode:
        im = im.resize((max(8, im.width // 2), max(8, im.height // 2)), Image.LANCZOS)
    if "resize128" in mode:
        s = 128 / max(im.size)
        im = im.resize((max(8, round(im.width * s)), max(8, round(im.height * s))), Image.LANCZOS)
    return im


def dih_hash(im, hs):
    import imagehash
    out = []
    for k in range(4):
        r = im.rotate(90 * k, expand=True) if k else im
        out.append(imagehash.phash(r, hash_size=hs).hash.flatten())
        out.append(imagehash.phash(r.transpose(Image.FLIP_LEFT_RIGHT), hash_size=hs).hash.flatten())
    return np.array(out)


def dmin(HA, HB):
    """Jarak Hamming minimum antara 8 varian dihedral A dan gambar asli B. Kembalikan (jarak, indeks varian)."""
    return min((int((HA[v] != HB[0]).sum()), v) for v in range(8))


def load(path):
    return Image.open(path).convert("RGB")


def pair_hash_report(A, B, modes, sizes=(16, 8)):
    out = {}
    for mode in modes:
        a, b = preprocess(A, mode), preprocess(B, mode)
        for hs in sizes:
            d, v = dmin(dih_hash(a, hs), dih_hash(b, hs))
            thr = THR16 if hs == 16 else THR8
            v0 = int((dih_hash(a, hs)[0] != dih_hash(b, hs)[0]).sum())
            out[f"h{hs}|{mode}"] = {"jarak_dihedral": d, "varian": v, "jarak_varian0": v0, "bit": hs * hs,
                                    "frak": round(d / (hs * hs), 4), "<=ambang": d <= thr}
    return out


def cos_pair(A, B):
    E = ad.embed_items([A, B], batch=2, workers=0)
    return float(max(E[0, v] @ E[1, 0] for v in range(8)))


def speckle_density(im):
    g = np.asarray(im.convert("L"), np.float32)
    med = np.asarray(im.convert("L").filter(ImageFilter.MedianFilter(5)), np.float32)
    return float(((g - med) > 50).mean())


def salt_pepper(im, p, rng):
    a = np.asarray(im).copy()
    m = rng.random(a.shape[:2])
    a[m < p / 2] = 255
    a[(m >= p / 2) & (m < p)] = 0
    return Image.fromarray(a)


# ------------------------------------------------------------------ main
def main():
    rep = {}
    blb = [r for r in MAN if r["class"] == "bacterial_leaf_blight"]
    for r in blb:
        r["_p"] = parse_name(r["orig_path"])
    byfile = {r["orig_path"]: r for r in MAN}

    # ========== 1. pasangan #01 / #09
    audit = list(csv.DictReader(open(ROOT / "reports/blb_audit/index.csv", encoding="utf-8")))
    pa, pb = audit[0]["orig_path"], audit[8]["orig_path"]
    ra, rb = byfile[pa], byfile[pb]
    A, B = load(ROOT / pa), load(ROOT / pb)
    modes = ["none", "median3", "median5", "resize50", "resize128", "median5+resize50", "median3+resize128"]
    ph = pair_hash_report(A, B, modes)
    cos_raw = cos_pair(A, B)
    cos_med3 = cos_pair(preprocess(A, "median3"), preprocess(B, "median3"))
    cos_med5 = cos_pair(preprocess(A, "median5"), preprocess(B, "median5"))
    members = defaultdict(list)
    for r in MAN:
        members[r["group"]].append(r)
    rep["1_pasangan"] = {
        "A": {"audit": "#01", "file": Path(pa).name, "split": ra["split"], "grup": ra["group"], "ukuran_grup": len(members[ra["group"]]),
              "ukuran_px": A.size, "kepadatan_bintik_putih": round(speckle_density(A), 5)},
        "B": {"audit": "#09", "file": Path(pb).name, "split": rb["split"], "grup": rb["group"], "ukuran_grup": len(members[rb["group"]]),
              "ukuran_px": B.size, "kepadatan_bintik_putih": round(speckle_density(B), 5)},
        "phash": ph,
        "cosine": {"raw": round(cos_raw, 4), "median3": round(cos_med3, 4), "median5": round(cos_med5, 4), "ambang": 0.97},
        "nama_dasar_roboflow": {"A": re.sub(r"\.rf\.[0-9a-f]+$", "", Path(pa).stem), "B": re.sub(r"\.rf\.[0-9a-f]+$", "", Path(pb).stem)},
        "anggota_grup_A": [Path(x["orig_path"]).name[:48] for x in members[ra["group"]]],
        "anggota_grup_B": [Path(x["orig_path"]).name[:48] for x in members[rb["group"]]],
        "split_asli_ds1": {"A": pa.split("/")[-2], "B": pb.split("/")[-2]},
    }
    print("[1] selesai", flush=True)

    # ========== 2. kontrol positif sintetis (garam-merica) pada gambar BLB train/val
    tv = [r for r in blb if r["split"] in ("train", "val")]
    rng = random.Random(7)
    samp = rng.sample(tv, 100)
    nrng = np.random.default_rng(11)
    dens = [0.002, 0.005, 0.01, 0.02, 0.05]
    O = [load(ROOT / r["orig_path"]) for r in samp]
    Eo_raw = ad.embed_items(O, batch=16, workers=0)
    Eo_m3 = ad.embed_items([preprocess(x, "median3") for x in O], batch=16, workers=0)
    Ho = {m: [dih_hash(preprocess(x, m), 16) for x in O] for m in ("none", "median3", "median5")}
    ctrl = {}
    for p in dens:
        N = [salt_pepper(x, p, nrng) for x in O]
        En_raw = ad.embed_items(N, batch=16, workers=0)
        En_m3 = ad.embed_items([preprocess(x, "median3") for x in N], batch=16, workers=0)
        row = {}
        for m in ("none", "median3", "median5"):
            hits = [dmin(dih_hash(preprocess(n, m), 16), Ho[m][i])[0] for i, n in enumerate(N)]
            row[f"phash16|{m}|<=14"] = round(float(np.mean([h <= THR16 for h in hits])), 3)
            row[f"phash16|{m}|jarak_median"] = float(np.median(hits))
        c_raw = [max(En_raw[i, v] @ Eo_raw[i, 0] for v in range(8)) for i in range(len(N))]
        c_m3 = [max(En_m3[i, v] @ Eo_m3[i, 0] for v in range(8)) for i in range(len(N))]
        row["embedding|raw|>=0.97"] = round(float(np.mean(np.array(c_raw) >= 0.97)), 3)
        row["embedding|raw|cos_median"] = round(float(np.median(c_raw)), 4)
        row["embedding|median3|>=0.97"] = round(float(np.mean(np.array(c_m3) >= 0.97)), 3)
        row["embedding|median3|cos_median"] = round(float(np.median(c_m3)), 4)
        ctrl[f"p={p}"] = row
    rep["2_kontrol_sintetis"] = {"n_gambar": len(samp), "kepadatan": dens, "hasil": ctrl,
                                 "catatan": "kepadatan = fraksi piksel yang diganti (setengah putih, setengah hitam)"}
    print("[2] selesai", flush=True)

    # ========== 3. ID foto
    rows = []
    for r in blb:
        d = r["_p"]
        rows.append({"file": Path(r["orig_path"]).name, "split_sekarang": r["split"], "folder_ds1": r["orig_path"].split("/")[-2],
                     "awalan": d["prefix"] if d else "", "fam": d["fam"] if d else "", "kind": d["kind"] if d else "",
                     "id": d["id"] if d else "", "ext": d["ext"] if d else "", "group": r["group"]})
    with open(ROOT / "reports/prov/blb_id_table.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    withid = [r for r in blb if r["_p"]]
    rep["3_id"] = {"gambar_BLB": len(blb), "dengan_ID_terparse": len(withid), "tanpa_ID": len(blb) - len(withid),
                   "tanpa_ID_contoh_bentuk": dict(Counter(re.sub(r"\d+", "N", Path(r["orig_path"]).name.split(".rf.")[0])
                                                          for r in blb if not r["_p"]).most_common(5)),
                   "keluarga": dict(Counter(r["_p"]["fam"] + ("/" + r["_p"]["kind"] if r["_p"]["kind"] != "-" else "") for r in withid)),
                   "awalan": dict(Counter(r["_p"]["prefix"] for r in withid)),
                   "ext": dict(Counter(r["_p"]["ext"] for r in withid))}
    keys = {"K1 (fam,id) [orig+rotated digabung]": lambda p: (p["fam"], p["id"]),
            "K2 (fam,kind,id)": lambda p: (p["fam"], p["kind"], p["id"]),
            "K3 (fam,id,ext)": lambda p: (p["fam"], p["id"], p["ext"])}
    stat = {}
    for kn, kf in keys.items():
        g = defaultdict(list)
        for r in withid:
            g[kf(r["_p"])].append(r)
        multi_img = {k: v for k, v in g.items() if len(v) > 1}
        multi_split = {k: v for k, v in g.items() if len({x["split"] for x in v}) > 1}
        multi_prefix = {k: v for k, v in g.items() if len({x["_p"]["prefix"] for x in v}) > 1}
        combos = Counter("+".join(sorted({x["split"] for x in v})) for v in multi_split.values())
        stat[kn] = {"ID_unik": len(g), "ID_dengan_>1_gambar": len(multi_img), "ID_di_>1_split": len(multi_split),
                    "kombinasi_split": dict(combos), "gambar_pada_ID_lintas_split": sum(len(v) for v in multi_split.values()),
                    "ID_dengan_>1_awalan": len(multi_prefix),
                    "ID_>1_awalan_tapi_1_split": len([k for k in multi_prefix if k not in multi_split])}
    rep["3_id"]["statistik"] = stat
    print("[3] selesai", flush=True)

    # ========== 4. uji hipotesis: ID kembar lintas split (train-val saja; test tidak dibuka)
    g = defaultdict(list)
    for r in withid:
        g[(r["_p"]["fam"], r["_p"]["id"])].append(r)
    pool = []
    for k, v in g.items():
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                a, b = v[i], v[j]
                if a["split"] != b["split"] and {a["split"], b["split"]} <= {"train", "val"}:
                    pool.append((a, b))
    rep["4_hipotesis"] = {"pasangan_ID_kembar_lintas_split_train_val": len(pool)}
    prng = random.Random(21)
    sel = prng.sample(pool, min(30, len(pool)))
    # pembanding: ID kembar lintas AWALAN dalam split yang sama (train-train)
    pool_same = [(v[i], v[j]) for v in g.values() for i in range(len(v)) for j in range(i + 1, len(v))
                 if v[i]["split"] == v[j]["split"] == "train" and v[i]["_p"]["prefix"] != v[j]["_p"]["prefix"]]
    sel_same = prng.sample(pool_same, min(30, len(pool_same)))
    # kontrol negatif: pasangan acak BLB train-val dengan ID BERBEDA
    pool_neg = []
    while len(pool_neg) < 30:
        a, b = prng.choice(withid), prng.choice(withid)
        if a["split"] in ("train", "val") and b["split"] in ("train", "val") and (a["_p"]["fam"], a["_p"]["id"]) != (b["_p"]["fam"], b["_p"]["id"]):
            pool_neg.append((a, b))

    def measure(pairs):
        out = []
        for a, b in pairs:
            IA, IB = load(ROOT / a["orig_path"]), load(ROOT / b["orig_path"])
            d_raw = dmin(dih_hash(IA, 16), dih_hash(IB, 16))
            d_med = dmin(dih_hash(preprocess(IA, "median5"), 16), dih_hash(preprocess(IB, "median5"), 16))
            c_raw = cos_pair(IA, IB)
            c_med = cos_pair(preprocess(IA, "median3"), preprocess(IB, "median3"))
            dmn, cmx = min(d_raw[0], d_med[0]), max(c_raw, c_med)
            if dmn <= THR16 or cmx >= 0.97:
                cat = "jelas sama"
            elif min(d_raw[0], d_med[0]) > 60 and cmx < 0.85:
                cat = "jelas berbeda"
            else:
                cat = "meragukan"
            pa_, pb_ = a["_p"], b["_p"]
            out.append({"a": Path(a["orig_path"]).name[:60], "b": Path(b["orig_path"]).name[:60], "a_split": a["split"], "b_split": b["split"],
                        "tipe": f"{pa_['fam']}/{pa_['kind']}-{pb_['kind']}" + ("" if pa_["ext"] == pb_["ext"] else " (ext beda)"),
                        "awalan": f"{pa_['prefix']}~{pb_['prefix']}",
                        "d16_raw": d_raw[0], "d16_median5": d_med[0], "cos_raw": round(c_raw, 4), "cos_median3": round(c_med, 4),
                        "kategori": cat, "_pa": a["orig_path"], "_pb": b["orig_path"], "_var": d_raw[1] if d_raw[0] <= d_med[0] else d_med[1]})
        return out
    m_cross = measure(sel)
    m_same = measure(sel_same)
    m_neg = measure(pool_neg)

    def summar(ms):
        c = Counter(m["kategori"] for m in ms)
        return {"n": len(ms), **{k: c.get(k, 0) for k in ("jelas sama", "jelas berbeda", "meragukan")},
                "d16_raw_median": float(np.median([m["d16_raw"] for m in ms])) if ms else None,
                "cos_raw_median": float(np.median([m["cos_raw"] for m in ms])) if ms else None}
    rep["4_hipotesis"].update({
        "kriteria": "jelas sama: min(phash16 raw, phash16 median5) <= 14 ATAU max(cos raw, cos median3) >= 0.97; jelas berbeda: min phash > 60 DAN cos < 0.85; selain itu meragukan",
        "lintas_split_train_val": summar(m_cross), "pembanding_lintas_awalan_train_train": summar(m_same),
        "kontrol_negatif_ID_berbeda": summar(m_neg),
        "per_tipe_lintas_split": {t: dict(Counter(m["kategori"] for m in m_cross if m["tipe"] == t)) for t in sorted({m["tipe"] for m in m_cross})},
        "30_pasangan": [{k: v for k, v in m.items() if not k.startswith("_")} for m in m_cross]})
    # montase 10 pasangan acak dari 30
    show = prng.sample(m_cross, min(10, len(m_cross)))
    T, W = 300, 34
    cv = Image.new("RGB", (2 * 2 * T, 5 * (T + W)), "white")
    dr = ImageDraw.Draw(cv)
    for k, m in enumerate(show):
        x0, y0 = (k % 2) * 2 * T, (k // 2) * (T + W)
        dr.text((x0 + 3, y0 + 3), f"#{k + 1} {m['tipe']} | {m['a_split']}~{m['b_split']}", fill="black")
        dr.text((x0 + 3, y0 + 17), f"d16={m['d16_raw']}/{m['d16_median5']}(med5) cos={m['cos_raw']}/{m['cos_median3']}(med3) -> {m['kategori']}", fill="black")
        cv.paste(load(ROOT / m["_pa"]).resize((T, T)), (x0, y0 + W))
        cv.paste(load(ROOT / m["_pb"]).resize((T, T)), (x0 + T, y0 + W))
    cv.save(ROOT / "reports/prov/blb_id_pairs.jpg", quality=88)
    rep["4_hipotesis"]["montase"] = [{"no": k + 1, "a": m["a"], "b": m["b"], "kategori": m["kategori"]} for k, m in enumerate(show)]
    print("[4] selesai", flush=True)

    # ========== 5. dampak aturan pengelompokan (tanpa membangun ulang)
    # pasangan ID kembar yang bisa diverifikasi piksel (train/val saja)
    tv_ids = [r for r in withid if r["split"] in ("train", "val")]
    paths_tv = [ROOT / r["orig_path"] for r in tv_ids]
    idx = {r["orig_path"]: i for i, r in enumerate(tv_ids)}
    Hmed = [dih_hash(preprocess(load(p), "median5"), 16) for p in paths_tv]
    Hraw = [dih_hash(load(p), 16) for p in paths_tv]
    Eall = ad.embeddings(paths_tv)

    def uf_groups(items, extra_pairs):
        parent = {id(x): id(x) for x in items}
        gid = {}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a, b):
            ra, rb_ = find(id(a)), find(id(b))
            if ra != rb_:
                parent[ra] = rb_
        first = {}
        for x in items:                                  # grup saat ini (manifest)
            if x["group"] in first:
                union(x, first[x["group"]])
            else:
                first[x["group"]] = x
        for a, b in extra_pairs:
            union(a, b)
        out = defaultdict(list)
        for x in items:
            out[find(id(x))].append(x)
        return list(out.values())

    def impact(items, extra_pairs, label):
        gs = uf_groups(items, extra_pairs)
        leak_val = sum(1 for g_ in gs for x in g_ if x["split"] == "val" and any(y["split"] == "train" for y in g_))
        leak_test = sum(1 for g_ in gs for x in g_ if x["split"] == "test" and any(y["split"] == "train" for y in g_))
        return {"aturan": label, "n_gambar": len(items), "n_grup": len(gs), "grup_terbesar": max(len(g_) for g_ in gs),
                "grup_>1_gambar": sum(len(g_) > 1 for g_ in gs), "val_dengan_saudara_di_train": leak_val,
                "test_dengan_saudara_di_train": leak_test}
    all_blb = withid                                     # semua BLB ber-ID (termasuk test: HANYA lewat metadata)
    id_pairs_all = []
    gid_ = defaultdict(list)
    for r in all_blb:
        gid_[(r["_p"]["fam"], r["_p"]["id"])].append(r)
    for v in gid_.values():
        for i in range(1, len(v)):
            id_pairs_all.append((v[0], v[i]))
    gid3 = defaultdict(list)
    for r in all_blb:
        gid3[(r["_p"]["fam"], r["_p"]["id"], r["_p"]["ext"])].append(r)
    id_pairs_ext = [(v[0], v[i]) for v in gid3.values() for i in range(1, len(v))]
    # R3: ID kembar DAN terverifikasi piksel (hanya pasangan train/val)
    ver = []
    for a, b in id_pairs_all:
        if a["split"] in ("train", "val") and b["split"] in ("train", "val"):
            i, j = idx[a["orig_path"]], idx[b["orig_path"]]
            d = min(dmin(Hmed[i], Hmed[j])[0], dmin(Hraw[i], Hraw[j])[0])
            c = float(max(Eall[i, v] @ Eall[j, 0] for v in range(8)))
            if d <= THR16 or c >= 0.97:
                ver.append((a, b))
    # R4: pixel saja (median5 + raw, dihedral) semua pasangan train/val
    H0 = np.array([h[0] for h in Hmed], np.float32)
    seen_pix = set()
    A_all = [np.array([Hmed[i][v] for i in range(len(Hmed))], np.float32) for v in range(8)]
    for v in range(8):
        d = A_all[v].sum(1)[:, None] + H0.sum(1)[None, :] - 2 * (A_all[v] @ H0.T)
        ii, jj = np.where(d <= THR16 + 0.5)
        seen_pix |= {(min(i, j), max(i, j)) for i, j in zip(ii.tolist(), jj.tolist()) if i != j}
    pix = [(tv_ids[i], tv_ids[j]) for i, j in sorted(seen_pix)]
    # R5: embedding saja >= 0.97 (dengan 8 varian) -- sudah dipakai pipeline; ditampilkan untuk pembanding
    res = []
    res.append(impact(all_blb, [], "R0 grup saat ini (phash14 + embedding0.97 + nama dasar Roboflow)"))
    res.append(impact(all_blb, id_pairs_all, "R1 + ID kembar (fam,id) lintas awalan [metadata, termasuk test]"))
    res.append(impact(all_blb, id_pairs_ext, "R2 + ID kembar dengan ext sama (fam,id,ext) [metadata, termasuk test]"))
    sub = [r for r in all_blb if r["split"] in ("train", "val")]
    res.append(impact(sub, [], "R0' (train+val saja) grup saat ini"))
    res.append(impact(sub, ver, "R3 + ID kembar yang terverifikasi piksel (phash16<=14 raw/median5 atau cos>=0.97) [train+val]"))
    res.append(impact(sub, pix, "R4 + pasangan piksel saja: phash16 median5 dihedral<=14, tanpa ID [train+val]"))
    res.append(impact(sub, pix + ver, "R5 = R3 + R4 [train+val]"))
    rep["5_dampak"] = {"n_ID_kembar_total": len(id_pairs_all), "n_ID_kembar_terverifikasi_piksel_train_val": len(ver),
                       "n_pasangan_piksel_median5_train_val": len(pix), "hasil": res,
                       "catatan": "R1/R2 menghitung test lewat metadata nama file saja; R3-R5 hanya train+val karena piksel test tidak dibuka."}
    print("[5] selesai", flush=True)

    # ========== 6. ds1 Blast / Brown spot (train-only) vs val/test ds4 dan ds2: kecocokan ID (angka saja)
    def num(name):
        m = re.findall(r"(\d+)", Path(name).stem)
        return int(m[-1]) if m else None
    ds1_train = [r for r in MAN if r["source"] == "ds1"]
    out6 = {}
    for cls in ("brown_spot", "leaf_blast"):
        rs = [r for r in ds1_train if r["class"] == cls]
        for r in rs:
            r["_p"] = parse_name(r["orig_path"])
        parsed = [r for r in rs if r["_p"]]
        id_set = {r["_p"]["id"] for r in parsed}
        entry = {"gambar_ds1": len(rs), "dengan_ID_terparse": len(parsed), "ID_unik": len(id_set)}
        for src, label in (("ds4", "ds4"), ("ds2", "ds2")):
            for sp in ("val", "test"):
                tgt = [r for r in MAN if r["source"] == src and r["split"] == sp and (r["class"] == cls or (src == "ds2"))]
                nums = {num(r["orig_path"]) for r in tgt} - {None}
                entry[f"{label}_{sp}_gambar_kelas_sama" if src == "ds4" else f"{label}_{sp}_gambar_(background)"] = len(tgt)
                entry[f"{label}_{sp}_ID_angka_sama"] = len(id_set & nums)
        out6[cls] = entry
    # pemeriksaan piksel hanya untuk val (test tidak dibuka): pasangan dengan ID angka sama ds1 Brown spot vs ds4 val Brown spot
    v_pairs = []
    ds1_brown = [r for r in ds1_train if r["class"] == "brown_spot" and r.get("_p")]
    ds4_val = [r for r in MAN if r["source"] == "ds4" and r["split"] == "val" and r["class"] == "brown_spot"]
    byn = defaultdict(list)
    for r in ds4_val:
        byn[num(r["orig_path"])].append(r)
    for r in ds1_brown:
        for q in byn.get(r["_p"]["id"], []):
            v_pairs.append((r, q))
    hits = 0
    for a, b in v_pairs:
        IA, IB = load(ROOT / a["orig_path"]), load(ROOT / b["orig_path"])
        d = min(dmin(dih_hash(IA, 16), dih_hash(IB, 16))[0], dmin(dih_hash(preprocess(IA, "median5"), 16), dih_hash(preprocess(IB, "median5"), 16))[0])
        c = cos_pair(IA, IB)
        hits += d <= THR16 or c >= 0.97
    out6["pemeriksaan_piksel_val_ds1_brown_vs_ds4_val_brown"] = {"pasangan_ID_angka_sama": len(v_pairs), "terdeteksi_duplikat": hits}
    rep["6_ds1_vs_ds4_ds2"] = out6
    print("[6] selesai", flush=True)

    (ROOT / "reports/prov/blb_pair_analysis.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k not in ("4_hipotesis",)}, indent=1, ensure_ascii=False, default=str)[:20000])
    h4 = rep["4_hipotesis"]
    print("\n== 4 ==\n", json.dumps({k: h4[k] for k in h4 if k not in ("30_pasangan", "montase")}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
