#!/usr/bin/env python3
"""
v2_followup.py - Analisis lanjutan untuk dataset prov_v2 (tidak mengubah dataset):
  A. pasangan SIFT lintas kelas (rincian + montase)
  B. detail sisa audit (gambar val/test dengan saudara di train pada k=50)
  C. alasan gambar test lama yang dibuang pada prov_v2
  D. metadata ds4 (Dataset metadata.xlsx): kolom, proksi waktu (epoch pada Original_Filename), proksi perangkat
     (dimensi asli), dan dampak pengelompokan berbasis metadata (TIDAK diterapkan)
Piksel test dibaca hanya untuk deteksi duplikat (izin pengguna). Tidak ada metrik model.
Keluaran: reports/prov_v2/followup.json (+ cross_class_pairs.jpg, audit_residual.jpg)
"""
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
import aug_detect as ad  # noqa: E402
import dedupe_rules as dr  # noqa: E402

ROOT = dr.ROOT
V2 = ROOT / "data/processed/prov_v2"
RD = ROOT / "reports/prov_v2"


def montage(pairs, out, labels):
    T, W = 300, 30
    n = len(pairs)
    cv = Image.new("RGB", (2 * 2 * T, ((n + 1) // 2) * (T + W)), "white")
    dd = ImageDraw.Draw(cv)
    for k, (a, b) in enumerate(pairs):
        x0, y0 = (k % 2) * 2 * T, (k // 2) * (T + W)
        dd.text((x0 + 3, y0 + 3), labels[k][0], fill="black")
        dd.text((x0 + 3, y0 + 16), labels[k][1], fill="black")
        for t, p in enumerate((a, b)):
            cv.paste(Image.open(ROOT / p).convert("RGB").resize((T, T)), (x0 + t * T, y0 + W))
    cv.save(out, quality=88)


def main():
    only_d = "--only-D" in sys.argv
    rep = {}
    man = list(csv.DictReader(open(V2 / "manifest.csv", encoding="utf-8")))
    byp = {r["orig_path"]: r for r in man}
    sp = pd.read_csv(RD / "sift_pairs.csv")
    sp21 = sp[sp["inlier"] >= dr.MIN_INLIERS]

    if not only_d:
        pass
    # ---------- A. lintas kelas
    cc = sp21[sp21["kelas_a"] != sp21["kelas_b"]]
    rep["A_lintas_kelas"] = {
        "total_pasangan>=21": int(len(sp21)), "lintas_kelas": int(len(cc)),
        "per_pasangan_kelas": {f"{a}~{b}": int(n) for (a, b), n in cc.assign(k=cc.apply(lambda r: tuple(sorted((r.kelas_a, r.kelas_b))), axis=1)).groupby("k").size().items()},
        "per_pasangan_sumber": {f"{a}~{b}": int(n) for (a, b), n in cc.assign(k=cc.apply(lambda r: tuple(sorted((r.sumber_a, r.sumber_b))), axis=1)).groupby("k").size().items()},
        "inlier_median/p90/maks": [float(cc.inlier.median()), float(cc.inlier.quantile(.9)), int(cc.inlier.max())],
        "gambar_unik_terlibat": int(len(set(cc.a) | set(cc.b)))}
    hi = cc.sort_values("inlier", ascending=False).head(4)
    lo = cc[(cc.inlier <= 35)].sample(min(4, int((cc.inlier <= 35).sum())), random_state=1)
    sel = pd.concat([hi, lo])
    montage([(r.a, r.b) for r in sel.itertuples()], RD / "cross_class_pairs.jpg",
            [(f"inlier={r.inlier} sk={r.skala} rot={r.rotasi_derajat}", f"{r.kelas_a}/{r.sumber_a} ~ {r.kelas_b}/{r.sumber_b}") for r in sel.itertuples()])
    rep["A_lintas_kelas"]["montase"] = [{"inlier": int(r.inlier), "a": Path(r.a).name, "b": Path(r.b).name} for r in sel.itertuples()]

    # ---------- B. detail sisa audit pada k=50 (ds4 dan ds1; ds2 dilewati karena path berkas diperkecil)
    keep = [r for r in man if r["source"] in ("ds4", "ds1", "ds1_provisional")]
    paths = [ROOT / r["orig_path"] for r in keep]
    E = ad.embeddings(paths)
    sp_ = np.array([r["split"] for r in keep])
    resid = []
    for q_sp in ("val", "test"):
        Q, R = np.where(sp_ == q_sp)[0], np.where(sp_ == "train")[0]
        P = dr.knn_pairs_cross(E[Q], E[R], 50)
        dr.ensure_descriptors([paths[Q[i]] for i in range(len(Q))] + [paths[R[j]] for j in range(len(R))])
        res = dr.verify_pairs([(paths[Q[i]], paths[R[j]]) for i, j in P])
        for (i, j), r in zip(P, res):
            if r[0] >= dr.MIN_INLIERS:
                resid.append({"split": q_sp, "query": keep[Q[i]]["orig_path"], "q_kelas": keep[Q[i]]["class"], "q_sumber": keep[Q[i]]["source"],
                              "train_partner": keep[R[j]]["orig_path"], "p_kelas": keep[R[j]]["class"], "p_sumber": keep[R[j]]["source"],
                              "inlier": r[0], "skala": r[1], "rot": r[2],
                              "rank_catatan": "di luar 30 kandidat grup bila partner tak berada dalam 30 tetangga"})
    rep["B_sisa_audit"] = {"n_pasangan": len(resid), "detail": resid[:20]}
    if resid:
        montage([(x["query"], x["train_partner"]) for x in resid[:8]], RD / "audit_residual.jpg",
                [(f"{x['split']} vs train | inlier={x['inlier']} sk={x['skala']} rot={x['rot']}", f"{x['q_kelas']}/{x['q_sumber']} ~ {x['p_kelas']}/{x['p_sumber']}") for x in resid[:8]])

    # ---------- C. test lama yang dibuang
    tc = pd.read_csv(RD / "test_changes.csv")
    dropped = tc[tc.split_baru == "dibuang"]
    rep["C_test_lama_dibuang"] = {"n": int(len(dropped)), "per_kelas_sumber": {f"{a}|{b}": int(n) for (a, b), n in dropped.groupby(["kelas", "sumber"]).size().items()},
                                  "per_kelas_sumber_masuk_test_baru": {f"{a}|{b}": int(n) for (a, b), n in tc[tc.perubahan == "masuk_ke_test"].groupby(["kelas", "sumber"]).size().items()}}

    # ---------- D. metadata ds4
    xl = pd.read_excel(ROOT / "data/raw/ds4/docs/Dataset metadata.xlsx", sheet_name="Image_Metadata")
    D = {"kolom": list(xl.columns), "n_baris": int(len(xl))}
    rx = xl["Original_Filename"].astype(str).str.extract(r"(20\d{2})(\d{2})(\d{2})[_-](\d{2})(\d{2})(\d{2})")
    ok = rx.notna().all(axis=1)
    ts = pd.to_datetime(rx[ok].astype(str).agg("".join, axis=1), format="%Y%m%d%H%M%S", errors="coerce")
    xl["ts"] = pd.NaT
    xl.loc[ok, "ts"] = ts
    xl["ts"] = pd.to_datetime(xl["ts"])
    xl["epoch"] = (xl["ts"] - pd.Timestamp("1970-01-01")).dt.total_seconds() * 1000.0
    ok = xl["ts"].notna()
    D["Original_Filename_bertanggal_waktu (YYYYMMDD_HHMMSS)"] = {"n": int(ok.sum()), "persen": round(float(ok.mean()) * 100, 1)}
    D["pola_nama_asli_tanpa_tanggal"] = dict(Counter(re.sub(r"\d+", "N", s_) for s_ in xl.loc[~ok, "Original_Filename"].astype(str)).most_common(6))
    D["rentang_waktu"] = [str(xl.ts.min()), str(xl.ts.max())]
    D["tanggal_unik"] = int(xl.ts.dt.date.nunique())
    D["per_kelas_bertanggal"] = {c: {"n": int(g.ts.notna().sum()), "dari": str(g.ts.min()), "sampai": str(g.ts.max()), "tanggal_unik": int(g.ts.dt.date.nunique())}
                                 for c, g in xl.groupby("Class")}
    D["gambar_per_tanggal_top8"] = {str(k): int(v) for k, v in xl.ts.dt.date.value_counts().head(8).items()}
    dup = xl[xl.duplicated("Original_Filename", keep=False)].sort_values("Original_Filename")
    D["Original_Filename_ganda"] = {"n_baris": int(len(dup)), "contoh": dup[["Class", "Original_Filename", "New_Filename"]].head(18).values.tolist()}
    D["dimensi_asli_kombinasi"] = {f"{w}x{h}": int(n) for (w, h), n in xl.groupby(["Original_Width_px", "Original_Height_px"]).size().sort_values(ascending=False).head(14).items()}
    # pool ds4 pada prov_v2: petakan New_Filename -> baris metadata
    m4 = [r for r in man if r["source"] == "ds4"]
    meta = {str(n).lower(): i for i, n in zip(xl.index, xl["New_Filename"])}
    rows4 = []
    for r in m4:
        i = meta.get(Path(r["orig_path"]).name.lower())
        rows4.append({**r, "epoch": float(xl.at[i, "epoch"]) if i is not None and not np.isnan(xl.at[i, "epoch"]) else None,
                      "dim": f"{xl.at[i, 'Original_Width_px']}x{xl.at[i, 'Original_Height_px']}" if i is not None else None})
    D["ds4_di_manifest"] = {"n": len(m4), "terpetakan_ke_metadata": sum(1 for r in rows4 if r["dim"]), "dengan_epoch": sum(1 for r in rows4 if r["epoch"])}
    # kombinasi dimensi per split (proksi perangkat)
    dimsplit = defaultdict(Counter)
    for r in rows4:
        dimsplit[r["dim"]][r["split"]] += 1
    D["dimensi_per_split_ds4"] = {k: dict(v) for k, v in sorted(dimsplit.items(), key=lambda kv: -sum(kv[1].values()))[:10]}
    # sebaran tanggal per split (apakah val/test berasal dari sesi/tanggal yang juga ada di train?)
    dsplit = defaultdict(set)
    ndate = Counter()
    for r in rows4:
        if r["epoch"]:
            d_ = str(pd.to_datetime(r["epoch"], unit="ms").date())
            dsplit[r["split"]].add(d_)
            ndate[(r["split"], d_)] += 1
    tr_dates = dsplit["train"]
    D["tanggal_per_split_ds4"] = {sp_: {"tanggal_unik": len(v), "tanggal_juga_ada_di_train": len(v & tr_dates)} for sp_, v in dsplit.items()}
    D["gambar_val/test_pada_tanggal_yang_juga_ada_di_train"] = {
        sp_: {"gambar_bertanggal": sum(n for (s2, d_), n in ndate.items() if s2 == sp_),
              "pada_tanggal_ada_di_train": sum(n for (s2, d_), n in ndate.items() if s2 == sp_ and d_ in tr_dates)} for sp_ in ("val", "test")}
    # pengelompokan berbasis waktu: gabungkan gambar ds4 dengan selisih epoch <= gap detik (berantai)
    idx = [i for i, r in enumerate(rows4) if r["epoch"]]
    idx.sort(key=lambda i: rows4[i]["epoch"])
    ep_ = np.array([rows4[i]["epoch"] for i in idx])
    gid = np.zeros(len(idx), int)
    pair_ds4 = set()
    nameidx = {rows4[i]["orig_path"]: k for k, i in enumerate(idx)}
    for a, b in zip(sp21.a, sp21.b):
        if a in nameidx and b in nameidx:
            pair_ds4.add((min(nameidx[a], nameidx[b]), max(nameidx[a], nameidx[b])))
    grp_v2 = {i: rows4[i]["group"] for i in idx}
    out = {}
    for gap in (2, 10, 60, 300, 1800):
        g = 0
        for k in range(1, len(idx)):
            if (ep_[k] - ep_[k - 1]) / 1000.0 > gap:
                g += 1
            gid[k] = g
        sizes = Counter(gid.tolist())
        cap = sum(1 for a, b in pair_ds4 if gid[a] == gid[b])
        # dampak: gabungkan dengan grup v2, hitung gambar val/test yang akan sekelompok dengan train
        parent = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for k, i in enumerate(idx):
            parent[find(("g", grp_v2[i]))] = find(("t", gid[k]))
        comp = defaultdict(list)
        for k, i in enumerate(idx):
            comp[find(("g", grp_v2[i]))].append(rows4[i]["split"])
        leak = {sp: sum(c.count(sp) for c in comp.values() if "train" in c) for sp in ("val", "test")}
        out[f"selisih<={gap}s"] = {"n_grup_waktu": len(sizes), "grup_waktu_terbesar": max(sizes.values()),
                                   "gambar_di_grup_waktu>1": int(sum(v for v in sizes.values() if v > 1)),
                                   "pasangan_SIFT_ds4_ditangkap": f"{cap}/{len(pair_ds4)}",
                                   "gambar_val/test_sekelompok_dengan_train_bila_digabung_dengan_v2": leak}
    D["pengelompokan_waktu_(TIDAK diterapkan)"] = out
    rep["D_metadata_ds4"] = D
    (RD / "followup.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items()}, indent=1, ensure_ascii=False, default=str)[:15000])


if __name__ == "__main__":
    main()
