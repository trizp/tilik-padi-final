#!/usr/bin/env python3
"""
v5_prep_time.py - Persiapan prov_v5 (TIDAK membangun dataset, TIDAK mengubah label):
  * jumlah foto bertanggal per sumber, sebelum (pembacaan lama) dan sesudah (pembacaan diperbaiki)
  * histogram selisih waktu antar foto berurutan (skala log) + pencarian jeda alami
  * tabel jendela 2/5/10/30 detik (+ beberapa jendela tambahan) terhadap pembagian prov_v4
  * audit waktu baseline prov_v4: gambar val/test dengan foto train dalam +-2 dan +-10 detik
Keluaran: reports/prov_v5_prep/ (time_counts.csv, time_gap_hist.png, window_table.csv, time_prep.json)
"""
import csv
import json
import re
import sys
from bisect import bisect_left
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports/prov_v5_prep"
V4 = ROOT / "data/processed/prov_v4"
RX_OLD = r"(20\d{2})(\d{2})(\d{2})[_-](\d{2})(\d{2})(\d{2})"
RX_NEW = r"(20\d{2})(\d{2})(\d{2})[_-]?(\d{2})(\d{2})(\d{2})"


def parse_ts(s, rx):
    m = re.search(rx, s)
    if not m:
        return None
    try:
        t = pd.Timestamp(f"{m.group(1)}-{m.group(2)}-{m.group(3)} {m.group(4)}:{m.group(5)}:{m.group(6)}")
    except Exception:
        return None
    return t if 2015 <= t.year <= 2026 else None


def ds4_map(rx):
    xl = pd.read_excel(ROOT / "data/raw/ds4/docs/Dataset metadata.xlsx", sheet_name="Image_Metadata")
    return {str(n).lower(): parse_ts(str(o), rx) for n, o in zip(xl["New_Filename"], xl["Original_Filename"])}


def ds1_ts(path, rx):
    stem = re.sub(r"\.rf\.[0-9a-f]+$", "", Path(path).stem)
    return parse_ts(stem, rx)


def chains(items, gap):
    """items: list (t, key). Gabung berantai bila selisih berurutan <= gap detik. Kembalikan dict key -> id_rantai."""
    items = sorted(items, key=lambda x: x[0])
    cid, out = 0, {}
    for k, (t, key) in enumerate(items):
        if k and (t - items[k - 1][0]).total_seconds() > gap:
            cid += 1
        out[key] = cid
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {}
    man = list(csv.DictReader(open(V4 / "manifest.csv", encoding="utf-8")))
    quar = [r["berkas"] for r in csv.DictReader(open(ROOT / "reports/prov_v4/quarantine.csv", encoding="utf-8"))]
    m4_old, m4_new = ds4_map(RX_OLD), ds4_map(RX_NEW)

    def ts_of(src, path, which):
        if src == "ds4":
            return (m4_new if which == "new" else m4_old).get(Path(path).name.lower())
        if src.startswith("ds1"):
            return ds1_ts(path, RX_NEW) if which == "new" else None      # pipeline lama tidak membaca jam ds1 sama sekali
        return None

    # ---------- 1. jumlah bertanggal per sumber (mentah dan dataset)
    raw = {"ds1": [p for p in (ROOT / "data/raw/ds1").rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}],
           "ds2": [p for p in (ROOT / "data/raw/ds2").rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}],
           "ds3": [p for p in (ROOT / "data/raw/ds3").rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}],
           "ds4": [p for p in (ROOT / "data/raw/ds4/images_labels/RiceLeafDiseaseBD/Original images").rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]}
    rows = []
    for src, ps in raw.items():
        old = sum(1 for p in ps if (m4_old.get(p.name.lower()) if src == "ds4" else None) is not None)
        new = sum(1 for p in ps if (m4_new.get(p.name.lower()) if src == "ds4" else (ds1_ts(p, RX_NEW) if src == "ds1" else parse_ts(p.stem, RX_NEW))) is not None)
        mm = [r for r in man if r["source"].split("_")[0] == src]
        rows.append({"sumber": src, "foto_mentah": len(ps), "bertanggal_lama": old, "bertanggal_baru": new,
                     "di_dataset_prov_v4": len(mm), "di_dataset_bertanggal_lama": sum(1 for r in mm if ts_of(r["source"], r["orig_path"], "old") is not None),
                     "di_dataset_bertanggal_baru": sum(1 for r in mm if ts_of(r["source"], r["orig_path"], "new") is not None)})
    pd.DataFrame(rows).to_csv(OUT / "time_counts.csv", index=False, encoding="utf-8")
    rep["1_bertanggal"] = rows

    # ---------- 2. histogram selisih berurutan (skala log)
    pool_paths = {r["orig_path"]: r for r in man}
    for q in quar:
        pool_paths.setdefault(q, {"orig_path": q, "source": "ds4" if "/ds4/" in q else "ds1", "split": "karantina", "group": f"q:{q}", "class": "?"})
    series = {}
    for src in ("ds4", "ds1"):
        ts = sorted(t for r in pool_paths.values() if r["source"].split("_")[0] == src for t in [ts_of(r["source"], r["orig_path"], "new")] if t is not None)
        d = np.diff(np.array([t.value for t in ts])) / 1e9
        series[src] = d
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.2))
    bins = np.logspace(np.log10(0.5), np.log10(10 ** 6.5), 60)
    gaps = {}
    for ax, src in zip(axs, ("ds4", "ds1")):
        d = np.maximum(series[src], 0.5)
        ax.hist(d, bins=bins, color="#4c78a8")
        ax.set_xscale("log")
        ax.set_title(f"{src}: selisih waktu antar foto berurutan (n={len(d)})")
        ax.set_xlabel("detik (skala log; 0 detik digambar di 0,5)")
        ax.set_ylabel("jumlah")
        for x in (2, 5, 10, 30):
            ax.axvline(x, color="#e45756", lw=1, ls="--")
        # jeda alami: titik terendah histogram (dihaluskan) antara 2 dan 600 detik
        h, e = np.histogram(np.log10(d), bins=np.linspace(np.log10(0.5), 6.5, 60))
        hs = np.convolve(h, np.ones(3) / 3, mode="same")
        cen = 10 ** ((e[:-1] + e[1:]) / 2)
        sel = (cen >= 2) & (cen <= 600)
        gaps[src] = {"jeda_alami_detik(perkiraan)": round(float(cen[sel][np.argmin(hs[sel])]), 1),
                     "persen_selisih": {f"<= {x}s": round(float((series[src] <= x).mean() * 100), 1) for x in (0, 1, 2, 5, 10, 30, 60, 300, 3600)}}
        ax.axvline(gaps[src]["jeda_alami_detik(perkiraan)"], color="#54a24b", lw=2)
    fig.tight_layout()
    fig.savefig(OUT / "time_gap_hist.png", dpi=130)
    rep["2_histogram"] = gaps

    # ---------- 3. tabel jendela
    sp = pd.read_csv(ROOT / "reports/prov_v4/sift_pairs.csv")
    acc = sp[sp.status == "diterima"]
    bb = acc[(acc.sumber_a == "ds4") & (acc.sumber_b == "ds4") & acc.apply(lambda r: {r.kelas_a, r.kelas_b} == {"leaf_blast", "brown_spot"}, axis=1)]
    oth4 = acc[(acc.sumber_a == "ds4") & (acc.sumber_b == "ds4")].drop(bb.index)
    oth1 = acc[(acc.sumber_a == "ds1") & (acc.sumber_b == "ds1")]
    itm = {src: [(t, r["orig_path"]) for r in pool_paths.values() if r["source"].split("_")[0] == src
                 for t in [ts_of(r["source"], r["orig_path"], "new")] if t is not None] for src in ("ds4", "ds1")}
    split_of = {r["orig_path"]: r["split"] for r in man}
    group_of = {r["orig_path"]: r["group"] for r in man}
    table = []
    for W in (2, 5, 10, 30, 3, 15, 20, 60):
        ch = {}
        for src in ("ds4", "ds1"):
            ch.update({k: f"{src}:{v}" for k, v in chains(itm[src], W).items()})

        def cap(df):
            ok = [(ch.get(a), ch.get(b)) for a, b in zip(df.a, df.b)]
            n_both = sum(1 for x, y in ok if x is not None and y is not None)
            return sum(1 for x, y in ok if x is not None and x == y), n_both, len(df)
        c_bb, nb_bb, n_bb = cap(bb)
        c_o4, nb_o4, n_o4 = cap(oth4)
        c_o1, nb_o1, n_o1 = cap(oth1)
        # gabungan grup prov_v4 + rantai waktu (hanya gambar di dataset)
        parent = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for r in man:
            parent[find("g:" + r["group"])] = find("p:" + r["orig_path"]) if False else find("g:" + r["group"])
        for r in man:
            c_ = ch.get(r["orig_path"])
            if c_ is not None:
                ra, rb = find("g:" + r["group"]), find("t:" + c_)
                if ra != rb:
                    parent[ra] = rb
        comp = defaultdict(list)
        for r in man:
            comp[find("g:" + r["group"])].append(r)
        sizes = sorted((len(v) for v in comp.values()), reverse=True)
        vt = defaultdict(int)
        for v in comp.values():
            if any(x["split"] == "train" for x in v):
                for x in v:
                    if x["split"] in ("val", "test"):
                        vt[x["source"]] += 1
        table.append({"jendela_detik": W, "jumlah_grup": len(comp), "grup_terbesar": sizes[0], "grup_terbesar_ke2": sizes[1],
                      "val_test_ds4_berbagi_grup_dengan_train": vt.get("ds4", 0), "val_test_ds1_BLB_berbagi_grup_dengan_train": vt.get("ds1_provisional", 0),
                      "pasangan_BlastBrown_tertangkap": f"{c_bb}/{nb_bb}", "persen_BlastBrown": round(100 * c_bb / max(1, nb_bb), 1),
                      "pasangan_SIFT_ds4_lain_tertangkap": f"{c_o4}/{nb_o4}", "persen_ds4_lain": round(100 * c_o4 / max(1, nb_o4), 1),
                      "pasangan_SIFT_ds1_tertangkap": f"{c_o1}/{nb_o1}", "persen_ds1": round(100 * c_o1 / max(1, nb_o1), 1),
                      "BlastBrown_tanpa_jam": n_bb - nb_bb})
    dt = pd.DataFrame(table)
    dt.to_csv(OUT / "window_table.csv", index=False, encoding="utf-8")
    rep["3_tabel_jendela"] = table
    n_vt4 = sum(1 for r in man if r["source"] == "ds4" and r["split"] in ("val", "test"))
    rep["3_catatan"] = {"val_test_ds4_total": n_vt4, "val_test_ds1_BLB_total": sum(1 for r in man if r["source"] == "ds1_provisional" and r["split"] in ("val", "test"))}
    # usulan: jendela terkecil dengan tangkapan >= 95% (Blast-Brown dan ds4 lain) dan grup terbesar < ~100
    cand = [t for t in sorted(table, key=lambda t: t["jendela_detik"]) if t["persen_BlastBrown"] >= 95 and t["persen_ds4_lain"] >= 95]
    rep["3_usulan"] = {"memenuhi_95%": [t["jendela_detik"] for t in cand],
                       "terkecil_dengan_grup_terbesar<100": next((t["jendela_detik"] for t in cand if t["grup_terbesar"] < 100), None)}

    # ---------- 5. audit waktu baseline prov_v4
    aud = {}
    for src in ("ds4", "ds1"):
        tr = sorted(t for r in man if r["source"].split("_")[0] == src and r["split"] == "train" for t in [ts_of(r["source"], r["orig_path"], "new")] if t is not None)
        for W in (2, 10):
            n = 0
            q = [r for r in man if r["source"].split("_")[0] == src and r["split"] in ("val", "test")]
            for r in q:
                t = ts_of(r["source"], r["orig_path"], "new")
                if t is None:
                    continue
                i = bisect_left(tr, t - pd.Timedelta(seconds=W))
                n += i < len(tr) and tr[i] <= t + pd.Timedelta(seconds=W)
            aud[f"{src}_+-{W}s"] = {"val_test": len(q), "dengan_foto_train_dalam_jendela": int(n), "persen": round(100 * n / max(1, len(q)), 2)}
    n_all = sum(1 for r in man if r["split"] in ("val", "test"))
    for W in (2, 10):
        tot = aud[f"ds4_+-{W}s"]["dengan_foto_train_dalam_jendela"] + aud[f"ds1_+-{W}s"]["dengan_foto_train_dalam_jendela"]
        aud[f"semua_+-{W}s"] = {"val_test": n_all, "dengan_foto_train_dalam_jendela": tot, "persen": round(100 * tot / n_all, 2)}
    rep["5_audit_waktu_prov_v4"] = aud
    (OUT / "time_prep.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(rep, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
