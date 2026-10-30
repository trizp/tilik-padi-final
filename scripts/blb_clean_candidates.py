#!/usr/bin/env python3
"""
blb_clean_candidates.py - Daftar kandidat val/test BLB resmi (ds2 BLB Original + ds3 blight) dan daftar ds2 yang punya
kembar di ds1, berdasarkan aturan SIFT v3 (inlier >= 21 dan sebaran inlier >= 15%, 50 tetangga embedding).

Pembanding: ds1 (semua kelas yang masuk dataset), ds4, ds2 (Healthy yang masuk dataset + BLB Original lain),
ds3 (seluruh isi ds3 selain dirinya). Piksel dibaca hanya untuk deteksi duplikat.

Keluaran (reports/<dataset>/, dari report_dir di config):
  blb_clean_eval_candidates.csv   - kandidat yang tidak punya kembar di ds1 maupun ds4, dengan status per dataset
  blb_ds2_with_ds1_twin.csv       - gambar ds2 yang punya kembar di ds1 (TIDAK boleh jadi val/test)
  blb_clean_candidates_summary.json
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import aug_detect as ad  # noqa: E402
import dedupe_rules as dr  # noqa: E402
from inventory import IMG_EXT  # noqa: E402

ROOT = dr.ROOT


def imgs(root, cond):
    return sorted(p for p in (ROOT / root).rglob("*") if p.suffix.lower() in IMG_EXT and cond(p.as_posix().lower()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/classes_v3.yaml")
    a = ap.parse_args()
    cfg, classes, cands = dr.build_pool(a.config)
    rd = ROOT / cfg["report_dir"]
    k = cfg["grouping"]["sift"]["knn"]
    min_frac = cfg["grouping"]["sift"].get("min_inlier_bbox_frac", dr.MIN_FRAC)
    man = {r["orig_path"]: r for r in csv.DictReader(open(ROOT / cfg["output_dir"] / "manifest.csv", encoding="utf-8"))}

    def rel(p):
        return Path(p).resolve().relative_to(ROOT).as_posix()

    ds2_blb = imgs("data/raw/ds2", lambda s: "/original/" in s and "bacterial leaf blight" in s)
    ds3_all = imgs("data/raw/ds3", lambda s: True)
    ds3_blight = [p for p in ds3_all if "/blight/" in p.as_posix().lower()]
    refs = {
        "ds1_BLB": [c for c in cands if c["tag"] == "ds1_provisional"],     # dicari terpisah agar tidak tersisih oleh gambar kelas lain
        "ds1_lain": [c for c in cands if c["src"] == "ds1" and c["tag"] != "ds1_provisional"],
        "ds4": [c for c in cands if c["src"] == "ds4"],
        "ds2_healthy": [c for c in cands if c["src"] == "ds2"],
    }
    ref_paths = {n: [c["path"] for c in v] for n, v in refs.items()}
    ref_names = {n: [rel(c.get("orig_path", c["path"])) for c in v] for n, v in refs.items()}
    ref_paths["ds2_BLB_lain"] = list(ds2_blb)
    ref_names["ds2_BLB_lain"] = [rel(p) for p in ds2_blb]
    ref_paths["ds3_semua"] = list(ds3_all)
    ref_names["ds3_semua"] = [rel(p) for p in ds3_all]
    queries = [("ds2", p) for p in ds2_blb] + [("ds3", p) for p in ds3_blight]
    qpaths = [p for _, p in queries]
    print(f"kandidat: ds2 BLB Original {len(ds2_blb)}, ds3 blight {len(ds3_blight)}; pembanding: " +
          ", ".join(f"{n}={len(v)}" for n, v in ref_paths.items()), flush=True)

    allp = list(qpaths) + [p for v in ref_paths.values() for p in v]
    Eq = ad.embeddings(qpaths)
    Er = {n: ad.embeddings(v) for n, v in ref_paths.items()}
    dr.ensure_descriptors(allp)

    hits = {i: defaultdict(list) for i in range(len(queries))}     # query -> ref -> [(nama, inlier)]
    for n, Et in Er.items():
        P = dr.knn_pairs_cross(Eq, Et, k)
        pairs = [(i, j) for i, j in P if Path(qpaths[i]) != Path(ref_paths[n][j])]
        res = dr.verify_pairs([(qpaths[i], ref_paths[n][j]) for i, j in pairs])
        for (i, j), r in zip(pairs, res):
            if r[0] >= dr.MIN_INLIERS:                        # KONSERVATIF: tanpa syarat sebaran (kebocoran lebih berbahaya daripada membuang foto bersih)
                hits[i][n].append((ref_names[n][j], int(r[0]), float(r[3])))
        print(f"  vs {n}: {len(pairs)} pasangan, terdeteksi {sum(1 for i in hits for _ in hits[i].get(n, []))}", flush=True)

    # grup antar-kandidat (ds2 BLB lain, ds3 semua) agar val/test tidak memecah foto kembar di dalam kandidat
    qidx = {rel(p): i for i, p in enumerate(qpaths)}
    parent = list(range(len(queries)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i in hits:
        for n in ("ds2_BLB_lain", "ds3_semua"):
            for nm, _, _ in hits[i].get(n, []):
                if nm in qidx:
                    parent[find(i)] = find(qidx[nm])
    rows, twin_rows = [], []

    def stat(i, names):
        m = [x for n in names for x in hits[i].get(n, [])]
        return len(m), (max(v for _, v, _ in m) if m else 0)
    for i, (src, p) in enumerate(queries):
        n1, m1 = stat(i, ["ds1_BLB", "ds1_lain"])
        n4, m4 = stat(i, ["ds4"])
        n2, m2 = stat(i, ["ds2_healthy"] + (["ds2_BLB_lain"] if src == "ds2" else []))
        n3, m3 = stat(i, ["ds3_semua"] if src == "ds3" or src == "ds2" else [])
        if src == "ds3":                                   # ds3 vs ds2 BLB
            n2b, m2b = stat(i, ["ds2_BLB_lain"])
            n2, m2 = n2 + n2b, max(m2, m2b)
        row = {"berkas": rel(p), "sumber": src, "grup_kandidat": find(i),
               "bersih_ds1": n1 == 0, "bersih_ds4": n4 == 0, "bersih_ds2": n2 == 0, "bersih_ds3": n3 == 0,
               "n_kembar_ds1": n1, "inlier_maks_ds1": m1, "n_kembar_ds4": n4, "n_kembar_ds2": n2, "n_kembar_ds3": n3,
               "n_kembar_ds1_bila_pakai_filter_sebaran": sum(1 for n_ in ("ds1_BLB", "ds1_lain") for _, _, fr in hits[i].get(n_, []) if fr >= min_frac)}
        if n1:
            twins = sorted(hits[i].get("ds1_BLB", []) + hits[i].get("ds1_lain", []), key=lambda x: -x[1])
            twins = [(a_, b_) for a_, b_, _ in twins]
            twin_rows.append({**{k_: row[k_] for k_ in ("berkas", "sumber", "n_kembar_ds1", "inlier_maks_ds1")},
                              "kembar_ds1_terbaik": twins[0][0], "split_kembar_terbaik_di_dataset": man.get(twins[0][0], {}).get("split", "-"),
                              "kelas_kembar_terbaik": man.get(twins[0][0], {}).get("class", "-"),
                              "split_semua_kembar_di_dataset": "|".join(sorted({man.get(nm, {}).get("split", "-") for nm, _ in twins})),
                              "alasan": "punya kembar di ds1 (SIFT): TIDAK boleh jadi val/test"})
        rows.append(row)
    clean = [r for r in rows if r["bersih_ds1"] and r["bersih_ds4"]]
    # ---- group_id (kelompok independen di antara kandidat) dan prioritas anotasi
    root_has_twin = defaultdict(bool)                      # kelompok yang anggotanya (siapa pun) punya kembar di ds1 -> terkontaminasi transitif
    for r in rows:
        if not r["bersih_ds1"] or not r["bersih_ds4"]:
            root_has_twin[r["grup_kandidat"]] = True
    members = defaultdict(list)
    for r in clean:
        members[r["grup_kandidat"]].append(r)
    order = sorted(members.items(), key=lambda kv: (root_has_twin[kv[0]], -len(kv[1]), min(x["sumber"] for x in kv[1]), min(x["berkas"] for x in kv[1])))
    gid = {root: f"G{n:03d}" for n, (root, _) in enumerate(order, 1)}
    reps_, rest_ = [], []
    for root, mem in order:
        mem.sort(key=lambda x: x["berkas"])
        reps_.append(mem[0])
        rest_ += mem[1:]
    ranked = {id(r): n for n, r in enumerate(reps_ + rest_, 1)}
    rep_ids = {id(r) for r in reps_}
    for r in rows:
        r["group_id"] = gid.get(r["grup_kandidat"], "")
        r["ukuran_kelompok"] = len(members.get(r["grup_kandidat"], [])) if r["group_id"] else ""
        r["kelompok_berisi_kembar_ds1"] = bool(root_has_twin[r["grup_kandidat"]]) if r["group_id"] else ""
        r["prioritas_anotasi"] = (1 if id(r) in rep_ids else 2) if r["group_id"] else ""
        r["urutan_anotasi"] = ranked.get(id(r), "")
    clean.sort(key=lambda r: r["urutan_anotasi"])
    fields = ["group_id", "urutan_anotasi", "prioritas_anotasi", "ukuran_kelompok", "kelompok_berisi_kembar_ds1"] + [k_ for k_ in rows[0] if k_ not in ("group_id", "urutan_anotasi", "prioritas_anotasi", "ukuran_kelompok", "kelompok_berisi_kembar_ds1")]
    with open(rd / "blb_clean_eval_candidates.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(clean)
    tw_fields = list(twin_rows[0]) if twin_rows else ["berkas"]
    with open(rd / "blb_ds2_with_ds1_twin.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=tw_fields)
        w.writeheader()
        w.writerows([r for r in twin_rows if r["sumber"] == "ds2"])
    summ = {"dataset": cfg["dataset_name"], "ds2_BLB_Original": len(ds2_blb), "ds3_blight": len(ds3_blight),
            "ds2_dengan_kembar_ds1": sum(1 for r in rows if r["sumber"] == "ds2" and not r["bersih_ds1"]),
            "ds2_bersih_ds1_dan_ds4": sum(1 for r in clean if r["sumber"] == "ds2"),
            "ds2_dengan_kembar_ds1_bila_pakai_filter_sebaran_15%": sum(1 for r in rows if r["sumber"] == "ds2" and r["n_kembar_ds1_bila_pakai_filter_sebaran"] > 0),
            "kriteria": f"kembar = SIFT inlier >= 21 (tanpa syarat sebaran; konservatif), k={k} tetangga embedding per himpunan pembanding",
            "kelompok_independen": {s_: {"gambar_bersih": sum(1 for r in clean if r["sumber"] == s_),
                                          "kelompok": len({r["group_id"] for r in clean if r["sumber"] == s_}),
                                          "gambar_dalam_kelompok_yang_berisi_kembar_ds1": sum(1 for r in clean if r["sumber"] == s_ and r["kelompok_berisi_kembar_ds1"] is True),
                                          "kelompok_bersih_transitif": len({r["group_id"] for r in clean if r["sumber"] == s_ and r["kelompok_berisi_kembar_ds1"] is False})}
                                    for s_ in ("ds2", "ds3")},
            "kelompok_independen_gabungan": len({r["group_id"] for r in clean}),
            "prioritas_1_(satu per kelompok)": sum(1 for r in clean if r["prioritas_anotasi"] == 1),
            "ds3_dengan_kembar_ds1": sum(1 for r in rows if r["sumber"] == "ds3" and not r["bersih_ds1"]),
            "ds3_bersih_ds1_dan_ds4": sum(1 for r in clean if r["sumber"] == "ds3"),
            "ds2_kembar_ds4": sum(1 for r in rows if r["sumber"] == "ds2" and not r["bersih_ds4"]),
            "ds3_kembar_ds4": sum(1 for r in rows if r["sumber"] == "ds3" and not r["bersih_ds4"]),
            "dalam_daftar_bersih_status_ds2/ds3_tidak_bersih": {s: {"tidak_bersih_ds2": sum(1 for r in clean if r["sumber"] == s and not r["bersih_ds2"]),
                                                                    "tidak_bersih_ds3": sum(1 for r in clean if r["sumber"] == s and not r["bersih_ds3"])} for s in ("ds2", "ds3")},
            "grup_kandidat_dalam_daftar_bersih": len({r["grup_kandidat"] for r in clean}),
            "ds2_kembar_ds1_menurut_split_kembar_di_dataset": {sp: sum(1 for r in twin_rows if r["sumber"] == "ds2" and sp in r["split_semua_kembar_di_dataset"].split("|"))
                                                                  for sp in ("train", "val", "test")}}
    (rd / "blb_clean_candidates_summary.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summ, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
