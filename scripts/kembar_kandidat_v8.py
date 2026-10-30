#!/usr/bin/env python3
"""
kembar_kandidat_v8.py - Cek kembaran kandidat dataset v8 terhadap ds1, ds2, ds3, ds4, antar kandidat, dan di dalam tiap kandidat.
TIDAK menjalankan model deteksi apa pun pada kandidat (piksel hanya dibaca untuk deteksi duplikat: phash, embedding DINOv2, SIFT),
sama seperti ds3_kembar.py.

Pembanding (foto MENTAH, konservatif):
  ds1 : data/raw/ds1 semua gambar
  ds2 : data/raw/ds2 cabang Original, semua folder kelas
  ds3 : data/raw/ds3 semua 240 foto (blast, blight, tungro)
  ds4 : data/raw/ds4 "Original images", semua folder
Kriteria kembar (salah satu cukup; ambang sama dengan dataset/ds3):
  * phash 16x16, Hamming <= 14 pada salah satu dari 8 varian dihedral
  * embedding DINOv2 (8 varian dihedral) cosine >= 0.97
  * SIFT+RANSAC inlier >= 21 tanpa syarat sebaran, pada k tetangga embedding terdekat per himpunan pembanding
SIFT kandidat -> pembanding: k=--knn per folder kelas (ds1 tidak berfolder: --knn-ds1; ds3: --knn-ds3).
Arah balik untuk ds3 (paling ketat, syarat "harus 0"): tiap foto ds3 -> k=--knn-balik-ds3 tetangga di tiap kandidat.
Antar kandidat: k=--knn-antar per pasangan kandidat (dua arah). Internal: k=--knn-internal di dalam kandidat (union-find -> kelompok).

Keluaran: reports/kandidat_v8/kembar/
  pasangan.csv           - semua pasangan kembar (kandidat vs pembanding / antar kandidat)
  pasangan_internal.csv  - pasangan kembar di dalam kandidat
  per_gambar.csv         - per gambar kandidat: jumlah kembar per sumber, group_id internal
  ringkasan.json, pasangan_teratas_<sumber>.jpg (untuk dicek sendiri)
"""
import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
import aug_detect as ad  # noqa: E402
import dedupe_rules as dr  # noqa: E402
import prepare_dataset as pd_  # noqa: E402
from inventory import IMG_EXT  # noqa: E402

ROOT = dr.ROOT
PHASH_THR, EMB_THR = 14, 0.97
KAND = ROOT / "data/raw/kandidat_v8"
OUT = ROOT / "reports/kandidat_v8/kembar"


def imgs(root, cond=lambda s: True):
    return sorted(p for p in (ROOT / root).rglob("*") if p.suffix.lower() in IMG_EXT and cond(p.as_posix().lower()))


def rel(p):
    return Path(p).resolve().relative_to(ROOT).as_posix()


def ref_sets():
    out = {"ds1|semua": imgs("data/raw/ds1")}
    for p in imgs("data/raw/ds2", lambda s: "/original/" in s):
        out.setdefault(f"ds2|{p.parent.name}", []).append(p)
    for p in imgs("data/raw/ds3"):
        out.setdefault(f"ds3|{p.parent.name}", []).append(p)
    for p in imgs("data/raw/ds4", lambda s: "/original images/" in s):
        out.setdefault(f"ds4|{p.parent.name}", []).append(p)
    return out


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def cari(Qp, var_q, ok_q, Eq, Rp, var_r, ok_r, Er, k):
    """Pasangan kembar Q x R. Kembalikan {(qi, rj): {metode: nilai}}."""
    found = defaultdict(dict)
    orig = var_r[:, 0]
    for v in range(8):
        for i, j in pd_.near_pairs(var_q[:, v], orig, PHASH_THR, False):
            if ok_q[i] and ok_r[j]:
                d = int(min(pd_.min_dist(var_q[i:i + 1, w], var_r[j:j + 1, 0])[0] for w in range(8)))
                found[(int(i), int(j))]["phash"] = d
    S = ad.sim_matrix(Eq, Er[:, 0])
    for i, j in zip(*np.where(S >= EMB_THR)):
        found[(int(i), int(j))]["emb"] = round(float(S[i, j]), 4)
    P = dr.knn_pairs_cross(Eq, Er, k)
    for (i, j), r in zip(P, dr.verify_pairs([(Qp[i], Rp[j]) for i, j in P])):
        if r[0] >= dr.MIN_INLIERS:
            found[(int(i), int(j))]["sift"] = int(r[0])
    extra = [ij for ij, d in found.items() if "sift" not in d]
    for ij, r in zip(extra, dr.verify_pairs([(Qp[i], Rp[j]) for i, j in extra]) if extra else []):
        found[ij]["sift_info"] = int(r[0])
    for ij, d in found.items():
        d.setdefault("cos", round(float(S[ij]), 4))
    return found, len(P)


def metode(d):
    return "+".join(m for m in ("phash", "emb", "sift") if m in d)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--knn", type=int, default=5)
    ap.add_argument("--knn-ds1", type=int, default=20)
    ap.add_argument("--knn-ds3", type=int, default=10)
    ap.add_argument("--knn-balik-ds3", type=int, default=100)
    ap.add_argument("--knn-antar", type=int, default=5)
    ap.add_argument("--knn-internal", type=int, default=10)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    kand = {d.name: sorted(p for p in d.rglob("*") if p.suffix.lower() in IMG_EXT) for d in sorted(KAND.iterdir()) if d.is_dir()}
    Q = [p for v in kand.values() for p in v]
    qk = [n for n, v in kand.items() for _ in v]
    sl = {}
    o = 0
    for n, v in kand.items():
        sl[n] = slice(o, o + len(v))
        o += len(v)
    refs = ref_sets()
    R = [p for v in refs.values() for p in v]
    log("kandidat:", {n: len(v) for n, v in kand.items()}, "| pembanding:", {k: len(v) for k, v in refs.items()})

    log("[1/5] phash 8 varian dihedral")
    var_q, ok_q = pd_.compute_hashes([{"path": p} for p in Q], 16)
    var_r, ok_r = pd_.compute_hashes([{"path": p} for p in R], 16)
    log("[2/5] embedding DINOv2 (hanya untuk deteksi duplikat)")
    Eq = ad.embeddings(Q)
    Er = ad.embeddings(R)
    log("[3/5] deskriptor SIFT")
    log("  dihitung baru:", dr.ensure_descriptors(Q + R))

    pasangan = []
    hit = defaultdict(lambda: defaultdict(set))    # qi -> sumber -> {berkas pembanding}

    def catat(qi, sumber, label, rp, d, arah):
        pasangan.append({"kandidat": qk[qi], "berkas": rel(Q[qi]), "pembanding": label, "berkas_pembanding": rel(rp), "metode": metode(d),
                         "phash_jarak": d.get("phash", ""), "emb_cos": d.get("emb", d.get("cos", "")),
                         "sift_inlier": d.get("sift", d.get("sift_info", "")), "arah": arah})
        hit[qi][sumber].add(rel(rp))

    log("[4/5] kandidat -> ds1/ds2/ds3/ds4")
    o = 0
    for name, ps in refs.items():
        n = len(ps)
        rs = slice(o, o + n)
        o += n
        src = name.split("|")[0]
        k = {"ds1": a.knn_ds1, "ds3": a.knn_ds3}.get(src, a.knn)
        f, np_ = cari(Q, var_q, ok_q, Eq, ps, var_r[rs], ok_r[rs], Er[rs], k)
        for (i, j), d in f.items():
            catat(i, src, name, ps[j], d, "kandidat->pembanding")
        log(f"  vs {name}: SIFT {np_} pasangan; kembar {len(f)} (gambar kandidat terdampak {len({i for i, _ in f})})")
        if src == "ds3":       # arah balik: foto ds3 sebagai query ke tiap kandidat
            for kn, s in sl.items():
                f2, np2 = cari(ps, var_r[rs], ok_r[rs], Er[rs], Q[s], var_q[s], ok_q[s], Eq[s], a.knn_balik_ds3)
                for (j, i), d in f2.items():
                    qi = s.start + i
                    if rel(ps[j]) not in hit[qi]["ds3"]:
                        catat(qi, "ds3", name, ps[j], d, "ds3->kandidat")
                log(f"    balik {name} -> {kn}: SIFT {np2} pasangan; kembar {len(f2)}")

    log("[4b/5] antar kandidat")
    names = list(kand)
    for x in names:
        for y in names:
            if x == y:
                continue
            sx, sy = sl[x], sl[y]
            f, np_ = cari(Q[sx], var_q[sx], ok_q[sx], Eq[sx], Q[sy], var_q[sy], ok_q[sy], Eq[sy], a.knn_antar)
            for (i, j), d in f.items():
                catat(sx.start + i, y, f"kandidat|{y}", Q[sy.start + j], d, f"{x}->{y}")
            log(f"  {x} -> {y}: SIFT {np_} pasangan; kembar {len(f)} (gambar {x} terdampak {len({i for i, _ in f})})")

    log("[5/5] kelompok internal tiap kandidat")
    internal, grp_id = [], {}
    for kn, s in sl.items():
        Qs, vq, Es = Q[s], var_q[s], Eq[s]
        ip = {}
        for i, j in pd_.phash_pairs(vq, PHASH_THR):
            ip.setdefault((int(i), int(j)), set()).add("phash")
        for i, j in ad.sim_pairs(Es, EMB_THR):
            ip.setdefault((int(i), int(j)), set()).add("emb")
        Pk = dr.knn_pairs(Es, min(a.knn_internal, len(Qs) - 1))
        for (i, j), r in zip(Pk, dr.verify_pairs([(Qs[i], Qs[j]) for i, j in Pk])):
            if r[0] >= dr.MIN_INLIERS:
                ip.setdefault((int(i), int(j)), set()).add("sift")
        g = pd_.union_groups(len(Qs), [np.array(list(ip), int).reshape(-1, 2)])
        uk = Counter(g)
        urut = {gg: n for n, gg in enumerate(sorted(uk, key=lambda gg: min(i for i in range(len(g)) if g[i] == gg)), 1)}
        for i in range(len(Qs)):
            grp_id[s.start + i] = (f"{kn}_G{urut[g[i]]:05d}", uk[g[i]])
        for (i, j), m in ip.items():
            internal.append({"kandidat": kn, "berkas": rel(Qs[i]), "berkas_pasangan": rel(Qs[j]), "metode": "+".join(sorted(m))})
        log(f"  {kn}: pasangan internal {len(ip)}; kelompok {len(uk)} dari {len(Qs)} gambar; kelompok ukuran>1: {sum(1 for v in uk.values() if v > 1)}")

    # ---- tulis
    pf = list(pasangan[0]) if pasangan else ["kandidat"]
    with open(OUT / "pasangan.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=pf)
        w.writeheader()
        w.writerows(pasangan)
    with open(OUT / "pasangan_internal.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["kandidat", "berkas", "berkas_pasangan", "metode"])
        w.writeheader()
        w.writerows(internal)
    sumber = ["ds1", "ds2", "ds3", "ds4"] + names
    rows = []
    for i, p in enumerate(Q):
        rows.append({"kandidat": qk[i], "berkas": rel(p), "group_id": grp_id[i][0], "ukuran_grup": grp_id[i][1],
                     **{f"n_kembar_{s}": len(hit[i].get(s, ())) for s in sumber}})
    with open(OUT / "per_gambar.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summ = {"dibuat": datetime.now().isoformat(timespec="seconds"), "model_deteksi_dijalankan_pada_kandidat": False,
            "kriteria": f"phash<= {PHASH_THR} (8 varian dihedral) ATAU emb cos>= {EMB_THR} ATAU SIFT inlier>= {dr.MIN_INLIERS}",
            "knn": vars(a), "pembanding": {k: len(v) for k, v in refs.items()}, "per_kandidat": {}}
    for kn, s in sl.items():
        rk = rows[s]
        grup_kena = defaultdict(set)
        for i in range(s.start, s.stop):
            for src in hit[i]:
                grup_kena[src].add(grp_id[i][0])
        gk_all = {g for v in grup_kena.values() for g in v}
        summ["per_kandidat"][kn] = {
            "gambar": len(rk), "kelompok_internal": len({r["group_id"] for r in rk}),
            "gambar_dengan_kembar_per_sumber": {src: sum(1 for r in rk if r[f"n_kembar_{src}"]) for src in sumber if src != kn},
            "kelompok_dengan_kembar_per_sumber": {src: len(grup_kena[src]) for src in sumber if src != kn},
            "gambar_dengan_kembar_ds3_termasuk_transitif": sum(1 for r in rk if r["group_id"] in grup_kena["ds3"]),
            "gambar_bebas_kembar_luar_termasuk_transitif": sum(1 for r in rk if r["group_id"] not in gk_all),
            "kembar_ds3_per_folder": dict(Counter(pp["pembanding"] for pp in pasangan if pp["kandidat"] == kn and pp["pembanding"].startswith("ds3"))),
            "kembar_ds1_ds4_per_folder": dict(Counter(pp["pembanding"] for pp in pasangan if pp["kandidat"] == kn and pp["pembanding"][:3] in ("ds1", "ds4"))),
            "metode": dict(Counter((pp["pembanding"].split("|")[0], pp["metode"]) for pp in pasangan if pp["kandidat"] == kn).most_common()),
        }
        summ["per_kandidat"][kn]["metode"] = {f"{k[0]}:{k[1]}": v for k, v in summ["per_kandidat"][kn]["metode"].items()}
    (OUT / "ringkasan.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False), encoding="utf-8")

    # lembar visual pasangan teratas per sumber, untuk dicek sendiri
    for src in ("ds1", "ds2", "ds3", "ds4", "kandidat"):
        pp = [r for r in pasangan if r["pembanding"].startswith(src)]
        rng = np.random.default_rng(8)
        kuat = sorted(pp, key=lambda r: -(r["sift_inlier"] if isinstance(r["sift_inlier"], int) else 0))[:6]
        lemah = [pp[i] for i in rng.choice(len(pp), min(6, len(pp)), replace=False)] if pp else []
        pilih = kuat + [r for r in lemah if r not in kuat]
        if not pilih:
            continue
        T, W = 240, 30
        cv = Image.new("RGB", (4 * T, ((len(pilih) + 1) // 2) * (T + W)), "white")
        dd = ImageDraw.Draw(cv)
        for n, r in enumerate(pilih):
            x0, y0 = (n % 2) * 2 * T, (n // 2) * (T + W)
            dd.text((x0 + 3, y0 + 2), f"{Path(r['berkas']).name[:28]} ~ {r['pembanding'][:28]}", fill="black")
            dd.text((x0 + 3, y0 + 15), f"{r['metode']} inl={r['sift_inlier']} ph={r['phash_jarak']} cos={r['emb_cos']}", fill="black")
            for k_, pth in enumerate((r["berkas"], r["berkas_pembanding"])):
                im = Image.open(ROOT / pth)
                im.draft("RGB", (T * 2, T * 2))
                cv.paste(im.convert("RGB").resize((T, T)), (x0 + k_ * T, y0 + W))
        cv.save(OUT / f"pasangan_teratas_{src}.jpg", quality=86)
    log(json.dumps(summ["per_kandidat"], indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
