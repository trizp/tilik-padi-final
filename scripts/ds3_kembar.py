#!/usr/bin/env python3
"""
ds3_kembar.py - Persiapan uji eksternal ds3: cek kembaran SEMUA kelas ds3 (blast, blight, tungro) terhadap ds1, ds2, ds4,
lalu buat daftar foto ds3 bersih per kelas lengkap dengan group_id. TIDAK menjalankan model deteksi apa pun pada ds3
(piksel hanya dibaca untuk deteksi duplikat: phash, embedding DINOv2, SIFT).

Pembanding (seluruh foto MENTAH, bukan hanya yang masuk dataset; lebih konservatif):
  ds1 : data/raw/ds1 semua gambar
  ds2 : data/raw/ds2 cabang Original, semua folder kelas (Augmented = turunan Original, tidak dipakai)
  ds4 : data/raw/ds4 "Original images", semua folder (termasuk Rice Tungro, Sheath blight, Leaf smut)
Kriteria kembar (salah satu cukup):
  * phash 16x16 (256 bit), jarak Hamming <= 14 pada salah satu dari 8 varian dihedral (ambang dataset)
  * embedding DINOv2 (8 varian dihedral) cosine >= 0.97 (ambang dataset)
  * SIFT+RANSAC inlier >= 21 TANPA syarat sebaran (konservatif), pada k tetangga embedding terdekat per
    himpunan pembanding (sumber x folder kelas, agar tidak tersisih oleh kelas lain)
Kelompok internal ds3 (group_id): union-find atas pasangan ds3-ds3 dengan kriteria yang sama (SIFT pada k=50 tetangga).
Foto ds3 BERSIH = tidak punya kembar di ds1/ds2/ds4, kelompoknya tidak memuat anggota yang punya kembar (transitif),
dan kelompoknya tidak mencakup dua folder kelas ds3 (label bertentangan).

Keluaran: reports/prov_v6/ds3_eksternal/
  ds3_daftar.csv          - semua 240 foto: status, alasan, group_id, rincian kembar
  ds3_bersih_<kelas>.txt  - daftar foto bersih per kelas (blight, blast; tungro = uji di luar cakupan)
  ds3_pasangan.csv        - semua pasangan kembar yang terdeteksi (untuk dicek visual)
  ds3_pasangan_teratas.jpg, ringkasan.json
"""
import argparse
import csv
import hashlib
import json
import sys
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
KELAS_DS3 = {"blight": ("bacterial_leaf_blight", "uji_eksternal"), "blast": ("leaf_blast", "uji_eksternal"),
             "tungro": ("luar_cakupan", "uji_luar_cakupan")}


def imgs(root, cond=lambda s: True):
    return sorted(p for p in (ROOT / root).rglob("*") if p.suffix.lower() in IMG_EXT and cond(p.as_posix().lower()))


def rel(p):
    return Path(p).resolve().relative_to(ROOT).as_posix()


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def ref_sets():
    """Himpunan pembanding per (sumber|folder kelas)."""
    out = {"ds1|semua": imgs("data/raw/ds1")}
    for p in imgs("data/raw/ds2", lambda s: "/original/" in s):
        out.setdefault(f"ds2|{p.parent.name}", []).append(p)
    for p in imgs("data/raw/ds4", lambda s: "/original images/" in s):
        out.setdefault(f"ds4|{p.parent.name}", []).append(p)
    return out


def hashes(paths):
    var, ok = pd_.compute_hashes([{"path": Path(p)} for p in paths], 16)
    return var, ok


def phash_cross(var_q, var_r):
    """{(qi, rj): jarak} untuk pasangan dengan jarak <= ambang pada salah satu varian dihedral query."""
    out = {}
    orig = var_r[:, 0]
    for v in range(8):
        for i, j in pd_.near_pairs(var_q[:, v], orig, PHASH_THR, False):
            out[(int(i), int(j))] = True
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--knn", type=int, default=100, help="tetangga embedding per himpunan pembanding untuk SIFT")
    ap.add_argument("--knn-ds1", type=int, default=200, help="ds1 tidak berfolder kelas -> k lebih besar")
    ap.add_argument("--manifest", default="data/processed/prov_v6/manifest.csv")
    ap.add_argument("--out", default="reports/prov_v6/ds3_eksternal")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)

    q = imgs("data/raw/ds3")
    qkelas = [p.parent.name.lower() for p in q]
    refs = ref_sets()
    print(f"ds3: {len(q)} foto {dict(Counter(qkelas))}; pembanding: " + ", ".join(f"{k}={len(v)}" for k, v in refs.items()), flush=True)
    man = {}
    if (ROOT / a.manifest).exists():
        with open(ROOT / a.manifest, encoding="utf-8") as f:
            man = {r["orig_path"]: r for r in csv.DictReader(f)}

    allr = [p for v in refs.values() for p in v]
    print("[1/4] phash 8 varian dihedral", flush=True)
    var_q, ok_q = hashes(q)
    var_all, ok_all = hashes(allr)
    print("[2/4] embedding DINOv2", flush=True)
    Eq = ad.embeddings(q)
    Eall = ad.embeddings(allr)
    print("[3/4] deskriptor SIFT", flush=True)
    dr.ensure_descriptors(list(q) + allr)

    # ---- pasangan ds3 -> pembanding
    hits = defaultdict(list)          # qi -> [dict]
    pairs_rows = []
    off = 0
    for name, ps in refs.items():
        n = len(ps)
        vr, Er = var_all[off:off + n], Eall[off:off + n]
        okr = ok_all[off:off + n]
        off += n
        found = defaultdict(dict)     # (qi, rj) -> {metode: nilai}
        for (i, j) in phash_cross(var_q, vr):
            if ok_q[i] and okr[j]:
                found[(i, j)]["phash"] = int(min(pd_.min_dist(var_q[i:i + 1, v], vr[j:j + 1, 0])[0] for v in range(8)))
        S = ad.sim_matrix(Eq, Er[:, 0])
        for i, j in zip(*np.where(S >= EMB_THR)):
            found[(int(i), int(j))]["emb"] = round(float(S[i, j]), 4)
        k = a.knn_ds1 if name.startswith("ds1") else a.knn
        P = dr.knn_pairs_cross(Eq, Er, k)
        res = dr.verify_pairs([(q[i], ps[j]) for i, j in P])
        for (i, j), r in zip(P, res):
            if r[0] >= dr.MIN_INLIERS:
                found[(int(i), int(j))]["sift"] = int(r[0])
        # verifikasi SIFT juga untuk pasangan phash/embedding yang tidak masuk kNN (untuk inlier di laporan)
        extra = [ij for ij, d in found.items() if "sift" not in d]
        if extra:
            for ij, r in zip(extra, dr.verify_pairs([(q[i], ps[j]) for i, j in extra])):
                found[ij]["sift_inlier_info"] = int(r[0])
        for (i, j), d in found.items():
            pr = rel(ps[j])
            mm = man.get(pr, {})
            row = {"ds3": rel(q[i]), "kelas_ds3": qkelas[i], "pembanding": name, "berkas_pembanding": pr,
                   "metode": "+".join(m for m in ("phash", "emb", "sift") if m in d),
                   "phash_jarak": d.get("phash", ""), "emb_cos": d.get("emb", round(float(S[i, j]), 4)),
                   "sift_inlier": d.get("sift", d.get("sift_inlier_info", "")),
                   "di_prov_v6": mm.get("split", "tidak"), "kelas_di_prov_v6": mm.get("class", "")}
            pairs_rows.append(row)
            hits[i].append(row)
        print(f"  vs {name}: kNN {len(P)} pasangan; kembar terdeteksi {len(found)} (foto ds3 terdampak {len({i for i, _ in found})})", flush=True)

    # ---- kelompok internal ds3
    print("[4/4] kelompok internal ds3", flush=True)
    ip = {}
    for i, j in pd_.phash_pairs(var_q, PHASH_THR):
        ip.setdefault((int(i), int(j)), set()).add("phash")
    for i, j in ad.sim_pairs(Eq, EMB_THR):
        ip.setdefault((int(i), int(j)), set()).add("emb")
    Pk = dr.knn_pairs(Eq, min(50, len(q) - 1))
    for (i, j), r in zip(Pk, dr.verify_pairs([(q[i], q[j]) for i, j in Pk])):
        if r[0] >= dr.MIN_INLIERS:
            ip.setdefault((int(i), int(j)), set()).add(f"sift{int(r[0])}")
    grp = pd_.union_groups(len(q), [np.array(list(ip), int).reshape(-1, 2)])
    for (i, j), m in ip.items():
        pairs_rows.append({"ds3": rel(q[i]), "kelas_ds3": qkelas[i], "pembanding": f"ds3|{qkelas[j]}", "berkas_pembanding": rel(q[j]),
                           "metode": "+".join(sorted(m)), "phash_jarak": "", "emb_cos": "", "sift_inlier": "", "di_prov_v6": "-", "kelas_di_prov_v6": ""})

    anggota = defaultdict(list)
    for i, g in enumerate(grp):
        anggota[g].append(i)
    g_kontam = {g: any(hits[i] for i in m) for g, m in anggota.items()}
    g_lintas = {g: len({qkelas[i] for i in m}) > 1 for g, m in anggota.items()}
    order = sorted(anggota, key=lambda g: min(rel(q[i]) for i in anggota[g]))
    gid = {g: f"D3G{n:03d}" for n, g in enumerate(order, 1)}

    rows = []
    for i, p in enumerate(q):
        g = grp[i]
        alasan = []
        if hits[i]:
            alasan.append("kembar di " + "|".join(sorted({h["pembanding"].split("|")[0] for h in hits[i]})))
        elif g_kontam[g]:
            alasan.append("satu kelompok dengan foto ds3 yang punya kembar di ds1/ds2/ds4")
        if g_lintas[g]:
            alasan.append("kelompok mencakup >1 folder kelas ds3")
        best = max(hits[i], key=lambda h: (h["sift_inlier"] or 0) if isinstance(h["sift_inlier"], int) else 0, default=None)
        kt, peran = KELAS_DS3[qkelas[i]]
        rows.append({"berkas": rel(p), "kelas_folder": qkelas[i], "kelas_target": kt, "peran": peran,
                     "bersih": not alasan, "alasan": "; ".join(alasan), "group_id": gid[g], "ukuran_grup": len(anggota[g]),
                     "grup_lintas_kelas": g_lintas[g], "grup_berisi_kembar_luar": g_kontam[g],
                     **{f"n_kembar_{s}": sum(1 for h in hits[i] if h["pembanding"].startswith(s)) for s in ("ds1", "ds2", "ds4")},
                     "kembar_terbaik": best["berkas_pembanding"] if best else "", "metode_kembar_terbaik": best["metode"] if best else "",
                     "kembar_di_prov_v6": "|".join(sorted({h["di_prov_v6"] for h in hits[i] if h["di_prov_v6"] != "tidak"})),
                     "sha256": sha256(p)})
    with open(out / "ds3_daftar.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    pf = ["ds3", "kelas_ds3", "pembanding", "berkas_pembanding", "metode", "phash_jarak", "emb_cos", "sift_inlier", "di_prov_v6", "kelas_di_prov_v6"]
    with open(out / "ds3_pasangan.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=pf)
        w.writeheader()
        w.writerows(pairs_rows)
    for kf in KELAS_DS3:
        (out / f"ds3_bersih_{kf}.txt").write_text("".join(f"{r['berkas']}\t{r['group_id']}\n" for r in rows if r["kelas_folder"] == kf and r["bersih"]),
                                                   encoding="utf-8")

    # lembar visual pasangan luar teratas (tertinggi inlier SIFT), untuk dicek sendiri
    luar = sorted([r for r in pairs_rows if not r["pembanding"].startswith("ds3")],
                  key=lambda r: -(r["sift_inlier"] if isinstance(r["sift_inlier"], int) else 0))[:12]
    if luar:
        T, W = 260, 30
        cv = Image.new("RGB", (4 * T, ((len(luar) + 1) // 2) * (T + W)), "white")
        dd = ImageDraw.Draw(cv)
        for n, r in enumerate(luar):
            x0, y0 = (n % 2) * 2 * T, (n // 2) * (T + W)
            dd.text((x0 + 3, y0 + 2), f"{Path(r['ds3']).name} ~ {r['pembanding']}", fill="black")
            dd.text((x0 + 3, y0 + 15), f"{r['metode']} inlier={r['sift_inlier']} phash={r['phash_jarak']} cos={r['emb_cos']}", fill="black")
            for k_, pth in enumerate((r["ds3"], r["berkas_pembanding"])):
                im = Image.open(ROOT / pth)
                im.draft("RGB", (T * 2, T * 2))
                cv.paste(im.convert("RGB").resize((T, T)), (x0 + k_ * T, y0 + W))
        cv.save(out / "ds3_pasangan_teratas.jpg", quality=88)

    summ = {"dibuat": datetime.now().isoformat(timespec="seconds"),
            "kriteria": f"phash<= {PHASH_THR} (8 varian dihedral) ATAU embedding cos>= {EMB_THR} ATAU SIFT inlier>= {dr.MIN_INLIERS} "
                        f"(tanpa syarat sebaran; k={a.knn} per sumber|folder, ds1 k={a.knn_ds1}); kelompok internal ds3 k=50",
            "pembanding": {k: len(v) for k, v in refs.items()},
            "model_dijalankan_pada_ds3": False,
            "per_kelas": {}}
    for kf in KELAS_DS3:
        rk = [r for r in rows if r["kelas_folder"] == kf]
        summ["per_kelas"][kf] = {
            "kelas_target": KELAS_DS3[kf][0], "peran": KELAS_DS3[kf][1], "foto": len(rk),
            "punya_kembar_luar": sum(1 for r in rk if any(r[f"n_kembar_{s}"] for s in ("ds1", "ds2", "ds4"))),
            "kembar_per_sumber": {s: sum(1 for r in rk if r[f"n_kembar_{s}"]) for s in ("ds1", "ds2", "ds4")},
            "kembar_yang_ada_di_prov_v6": sum(1 for r in rk if r["kembar_di_prov_v6"]),
            "dibuang_transitif": sum(1 for r in rk if r["alasan"].startswith("satu kelompok")),
            "grup_lintas_kelas": sum(1 for r in rk if r["grup_lintas_kelas"]),
            "bersih": sum(1 for r in rk if r["bersih"]),
            "kelompok_bersih": len({r["group_id"] for r in rk if r["bersih"]}),
            "metode_kembar": dict(Counter(h["metode"] for i, p in enumerate(q) if qkelas[i] == kf for h in hits[i]))}
    summ["pasangan_internal_ds3"] = len(ip)
    summ["kelompok_internal_ds3_ukuran>1"] = sum(1 for m in anggota.values() if len(m) > 1)
    (out / "ringkasan.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summ, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
