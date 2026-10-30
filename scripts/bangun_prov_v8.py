#!/usr/bin/env python3
"""
bangun_prov_v8.py - prov_v8 = prov_v7 (SEMUA baris, split, kelompok, label identik) + daunpadi (train/val) + Tungro ds4 (negatif,
train saja). Keputusan: KEPUTUSAN_PROYEK.md 2026-10-14 "KEPUTUSAN v8". TIDAK menjalankan model apa pun.

  python scripts/bangun_prov_v8.py seleksi   # graf kelompok + pilih daunpadi/Tungro/daftar luar cakupan -> reports/prov_v8/seleksi/
  python scripts/bangun_prov_v8.py bangun    # tulis data/processed/prov_v8 (belum beku) + pemeriksaan kesetaraan dengan prov_v7
  python scripts/bangun_prov_v8.py audit     # audit SIFT k=200, audit waktu, kembar lintas set uji -> reports/prov_v8/audit.json
  (verify_dataset: python scripts/verify_dataset.py --config configs/classes_v8.yaml > reports/prov_v8/verify_dataset.log)
  python scripts/bangun_prov_v8.py bekukan   # hanya bila kesetaraan, audit, dan verify_dataset LOLOS

Graf kelompok (konservatif; aturan v5 + SIFT tanpa syarat sebaran):
  simpul = semua foto mentah ds1, ds2 Original, ds3, ds4 Original (semua folder), aditia, daunpadi (semua) + baris prov_v7 (lewat orig_path)
  sisi   = (a) kelompok prov_v7 yang sama; (b) jendela waktu geser 5 dtk (ds4 dari metadata, ds1 dari nama berkas);
           (c) untuk simpul BARU (Tungro, Leaf smut, Sheath blight, daunpadi) terhadap SEMUA simpul: phash<=14 (8 varian dihedral),
               cos DINOv2>=0,97, SIFT>=21 inlier pada 100 tetangga embedding (tanpa syarat sebaran)
  komponen terhubung = kelompok.
Aturan pilih:
  daunpadi : satu kelas (Brownspot -> brown_spot, leafblast -> leaf_blast), Hispa tidak dipakai; kelompoknya tidak memuat simpul di
             luar daunpadi (kembar ds1 atau sumber lain). Kelompok dibagi 80/20 train/val per kelas (seed 42).
  Tungro   : kelompoknya tidak memuat val/test/test_new_session prov_v7, ds3, aditia, Leaf smut/Sheath blight ds4, atau daunpadi;
             tidak berada di rentang sesi test_new_session (+-600 dtk); acak seed 42, maks 600; class=background, source=ds4_tungro.
  Luar cakupan ds4 (Leaf smut, Sheath blight): kelompoknya tidak memuat train/val prov_v8.
"""
import csv
import hashlib
import json
import os
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
import aug_detect as ad  # noqa: E402
import dedupe_rules as dr  # noqa: E402
import prepare_dataset as pd_  # noqa: E402
from inventory import IMG_EXT  # noqa: E402

ROOT = dr.ROOT
V7 = ROOT / "data/processed/prov_v7"
V8 = ROOT / "data/processed/prov_v8"
REP = ROOT / "reports/prov_v8"
SEL = REP / "seleksi"
DP = ROOT / "data/raw/kandidat_v8/rf_daunpadi_b1cjw"
AD = ROOT / "data/raw/kandidat_v8/rf_aditia_4ued0"
DS4 = ROOT / "data/raw/ds4/images_labels/RiceLeafDiseaseBD/Original images"
XLSX = ROOT / "data/raw/ds4/docs/Dataset metadata.xlsx"
SEED, MAKS_TUNGRO, VAL_FRAC = 42, 600, 0.20
PHASH_THR, EMB_THR, KNN, KNN_AUDIT = 14, 0.97, 100, 200
SPL = ("train", "val", "test", "test_new_session")
NAMES = ["leaf_blast", "brown_spot", "bacterial_leaf_blight"]
DP_MAP = {"leafblast": "leaf_blast", "Brownspot": "brown_spot"}
PIX_TH = {"corner_black": 0.25, "border_black": 0.08, "smear": 0.25}     # ambang piksel prov_v7 (classes_v7.yaml)


def rel(p):
    return Path(p).resolve().relative_to(ROOT).as_posix()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def imgs(d, cond=lambda s: True):
    return sorted(p for p in Path(d).rglob("*") if p.suffix.lower() in IMG_EXT and cond(p.as_posix().lower()))


def tautkan(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def labels_sha(D):
    h = hashlib.sha256()
    for sp in SPL:
        for lf in sorted((D / "labels" / sp).iterdir()):
            h.update(f"{sp}/{lf.name}\n".encode())
            h.update(lf.read_bytes())
    return h.hexdigest()


def dp_label(p):
    names = yaml.safe_load((DP / "data.yaml").read_text(encoding="utf-8"))["names"]
    lab = p.parent.parent / "labels" / (p.stem + ".txt")
    rows = [ln.split() for ln in lab.read_text().splitlines() if ln.strip()] if lab.exists() else []
    return rows, {names[int(r[0])] for r in rows}, names


# ============================================================ simpul
def simpul():
    man = list(csv.DictReader(open(V7 / "manifest.csv", encoding="utf-8")))
    N, idx = [], {}

    def tambah(path, src, **kw):
        r = rel(path)
        if r in idx:
            N[idx[r]].update({k: v for k, v in kw.items() if v is not None})
            return idx[r]
        idx[r] = len(N)
        N.append({"rel": r, "path": ROOT / r, "src": src, "split": None, "group_v7": None, "kelas_v7": None, "folder": Path(r).parent.name, **kw})
        return idx[r]

    for p in imgs(ROOT / "data/raw/ds1"):
        tambah(p, "ds1")
    for p in imgs(ROOT / "data/raw/ds2", lambda s: "/original/" in s):
        tambah(p, "ds2")
    for p in imgs(ROOT / "data/raw/ds3"):
        tambah(p, "ds3")
    for p in imgs(DS4):
        tambah(p, "ds4")
    for p in imgs(AD):
        tambah(p, "aditia")
    for p in imgs(DP):
        tambah(p, "daunpadi")
    for r in man:
        i = idx.get(r["orig_path"])
        if i is None:
            sys.exit(f"[GAGAL] orig_path prov_v7 tidak ditemukan di simpul mentah: {r['orig_path']}")
        N[i].update(split=r["split"], group_v7=int(r["group"]), kelas_v7=r["class"], file_v7=r["file"], source_v7=r["source"])
    return N, man


def ts_simpul(N):
    m4 = dr.ds4_timestamps(XLSX)
    out = []
    for c in N:
        if c["src"] == "ds4":
            out.append(m4.get(Path(c["rel"]).name.lower()))
        elif c["src"] == "ds1":
            out.append(dr.ds1_timestamp(c["rel"]))
        else:
            out.append(None)
    return out


def pasangan_waktu(N, ts, gap=5):
    out = []
    for src in ("ds4", "ds1"):
        it = sorted((ts[i], i) for i, c in enumerate(N) if c["src"] == src and ts[i] is not None)
        for k in range(len(it)):
            j = k + 1
            while j < len(it) and (it[j][0] - it[k][0]).total_seconds() <= gap:
                out.append((it[k][1], it[j][1]))
                j += 1
    return out


def visual_baru(N, Q):
    """Sisi visual simpul baru Q terhadap semua simpul. Kembalikan list (i, j, metode, nilai)."""
    P = [c["path"] for c in N]
    var, ok = pd_.compute_hashes([{"path": p} for p in P], 16)
    E = ad.embeddings(P)
    dr.ensure_descriptors(P)
    Qa = np.array(Q)
    sisi = {}
    orig = var[:, 0]
    for v in range(8):
        for a, j in pd_.near_pairs(var[Qa, v], orig, PHASH_THR, False):
            i = int(Qa[a])
            if i != int(j) and ok[i] and ok[j]:
                sisi.setdefault((min(i, int(j)), max(i, int(j))), {})["phash"] = 1
    O = E[:, 0]
    knn = []
    for s in range(0, len(Qa), 512):
        q = Qa[s:s + 512]
        S = np.max([E[q, v] @ O.T for v in range(8)], axis=0)
        for r, i in enumerate(q):
            S[r, i] = -9
        for r, j in zip(*np.where(S >= EMB_THR)):
            i = int(q[r])
            sisi.setdefault((min(i, int(j)), max(i, int(j))), {})["emb"] = round(float(S[r, j]), 4)
        nb = np.argpartition(-S, KNN, axis=1)[:, :KNN]
        for r, i in enumerate(q):
            knn += [(int(i), int(j)) for j in nb[r]]
    knn = sorted({(min(a, b), max(a, b)) for a, b in knn})
    print(f"  SIFT: {len(knn)} pasangan kNN", flush=True)
    for (i, j), r in zip(knn, dr.verify_pairs([(P[i], P[j]) for i, j in knn])):
        if r[0] >= dr.MIN_INLIERS:
            sisi.setdefault((i, j), {})["sift"] = int(r[0])
    return sisi


# ============================================================ seleksi
def seleksi():
    SEL.mkdir(parents=True, exist_ok=True)
    N, man = simpul()
    ts = ts_simpul(N)
    n = len(N)
    print(f"simpul {n}: {dict(Counter(c['src'] for c in N))}; baris prov_v7 {len(man)}", flush=True)
    Q = [i for i, c in enumerate(N) if c["src"] == "daunpadi" or (c["src"] == "ds4" and c["folder"] in ("Rice Tungro", "Leaf smut", "Sheath blight"))]
    print(f"simpul baru (sisi visual dihitung): {len(Q)}", flush=True)
    pw = pasangan_waktu(N, ts)
    vis = visual_baru(N, Q)
    gv7 = defaultdict(list)
    for i, c in enumerate(N):
        if c["group_v7"] is not None:
            gv7[c["group_v7"]].append(i)
    pg = [(m[0], x) for m in gv7.values() for x in m[1:]]
    allp = np.array(pw + list(vis) + pg, int).reshape(-1, 2)
    comp = pd_.union_groups(n, [allp])
    anggota = defaultdict(list)
    for i, g in enumerate(comp):
        anggota[g].append(i)

    def peran(c):
        if c["split"]:
            return f"v7_{c['split']}"
        if c["src"] in ("ds3", "aditia", "daunpadi"):
            return c["src"]
        if c["src"] == "ds4" and c["folder"] in ("Leaf smut", "Sheath blight"):
            return "ds4_luar_cakupan"
        if c["src"] == "ds4" and c["folder"] == "Rice Tungro":
            return "ds4_tungro"
        return f"{c['src']}_tidak_di_v7"
    pr = [peran(c) for c in N]
    isi = {g: Counter(pr[i] for i in m) for g, m in anggota.items()}

    # ---- daunpadi
    dp_rows = []
    for i, c in enumerate(N):
        if c["src"] != "daunpadi":
            continue
        rows, kel, _ = dp_label(c["path"])
        g = comp[i]
        luar = {k: v for k, v in isi[g].items() if k != "daunpadi"}
        satu = len(kel) == 1 and next(iter(kel)) in DP_MAP
        alasan = []
        if not satu:
            alasan.append("bukan satu kelas Brownspot/leafblast" if kel else "tanpa label")
        if luar:
            alasan.append("kelompok memuat " + "|".join(f"{k}:{v}" for k, v in sorted(luar.items())))
        dp_rows.append({"i": i, "berkas": c["rel"], "kelas_asli": " + ".join(sorted(kel)), "kelas": DP_MAP.get(next(iter(kel)), "") if satu else "",
                        "komponen": int(g), "dipakai": not alasan, "alasan": "; ".join(alasan), "n_kotak": len(rows),
                        "split_roboflow": c["path"].parent.parent.name})
    pakai = [r for r in dp_rows if r["dipakai"]]
    kom_kelas = defaultdict(Counter)
    for r in pakai:
        kom_kelas[r["komponen"]][r["kelas"]] += 1
    rng = random.Random(SEED)
    val_kom = set()
    for kelas in ("brown_spot", "leaf_blast"):
        ks = sorted(k for k, c in kom_kelas.items() if c.most_common(1)[0][0] == kelas)
        rng.shuffle(ks)
        val_kom |= set(ks[:round(VAL_FRAC * len(ks))])
    pm = ad.pixel_metrics([ROOT / r["berkas"] for r in pakai])
    for r, m in zip(pakai, pm):
        r["split"] = "val" if r["komponen"] in val_kom else "train"
        r["aug_reasons"] = ad.pixel_flag(m, PIX_TH)
    for r in pakai:
        if r["aug_reasons"] and r["split"] == "val":          # aturan v7: gambar bertanda augmentasi tidak boleh di val/test
            r["split"] = "train"
            r["catatan"] = "dipindah ke train: tanda augmentasi piksel (seluruh kelompok ikut, lihat di bawah)"
    # satu kelompok satu split: bila ada anggota dipindah ke train, seluruh kelompoknya ke train
    kom_train = {r["komponen"] for r in pakai if r["split"] == "train"}
    for r in pakai:
        if r["komponen"] in kom_train and r["split"] == "val":
            r["split"] = "train"

    # ---- rentang sesi test_new_session
    t_tns = [ts[i] for i, c in enumerate(N) if c["split"] == "test_new_session" and ts[i] is not None]
    lo, hi = min(t_tns) - timedelta(seconds=600), max(t_tns) + timedelta(seconds=600)

    # ---- Tungro
    terlarang = {"v7_val", "v7_test", "v7_test_new_session", "ds3", "aditia", "ds4_luar_cakupan", "daunpadi"}
    tg_rows = []
    for i, c in enumerate(N):
        if pr[i] != "ds4_tungro":
            continue
        g = comp[i]
        bad = {k: v for k, v in isi[g].items() if k in terlarang}
        alasan = []
        if bad:
            alasan.append("kelompok memuat " + "|".join(f"{k}:{v}" for k, v in sorted(bad.items())))
        if ts[i] is not None and lo <= ts[i] <= hi:
            alasan.append("dalam rentang sesi test_new_session")
        tg_rows.append({"i": i, "berkas": c["rel"], "komponen": int(g), "waktu": ts[i].isoformat() if ts[i] else "",
                        "kelompok_memuat_v7_train": isi[g].get("v7_train", 0), "layak": not alasan, "alasan": "; ".join(alasan)})
    layak = [r for r in tg_rows if r["layak"]]
    pilih = set(r["i"] for r in random.Random(SEED).sample(layak, min(MAKS_TUNGRO, len(layak))))
    for r in tg_rows:
        r["dipilih"] = r["i"] in pilih
    pm_t = ad.pixel_metrics([ROOT / r["berkas"] for r in tg_rows if r["dipilih"]])
    for r, m in zip([r for r in tg_rows if r["dipilih"]], pm_t):
        r["aug_reasons"] = ad.pixel_flag(m, PIX_TH)

    # ---- luar cakupan ds4 (setelah train/val v8 diketahui)
    train_val_v8 = set()
    for i, c in enumerate(N):
        if c["split"] in ("train", "val"):
            train_val_v8.add(comp[i])
    for r in pakai:
        train_val_v8.add(r["komponen"])
    for r in tg_rows:
        if r["dipilih"]:
            train_val_v8.add(r["komponen"])
    lc_rows = []
    for i, c in enumerate(N):
        if pr[i] != "ds4_luar_cakupan":
            continue
        g = comp[i]
        kena = g in train_val_v8
        lc_rows.append({"berkas": c["rel"], "kelas_luar_cakupan": c["folder"], "komponen": int(g), "masuk_daftar": not kena,
                        "alasan": "kelompok memuat train/val prov_v8" if kena else "", "isi_kelompok": json.dumps(dict(isi[g]), ensure_ascii=False)})

    # ---- tulis
    def simpan(nama, rows):
        if rows:
            with open(SEL / nama, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
                w.writeheader()
                w.writerows(rows)
    simpan("daunpadi.csv", dp_rows)
    simpan("tungro.csv", tg_rows)
    simpan("luar_cakupan_ds4.csv", lc_rows)
    simpan("sisi_visual_baru.csv", [{"a": N[i]["rel"], "b": N[j]["rel"], **d} for (i, j), d in vis.items()])
    kv = []
    for k in sorted({r["komponen"] for r in tg_rows if r["dipilih"]}):
        for g in sorted({N[i]["group_v7"] for i in anggota[k] if N[i]["group_v7"] is not None}):
            kv.append({"komponen": k, "group_v7": g})
    with open(SEL / "komponen_v7.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["komponen", "group_v7"])
        w.writeheader()
        w.writerows(kv)
    ring = {"dibuat": datetime.now().isoformat(timespec="seconds"), "model_dijalankan": False,
            "simpul": dict(Counter(c["src"] for c in N)), "sisi": {"waktu_5dtk": len(pw), "visual_baru": len(vis), "kelompok_v7": len(pg)},
            "rentang_sesi_tns": [lo.isoformat(), hi.isoformat()],
            "daunpadi": {"gambar": len(dp_rows), "dipakai": len(pakai),
                         "alasan_tidak_dipakai": dict(Counter(r["alasan"].split(" kelompok memuat")[0].split("|")[0][:60] for r in dp_rows if not r["dipakai"])),
                         "dipakai_per_split_kelas": dict(Counter(f"{r['split']}|{r['kelas']}" for r in pakai)),
                         "kelompok_per_split_kelas": {f"{sp}|{k}": len({r["komponen"] for r in pakai if r["split"] == sp and r["kelas"] == k})
                                                      for sp in ("train", "val") for k in ("brown_spot", "leaf_blast")},
                         "kotak_per_split_kelas": dict(Counter({f"{r['split']}|{r['kelas']}": 0 for r in pakai}) + Counter()),
                         "tanda_aug_piksel": sum(1 for r in pakai if r["aug_reasons"])},
            "tungro": {"total": len(tg_rows), "layak": len(layak), "dipilih": len(pilih),
                       "kelompok_dipilih": len({r["komponen"] for r in tg_rows if r["dipilih"]}),
                       "dipilih_sekelompok_dengan_v7_train": sum(1 for r in tg_rows if r["dipilih"] and r["kelompok_memuat_v7_train"]),
                       "alasan_tidak_layak": dict(Counter(a.split(":")[0] for r in tg_rows if not r["layak"] for a in r["alasan"].replace("kelompok memuat ", "").replace("|", "; ").split("; "))),
                       "tanda_aug_piksel_dipilih": sum(1 for r in tg_rows if r["dipilih"] and r.get("aug_reasons"))},
            "luar_cakupan_ds4": {k: {"total": sum(1 for r in lc_rows if r["kelas_luar_cakupan"] == k),
                                     "masuk_daftar": sum(1 for r in lc_rows if r["kelas_luar_cakupan"] == k and r["masuk_daftar"]),
                                     "kelompok_masuk": len({r["komponen"] for r in lc_rows if r["kelas_luar_cakupan"] == k and r["masuk_daftar"]})}
                                 for k in ("Leaf smut", "Sheath blight")}}
    kp = Counter()
    for r in pakai:
        kp[f"{r['split']}|{r['kelas']}"] += r["n_kotak"]
    ring["daunpadi"]["kotak_per_split_kelas"] = dict(kp)
    (SEL / "ringkasan_seleksi.json").write_text(json.dumps(ring, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(ring, indent=1, ensure_ascii=False))


# ============================================================ bangun
def nama_baru(p, prefix):
    stem = re.sub(r"\.rf\.[0-9a-f]+$", "", p.stem)
    stem = re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")
    return f"{prefix}{stem}_{sha(p)[:8]}{p.suffix.lower()}"


def bangun():
    if V8.exists():
        sys.exit(f"[DITOLAK] {V8} sudah ada")
    dy7 = yaml.safe_load((V7 / "data.yaml").read_text(encoding="utf-8"))
    if sha(V7 / "manifest.csv") != dy7["manifest_sha256"] or not dy7.get("frozen"):
        sys.exit("[GAGAL] manifest prov_v7 berubah atau belum beku")
    man = list(csv.DictReader(open(V7 / "manifest.csv", encoding="utf-8")))
    kol = list(man[0].keys())
    dp = [r for r in csv.DictReader(open(SEL / "daunpadi.csv", encoding="utf-8")) if r["dipakai"] == "True"]
    tg = [r for r in csv.DictReader(open(SEL / "tungro.csv", encoding="utf-8")) if r["dipilih"] == "True"]
    lc = [r for r in csv.DictReader(open(SEL / "luar_cakupan_ds4.csv", encoding="utf-8")) if r["masuk_daftar"] == "True"]
    for sp in SPL:
        (V8 / "images" / sp).mkdir(parents=True)
        (V8 / "labels" / sp).mkdir(parents=True)
    for r in man:
        sp, f = r["split"], r["file"]
        tautkan(V7 / "images" / sp / f, V8 / "images" / sp / f)
        lab = V7 / "labels" / sp / (Path(f).stem + ".txt")
        if lab.exists():
            tautkan(lab, V8 / "labels" / sp / lab.name)
    gmax = max(int(r["group"]) for r in man)
    # id kelompok baru: komponen graf -> id baru (> id prov_v7 terbesar); komponen Tungro yang memuat kelompok train prov_v7
    # memakai id prov_v7 terkecil di komponen itu (komponen_v7.csv dari tahap seleksi)
    gid = {}
    nxt = gmax + 1
    new_rows, kliping = [], 0
    names_dp = yaml.safe_load((DP / "data.yaml").read_text(encoding="utf-8"))["names"]
    for r in sorted(dp, key=lambda r: r["berkas"]):
        p = ROOT / r["berkas"]
        g = gid.setdefault(("dp", r["komponen"]), None)
        if g is None:
            gid[("dp", r["komponen"])] = g = nxt
            nxt += 1
        f = nama_baru(p, "dp_")
        sp = r["split"]
        tautkan(p, V8 / "images" / sp / f)
        out = []
        for ln in (p.parent.parent / "labels" / (p.stem + ".txt")).read_text().splitlines():
            t = ln.split()
            if not t:
                continue
            if len(t) != 5:
                sys.exit(f"[GAGAL] label daunpadi bukan kotak: {p}")
            c = NAMES.index(DP_MAP[names_dp[int(t[0])]])
            x, y, w, h = map(float, t[1:])
            x1, y1, x2, y2 = max(0.0, x - w / 2), max(0.0, y - h / 2), min(1.0, x + w / 2), min(1.0, y + h / 2)
            if (x1, y1, x2, y2) != (x - w / 2, y - h / 2, x + w / 2, y + h / 2):
                kliping += 1
            out.append(f"{c} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}")
        (V8 / "labels" / sp / (Path(f).stem + ".txt")).write_text("\n".join(out) + "\n", encoding="utf-8")
        new_rows.append({"file": f, "split": sp, "class": r["kelas"], "source": "daunpadi", "provisional": "False", "train_only": "False",
                         "aug_possible": str(bool(r["aug_reasons"])), "aug_reasons": r["aug_reasons"], "group": str(g),
                         "n_boxes": str(len(out)), "n_boxes_from_polygon": "0", "orig_split": r["split_roboflow"], "orig_path": r["berkas"],
                         "perbaikan_orientasi": "-"})
    comp_members = defaultdict(list)
    for r in csv.DictReader(open(SEL / "komponen_v7.csv", encoding="utf-8")):
        comp_members[r["komponen"]].append(int(r["group_v7"]))
    for r in sorted(tg, key=lambda r: r["berkas"]):
        p = ROOT / r["berkas"]
        if comp_members.get(r["komponen"]):
            g = min(comp_members[r["komponen"]])
        else:
            g = gid.setdefault(("tg", r["komponen"]), None)
            if g is None:
                gid[("tg", r["komponen"])] = g = nxt
                nxt += 1
        f = nama_baru(p, "ds4_")
        tautkan(p, V8 / "images" / "train" / f)
        (V8 / "labels" / "train" / (Path(f).stem + ".txt")).write_text("", encoding="utf-8")
        new_rows.append({"file": f, "split": "train", "class": "background", "source": "ds4_tungro", "provisional": "False", "train_only": "True",
                         "aug_possible": str(bool(r.get("aug_reasons"))), "aug_reasons": r.get("aug_reasons", ""), "group": str(g),
                         "n_boxes": "0", "n_boxes_from_polygon": "0", "orig_split": "-", "orig_path": r["berkas"], "perbaikan_orientasi": "-"})
    semua = man + new_rows
    with open(V8 / "manifest.csv", "w", newline="", encoding="utf-8") as fo:
        w = csv.DictWriter(fo, fieldnames=kol)
        w.writeheader()
        w.writerows(semua)
    shutil.copytree(V7 / "lists", V8 / "lists")
    for (sp, src, nama) in (("train", "daunpadi", "train_daunpadi.txt"), ("val", "daunpadi", "val_daunpadi.txt"), ("train", "ds4_tungro", "train_ds4_tungro.txt")):
        (V8 / "lists" / nama).write_text("".join(f"images/{sp}/{r['file']}\n" for r in new_rows if r["split"] == sp and r["source"] == src), encoding="utf-8")
    (V8 / "luar_cakupan").mkdir()
    for k, nama in (("Leaf smut", "ds4_leaf_smut.txt"), ("Sheath blight", "ds4_sheath_blight.txt")):
        (V8 / "luar_cakupan" / nama).write_text("".join(f"{r['berkas']}\tLC{int(r['komponen']):06d}\n" for r in lc if r["kelas_luar_cakupan"] == k), encoding="utf-8")
    dy8 = dict(dy7, path=str(V8).replace("\\", "/"), dataset_name="prov_v8", run_prefix="v8_", manifest_sha256=sha(V8 / "manifest.csv"), frozen=False,
               induk_dataset="prov_v7", induk_manifest_sha256=dy7["manifest_sha256"],
               perbaikan="prov_v7 + daunpadi (train/val) + Tungro ds4 (background negatif, train saja); baris prov_v7 identik",
               negatif_background_sources=["ds4_tungro"])
    dy8.pop("induk_dataset_v6", None)
    (V8 / "data.yaml").write_text(yaml.safe_dump(dy8, allow_unicode=True, sort_keys=False), encoding="utf-8")
    m7 = json.loads((V7 / "dataset_meta.json").read_text(encoding="utf-8"))
    ring = json.loads((SEL / "ringkasan_seleksi.json").read_text(encoding="utf-8"))
    m8 = dict(m7, dataset_name="prov_v8", dibuat=datetime.now().isoformat(timespec="seconds"), manifest_sha256=dy8["manifest_sha256"], frozen=False,
              freeze={"frozen": False, "alasan_tidak_dibekukan": "menunggu pemeriksaan"},
              induk={"dataset": "prov_v7", "manifest_sha256": dy7["manifest_sha256"]},
              jumlah={sp: sum(1 for r in semua if r["split"] == sp) for sp in SPL},
              tambahan_v8={"keputusan": "KEPUTUSAN_PROYEK.md 2026-10-14 KEPUTUSAN v8", "seleksi": "reports/prov_v8/seleksi/ringkasan_seleksi.json",
                           "daunpadi": ring["daunpadi"], "tungro": ring["tungro"], "luar_cakupan_ds4": ring["luar_cakupan_ds4"],
                           "kotak_dikliping_ke_bingkai": kliping, "aturan_kelompok": __doc__.split("Graf kelompok")[1].split("Aturan pilih")[0].strip()})
    m8["catatan"] = list(m7.get("catatan", [])) + ["prov_v8: semua baris prov_v7 identik; tambahan daunpadi (Brownspot/leafblast satu kelas, kelompok tanpa kembar "
                                                   "luar) dan Tungro ds4 sebagai background negatif (source ds4_tungro, train saja, tidak dihitung dalam batas "
                                                   "background 10% daun sehat)."]
    m8.pop("perbaikan_orientasi", None)
    (V8 / "dataset_meta.json").write_text(json.dumps(m8, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- pemeriksaan kesetaraan baris prov_v7
    m8r = list(csv.DictReader(open(V8 / "manifest.csv", encoding="utf-8")))
    k7 = {(r["split"], r["file"], r["group"], r["class"], r["source"]) for r in man}
    k8 = {(r["split"], r["file"], r["group"], r["class"], r["source"]) for r in m8r if r["source"] not in ("daunpadi", "ds4_tungro")}
    beda_img, beda_lab = 0, 0
    for r in man:
        sp, f = r["split"], r["file"]
        if sha(V7 / "images" / sp / f) != sha(V8 / "images" / sp / f):
            beda_img += 1
        l7, l8 = V7 / "labels" / sp / (Path(f).stem + ".txt"), V8 / "labels" / sp / (Path(f).stem + ".txt")
        if l7.exists() != l8.exists() or (l7.exists() and sha(l7) != sha(l8)):
            beda_lab += 1
    baru = [r for r in m8r if r["source"] in ("daunpadi", "ds4_tungro")]
    hasil = {"baris_v7_identik": k7 == k8, "selisih_baris_v7": len(k7 ^ k8), "gambar_v7_berbeda": beda_img, "label_v7_berbeda": beda_lab,
             "test_dan_tns_tidak_berubah": all(r["split"] in ("train", "val") for r in baru),
             "jumlah_per_split_v7": {sp: sum(1 for r in man if r["split"] == sp) for sp in SPL},
             "jumlah_per_split_v8": {sp: len(list((V8 / "images" / sp).iterdir())) for sp in SPL},
             "baris_baru": dict(Counter(f"{r['split']}|{r['source']}|{r['class']}" for r in baru)),
             "kelompok_baru_lintas_split": sum(1 for g, sps in _grup_split(m8r).items() if len(sps) > 1),
             "kotak_dikliping": kliping}
    hasil["LOLOS"] = (hasil["baris_v7_identik"] and beda_img == 0 and beda_lab == 0 and hasil["test_dan_tns_tidak_berubah"]
                      and hasil["kelompok_baru_lintas_split"] == 0)
    (REP / "pemeriksaan_kesetaraan.json").write_text(json.dumps(hasil, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(hasil, indent=1, ensure_ascii=False))
    sys.exit(0 if hasil["LOLOS"] else 1)


def _grup_split(rows):
    g = defaultdict(set)
    for r in rows:
        g[r["group"]].add(r["split"])
    return g


# ============================================================ audit
def audit():
    man = list(csv.DictReader(open(V8 / "manifest.csv", encoding="utf-8")))
    tr = [r for r in man if r["split"] == "train"]
    uji = [r for r in man if r["split"] != "train"]
    Pt = [ROOT / r["orig_path"] for r in tr]
    Pu = [ROOT / r["orig_path"] for r in uji]
    Et, Eu = ad.embeddings(Pt), ad.embeddings(Pu)
    dr.ensure_descriptors(Pt + Pu)
    nb = dr.knn_pairs_cross(Eu, Et, KNN_AUDIT)
    print(f"[audit SIFT] {len(nb)} pasangan", flush=True)
    res = dr.verify_pairs([(Pu[i], Pt[j]) for i, j in nb])
    kena = defaultdict(list)
    for (i, j), r in zip(nb, res):
        if r[0] >= dr.MIN_INLIERS:
            kena[int(i)].append((int(j), int(r[0])))
    per = {}
    for sp in ("val", "test", "test_new_session"):
        idx = [i for i, r in enumerate(uji) if r["split"] == sp]
        k = [i for i in idx if kena.get(i)]
        k_baru = [i for i in idx if any(tr[j]["source"] in ("daunpadi", "ds4_tungro") for j, _ in kena.get(i, []))]
        u_baru = [i for i in idx if uji[i]["source"] == "daunpadi" and kena.get(i)]
        per[sp] = {"gambar": len(idx), "punya_kembar_train": len(k), "persen": round(100 * len(k) / max(1, len(idx)), 3),
                   "kembar_dengan_train_BARU": len(k_baru), "gambar_uji_baru_berkembar": len(u_baru)}
    tot = sum(v["punya_kembar_train"] for v in per.values())
    ntot = sum(v["gambar"] for v in per.values())
    rinci = [{"uji": uji[i]["file"], "split": uji[i]["split"], "sumber_uji": uji[i]["source"], "train": tr[j]["file"], "sumber_train": tr[j]["source"],
              "inlier": n_} for i, v in kena.items() for j, n_ in v]
    # audit waktu ds4 (train termasuk Tungro) vs val/test/tns
    m4 = dr.ds4_timestamps(XLSX)

    def t4(r):
        return m4.get(Path(r["orig_path"]).name.lower()) if r["source"].startswith("ds4") else None
    t_tr = sorted(t for t in (t4(r) for r in tr) if t)
    t_tr_tg = sorted(t for t in (t4(r) for r in tr if r["source"] == "ds4_tungro") if t)
    import bisect

    def dekat(t, arr, s):
        a = bisect.bisect_left(arr, t - timedelta(seconds=s))
        return a < len(arr) and arr[a] <= t + timedelta(seconds=s)
    waktu = {}
    for s in (5, 10):
        waktu[f"+-{s}dtk"] = {sp: sum(1 for r in uji if r["split"] == sp and t4(r) and dekat(t4(r), t_tr, s)) for sp in ("val", "test", "test_new_session")}
        waktu[f"+-{s}dtk_Tungro"] = {sp: sum(1 for r in uji if r["split"] == sp and t4(r) and dekat(t4(r), t_tr_tg, s)) for sp in ("val", "test", "test_new_session")}
    # kembar langsung: set uji lain -> train BARU (daunpadi train + Tungro) dan -> semua train
    baru = [i for i, r in enumerate(tr) if r["source"] in ("daunpadi", "ds4_tungro")]
    Pb, Eb = [Pt[i] for i in baru], Et[baru]
    lintas = {}
    set_uji = {"ds3": imgs(ROOT / "data/raw/ds3"), "aditia": imgs(AD),
               "luar_cakupan_ds4": [ROOT / ln.split("\t")[0] for f in ("ds4_leaf_smut.txt", "ds4_sheath_blight.txt")
                                    for ln in (V8 / "luar_cakupan" / f).read_text(encoding="utf-8").splitlines() if ln.strip()]}
    var_b, ok_b = pd_.compute_hashes([{"path": p} for p in Pb], 16)
    var_t, ok_t = pd_.compute_hashes([{"path": p} for p in Pt], 16)
    for nama, Ps in set_uji.items():
        Es = ad.embeddings(Ps)
        dr.ensure_descriptors(Ps)
        var_s, ok_s = pd_.compute_hashes([{"path": p} for p in Ps], 16)
        hasil = {}
        for lbl, (P_, E_, vv, oo) in {"train_baru": (Pb, Eb, var_b, ok_b), "semua_train": (Pt, Et, var_t, ok_t)}.items():
            ph = {int(i) for v in range(8) for i, j in pd_.near_pairs(var_s[:, v], vv[:, 0], PHASH_THR, False) if ok_s[i] and oo[j]}
            S = np.max([Es[:, v] @ E_[:, 0].T for v in range(8)], axis=0)
            em = {int(i) for i in np.where((S >= EMB_THR).any(1))[0]}
            nb2 = dr.knn_pairs_cross(Es, E_, KNN_AUDIT if lbl == "train_baru" else 50)
            si = {int(i) for (i, j), r in zip(nb2, dr.verify_pairs([(Ps[i], P_[j]) for i, j in nb2])) if r[0] >= dr.MIN_INLIERS}
            hasil[lbl] = {"phash": len(ph), "emb": len(em), "sift": len(si), "gambar_uji_berkembar": len(ph | em | si)}
        lintas[nama] = {"gambar": len(Ps), **hasil}
        print(f"[lintas] {nama}: {lintas[nama]}", flush=True)
    lolos = (tot / ntot * 100 < 1.0 and all(v["kembar_dengan_train_BARU"] == 0 and v["gambar_uji_baru_berkembar"] == 0 for v in per.values())
             and all(v == 0 for v in waktu["+-5dtk"].values())
             and all(v["train_baru"]["gambar_uji_berkembar"] == 0 for v in lintas.values()))
    out = {"dibuat": datetime.now().isoformat(timespec="seconds"), "model_dijalankan": False,
           "kriteria_lolos": "SIFT residual val+test+tns < 1% (k=200, inlier>=21 tanpa sebaran, sama dengan v7) DAN 0 kembar yang melibatkan gambar baru "
                             "DAN audit waktu +-5 dtk = 0 DAN 0 kembar langsung train baru vs ds3/aditia/luar cakupan ds4",
           "sift_k200": per, "sift_residual_persen_total": round(100 * tot / ntot, 3), "waktu_ds4": waktu, "lintas_set_uji": lintas, "LOLOS": lolos}
    (REP / "audit.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(REP / "audit_pasangan_sift.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["uji", "split", "sumber_uji", "train", "sumber_train", "inlier"])
        w.writeheader()
        w.writerows(rinci)
    print(json.dumps(out, indent=1, ensure_ascii=False))
    sys.exit(0 if lolos else 1)


# ============================================================ bekukan
def bekukan():
    pk = json.loads((REP / "pemeriksaan_kesetaraan.json").read_text(encoding="utf-8"))
    au = json.loads((REP / "audit.json").read_text(encoding="utf-8"))
    vd = (REP / "verify_dataset.log").read_text(encoding="utf-8", errors="replace")
    if not (pk["LOLOS"] and au["LOLOS"] and "HASIL: LOLOS" in vd):
        sys.exit(f"[DITOLAK] pemeriksaan belum lolos: kesetaraan={pk['LOLOS']} audit={au['LOLOS']} verify={'HASIL: LOLOS' in vd}")
    now = datetime.now().isoformat(timespec="seconds")
    dy = yaml.safe_load((V8 / "data.yaml").read_text(encoding="utf-8"))
    dy["frozen"] = True
    (V8 / "data.yaml").write_text(yaml.safe_dump(dy, allow_unicode=True, sort_keys=False), encoding="utf-8")
    m = json.loads((V8 / "dataset_meta.json").read_text(encoding="utf-8"))
    m["frozen"] = True
    m["freeze"] = {"frozen": True, "frozen_at": now, "alasan_tidak_dibekukan": None, "manifest_sha256": sha(V8 / "manifest.csv"),
                   "data_yaml_sha256": sha(V8 / "data.yaml"), "labels_sha256": labels_sha(V8),
                   "pemeriksaan": {"kesetaraan": "reports/prov_v8/pemeriksaan_kesetaraan.json", "audit": "reports/prov_v8/audit.json",
                                   "verify_dataset": "reports/prov_v8/verify_dataset.log"}}
    (V8 / "dataset_meta.json").write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    (V8 / "FROZEN.txt").write_text(f"DIBEKUKAN {now}\nmanifest_sha256={m['freeze']['manifest_sha256']}\nPerubahan lanjutan harus menjadi versi baru.\n",
                                   encoding="utf-8")
    print(f"[OK] prov_v8 dibekukan {now}; manifest {m['freeze']['manifest_sha256']}")


if __name__ == "__main__":
    {"seleksi": seleksi, "bangun": bangun, "audit": audit, "bekukan": bekukan}[sys.argv[1]]()
