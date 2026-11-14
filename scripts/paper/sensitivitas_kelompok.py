"""Sensitivitas ambang pengelompokan kembar (ANALISIS UNTUK REVISI PAPER SOFTT, tanpa training).

Graf kembar prov_v7 dibangun ulang dari data kemiripan yang SUDAH TERSIMPAN, lalu ambang dilonggarkan:
  * DINOv2-small cosine 0,95 (bawaan 0,97)   : data/interim/dino_emb.npz (8 varian dihedral, cache prepare_dataset)
  * SIFT 15 inlier (bawaan 21)                : reports/prov_v6/sift_pairs.csv (semua pasangan kNN k=100 yang
                                                 diverifikasi dengan inlier >= 8; aturan sebaran >= 2% tetap dipakai)
  * selang waktu 10 detik (bawaan 5)          : stempel waktu ds4 (Dataset metadata.xlsx) dan ds1 (nama berkas)
  * gabungan ketiganya.
Split TIDAK diubah: split beku prov_v7 (manifest.csv) hanya dibaca. Kelompok bawaan (kolom `group`, sudah memuat
phash, nama dasar Roboflow, K4, dan rantai lewat foto yang dibuang) dipakai sebagai sisi dasar, lalu sisi longgar
ditambahkan. Simpul = 8959 foto prov_v7 (foto di luar dataset tidak ikut sebagai jembatan rantai).

Keluaran: reports/revisi_paper/sensitivitas_kelompok/
"""
import csv
import datetime
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import dedupe_rules as dr  # noqa: E402  (fungsi waktu yang sama dengan pembangun dataset)

MANIFEST = ROOT / "data/processed/prov_v7/manifest.csv"
SIFT_CSV = ROOT / "reports/prov_v6/sift_pairs.csv"
EMB = ROOT / "data/interim/dino_emb.npz"
XLSX = ROOT / "data/raw/ds4/docs/Dataset metadata.xlsx"
OUT = ROOT / "reports/revisi_paper/sensitivitas_kelompok"
MIN_FRAC = 0.02          # aturan sebaran kotak inlier prov_v6/v7 (classes_v7.yaml)
SPLIT_Q = ["test", "test_new_session", "val"]


def jalur_embedding(r):
    """Jalur yang dipakai prepare_dataset saat menghitung embedding (ds2 diperkecil ke data/interim/resized)."""
    if r.source == "ds2":
        return "data/interim/resized/ds2/" + hashlib.sha1(r.orig_path.encode()).hexdigest()[:16] + ".jpg"
    return r.orig_path


def muat_embedding(m):
    z = np.load(EMB, allow_pickle=True)
    per_jalur = defaultdict(list)
    for k, s in enumerate(z["sig"]):
        p, size, mt = str(s).rsplit("|", 2)
        per_jalur[p].append((int(size), int(mt), k))
    idx, cocok_stat = [], 0
    for r in m.itertuples():
        p = jalur_embedding(r)
        cand = per_jalur.get(p)
        if not cand:
            raise SystemExit(f"embedding tidak ada di cache: {p}")
        st = (ROOT / p).stat()
        hit = [k for size, mt, k in cand if size == st.st_size and mt == st.st_mtime_ns]
        cocok_stat += bool(hit)
        idx.append(hit[0] if hit else cand[-1][2])
    E = z["emb"][np.array(idx)].astype(np.float32)
    return E, cocok_stat


def pasangan_embedding(E, thr, chunk=1024):
    """Sama dengan aug_detect.sim_pairs: maks atas 8 varian query vs varian asli acuan, dua arah."""
    O = E[:, 0]
    out = []
    for s in range(0, len(E), chunk):
        S = np.max([E[s:s + chunk, v] @ O.T for v in range(E.shape[1])], axis=0)
        ii, jj = np.where(S >= thr)
        ii = ii + s
        k = ii != jj
        out.append(np.stack([ii[k], jj[k], (S[ii[k] - s, jj[k]] * 1e4).astype(int)], 1))
    P = np.concatenate(out)
    P[:, :2] = np.sort(P[:, :2], axis=1)
    # simpan nilai maks per pasangan
    df = pd.DataFrame(P, columns=["i", "j", "v"]).groupby(["i", "j"], as_index=False)["v"].max()
    return [(int(a), int(b), v / 1e4) for a, b, v in df.itertuples(index=False)]


def pasangan_sift(m, min_inl):
    pos = {p: k for k, p in enumerate(m.orig_path)}
    d = pd.read_csv(SIFT_CSV)
    d = d[(d.inlier >= min_inl) & (d.frac_kotak_inlier >= MIN_FRAC)]
    out, luar = [], 0
    for a, b, n in zip(d.a, d.b, d.inlier):
        i, j = pos.get(a), pos.get(b)
        if i is None or j is None:
            luar += 1
            continue
        out.append((min(i, j), max(i, j), int(n)))
    return out, luar


def pasangan_waktu(m, gap):
    ts_map = dr.ds4_timestamps(XLSX)
    cands = [{"src": "ds1" if s.startswith("ds1") else s, "path": p, "orig_path": p} for s, p in zip(m.source, m.orig_path)]
    ts = [dr.timestamp_of(c, ts_map) for c in cands]
    P, n_ts = dr.time_pairs(cands, lambda c: dr.timestamp_of(c, ts_map), gap, ("ds4", "ds1"))
    return [(int(min(i, j)), int(max(i, j)), abs((ts[i] - ts[j]).total_seconds())) for i, j in P], n_ts


def union_find(n, edges):
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    return np.array([find(i) for i in range(n)])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(MANIFEST)
    n = len(m)
    split = m.split.to_numpy()
    print(f"manifest prov_v7: {n} foto; kelompok beku: {m.group.nunique()}", flush=True)

    # sisi dasar: anggota kelompok beku dihubungkan berantai
    dasar = []
    for _, idx in m.groupby("group").indices.items():
        idx = sorted(idx)
        dasar += [(idx[0], k) for k in idx[1:]]

    E, cocok = muat_embedding(m)
    print(f"embedding: {n} foto (ukuran+mtime cocok dengan cache: {cocok})", flush=True)
    kriteria = {}
    kriteria["dino_0.97"] = pasangan_embedding(E, 0.97)
    kriteria["dino_0.95"] = pasangan_embedding(E, 0.95)
    sift21, luar21 = pasangan_sift(m, 21)
    sift15, luar15 = pasangan_sift(m, 15)
    kriteria["sift_21"], kriteria["sift_15"] = sift21, sift15
    kriteria["waktu_5"], n_ts = pasangan_waktu(m, 5)
    kriteria["waktu_10"], _ = pasangan_waktu(m, 10)
    print("pasangan per kriteria:", {k: len(v) for k, v in kriteria.items()}, f"| SIFT dgn ujung di luar dataset: 21={luar21}, 15={luar15}",
          f"| bertanggal: {n_ts}", flush=True)

    pengaturan = {
        "bawaan (0,97 / 21 / 5 dtk)": ["dino_0.97", "sift_21", "waktu_5"],
        "DINOv2 0,95": ["dino_0.95", "sift_21", "waktu_5"],
        "SIFT 15 inlier": ["dino_0.97", "sift_15", "waktu_5"],
        "waktu 10 dtk": ["dino_0.97", "sift_21", "waktu_10"],
        "gabungan (0,95 / 15 / 10 dtk)": ["dino_0.95", "sift_15", "waktu_10"],
        # tambahan: DINOv2 tetap 0,97 (pada 0,95 test_new_session ikut 'kembar' 30%, lihat CATATAN.md)
        "SIFT 15 + waktu 10 dtk (DINOv2 0,97)": ["dino_0.97", "sift_15", "waktu_10"],
    }
    total = {s: int((split == s).sum()) for s in SPLIT_Q}
    baris, per_foto = [], []
    for nama, ks in pengaturan.items():
        sisi = [(i, j) for k in ks for i, j, _ in kriteria[k]]
        g = union_find(n, dasar + sisi)
        ukuran = Counter(g)
        g_train = {x for x, s in zip(g, split) if s == "train"}
        # kembaran langsung di train: ada sisi kriteria (ambang pengaturan ini) ke foto train
        kembar = defaultdict(list)
        for k in ks:
            for i, j, v in kriteria[k]:
                for a, b in ((i, j), (j, i)):
                    if split[a] != "train" and split[b] == "train":
                        kembar[a].append((k, b, v))
        lintas = sum(len({split[i] for i in idx}) > 1 for idx in pd.Series(range(n)).groupby(g).groups.values())
        r = {"pengaturan": nama, "kriteria": "+".join(ks), "jumlah_kelompok": len(ukuran), "kelompok_terbesar": max(ukuran.values()),
             "kelompok_lintas_split": lintas}
        for s in SPLIT_Q:
            idx = np.where(split == s)[0]
            langsung = [i for i in idx if i in kembar]
            se_kel = [i for i in idx if g[i] in g_train]
            r[f"{s}_kembar_langsung"] = len(langsung)
            r[f"{s}_kembar_langsung_pct"] = round(100 * len(langsung) / total[s], 2)
            r[f"{s}_sekelompok_train"] = len(se_kel)
            r[f"{s}_sekelompok_train_pct"] = round(100 * len(se_kel) / total[s], 2)
            for k in ks:
                r[f"{s}_langsung_{k}"] = sum(any(t[0] == k for t in kembar[i]) for i in idx)
            if s in ("test", "test_new_session"):
                for i in langsung:
                    k, b, v = max(kembar[i], key=lambda t: t[2] if t[0].startswith("dino") or t[0].startswith("sift") else -t[2])
                    per_foto.append({"pengaturan": nama, "foto": m.file[i], "split": s, "kelas": m["class"][i], "sumber": m.source[i],
                                     "kriteria": ";".join(sorted({t[0] for t in kembar[i]})), "contoh_kriteria": k,
                                     "contoh_kembaran_train": m.file[b], "nilai": round(v, 4)})
        baris.append(r)
        print(nama, {k: r[k] for k in r if k.endswith(("kelompok", "terbesar", "langsung", "langsung_pct", "train_pct"))}, flush=True)

    pd.DataFrame(baris).to_csv(OUT / "ringkasan.csv", index=False, encoding="utf-8")
    pd.DataFrame(per_foto).to_csv(OUT / "per_foto_kembar.csv", index=False, encoding="utf-8")
    json.dump({"dibuat": datetime.datetime.now().isoformat(timespec="seconds"), "manifest": str(MANIFEST.relative_to(ROOT)),
               "jumlah_foto": n, "total_per_split": total, "pasangan_per_kriteria": {k: len(v) for k, v in kriteria.items()},
               "sift_ujung_di_luar_dataset": {"21": luar21, "15": luar15}, "bertanggal": n_ts, "embedding_stat_cocok": cocok,
               "hasil": baris}, open(OUT / "ringkasan.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    md = ["# Sensitivitas ambang pengelompokan kembar (ANALISIS UNTUK REVISI PAPER; tanpa training, split beku prov_v7)", "",
          f"Foto: {n} (test {total['test']}, test_new_session {total['test_new_session']}, val {total['val']}). "
          "Kelompok beku dipakai sebagai dasar, lalu sisi dari ambang longgar ditambahkan. "
          "'Kembar langsung' = foto punya pasangan di TRAIN menurut kriteria pengaturan itu; "
          "'sekelompok' = kelompok (rantai) foto memuat foto train.", "",
          "| Pengaturan | Kelompok | Terbesar | Test kembar langsung | Test sekelompok | TNS kembar langsung | TNS sekelompok | Val kembar langsung |",
          "|---|---|---|---|---|---|---|---|"]
    for r in baris:
        md.append(f"| {r['pengaturan']} | {r['jumlah_kelompok']} | {r['kelompok_terbesar']} | "
                  f"{r['test_kembar_langsung']} ({r['test_kembar_langsung_pct']}%) | {r['test_sekelompok_train']} ({r['test_sekelompok_train_pct']}%) | "
                  f"{r['test_new_session_kembar_langsung']} ({r['test_new_session_kembar_langsung_pct']}%) | "
                  f"{r['test_new_session_sekelompok_train']} ({r['test_new_session_sekelompok_train_pct']}%) | "
                  f"{r['val_kembar_langsung']} ({r['val_kembar_langsung_pct']}%) |")
    md += ["", "Rincian per kriteria (foto test dengan kembaran langsung di train):", ""]
    for r in baris:
        md.append(f"- {r['pengaturan']}: " + ", ".join(f"{k}={r[f'test_langsung_{k}']}" for k in r["kriteria"].split("+")))
    (OUT / "RINGKASAN.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("selesai ->", OUT)


if __name__ == "__main__":
    main()
