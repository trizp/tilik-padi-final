#!/usr/bin/env python3
"""
prepare_dataset.py - Menyusun dataset YOLO final dari sumber mentah, sesuai configs/classes.yaml.

Langkah:
  1. Kumpulkan kandidat dari tiap sumber (yolo / yolo_folders / classification), filter dan
     remap kelas berdasarkan NAMA (bukan ID). Gambar yang memuat kelas di luar target dibuang.
     Gambar sehat dipakai sebagai background (tanpa box). Label kosong pada kelas penyakit dibuang.
  2. Hitung perceptual hash (phash, termasuk versi flip horizontal) untuk semua kandidat.
  3. Kelompokkan duplikat (lintas sumber) -> satu grup selalu berada di SATU split.
  4. Split 70:20:10 stratified per (kelas, sumber) dengan grup utuh.
  5. Tulis images/labels/{train,val,test}, data.yaml, manifest.csv, daftar test multi-penyakit,
     dan laporan statistik (Markdown + CSV + JSON) ke report_dir.
  6. Verifikasi mandiri: tidak ada pasangan gambar lintas split dengan jarak phash <= ambang.

Pemakaian:
  python scripts/prepare_dataset.py                 # gagal jika output_dir sudah ada
  python scripts/prepare_dataset.py --force         # bangun ulang output_dir (data mentah tidak disentuh)
  python scripts/prepare_dataset.py --dry-run       # hitung + laporan saja, tanpa menulis gambar
"""
import argparse
import csv
import datetime
import hashlib
import json
import os
import platform
import random
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from inventory import IMG_EXT, find_yolo_label, load_yaml_names, split_of  # noqa: E402
import aug_detect as ad  # noqa: E402
import dedupe_rules as dr  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BG = "__background__"
SPLITS = ("train", "val", "test")
HOLD = "test_new_session"     # split terpisah opsional (sesi pemotretan utuh), tidak masuk train/val/test
CACHE = ROOT / "data/interim/phash_prepare.npz"


def norm(s: str) -> str:
    """Normalisasi nama: huruf kecil, deretan non-alfanumerik -> satu '_'."""
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def rel(p: Path) -> str:
    try:
        return Path(p).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return Path(p).as_posix()


# ============================================================ 1. KUMPULKAN KANDIDAT
class Excl:
    """Pencatat gambar yang dibuang beserta alasannya (untuk laporan). `items` menyimpan (sumber, alasan, path) per gambar."""

    def __init__(self):
        self.c = Counter()
        self.items = []

    def add(self, source, reason, n=1, path=None):
        self.c[(source, reason)] += n
        if path is not None:
            self.items.append((source, reason, rel(path)))


def clean_boxes(rows, excl, src):
    """
    rows: list (cls, cx, cy, w, h[, dari_poligon]). Klip ke [0,1], buang box degenerate dan baris
    ganda persis. Keluaran: tuple 6 elemen (cls, cx, cy, w, h, dari_poligon).
    """
    out, seen = [], set()
    for c, cx, cy, w, h, *fl in rows:
        x1, y1 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
        x2, y2 = min(1.0, cx + w / 2), min(1.0, cy + h / 2)
        nw, nh = x2 - x1, y2 - y1
        if nw <= 1e-4 or nh <= 1e-4:
            excl.add(src, "box_degenerate")
            continue
        key = (c, round(x1, 5), round(y1, 5), round(x2, 5), round(y2, 5))
        if key in seen:
            excl.add(src, "box_duplikat_persis")
            continue
        seen.add(key)
        out.append((c, (x1 + x2) / 2, (y1 + y2) / 2, nw, nh, bool(fl and fl[0])))
    return out


def read_label_lines(lab: Path, polygon_to_bbox=False):
    """
    Baca label YOLO. Kembalikan (rows, n_tak_valid); rows: (cid, cx, cy, w, h, dari_poligon).
    Baris 5 kolom = box. Baris poligon (cls x1 y1 x2 y2 x3 y3 ... , >= 3 titik) dikonversi ke bbox
    pembatas jika polygon_to_bbox=True, selain itu dihitung tak valid.
    """
    rows, bad = [], 0
    for line in lab.read_text(errors="ignore").splitlines():
        t = line.split()
        if not t:
            continue
        try:
            if len(t) == 5:
                rows.append((int(float(t[0])), *map(float, t[1:5]), False))
            elif polygon_to_bbox and len(t) >= 7 and (len(t) - 1) % 2 == 0:
                v = np.array(list(map(float, t[1:])))
                xs, ys = v[0::2], v[1::2]
                x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
                rows.append((int(float(t[0])), (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, True))
            else:
                bad += 1
        except ValueError:
            bad += 1
    return rows, bad


def collect_yolo(cfg, classes, excl, meta):
    src, root = cfg["name"], ROOT / cfg["path"]
    names, _ = load_yaml_names(root)
    if not names:
        raise SystemExit(f"[{src}] data.yaml (names) tidak ditemukan di {root}")
    cmap = {norm(k): v for k, v in cfg["map"].items()}
    tags = cfg.get("source_tag_by_class", {})
    unmapped_seen = Counter()
    out = []
    imgs = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT)
    for img in imgs:
        lab = find_yolo_label(img)
        if lab is None:
            excl.add(src, "tanpa_label", path=img)
            continue
        raw, bad = read_label_lines(lab, cfg.get("polygon_to_bbox", False))
        if bad:
            excl.add(src, "baris_label_tak_valid", bad)
        rows, bad_img = [], False
        for cid, cx, cy, w, h, poly in raw:
            sname = norm(names[cid]) if 0 <= cid < len(names) else f"id{cid}"
            tgt = cmap.get(sname)
            if tgt is None:
                unmapped_seen[sname] += 1
                if cfg.get("drop_unmapped_images", True):
                    bad_img = True
                    break
                continue
            rows.append((classes.index(tgt), cx, cy, w, h, poly))
        if bad_img:
            excl.add(src, "memuat_kelas_di_luar_target", path=img)
            continue
        rows = clean_boxes(rows, excl, src)
        if not rows:
            excl.add(src, "tanpa_box_valid", path=img)
            continue
        cls_set = sorted({r[0] for r in rows})
        tag_by_cls = {c: tags.get(classes[c], src) for c in cls_set}
        t_only = any(classes[c] in cfg.get("train_only_classes", []) for c in cls_set)
        out.append(dict(src=src, path=img, boxes=rows, cls_ids=cls_set, train_only=t_only,
                        tag=tag_by_cls[cls_set[0]] if len(cls_set) == 1 else "+".join(tag_by_cls[c] for c in cls_set),
                        orig_split=split_of(img.relative_to(root)), max_side=None))
    meta[src] = {"kelas_sumber_di_luar_target_(jumlah_box)": dict(unmapped_seen.most_common())}
    return out


def collect_yolo_folders(cfg, classes, excl, meta):
    src = cfg["name"]
    idir, ldir = ROOT / cfg["images_dir"], ROOT / cfg["labels_dir"]
    fmap = {norm(k): v for k, v in cfg["folder_map"].items()}
    lab_folders = {norm(p.name): p for p in ldir.iterdir() if p.is_dir()}
    ids_by_folder = defaultdict(Counter)
    out = []
    for fdir in sorted(p for p in idir.iterdir() if p.is_dir()):
        key = norm(fdir.name)
        tgt = fmap.get(key)
        imgs = sorted(p for p in fdir.iterdir() if p.suffix.lower() in IMG_EXT)
        if tgt is None:
            excl.add(src, f"folder_tidak_dipakai:{fdir.name}", len(imgs))
            for im_ in imgs:
                excl.add(src, f"folder_tidak_dipakai:{fdir.name}", 0, path=im_)
            continue
        if tgt == BG:
            for img in imgs:
                out.append(dict(src=src, path=img, boxes=[], cls_ids=[], tag=src, orig_split="-", max_side=None))
            continue
        cid = classes.index(tgt)
        lfolder = lab_folders.get(key)
        for img in imgs:
            lab = (lfolder / "labels" / (img.stem + ".txt")) if lfolder else None
            if lab is None or not lab.exists():
                excl.add(src, "label_tidak_ditemukan", path=img)
                continue
            raw, bad = read_label_lines(lab)
            if bad:
                excl.add(src, "baris_label_tak_valid", bad)
            for r in raw:
                ids_by_folder[fdir.name][r[0]] += 1
            rows = clean_boxes([(cid, *r[1:]) for r in raw], excl, src)   # ID file label DIABAIKAN
            if not rows:
                excl.add(src, "label_kosong" if not raw else "tanpa_box_valid", path=img)
                continue
            out.append(dict(src=src, path=img, boxes=rows, cls_ids=[cid], tag=src, orig_split="-", max_side=None))
    meta[src] = {"id_label_mentah_per_folder_(diabaikan)": {k: dict(v) for k, v in ids_by_folder.items()}}
    return out


def collect_classification(cfg, classes, excl, meta, rng):
    src, root = cfg["name"], ROOT / cfg["path"]
    fmap = {norm(k): v for k, v in cfg["folder_map"].items()}
    bg = []
    skip = [s.lower() for s in cfg.get("exclude_path_contains", [])]
    for img in sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT):
        hit = next((s for s in skip if s in img.as_posix().lower()), None)
        if hit:                       # mis. cabang "Augmented": gambar hasil augmentasi tidak boleh masuk val/test
            excl.add(src, f"path_dikecualikan:{hit}", path=img)
            continue
        tgt = fmap.get(norm(img.parent.name))
        if tgt is None:
            excl.add(src, f"folder_tidak_dipakai:{img.parent.name}", path=img)
        elif tgt == BG:
            bg.append(img)
        else:
            raise SystemExit(f"[{src}] sumber klasifikasi hanya boleh dipetakan ke {BG} (tidak ada box).")
    mx = cfg.get("max_images")
    if mx and len(bg) > mx:
        keep_bg = sorted(rng.sample(bg, mx))
        excl.add(src, "melebihi_max_images", len(bg) - mx)
        for im_ in set(bg) - set(keep_bg):
            excl.add(src, "melebihi_max_images", 0, path=im_)
        bg = keep_bg
    meta[src] = {"background_dipakai": len(bg)}
    return [dict(src=src, path=p, boxes=[], cls_ids=[], tag=src, orig_split="-", max_side=cfg.get("max_side"))
            for p in bg]


# ============================================================ 2. PHASH
def _hash_one(args):
    """
    phash 256-bit untuk 8 varian dihedral (rotasi 0/90/180/270 x flip horizontal).
    Kembalikan array bool (8, hs*hs); indeks 0 = gambar asli. None jika gambar tak terbaca.
    """
    path, hs = args
    import imagehash
    try:
        with Image.open(path) as im:
            im.draft("RGB", (512, 512))
            im = im.convert("RGB")
            out = []
            for k in range(4):
                r = im.rotate(90 * k, expand=True) if k else im
                out.append(imagehash.phash(r, hash_size=hs).hash.flatten())
                out.append(imagehash.phash(r.transpose(Image.FLIP_LEFT_RIGHT), hash_size=hs).hash.flatten())
        return np.array(out)
    except Exception:
        return None


def sig(p: Path):
    st = p.stat()
    return f"{rel(p)}|{st.st_size}|{st.st_mtime_ns}"


HASH_VER = 2   # 2 = 8 varian dihedral


def compute_hashes(cands, hs):
    """Hash 8 varian dihedral semua kandidat (dengan cache). Kembalikan (var[N,8,bits], ok_mask)."""
    cache = {}
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        if int(z["hs"]) == hs and "ver" in z.files and int(z["ver"]) == HASH_VER:
            for s_, v in zip(z["sig"], z["var"]):
                cache[str(s_)] = v
    sigs = [sig(c["path"]) for c in cands]
    todo = [i for i, s_ in enumerate(sigs) if s_ not in cache]
    print(f"  phash (8 varian dihedral): {len(cands)} gambar, {len(todo)} perlu dihitung (sisanya dari cache)", flush=True)
    if todo:
        with ProcessPoolExecutor() as ex:
            res = list(ex.map(_hash_one, [(str(cands[i]["path"]), hs) for i in todo], chunksize=16))
        for i, r in zip(todo, res):
            if r is not None:
                cache[sigs[i]] = r
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        keys = list(cache)
        np.savez_compressed(CACHE, hs=hs, ver=HASH_VER, sig=np.array(keys), var=np.array([cache[k] for k in keys]))
    var = np.zeros((len(cands), 8, hs * hs), bool)
    ok = np.zeros(len(cands), bool)
    for i, s_ in enumerate(sigs):
        if s_ in cache:
            var[i] = cache[s_]
            ok[i] = True
    return var, ok


def near_pairs(A, B, thr, same, chunk=1024):
    """Pasangan (i, j) dengan Hamming(A_i, B_j) <= thr. same=True: buang i==j dan i>j."""
    Af, Bf = A.astype(np.float32), B.astype(np.float32)
    bs = Bf.sum(1)
    pairs = []
    for s in range(0, len(Af), chunk):
        x = Af[s:s + chunk]
        d = x.sum(1)[:, None] + bs[None, :] - 2.0 * (x @ Bf.T)
        ii, jj = np.where(d <= thr + 0.5)
        ii = ii + s
        if same:
            m = ii < jj
            ii, jj = ii[m], jj[m]
        pairs.append(np.stack([ii, jj], 1))
    return np.concatenate(pairs) if pairs else np.zeros((0, 2), int)


def min_dist(A, B, chunk=1024):
    """Jarak Hamming minimum tiap baris A ke seluruh B."""
    Af, Bf = A.astype(np.float32), B.astype(np.float32)
    bs = Bf.sum(1)
    out = np.empty(len(Af))
    for s in range(0, len(Af), chunk):
        x = Af[s:s + chunk]
        out[s:s + chunk] = (x.sum(1)[:, None] + bs[None, :] - 2.0 * (x @ Bf.T)).min(1)
    return out


def phash_pairs(var, thr):
    """Pasangan (i<j) dengan Hamming(varian_v(i), asli(j)) <= thr untuk salah satu dari 8 varian dihedral."""
    orig = var[:, 0]
    P = [near_pairs(orig, orig, thr, True)]
    for v in range(1, var.shape[1]):
        P.append(near_pairs(var[:, v], orig, thr, False))
    P = np.concatenate(P)
    P = P[P[:, 0] != P[:, 1]]
    return np.unique(np.sort(P, axis=1), axis=0) if len(P) else P


def union_groups(n, pair_sets):
    """Union-find atas gabungan beberapa himpunan pasangan. Kembalikan array id-grup."""
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for P in pair_sets:
        for i, j in P:
            ri, rj = find(int(i)), find(int(j))
            if ri != rj:
                parent[ri] = rj
    return np.array([find(i) for i in range(n)])


def basename_pairs(cands):
    """Versi Roboflow (".rf.<hash>") dengan nama dasar sama = satu gambar sumber -> pasangan."""
    by = defaultdict(list)
    for i, c in enumerate(cands):
        st = c.get("orig_path", c["path"]).stem
        if ".rf." in st:
            by[(c["src"], re.sub(r"\.rf\.[0-9a-f]+.*$", "", st))].append(i)
    out = [(v[0], j) for v in by.values() if len(v) > 1 for j in v[1:]]
    return np.array(out, int).reshape(-1, 2)


def min_dist_dihedral(var_a, orig_b):
    """Jarak minimum tiap gambar di A (semua varian dihedral) ke seluruh gambar asli di B."""
    return np.min([min_dist(var_a[:, v], orig_b) for v in range(var_a.shape[1])], axis=0)


# ============================================================ 3. SPLIT
def stratified_group_split(groups, strata, fracs, seed, forced_train=None):
    """
    groups: array id-grup per gambar; strata: list kunci stratum per gambar.
    forced_train: bool per gambar; grup yang memuat satu saja gambar 'wajib train' masuk TRAIN seluruhnya.
    Sisanya (yang boleh ke val/test): greedy, grup besar dulu, tiap grup ke split dengan rasio terisi
    (terhadap target) paling rendah. Target val/test = fraksi x jumlah gambar yang BOLEH ke val/test
    pada stratum itu; sisanya (termasuk yang wajib train) otomatis train.
    """
    rng = random.Random(seed)
    forced_train = forced_train if forced_train is not None else [False] * len(groups)
    members = defaultdict(list)
    for i, g in enumerate(groups):
        members[int(g)].append(i)
    forced_g = {g for g, m in members.items() if any(forced_train[i] for i in m)}
    free = [i for i, g in enumerate(groups) if int(g) not in forced_g]
    total = Counter(strata[i] for i in free)
    target = {s: {sp: total[s] * fracs[sp] for sp in SPLITS} for s in total}
    cnt = {s: {sp: 0 for sp in SPLITS} for s in total}
    order = [g for g in members if g not in forced_g]
    rng.shuffle(order)
    order.sort(key=lambda g: -len(members[g]))
    assign = {g: "train" for g in forced_g}
    for g in order:
        add = Counter(strata[i] for i in members[g])
        best, best_cost = None, None
        for sp in SPLITS:
            cost = sum((cnt[s][sp] + k) / max(target[s][sp], 1e-9) for s, k in add.items())
            cost += rng.random() * 1e-6
            if best_cost is None or cost < best_cost:
                best, best_cost = sp, cost
        assign[g] = best
        for s, k in add.items():
            cnt[s][best] += k
    return [assign[int(g)] for g in groups]


# ============================================================ 4. TULIS DATASET
def resize_before_hash(cands):
    """
    Gambar bersumber max_side diperkecil SEKALI ke data/interim/resized/<sumber>/ lalu seluruh proses
    berikutnya (phash, split, penulisan) memakai file itu. Dengan begitu hash yang dipakai untuk
    pengelompokan sama persis dengan file akhir di dataset (verifikasi konsisten).
    """
    n = 0
    for c in cands:
        if not c["max_side"]:
            continue
        dst = ROOT / "data/interim/resized" / c["src"] / (hashlib.sha1(rel(c["path"]).encode()).hexdigest()[:16] + ".jpg")
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(c["path"]) as im:
                im = im.convert("RGB")
                s = c["max_side"] / max(im.size)
                if s < 1:
                    im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
                im.save(dst, quality=95)
            n += 1
        c["orig_path"], c["path"], c["max_side"] = c["path"], dst, None
    if n:
        print(f"  {n} gambar diperkecil ke data/interim/resized", flush=True)


def safe_name(c, used):
    src_path = c.get("orig_path", c["path"])
    stem = re.sub(r"[^A-Za-z0-9]+", "-", src_path.stem)[:40].strip("-")
    h = hashlib.sha1(rel(src_path).encode()).hexdigest()[:8]
    name = f'{c["tag"].replace("+", "-")}_{stem}_{h}{c["path"].suffix.lower()}'
    assert name not in used, name
    used.add(name)
    return name


def materialize(cands, splits, out, classes, link_mode):
    used = set()
    for sp in [s_ for s_ in SPLITS + (HOLD,) if s_ in set(splits)]:
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sp).mkdir(parents=True, exist_ok=True)
    for c, sp in zip(cands, splits):
        name = safe_name(c, used)
        c["file"], c["split"] = name, sp
        dst = out / "images" / sp / name
        if link_mode == "hardlink":
            try:
                os.link(c["path"], dst)
            except OSError:
                shutil.copy2(c["path"], dst)
        else:
            shutil.copy2(c["path"], dst)
        lab = "\n".join(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cid, cx, cy, w, h, _ in c["boxes"])
        (out / "labels" / sp / (Path(name).stem + ".txt")).write_text(lab + ("\n" if lab else ""))


# ============================================================ 5. LAPORAN
def pct(x):
    return f"{x * 100:.1f}%"


def md_table(header, rows):
    s = "| " + " | ".join(header) + " |\n|" + "|".join("---" for _ in header) + "|\n"
    return s + "\n".join("| " + " | ".join(str(x) for x in r) + " |" for r in rows) + "\n"


def pairs_montage(cands, var, classes, lo, hi, n, seed, out_path):
    """Pilih n pasangan acak dengan jarak phash minimum (8 varian dihedral) pada [lo, hi] dan gambar berdampingan."""
    from PIL import ImageDraw
    P = phash_pairs(var, hi)
    rows = []
    for i, j in P:
        dm, vm = min((int((var[i, v] != var[j, 0]).sum()), v) for v in range(var.shape[1]))
        if lo <= dm <= hi:
            rows.append((int(i), int(j), dm, vm))
    sel = random.Random(seed).sample(rows, min(n, len(rows)))
    T, W = 300, 18
    per_row = 2
    nrows = (len(sel) + per_row - 1) // per_row
    cv = Image.new("RGB", (per_row * 2 * T, nrows * (T + W)), "white")
    dr = ImageDraw.Draw(cv)

    def load(c, v=0):
        with Image.open(c["path"]) as im:
            im = im.convert("RGB")
            r = im.rotate(90 * (v // 2), expand=True) if v // 2 else im
            if v % 2:
                r = r.transpose(Image.FLIP_LEFT_RIGHT)
            return r.resize((T, T))

    for k, (i, j, dm, vm) in enumerate(sel):
        x0, y0 = (k % per_row) * 2 * T, (k // per_row) * (T + W)
        a, b = cands[i], cands[j]
        dr.text((x0 + 3, y0 + 3), f"#{k + 1} d={dm} varian={vm} | {class_label(a, classes)}/{a['tag']} ~ {class_label(b, classes)}/{b['tag']}", fill="black")
        cv.paste(load(a, vm), (x0, y0 + W))
        cv.paste(load(b), (x0 + T, y0 + W))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv.save(out_path, quality=88)
    return {"n_pasangan_di_pita": len(rows), "n_ditampilkan": len(sel), "file": rel(out_path),
            "pasangan": [{"no": k + 1, "jarak": r[2], "varian": r[3], "a": rel(cands[r[0]].get("orig_path", cands[r[0]]["path"])),
                          "b": rel(cands[r[1]].get("orig_path", cands[r[1]]["path"]))} for k, r in enumerate(sel)]}


def sibling_audit(cands, splits, E, var, cfg, k):
    """
    Audit INDEPENDEN gambar val/test yang masih punya saudara di train, dengan sinyal berbeda dari pembentuk grup:
    SIFT pada k tetangga embedding TERDEKAT di train (k lebih besar dari kandidat grup), kunci K4, phash, embedding.
    Piksel test dibaca HANYA untuk deteksi duplikat (izin pengguna); tidak ada metrik model.
    """
    thr = cfg["dup_threshold"]
    cos = cfg["augmentation"]["embedding"]["cosine"]
    paths = [c["path"] for c in cands]
    ix = {sp: [i for i, x in enumerate(splits) if x == sp] for sp in SPLITS + (HOLD,)}
    pairs_k4 = dr.k4_pairs(cands)
    out = {}
    ignore_frac = bool(cfg["grouping"].get("audit_ignore_frac"))
    min_frac = 0.0 if ignore_frac else cfg["grouping"]["sift"].get("min_inlier_bbox_frac", dr.MIN_FRAC)
    qpairs = [("val", "train"), ("test", "train"), ("test", "val")] + ([(HOLD, "train"), (HOLD, "val"), (HOLD, "test")] if ix[HOLD] else [])
    for q_sp, r_sp in qpairs:
        Q, R = ix[q_sp], ix[r_sp]
        sets = {}
        # SIFT pada k tetangga terdekat
        P = dr.knn_pairs_cross(E[Q], E[R], k)
        dr.ensure_descriptors([paths[Q[a]] for a in sorted({int(a) for a, _ in P})] + [paths[R[b]] for b in sorted({int(b) for _, b in P})])
        res = dr.verify_pairs([(paths[Q[a]], paths[R[b]]) for a, b in P])
        sets["SIFT>=%d (k=%d)" % (dr.MIN_INLIERS, k)] = {Q[int(a)] for (a, b), r in zip(P, res) if r[0] >= dr.MIN_INLIERS and r[3] >= min_frac}
        # K4 lintas split
        sq, sr = set(Q), set(R)
        sets["kunci K4"] = {i for i, j in pairs_k4 if i in sq and j in sr} | {j for i, j in pairs_k4 if j in sq and i in sr}
        # phash dan embedding
        d = min_dist_dihedral(var[Q], var[R][:, 0])
        sets[f"phash<={thr}"] = {Q[a] for a in range(len(Q)) if d[a] <= thr}
        Sm = ad.sim_matrix(E[Q], E[R][:, 0])
        sets[f"embedding>={cos}"] = {Q[a] for a in np.where((Sm >= cos).any(1))[0]}
        union = set().union(*sets.values())
        out[f"{q_sp}-{r_sp}"] = {"n_gambar": len(Q), "dengan_saudara_per_sinyal": {kk: len(v) for kk, v in sets.items()},
                                 "dengan_saudara_gabungan": len(union),
                                 "contoh": [rel(cands[i].get("orig_path", cands[i]["path"])) for i in sorted(union)[:5]],
                                 "indeks": sorted(int(i) for i in union)}
    return out


def log_test_changes(cands, cfg, rep_dir):
    """Catat semua perubahan keanggotaan split TEST dibanding dataset pembanding (compare_manifest)."""
    cm = cfg.get("compare_manifest")
    if not cm or not (ROOT / cm).exists():
        return None
    old = {r["orig_path"]: r for r in csv.DictReader(open(ROOT / cm, encoding="utf-8"))}
    new = {rel(c.get("orig_path", c["path"])): c for c in cands}
    rows = []
    for k in sorted(set(old) | set(new)):
        o, n = old.get(k), new.get(k)
        os_, ns = (o["split"] if o else "-"), (n["split"] if n else "dibuang")
        if os_ == "test" or ns == "test":
            kind = ("tetap_test" if os_ == ns else ("keluar_dari_test" if os_ == "test" else "masuk_ke_test"))
            rows.append({"orig_path": k, "kelas": (o or {}).get("class") or class_label(n, cfg["disease_classes"]),
                         "sumber": (o or {}).get("source") or n["tag"], "split_lama": os_, "split_baru": ns, "perubahan": kind})
    rep_dir.mkdir(parents=True, exist_ok=True)
    with open(rep_dir / "test_changes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["orig_path", "kelas", "sumber", "split_lama", "split_baru", "perubahan"])
        w.writeheader()
        w.writerows(rows)
    c = Counter(r["perubahan"] for r in rows)
    dest = Counter(r["split_baru"] for r in rows if r["perubahan"] == "keluar_dari_test")
    src = Counter(r["split_lama"] for r in rows if r["perubahan"] == "masuk_ke_test")
    old_test = sum(1 for r in old.values() if r["split"] == "test")
    new_test = sum(1 for c_ in cands if c_["split"] == "test")
    return {"pembanding": cm, "test_lama": old_test, "test_baru": new_test, "ringkas": dict(c),
            "keluar_dari_test_ke": dict(dest), "masuk_ke_test_dari": dict(src), "berkas": rel(rep_dir / "test_changes.csv")}


def sha256_file(p, block=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def write_dataset_meta(out, cfg, cfg_path, info, cands, classes):
    """Metadata dataset: hash manifest, versi aturan pengelompokan, parameter, versi library, hash skrip/config."""
    import cv2
    ck = Path.home() / ".cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth"
    gh = hashlib.sha256("\n".join(f"{c['file']}\t{c['group']}\t{c['split']}" for c in sorted(cands, key=lambda c: c["file"])).encode()).hexdigest()
    libs = {"python": sys.version.split()[0], "numpy": np.__version__, "opencv": cv2.__version__, "platform": platform.platform()}
    for mod in ("torch", "ultralytics", "PIL", "imagehash", "yaml"):
        try:
            libs[mod] = __import__(mod).__version__
        except Exception:
            pass
    meta_ = {
        "dataset_name": cfg["dataset_name"], "dibuat": datetime.datetime.now().isoformat(timespec="seconds"),
        "manifest_sha256": sha256_file(out / "manifest.csv"), "grup_split_sha256": gh,
        "grouping_rule_version": dr.RULE_VERSION, "grouping_rule": dr.RULE_DESC,
        "parameter": {"seed": cfg["seed"], "split": cfg["split"], "hash_size": cfg["hash_size"], "dup_threshold": cfg["dup_threshold"],
                      "embedding": {"model": "dinov2_vits14 (torch.hub facebookresearch/dinov2)", "cosine": cfg["augmentation"]["embedding"]["cosine"],
                                    "checkpoint_sha256": sha256_file(ck) if ck.exists() else None},
                      "sift": {"nfeatures": dr.NFEAT, "max_px": dr.MAXPX, "ratio": dr.RATIO, "ransac_px": dr.RANSAC_PX,
                               "min_inliers": dr.MIN_INLIERS, "knn": cfg["grouping"]["sift"]["knn"], "model": "similarity (estimateAffinePartial2D)",
                               "orientasi": "asli dan flip horizontal"},
                      "k4": "ds1: (keluarga, jenis, ID, ekstensi) sama dan awalan berbeda",
                      "sift_min_inlier_bbox_frac": cfg["grouping"]["sift"].get("min_inlier_bbox_frac"),
                      "jendela_waktu": cfg["grouping"].get("time_window"), "karantina": cfg["grouping"].get("quarantine"),
                      "max_background_frac_train": cfg.get("max_background_frac_train"), "provisional_sources": cfg.get("provisional_sources", [])},
        "jumlah": {sp: sum(1 for c in cands if c["split"] == sp) for sp in SPLITS + (HOLD,) if any(c["split"] == sp for c in cands)},
        "hash_berkas": {"config": sha256_file(ROOT / cfg_path),
                        **{n: sha256_file(Path(__file__).parent / n) for n in ("prepare_dataset.py", "dedupe_rules.py", "aug_detect.py", "inventory.py")}},
        "versi_library": libs, "sumber_pipeline": info.get("rule_pairs", {}),
        "karantina": {"jumlah_gambar": (info.get("quarantine") or {}).get("n_gambar"), "berkas": "quarantine.csv"},
        "catatan": [],
    }
    dt = info.get("ds4_tanggal")
    if dt:
        if dt["semua_tanggal_val_test_ada_di_train"]:
            meta_["catatan"].append("Seluruh tanggal pemotretan ds4 pada val dan test juga ada di train (sesi pemotretan bersama); "
                                    "split berbasis grup tidak menahan sesi pemotretan, jadi val/test tidak mengukur generalisasi ke sesi/tanggal baru.")
        else:
            meta_["catatan"].append("Sebagian tanggal pemotretan ds4 pada val/test tidak ada di train; lihat ds4_tanggal.")
        meta_["ds4_tanggal"] = dt
    (out / "dataset_meta.json").write_text(json.dumps(meta_, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta_


def remap_pairs(P, new_index):
    """Petakan indeks pasangan ke indeks baru (new_index[i] = -1 jika gambar dibuang); buang pasangan yang kehilangan salah satu ujung."""
    if len(P) == 0:
        return P
    a, b = new_index[P[:, 0]], new_index[P[:, 1]]
    m = (a >= 0) & (b >= 0)
    return np.stack([a[m], b[m]], 1)


def grp_stats(groups, mask):
    """Statistik grup untuk himpunan gambar `mask`: jumlah grup, gambar di grup >1, grup terbesar (ukuran grup global)."""
    size = Counter(groups.tolist())
    idx = np.where(mask)[0]
    gs = [int(groups[i]) for i in idx]
    return {"n_gambar": int(len(idx)), "n_grup": len(set(gs)), "gambar_di_grup>1": int(sum(1 for g in gs if size[g] > 1)),
            "grup_terbesar": int(max((size[g] for g in gs), default=0))}


def select_session_holdout(cands, groups, classes, grp, ts_map, info):
    """
    Sesi pemotretan ds4 = rangkaian foto yang dipisah jeda >= gap_s detik. Pilih SATU sesi berukuran min_frac..max_frac
    dari gambar ds4 terpakai yang memuat semua required_classes; sesi diambil beserta seluruh grup duplikatnya (agar tidak
    bocor) dan tidak boleh memuat gambar wajib-train. Kembalikan mask per gambar (True = test_new_session).
    """
    hc = grp.get("session_holdout") or {}
    mask = [False] * len(cands)
    info["session_holdout"] = None
    if not hc.get("enabled") or not ts_map:
        return mask, set()
    ts_fn = lambda c_: dr.timestamp_of(c_, ts_map)
    items = sorted((t, i) for i, c in enumerate(cands) if c["src"] == "ds4" for t in [ts_fn(c)] if t is not None)
    sess, cur = [], [items[0][1]]
    for k in range(1, len(items)):
        if (items[k][0] - items[k - 1][0]).total_seconds() >= hc.get("gap_s", 600):
            sess.append(cur)
            cur = []
        cur.append(items[k][1])
    sess.append(cur)
    n4 = sum(1 for c in cands if c["src"] == "ds4")
    req = hc.get("required_classes", ["leaf_blast", "brown_spot", "background"])
    gmem = defaultdict(list)
    for i, g in enumerate(groups):
        gmem[int(g)].append(i)
    rows, fulls = [], {}
    for sid, mem in enumerate(sess):
        mset = set(mem)
        full = sorted({j for i in mem for j in gmem[int(groups[i])]})
        fulls[sid] = full
        cls = Counter(class_label(cands[i], classes) for i in full)
        forced = sum(1 for i in full if cands[i]["forced_reason"])
        frac = len(mem) / n4
        rows.append({"sesi": sid, "mulai": str(ts_fn(cands[mem[0]])), "selesai": str(ts_fn(cands[mem[-1]])), "gambar_sesi": len(mem),
                     "persen_ds4": round(100 * frac, 2), "gambar_termasuk_grup": len(full), "gambar_di_luar_sesi": len(full) - len(mset & set(full)),
                     "kelas": dict(cls), "wajib_train": forced,
                     "memenuhi": hc["min_frac"] <= frac <= hc["max_frac"] and all(cls.get(k, 0) > 0 for k in req) and forced == 0})
    ok = [r for r in rows if r["memenuhi"]]
    chosen = None
    drop_out = set()
    if ok:
        mid = 100 * (hc["min_frac"] + hc["max_frac"]) / 2
        chosen = sorted(ok, key=lambda r: (r["gambar_di_luar_sesi"], -min(r["kelas"].get(k, 0) for k in req) / r["gambar_termasuk_grup"],
                                           abs(r["persen_ds4"] - mid)))[0]
        sess_set = set(sess[chosen["sesi"]])
        for i in fulls[chosen["sesi"]]:
            if i in sess_set:
                mask[i] = True
            else:
                drop_out.add(i)
        chosen["gambar_luar_sesi_dibuang"] = len(drop_out)
        chosen["kelas_setelah_buang_luar_sesi"] = dict(Counter(class_label(cands[i], classes) for i in fulls[chosen["sesi"]] if i in sess_set))
    reason = None
    if not chosen:
        inr = [r for r in rows if hc["min_frac"] <= r["gambar_sesi"] / n4 <= hc["max_frac"]]
        reason = (f"tidak ada sesi berukuran {hc['min_frac']:.0%}-{hc['max_frac']:.0%}" if not inr else
                  f"{len(inr)} sesi berukuran sesuai, tetapi tidak ada yang memuat semua kelas {req} tanpa gambar wajib-train")
    info["session_holdout"] = {"n_sesi": len(sess), "gap_s": hc.get("gap_s", 600), "gambar_ds4_terpakai": n4, "syarat": hc,
                               "dipilih": chosen, "alasan_tidak_ada": reason,
                               "kandidat_ukuran_sesuai": [r for r in rows if hc["min_frac"] <= r["gambar_sesi"] / n4 <= hc["max_frac"]],
                               "semua_sesi_ringkas": [{k: r[k] for k in ("sesi", "mulai", "gambar_sesi", "persen_ds4", "memenuhi")} for r in rows]}
    return mask, drop_out


def time_audit(cands, splits, ts_fn, windows, sources):
    """Gambar val/test/test_new_session yang punya foto (sumber sama) di split lain dalam +-W detik."""
    import bisect
    tsv = [ts_fn(c) if c["src"] in sources else None for c in cands]
    by = defaultdict(lambda: defaultdict(list))
    for i, (c, s_) in enumerate(zip(cands, splits)):
        if tsv[i] is not None:
            by[c["src"]][s_].append(tsv[i])
    for d_ in by.values():
        for v_ in d_.values():
            v_.sort()
    qp = [("val", ("train",)), ("test", ("train",)), ("test", ("val",))]
    if HOLD in set(splits):
        qp.append((HOLD, ("train", "val", "test")))
    out = {}
    for W in windows:
        dt = datetime.timedelta(seconds=W)
        for q, refs in qp:
            n_q = n_hit = 0
            for i, (c, s_) in enumerate(zip(cands, splits)):
                if s_ != q or tsv[i] is None:
                    continue
                n_q += 1
                for r_ in refs:
                    arr = by[c["src"]].get(r_, [])
                    k = bisect.bisect_left(arr, tsv[i] - dt)
                    if k < len(arr) and arr[k] <= tsv[i] + dt:
                        n_hit += 1
                        break
            n_tot = sum(1 for x in splits if x == q)
            out[f"+-{W:g}s|{q}-{'+'.join(refs)}"] = {"gambar": n_tot, "bertanggal": n_q, "dengan_foto_dalam_jendela": n_hit,
                                                    "persen": round(100 * n_hit / max(1, n_tot), 3)}
    return out


def class_label(c, classes):
    return "background" if not c["cls_ids"] else "+".join(classes[i] for i in c["cls_ids"])


def build_report(cands, classes, cfg, excl, meta, groups, leak, args_dry, info=None):
    info = info or {}
    rep_dir = ROOT / cfg["report_dir"]
    rep_dir.mkdir(parents=True, exist_ok=True)
    thr = cfg["dup_threshold"]
    prov = set(cfg.get("provisional_sources", []))
    lab = [class_label(c, classes) for c in cands]
    tag = [c["tag"] for c in cands]
    spl = [c["split"] for c in cands]
    SP = [s_ for s_ in SPLITS + (HOLD,) if s_ in set(spl)]

    # --- hitung per (kelas, sumber, split)
    img_cnt, box_cnt = Counter(), Counter()
    for c, l, t, s in zip(cands, lab, tag, spl):
        img_cnt[(l, t, s)] += 1
        box_cnt[(l, t, s)] += len(c["boxes"])
    keys = sorted({(l, t) for l, t, _ in img_cnt}, key=lambda k: (k[0] == "background", k[0], k[1]))

    with open(rep_dir / "split_counts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "source", "provisional", "split", "images", "boxes"])
        for (l, t) in keys:
            for s in SP:
                w.writerow([l, t, t in prov, s, img_cnt[(l, t, s)], box_cnt[(l, t, s)]])

    md = [f"# Laporan split dataset `{cfg['dataset_name']}`\n"]
    md.append(f"- Seed: {cfg['seed']} | rasio split: {cfg['split']} | phash {cfg['hash_size']}x{cfg['hash_size']}, ambang {thr} (8 varian dihedral: rotasi 0/90/180/270 x flip)")
    md.append(f"- Total gambar: **{len(cands)}** | `healthy_mode`: {cfg['healthy_mode']}")
    if prov:
        md.append(f"- **PROVISIONAL**: sumber `{', '.join(sorted(prov))}` belum final. Run diberi awalan `{cfg['run_prefix']}` "
                  f"dan **tidak boleh dipakai untuk tabel paper**.")
    md.append("")

    # 1. gambar per kelas per split (total)
    md.append("## 1. Gambar dan box per kelas per split\n")
    rows = []
    for l in classes + ["background"]:
        r = [l]
        for s in SP:
            ni = sum(v for (a, _, b), v in img_cnt.items() if a == l and b == s)
            nb = sum(v for (a, _, b), v in box_cnt.items() if a == l and b == s)
            r.append(f"{ni} ({nb} box)" if l != "background" else f"{ni}")
        r.append(sum(v for (a, _, _), v in img_cnt.items() if a == l))
        r.append(len({c["group"] for c, x in zip(cands, lab) if x == l}))
        rows.append(r)
    tot = ["**total**"] + [sum(v for (_, _, b), v in img_cnt.items() if b == s) for s in SP] + [len(cands), len({c["group"] for c in cands})]
    rows.append(tot)
    md.append(md_table(["kelas", *SP, "total gambar", "grup unik"], rows))
    nb_all = sum(1 for l in lab if l == "background")
    md.append(f"Porsi background: **{pct(nb_all / len(cands))}** dari total "
              f"(train {pct(img_cnt_sum(img_cnt, 'background', 'train') / max(1, split_n(spl, 'train')))}, "
              f"val {pct(img_cnt_sum(img_cnt, 'background', 'val') / max(1, split_n(spl, 'val')))}, "
              f"test {pct(img_cnt_sum(img_cnt, 'background', 'test') / max(1, split_n(spl, 'test')))}). "
              f"Ultralytics menyarankan 0-10%.\n")

    # 2. per kelas per SUMBER per split
    md.append("## 2. Gambar per kelas per SUMBER per split\n")
    rows = []
    gset = defaultdict(set)
    for c, l, t, s in zip(cands, lab, tag, spl):
        gset[(l, t, s)].add(c["group"])
    for (l, t) in keys:
        n = [img_cnt[(l, t, s)] for s in SP]
        tot_c = sum(img_cnt[(l, tt, s)] for tt in {k[1] for k in keys if k[0] == l} for s in SP)
        share = f"{sum(n) / tot_c * 100:.0f}%"
        cells = [f"{img_cnt[(l, t, s)]} ({len(gset[(l, t, s)])} grup)" for s in SP]
        allg = len(set().union(*[gset[(l, t, s)] for s in SP]))
        rows.append([l, t + (" ⚠ provisional" if t in prov else ""), *cells, f"{sum(n)} ({allg} grup)", share])
    md.append(md_table(["kelas", "sumber", *SP, "total", "porsi dari kelas"], rows))
    single = []
    for l in classes + ["background"]:
        srcs = {k[1] for k in keys if k[0] == l}
        if len(srcs) == 1:
            single.append(f"`{l}` hanya dari `{next(iter(srcs))}`")
    md.append("**Kelas dengan satu sumber saja:** " + (", ".join(single) if single else "tidak ada") + ".\n")
    md.append("Jumlah **box** per kelas per sumber per split ada di `split_counts.csv`.\n")

    # 3. statistik box per kelas (dan per kelas+sumber)
    md.append("## 3. Statistik luas box per kelas (relatif terhadap gambar, semua split)\n")
    areas = defaultdict(list)
    npoly = Counter()
    nimg = Counter()
    for c, l, t in zip(cands, lab, tag):
        if c["boxes"]:
            nimg[(l, t)] += 1
            nimg[(l, "*semua sumber*")] += 1
            for _, _, _, w, h, poly in c["boxes"]:
                for k in ((l, t), (l, "*semua sumber*")):
                    areas[k].append(w * h)
                    npoly[k] += poly
    srows, csvrows = [], []
    for key in sorted(areas, key=lambda k: (classes.index(k[0]), k[1] != "*semua sumber*", k[1])):
        a = np.array(areas[key])
        row = dict(kelas=key[0], sumber=key[1], gambar=nimg[key], box=len(a), box_per_gambar=round(len(a) / nimg[key], 2),
                   median_luas_pct=round(float(np.median(a)) * 100, 3), rata2_luas_pct=round(float(a.mean()) * 100, 3),
                   box_di_atas_1pct=round(float((a > 0.01).mean()) * 100, 1),
                   box_dari_poligon_pct=round(npoly[key] / len(a) * 100, 1))
        csvrows.append(row)
        srows.append([row["kelas"], row["sumber"], row["gambar"], row["box"], row["box_per_gambar"],
                      f'{row["median_luas_pct"]:.2f}%', f'{row["rata2_luas_pct"]:.2f}%', f'{row["box_di_atas_1pct"]:.1f}%',
                      f'{row["box_dari_poligon_pct"]:.1f}%'])
    with open(rep_dir / "class_box_stats.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(csvrows[0]))
        w.writeheader()
        w.writerows(csvrows)
    md.append(md_table(["kelas", "sumber", "gambar", "box", "box/gambar", "median luas", "rata2 luas",
                        "box > 1% luas gambar", "box dari poligon"], srows))
    md.append("\"Box dari poligon\" = bbox pembatas yang diturunkan dari label poligon (segmentasi) sumber; "
              "sisanya box asli.\n")
    med = {r["kelas"]: r["median_luas_pct"] for r in csvrows if r["sumber"] == "*semua sumber*"}
    if med:
        hi, lo = max(med, key=med.get), min(med, key=med.get)
        ratio = med[hi] / med[lo]
        md.append(f"Perbandingan antar kelas: median luas terbesar `{hi}` ({med[hi]:.2f}%) vs terkecil `{lo}` ({med[lo]:.2f}%) "
                  f"= **{ratio:.0f}x**. " + ("**Skala box antar kelas tidak seragam** (unit anotasi berbeda) -> baca metrik per kelas terpisah.\n"
                                           if ratio > 10 else "Skala box antar kelas relatif seragam.\n"))

    # 4. duplikat
    md.append("## 4. Deduplikasi (phash)\n")
    gsize = Counter(groups.tolist())
    sizes = Counter(gsize.values())
    multi = [g for g, n in gsize.items() if n > 1]
    cross_src, cross_cls, cls_examples = 0, 0, []
    by_g = defaultdict(list)
    for i, g in enumerate(groups):
        by_g[int(g)].append(i)
    for g in multi:
        ss = {tag[i].replace("ds1_provisional", "ds1") for i in by_g[g]}
        cc = {lab[i] for i in by_g[g]}
        cross_src += len(ss) > 1
        if len(cc) > 1:
            cross_cls += 1
            cls_examples.append((g, Counter(lab[i] for i in by_g[g]), [Path(cands[i].get("orig_path", cands[i]["path"])).name for i in by_g[g]][:3]))
    md.append(f"- Grup unik: {len(gsize)} dari {len(cands)} gambar; grup berisi >1 gambar: **{len(multi)}** "
              f"(mencakup {sum(n for n in gsize.values() if n > 1)} gambar).")
    md.append(f"- Grup terbesar: {max(gsize.values())} gambar. Distribusi ukuran grup: "
              f"{dict(sorted(sizes.items())[:8])}{' ...' if len(sizes) > 8 else ''}")
    md.append(f"- Grup lintas SUMBER: **{cross_src}**; grup lintas KELAS pada dataset akhir: **{cross_cls}** "
              f"(kemiripan adegan embedding yang memuat >1 kelas sebelum penyaringan: {info.get('grup_adegan_lintas_kelas', '?')}; "
              "tetap dijaga dalam satu split, tidak dibuang).")
    per_src_dup = Counter()
    for g in multi:
        for i in by_g[g]:
            per_src_dup[tag[i]] += 1
    md.append(f"- Gambar yang berada di grup duplikat per sumber: {dict(per_src_dup)}")
    for g_, cc_, nm_ in cls_examples[:5]:
        md.append(f"- Contoh grup lintas kelas: {sum(cc_.values())} gambar {dict(cc_)}; mis. {', '.join(nm_)}")
    for name_, n_ in info.get("rule_pairs", {}).items():
        md.append(f"- Pasangan yang menyatukan grup: {name_}: **{n_}**")
    md.append(f"- **Verifikasi leakage** (jarak phash minimum antar split, 8 varian dihedral; ambang {thr}): "
              + ", ".join(f"{k}: {v['pairs_leq_thr']} pasangan <= {thr} (jarak min {v['min']:.0f})" for k, v in leak.items())
              + ". " + ("**LOLOS (0 pasangan).**" if all(v["pairs_leq_thr"] == 0 for v in leak.values()) else "**GAGAL - ada kebocoran!**"))
    if any("emb_max_cos" in v for v in leak.values()):
        md.append("- **Verifikasi embedding** (cosine maks antar split, 8 varian dihedral): "
                  + ", ".join(f"{k}: maks {v['emb_max_cos']} ({v['emb_pairs_ge_thr']} pasangan >= ambang)" for k, v in leak.items() if "emb_max_cos" in v)
                  + f". Versi Roboflow dengan nama dasar sama yang terpecah antar split: **{info.get('rf_basename_terpecah', '?')}**.\n")

    # 5. test multi-penyakit
    md.append("## 5. Test multi-penyakit (gambar dengan >= 2 kelas)\n")
    multi_test = [c for c in cands if c["split"] == "test" and len(c["cls_ids"]) >= 2]
    n_multi_all = sum(1 for c in cands if len(c["cls_ids"]) >= 2)
    md.append(f"- Gambar >= 2 kelas di seluruh dataset: **{n_multi_all}**; di split test: **{len(multi_test)}**. "
              f"Daftar: `{cfg['output_dir']}/lists/test_multidisease.txt`.")
    if n_multi_all == 0:
        md.append("- Tidak ada satu pun gambar multi-penyakit pada sumber yang dipakai (semua label satu kelas per gambar). "
                  "Syarat test multi-penyakit pada CLAUDE.md **belum bisa dipenuhi**; perlu foto yang dianotasi manual.")
    md.append("")

    # 6. dibuang
    md.append("## 6. Gambar/box yang dibuang\n")
    rows = [[s, r, n] for (s, r), n in sorted(excl.c.items())]
    md.append(md_table(["sumber", "alasan", "jumlah"], rows))
    with open(rep_dir / "exclusions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "reason", "count"])
        w.writerows(rows)
    md.append("Catatan per sumber:\n```json\n" + json.dumps(meta, indent=2, ensure_ascii=False) + "\n```\n")

    # 7. deviasi stratifikasi
    md.append("## 7. Kesesuaian rasio split per stratum (kelas|sumber)\n")
    rows = []
    for (l, t) in keys:
        n = [img_cnt[(l, t, s)] for s in SP]
        T = sum(n)
        rows.append([f"{l}|{t}", T, *[f"{x} ({x / T * 100:.0f}%)" for x in n]])
    md.append(md_table(["stratum", "total", *SP], rows))
    md.append("Target 70/20/10. Stratum kecil atau berisi grup duplikat besar bisa menyimpang karena grup tidak boleh dipecah.\n")

    # 8. deteksi augmentasi dan aturan val/test
    aug = cfg.get("augmentation", {}) or {}
    if aug.get("enabled"):
        md.append("## 8. Deteksi augmentasi dan kelayakan val/test\n")
        md.append(f"Detektor: (a) pola nama (sumber {aug.get('name_sources')}), (b) heuristik piksel {aug.get('pixel')} "
                  f"(sumber {aug.get('pixel_sources')}), (c) embedding DINOv2-small cosine >= {aug['embedding']['cosine']} "
                  f"(flag 'punya saudara' hanya sumber {aug.get('embedding_sources')}; juga menggabungkan grup). "
                  f"Kelas exempt (tetap boleh ke val/test, hanya diberi flag `aug_possible`): {aug.get('exempt_classes')}. "
                  "Kalibrasi dan kontrol positif: `reports/aug_detect/`.\n")
        rows = []
        for key in sorted({(class_label(c, classes), c["tag"]) for c in cands}, key=lambda k: (k[0] == "background", k[0], k[1])):
            cs = [c for c in cands if (class_label(c, classes), c["tag"]) == key]

            def cnt(prefix):
                return sum(any(r.startswith(prefix) for r in c["aug_reasons"]) for c in cs)
            rows.append([key[0], key[1], len(cs), cnt("nama"), cnt("piksel"), cnt("embedding"),
                         sum(c["aug_possible"] for c in cs), sum(bool(c["forced_reason"]) for c in cs),
                         sum(c["split"] != "train" and c["aug_possible"] for c in cs)])
        md.append(md_table(["kelas", "sumber", "gambar", "flag nama", "flag piksel", "flag embedding", "aug_possible (salah satu)",
                            "wajib train (train-only atau aug)", "aug_possible di val/test"], rows))
        reasons = Counter()
        for c in cands:
            if c["forced_reason"]:
                reasons[c["forced_reason"].split(":")[0] if c["forced_reason"] != "train_only" else "train_only"] += 1
        md.append(f"Gambar wajib-train menurut sebab: {dict(reasons)}. "
                  "Gambar kelas exempt yang `aug_possible` tetap boleh ke val/test; metrik pada subset itu sebaiknya "
                  "dilaporkan terpisah (kolom `aug_possible` di manifest).\n")

    # 9. background di train
    if info.get("bg"):
        b = info["bg"]
        md.append("## 9. Pemangkasan background (hanya train)\n")
        md.append(f"- Batas {pct(b['max_frac'])} dari gambar train. Gambar train berpenyakit: {b['train_penyakit']} -> batas background {b['cap']}.")
        md.append(f"- Background train: {b['bg_train_sebelum']} -> **{b['bg_train_sesudah']}** (dipangkas {b['dipangkas']} dari sumber {b['sumber_prunable']}, "
                  f"acak dengan seed {cfg['seed']}); dipertahankan per sumber: {b['dipertahankan_per_sumber']}.")
        md.append(f"- Batas tercapai: **{b['cap_tercapai']}**. Val dan test tidak diubah (porsi background val/test lebih besar daripada train; "
                  "metrik background/false positive di val/test tidak sebanding langsung dengan distribusi train).\n")

    # 10. pasangan jarak 12..ambang
    if info.get("montage"):
        m = info["montage"]
        md.append("## 10. Pasangan acak pada jarak phash 12-14 (pemeriksaan visual)\n")
        md.append(f"- Jumlah pasangan di pita jarak 12-{cfg['dup_threshold']} (dalam dataset akhir): **{m['n_pasangan_di_pita']}**; "
                  f"ditampilkan {m['n_ditampilkan']} acak (seed {cfg['seed']}): `{m['file']}`. "
                  "Gambar kiri sudah diputar/di-flip ke varian yang cocok dengan gambar kanan.")
        md.append(md_table(["#", "jarak", "varian", "gambar A", "gambar B"],
                           [[x["no"], x["jarak"], x["varian"], x["a"], x["b"]] for x in m["pasangan"]]))

    # 11. SIFT
    if info.get("sift"):
        sf = info["sift"]
        md.append("## 11. Verifikasi SIFT pada kandidat kNN embedding (aturan v2)\n")
        md.append(f"- Kandidat unik (k={sf['k']} tetangga terdekat per gambar, tanpa ambang cosine): **{sf['kandidat_unik']}**; "
                  f"sudah satu grup sebelum SIFT: {sf['sudah_satu_grup']}; diverifikasi: {sf['diverifikasi']}; "
                  f"**terverifikasi >= {sf['min_inliers']} inlier: {sf['terverifikasi']}**.")
        md.append(f"- Waktu: deskriptor {sf['waktu_deskriptor_s']} s, verifikasi {sf['waktu_verifikasi_s']} s.")
        md.append(f"- Aturan sebaran: pasangan dengan inlier >= {sf['min_inliers']} tetapi kotak pembatas inlier < {sf['min_inlier_bbox_frac']:.0%} luas gambar "
                  f"DITOLAK (dugaan kecocokan watermark/teks kecil): **{sf['ditolak_sebaran_kecil']}** pasangan. Daftar dengan status di `sift_pairs.csv`.")
        md.append("- Histogram inlier pada pasangan yang diverifikasi: " + ", ".join(f"{k_}: {v_}" for k_, v_ in sf["histogram_inlier"].items()))
        md.append(f"- Pasangan terverifikasi lintas KELAS: **{sf['lintas_kelas']}**; lintas SUMBER: **{sf['lintas_sumber']}**. "
                  "Pasangan lintas kelas tetap digabung dalam satu split tetapi TIDAK dibuang (belum dinilai manusia). Pasangan lengkap: `sift_pairs.csv`.")
        for ex_ in sf["lintas_kelas_contoh"]:
            md.append(f"  - contoh lintas kelas: {ex_[1]} `{Path(ex_[0]).name}` ~ {ex_[3]} `{Path(ex_[2]).name}`")
        md.append("")
    # 12. audit saudara
    if info.get("audit"):
        md.append("## 12. Audit independen: gambar val/test yang masih punya saudara di train\n")
        md.append("Target 0 menurut aturan baru. Sinyal audit berbeda dari pembentuk grup: SIFT pada **k lebih besar** tetangga embedding di train, "
                  "kunci K4, phash, embedding. Piksel test dibaca hanya untuk deteksi duplikat.\n")
        rows = []
        for k_, v_ in info["audit"].items():
            rows.append([k_, v_["n_gambar"], *[f"{kk}: {vv}" for kk, vv in v_["dengan_saudara_per_sinyal"].items()], f"**{v_['dengan_saudara_gabungan']}**"])
        hdr = ["pasangan split", "gambar query"] + list(next(iter(info["audit"].values()))["dengan_saudara_per_sinyal"]) + ["gabungan"]
        md.append(md_table(hdr, [[r_[0], r_[1], *[x.split(": ")[1] for x in r_[2:-1]], r_[-1]] for r_ in rows]))
        for k_, v_ in info["audit"].items():
            if v_["dengan_saudara_gabungan"]:
                md.append(f"- {k_}: contoh {v_['contoh']}")
        ak = info.get("audit_kriteria")
        if ak:
            md.append(f"- **Kriteria penerimaan** (val/test yang masih punya saudara di train, SIFT k={ak['k']} + K4 + phash + embedding): "
                      f"{ak['n_dengan_saudara_di_train']} dari {ak['n_val_test']} gambar val+test = **{ak['persen']}%**; ambang < {ak['ambang_persen']}% -> "
                      f"**{'LOLOS' if ak.get('lolos_sift', ak['lolos']) else 'TIDAK LOLOS'}** (aturan sebaran {'TIDAK dipakai' if ak.get('tanpa_aturan_sebaran') else 'dipakai'}). "
                      f"(test~val tambahan: {ak['test_val_tambahan']} gambar test punya saudara di val; test_new_session: {ak.get('test_new_session_dengan_saudara')}.)")
            if "lolos_waktu" in ak:
                md.append(f"- **Kriteria waktu**: pelanggaran jendela terpilih (harus 0): **{ak['waktu_jendela_terpilih_pelanggaran']}** -> "
                          f"**{'LOLOS' if ak['lolos_waktu'] else 'TIDAK LOLOS'}**. Putusan gabungan: **{'LOLOS' if ak['lolos'] else 'TIDAK LOLOS'}**.")
        md.append("")
    # 13. perubahan test
    if info.get("test_changes"):
        tc = info["test_changes"]
        md.append("## 13. Perubahan keanggotaan split TEST dibanding `prov`\n")
        md.append(f"- Test lama (`{tc['pembanding']}`): **{tc['test_lama']}**; test baru: **{tc['test_baru']}**.")
        md.append(f"- Ringkas: {tc['ringkas']}. Gambar yang keluar dari test pindah ke: {tc['keluar_dari_test_ke']}; "
                  f"gambar yang masuk test berasal dari: {tc['masuk_ke_test_dari']}.")
        md.append(f"- Daftar lengkap: `{tc['berkas']}`.\n")
    if info.get("freeze"):
        fz = info["freeze"]
        md.append("## Status pembekuan\n")
        md.append(f"- **frozen: {fz['frozen']}**" + (f" pada {fz['frozen_at']}" if fz["frozen"] else f" ({fz['alasan_tidak_dibekukan']})") +
                  f". `manifest_sha256` `{fz['manifest_sha256']}`; `data_yaml_sha256` `{fz['data_yaml_sha256']}`; `labels_sha256` `{fz['labels_sha256']}`; "
                  f"`daftar_gambar_sha256` `{fz['daftar_gambar_sha256']}`.\n")
    if info.get("meta"):
        md.append("## 14. Metadata dataset\n")
        md.append(f"- Versi aturan pengelompokan: `{dr.RULE_VERSION}`. {dr.RULE_DESC}")
        md.append(f"- `manifest_sha256`: `{info['meta']['manifest_sha256']}`; `grup_split_sha256`: `{info['meta']['grup_split_sha256']}`; "
                  f"`config_sha256`: `{info['meta']['config_sha256']}` (rinci di `dataset_meta.json`).\n")

    # 15. karantina
    if info.get("quarantine") is not None:
        q = info["quarantine"]
        md.append("## 15. Karantina (pasangan lintas kelas dengan inlier > %d)\n" % q["ambang_inlier"])
        md.append(f"- **{q['n_gambar']} gambar** dikeluarkan dari train/val/test pada build ini (berkas asli dan label TIDAK diubah/dihapus); "
                  f"dari {q['n_pasangan']} pasangan baru. Daftar: `quarantine.csv`.")
        if "n_diwarisi" in q:
            md.append(f"- Asal: diwarisi dari versi sebelumnya {q['n_diwarisi']} (tidak ditemukan lagi: {q['n_diwarisi_tidak_ditemukan']}); "
                      f"tambahan baru oleh aturan pada versi ini: **{q['n_baru_oleh_aturan']}**.")
        md.append(md_table(["kelas | sumber", "gambar dikarantina"], [[k_, v_] for k_, v_ in sorted(q["per_kelas_sumber"].items())]))
        md.append("Per pasangan kelas: " + ", ".join(f"{k_}: {v_}" for k_, v_ in q["per_pasangan_kelas"].items()) + "\n")
    # 16. jendela waktu
    if info.get("time_window"):
        tw_ = info["time_window"]
        md.append("## 16. Aturan jendela waktu\n")
        md.append(f"- Selisih waktu pemotretan <= {tw_['gap_s']} detik (jendela geser); gambar bertanggal per sumber (sebelum foto karantina/buangan dikeluarkan): "
                  f"{tw_['gambar_bertanggal']}; pasangan dalam jendela: {tw_['pasangan']}. Kelompok dihitung pada himpunan lengkap.")
        dm = tw_["dampak"]
        md.append(md_table(["", f"gambar sumber berjendela waktu ({'+'.join(tw_.get('sumber', ['ds4']))})", "grup unik", "gambar di grup > 1", "grup terbesar"],
                           [[k_.replace("_", " "), v_["n_gambar"], v_["n_grup"], v_["gambar_di_grup>1"], v_["grup_terbesar"]] for k_, v_ in dm.items()]))
        md.append("Rasio split akhir per stratum ada di bagian 7.\n")
    if info.get("ds4_tanggal"):
        d_ = info["ds4_tanggal"]
        md.append(f"- ds4 bertanggal per split: {d_['gambar_bertanggal']}; tanggal unik: {d_['tanggal_unik']}; "
                  f"seluruh tanggal val/test juga ada di train: **{d_['semua_tanggal_val_test_ada_di_train']}** (dicatat di `dataset_meta.json`).\n")

    # 17. uji sesi baru
    sh = info.get("session_holdout")
    if sh:
        md.append("## 17. Uji sesi baru (`test_new_session`)\n")
        md.append(f"- Sesi ds4 (jeda >= {sh['gap_s']} detik): **{sh['n_sesi']}** sesi dari {sh['gambar_ds4_terpakai']} gambar ds4 terpakai. "
                  f"Syarat: ukuran {sh['syarat']['min_frac']:.0%}-{sh['syarat']['max_frac']:.0%}, memuat {sh['syarat'].get('required_classes')}, tanpa gambar wajib-train.")
        if sh["dipilih"]:
            d_ = sh["dipilih"]
            md.append(f"- **Dipilih sesi {d_['sesi']}** ({d_['mulai']} s/d {d_['selesai']}): {d_['gambar_sesi']} gambar sesi ({d_['persen_ds4']}% ds4); "
                      f"bersama grup duplikatnya {d_['gambar_termasuk_grup']} gambar (di luar sesi: {d_['gambar_di_luar_sesi']}). "
                      f"Anggota dari luar sesi DIBUANG dari dataset: {d_.get('gambar_luar_sesi_dibuang', 0)} (alasan: kelompok duplikat lintas sesi). "
                      f"Isi akhir per kelas: {d_.get('kelas_setelah_buang_luar_sesi', d_['kelas'])}.")
        else:
            md.append(f"- **Tidak ada sesi yang memenuhi**: {sh['alasan_tidak_ada']}. Split `test_new_session` tidak dibuat.")
        if sh["kandidat_ukuran_sesuai"]:
            md.append(md_table(["sesi", "mulai", "gambar sesi", "% ds4", "termasuk grup", "di luar sesi", "kelas", "wajib train", "memenuhi"],
                               [[r_["sesi"], r_["mulai"], r_["gambar_sesi"], r_["persen_ds4"], r_["gambar_termasuk_grup"], r_["gambar_di_luar_sesi"],
                                 r_["kelas"], r_["wajib_train"], r_["memenuhi"]] for r_ in sh["kandidat_ukuran_sesuai"]]))
    # 18. audit waktu
    if info.get("audit_waktu"):
        md.append("## 18. Audit waktu (foto sumber yang sama di split lain dalam +-W detik)\n")
        md.append(md_table(["jendela | split query - split acuan", "gambar", "bertanggal", "punya foto dalam jendela", "%"],
                           [[k_, v_["gambar"], v_["bertanggal"], v_["dengan_foto_dalam_jendela"], v_["persen"]] for k_, v_ in info["audit_waktu"].items()]))
        md.append("Jendela terpilih harus 0 (dijamin aturan rantai waktu); jendela lebih lebar hanya dilaporkan.\n")

    text = "\n".join(md)
    (rep_dir / "split_report.md").write_text(text, encoding="utf-8")
    return text


def img_cnt_sum(cnt, label, split):
    return sum(v for (a, _, b), v in cnt.items() if a == label and b == split)


def split_n(spl, s):
    return sum(1 for x in spl if x == s)


# ============================================================ MAIN
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/classes.yaml")
    ap.add_argument("--force", action="store_true", help="bangun ulang output_dir")
    ap.add_argument("--dry-run", action="store_true", help="tanpa menulis gambar/label")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    classes = cfg["disease_classes"]
    if cfg.get("healthy_mode") != "background":
        raise SystemExit("Hanya healthy_mode: background yang didukung (keputusan proyek).")
    out = ROOT / cfg["output_dir"]
    if not args.dry_run and out.exists():
        if not args.force:
            raise SystemExit(f"{out} sudah ada. Pakai --force untuk membangun ulang.")
        if "processed" not in out.parts:
            raise SystemExit("Penghapusan hanya diizinkan di dalam folder 'processed'.")
        shutil.rmtree(out)

    rng = random.Random(cfg["seed"])
    excl, meta = Excl(), {}
    cands = []
    print("[1/5] mengumpulkan kandidat ...", flush=True)
    for s in cfg["sources"]:
        if s.get("role", "pool") != "pool":
            meta[s["name"]] = {"tidak_diproses": f"role={s['role']}"}
            continue
        if s["type"] == "yolo":
            got = collect_yolo(s, classes, excl, meta)
        elif s["type"] == "yolo_folders":
            got = collect_yolo_folders(s, classes, excl, meta)
        elif s["type"] == "classification":
            got = collect_classification(s, classes, excl, meta, rng)
        else:
            raise SystemExit(f"type sumber tidak dikenal: {s['type']}")
        print(f"  {s['name']}: {len(got)} kandidat", flush=True)
        cands += got

    print("[2/5] phash ...", flush=True)
    resize_before_hash(cands)
    var, ok = compute_hashes(cands, cfg["hash_size"])
    if (~ok).any():
        for i in np.where(~ok)[0]:
            excl.add(cands[i]["src"], "gambar_tidak_terbaca", path=cands[i].get("orig_path", cands[i]["path"]))
        cands = [c for c, k in zip(cands, ok) if k]
        var = var[ok]

    aug = cfg.get("augmentation", {}) or {}
    E = None
    info = {"rule_pairs": {}}
    pair_sets = []
    P = phash_pairs(var, cfg["dup_threshold"])
    pair_sets.append(P)
    strict_sets = [P]          # duplikat sungguhan (phash, nama dasar Roboflow): dipakai untuk deteksi label bertentangan
    info["rule_pairs"][f"phash <= {cfg['dup_threshold']} (8 varian dihedral)"] = len(P)
    for c in cands:
        c["aug_reasons"] = []
    if aug.get("enabled"):
        print("  deteksi augmentasi (nama, piksel, embedding) ...", flush=True)
        # (a) pola nama file
        ni = [i for i, c in enumerate(cands) if c["src"] in aug.get("name_sources", [])]
        for i, r in zip(ni, ad.name_flags([cands[i].get("orig_path", cands[i]["path"]) for i in ni])):
            if r:
                cands[i]["aug_reasons"].append("nama:" + r)
        # (b) heuristik piksel
        pi = [i for i, c in enumerate(cands) if c["src"] in aug.get("pixel_sources", [])]
        for i, m in zip(pi, ad.pixel_metrics([cands[i]["path"] for i in pi])):
            r = ad.pixel_flag(m, aug["pixel"])
            if r:
                cands[i]["aug_reasons"].append("piksel:" + r)
        # (c) embedding DINOv2: flag "punya saudara" (sumber tertentu) + penggabungan grup
        E = ad.embeddings([c["path"] for c in cands])
        cos = aug["embedding"]["cosine"]
        Pe = ad.sim_pairs(E, cos)
        info["rule_pairs"][f"embedding cosine >= {cos} (DINOv2-small, 8 varian dihedral)"] = len(Pe)
        for i, j in Pe:
            i, j = int(i), int(j)
            if cands[i]["src"] == cands[j]["src"] and cands[i]["src"] in aug.get("embedding_sources", []):
                for k in (i, j):
                    if "embedding:saudara" not in cands[k]["aug_reasons"]:
                        cands[k]["aug_reasons"].append("embedding:saudara")
        if aug["embedding"].get("merge_groups", True):
            pair_sets.append(Pe)
        if aug.get("roboflow_basename_groups", True):
            Pb = basename_pairs(cands)
            pair_sets.append(Pb)
            strict_sets.append(Pb)
            info["rule_pairs"]["nama dasar Roboflow sama (.rf.<hash>)"] = len(Pb)
    # ---- aturan pengelompokan v2: kunci ID K4 + pasangan terverifikasi SIFT (kNN embedding, semua kelas/sumber)
    grp = cfg.get("grouping", {}) or {}
    info["sift"] = None
    if grp.get("k4"):
        Pk = dr.k4_pairs(cands)
        pair_sets.append(Pk)
        strict_sets.append(Pk)
        info["rule_pairs"]["kunci ID K4 (fam,jenis,id,ekstensi; lintas awalan; ds1)"] = len(Pk)
    # ---- aturan waktu (ds4): foto yang dipotret berselisih <= gap_s detik digabung (tanggal-jam dari nama berkas asli)
    tw = grp.get("time_window", {}) or {}
    info["time_window"] = None
    Pt = np.zeros((0, 2), int)
    ts_map = None
    if tw.get("enabled"):
        ts_map = dr.ds4_timestamps(ROOT / tw["metadata"])
        Pt, n_ts = dr.time_pairs(cands, lambda c_: dr.timestamp_of(c_, ts_map), tw["gap_s"], tuple(tw.get("sources", ["ds4"])))
        info["time_window"] = {"gap_s": tw["gap_s"], "sumber": tw.get("sources", ["ds4"]), "gambar_bertanggal": n_ts, "pasangan": int(len(Pt))}
        info["rule_pairs"][f"jendela waktu <= {tw['gap_s']} s (ds4; tanggal-jam dari nama berkas asli)"] = len(Pt)
    sc = grp.get("sift", {}) or {}
    qc = grp.get("quarantine", {}) or {}
    info["quarantine"] = None
    quarantined = {}
    if sc.get("enabled"):
        if E is None:
            raise SystemExit("SIFT memakai kandidat embedding: set augmentation.enabled: true")
        t0 = time.time()
        paths_all = [c["path"] for c in cands]
        lab_ = [class_label(c, classes) for c in cands]
        labarr = np.array(lab_)
        gcur = union_groups(len(cands), pair_sets + [Pt])
        Pc = dr.knn_pairs(E, sc["knn"])
        # diverifikasi: pasangan yang belum satu grup, DAN pasangan lintas kelas (untuk karantina)
        todo = Pc[(gcur[Pc[:, 0]] != gcur[Pc[:, 1]]) | (labarr[Pc[:, 0]] != labarr[Pc[:, 1]])]
        n_desc = dr.ensure_descriptors(paths_all)
        t1 = time.time()
        print(f"  SIFT: {len(Pc)} kandidat kNN (k={sc['knn']}), {len(Pc) - len(todo)} dilewati (sudah satu grup & sekelas), {len(todo)} diverifikasi "
              f"(deskriptor baru {n_desc}, {t1 - t0:.0f} s)", flush=True)
        res = dr.verify_pairs([(paths_all[i], paths_all[j]) for i, j in todo])
        inl = np.array([r_[0] for r_ in res])
        frac = np.array([r_[3] for r_ in res])
        min_frac = sc.get("min_inlier_bbox_frac", dr.MIN_FRAC)
        hit = inl >= sc["min_inliers"]
        acc, rej = hit & (frac >= min_frac), hit & (frac < min_frac)
        Pv = todo[acc]
        pair_sets.append(Pv)
        t2 = time.time()
        bins = [(0, 0), (1, 5), (6, 10), (11, 20), (21, 40), (41, 100), (101, 10 ** 9)]
        cross_cls = [(i, j) for i, j in Pv if lab_[i] != lab_[j]]
        cross_src = [(i, j) for i, j in Pv if cands[i]["src"] != cands[j]["src"]]
        info["sift"] = {"k": sc["knn"], "min_inliers": sc["min_inliers"], "min_inlier_bbox_frac": min_frac, "kandidat_unik": int(len(Pc)),
                        "sudah_satu_grup": int(len(Pc) - len(todo)), "diverifikasi": int(len(todo)), "terverifikasi": int(len(Pv)),
                        "ditolak_sebaran_kecil": int(rej.sum()),
                        "histogram_inlier": {f"{a_}-{b_ if b_ < 10 ** 9 else '+'}": int(((inl >= a_) & (inl <= b_)).sum()) for a_, b_ in bins},
                        "lintas_kelas": len(cross_cls), "lintas_sumber": len(cross_src),
                        "lintas_kelas_contoh": [[rel(cands[i].get("orig_path", cands[i]["path"])), lab_[i], rel(cands[j].get("orig_path", cands[j]["path"])), lab_[j]]
                                                 for i, j in cross_cls[:5]],
                        "waktu_deskriptor_s": round(t1 - t0, 1), "waktu_verifikasi_s": round(t2 - t1, 1)}
        rd = ROOT / cfg["report_dir"]
        rd.mkdir(parents=True, exist_ok=True)
        with open(rd / "sift_pairs.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["a", "b", "kelas_a", "kelas_b", "sumber_a", "sumber_b", "inlier", "skala", "rotasi_derajat", "frac_kotak_inlier", "status"])
            for (i, j), r_, a_, rj_ in zip(todo, res, acc, rej):
                if r_[0] >= 8:
                    w.writerow([rel(cands[i].get("orig_path", cands[i]["path"])), rel(cands[j].get("orig_path", cands[j]["path"])),
                                lab_[i], lab_[j], cands[i]["src"], cands[j]["src"], r_[0], r_[1], r_[2], r_[3],
                                "diterima" if a_ else ("ditolak_sebaran_kecil" if rj_ else "inlier<21")])
        info["rule_pairs"][f"SIFT >= {sc['min_inliers']} inlier dan sebaran >= {min_frac:.0%} pada {sc['knn']} tetangga embedding"] = len(Pv)
        # karantina: pasangan lintas kelas dengan inlier > ambang (hanya yang diterima aturan sebaran) + warisan dari versi sebelumnya
        if qc:
            qthr = qc["cross_class_min_inliers"]
            qp = [(i, j, r_[0]) for (i, j), r_, a_ in zip(todo, res, acc) if a_ and r_[0] > qthr and lab_[i] != lab_[j]]
            new_hits = set()
            for i, j, n_ in qp:
                quarantined.setdefault(int(i), []).append((int(j), int(n_)))
                quarantined.setdefault(int(j), []).append((int(i), int(n_)))
                new_hits |= {int(i), int(j)}
            carry, n_prev = {}, 0
            if qc.get("carry_over_csv"):
                path_idx = {rel(c.get("orig_path", c["path"])): k_ for k_, c in enumerate(cands)}
                prev = list(csv.DictReader(open(ROOT / qc["carry_over_csv"], encoding="utf-8")))
                n_prev = len(prev)
                for r_ in prev:
                    k_ = path_idx.get(r_["berkas"])
                    if k_ is not None:
                        carry[k_] = r_
                        if k_ not in quarantined:
                            quarantined[k_] = [(path_idx.get(r_["pasangan_terbaik"], k_), int(r_["inlier_terbaik"]))]
            rows_q = []
            for i, lst in sorted(quarantined.items()):
                j, n_ = max(lst, key=lambda x: x[1])
                asal = ("diwarisi_dan_aturan_baru" if (i in carry and i in new_hits) else "diwarisi" if i in carry else "aturan_baru")
                rows_q.append({"berkas": rel(cands[i].get("orig_path", cands[i]["path"])), "kelas": lab_[i], "sumber": cands[i]["tag"],
                               "n_pasangan_lintas_kelas": len(lst), "pasangan_terbaik": rel(cands[j].get("orig_path", cands[j]["path"])),
                               "kelas_pasangan": lab_[j], "sumber_pasangan": cands[j]["tag"], "inlier_terbaik": n_, "asal": asal,
                               "alasan": f"pasangan lintas kelas dengan inlier > {qthr} (SIFT)"})
            with open(rd / "quarantine.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows_q[0]) if rows_q else ["berkas"])
                w.writeheader()
                w.writerows(rows_q)
            info["quarantine"] = {"ambang_inlier": qthr, "n_pasangan": len(qp), "n_gambar": len(quarantined),
                                  "n_diwarisi": len(carry), "n_diwarisi_tidak_ditemukan": n_prev - len(carry),
                                  "n_baru_oleh_aturan": len(new_hits - set(carry)),
                                  "per_kelas_sumber": dict(Counter(f"{r_['kelas']}|{r_['sumber']}" for r_ in rows_q)),
                                  "per_pasangan_kelas": dict(Counter("~".join(sorted((r_["kelas"], r_["kelas_pasangan"]))) for r_ in rows_q))}
            for i in quarantined:
                excl.add(cands[i]["src"], f"karantina_lintas_kelas_inlier>{qthr}", path=cands[i].get("orig_path", cands[i]["path"]))
    # ---- v5/v6: SEMUA kelompok dibentuk pada himpunan LENGKAP (termasuk foto karantina dan duplikat berlabel bertentangan).
    #      Baru setelah kelompok terbentuk, foto karantina/buangan dikeluarkan; id kelompok anggota lain DIPERTAHANKAN,
    #      sehingga foto yang tersambung lewat foto karantina tetap satu kelompok.
    print("[3/5] mengelompokkan duplikat (himpunan lengkap) ...", flush=True)
    groups_nt = union_groups(len(cands), pair_sets)
    groups = union_groups(len(cands), pair_sets + [Pt]) if len(Pt) else groups_nt     # untuk split
    gstrict = union_groups(len(cands), strict_sets)       # hanya duplikat sungguhan

    # Duplikat SUNGGUHAN dengan label BERTENTANGAN tidak bisa dipercaya -> dibuang (setelah pengelompokan).
    labs0 = [class_label(c, classes) for c in cands]
    gl = defaultdict(set)
    for g, l in zip(gstrict, labs0):
        gl[int(g)].add(l)
    bad_g = {g for g, s_ in gl.items() if len(s_) > 1}
    conflict_idx = {i for i in range(len(cands)) if int(gstrict[i]) in bad_g}
    if bad_g:
        meta["grup_label_bertentangan_dibuang"] = [[f"{labs0[i]} <- {rel(cands[i]['path'])}" for i in range(len(cands)) if int(gstrict[i]) == g]
                                                   for g in sorted(bad_g)]
        for i in sorted(conflict_idx - set(quarantined)):
            excl.add(cands[i]["src"], "duplikat_label_bertentangan", path=cands[i].get("orig_path", cands[i]["path"]))
    drop_idx = set(quarantined) | conflict_idx
    info["dibuang_setelah_pengelompokan"] = {"karantina": len(quarantined), "duplikat_bertentangan_saja": len(conflict_idx - set(quarantined)),
                                             "keduanya": len(conflict_idx & set(quarantined))}
    keep = [i for i in range(len(cands)) if i not in drop_idx]
    cands = [cands[i] for i in keep]
    var, groups, groups_nt = var[keep], groups[keep], groups_nt[keep]
    E = E[keep] if E is not None else None
    if info.get("time_window"):
        src4 = set(tw.get("sources", ["ds4"]))
        m4 = np.array([c["src"] in src4 for c in cands])
        info["time_window"]["dampak"] = {"tanpa_aturan_waktu": grp_stats(groups_nt, m4), "dengan_aturan_waktu": grp_stats(groups, m4)}
    labs1 = [class_label(c, classes) for c in cands]
    gcls = defaultdict(set)
    for g, l in zip(groups, labs1):
        gcls[int(g)].add(l)
    info["grup_adegan_lintas_kelas"] = sum(len(v) > 1 for v in gcls.values())

    print("[4/5] split stratified per grup ...", flush=True)
    exempt = set(aug.get("exempt_classes", []))
    for c in cands:
        names_ = {classes[i] for i in c["cls_ids"]}
        c["aug_possible"] = bool(c["aug_reasons"])
        if c.get("train_only"):
            c["forced_reason"] = "train_only"
        elif c["aug_reasons"] and not (names_ and names_ <= exempt):
            c["forced_reason"] = "aug:" + ",".join(c["aug_reasons"])
        else:
            c["forced_reason"] = ""
    hold_mask, drop_out = select_session_holdout(cands, groups, classes, grp, ts_map, info)
    if drop_out:      # anggota kelompok dari LUAR sesi terpilih: dibuang dari dataset (tidak dipindah ke split lain)
        for i in sorted(drop_out):
            excl.add(cands[i]["src"], "kelompok_duplikat_lintas_sesi", path=cands[i].get("orig_path", cands[i]["path"]))
        keep = [i for i in range(len(cands)) if i not in drop_out]
        cands = [cands[i] for i in keep]
        var, groups = var[keep], groups[keep]
        E = E[keep] if E is not None else None
        hold_mask = [hold_mask[i] for i in keep]
    strata = [f"{class_label(c, classes)}|{c['tag']}" for c in cands]
    free = [i for i in range(len(cands)) if not hold_mask[i]]
    sub = stratified_group_split(groups[free], [strata[i] for i in free], cfg["split"], cfg["seed"], [bool(cands[i]["forced_reason"]) for i in free])
    splits = [HOLD] * len(cands)
    for k_, i in enumerate(free):
        splits[i] = sub[k_]

    # pangkas background HANYA di train (sumber prunable); val/test tidak diubah
    mbf = cfg.get("max_background_frac_train")
    info["bg"] = None
    if mbf is not None:
        tr = [i for i, s_ in enumerate(splits) if s_ == "train"]
        n_dis = sum(1 for i in tr if cands[i]["cls_ids"])
        cap = int(n_dis * mbf / (1 - mbf))
        prun = set(cfg.get("background_prunable_sources", []))
        bg_all = [i for i in tr if not cands[i]["cls_ids"]]
        bg_prun = [i for i in bg_all if cands[i]["src"] in prun]
        n_remove = max(0, min(len(bg_all) - cap, len(bg_prun)))
        drop = set(rng.sample(sorted(bg_prun), n_remove)) if n_remove else set()
        info["bg"] = {"max_frac": mbf, "train_penyakit": n_dis, "cap": cap, "bg_train_sebelum": len(bg_all),
                      "dipangkas": len(drop), "bg_train_sesudah": len(bg_all) - len(drop),
                      "sumber_prunable": sorted(prun),
                      "dipertahankan_per_sumber": dict(Counter(cands[i]["src"] for i in bg_all if i not in drop)),
                      "cap_tercapai": (len(bg_all) - len(drop)) <= cap}
        if drop:
            for i in drop:
                excl.add(cands[i]["src"], "dipangkas_max_background_frac_train", path=cands[i].get("orig_path", cands[i]["path"]))
            keep = [i for i in range(len(cands)) if i not in drop]
            cands = [cands[i] for i in keep]
            var, groups = var[keep], groups[keep]
            E = E[keep] if E is not None else None
            splits = [splits[i] for i in keep]
    for c, g, s_ in zip(cands, groups, splits):
        c["group"], c["split"] = int(g), s_
    # integritas: tiap grup tepat satu split
    gs = defaultdict(set)
    for g, s_ in zip(groups, splits):
        gs[int(g)].add(s_)
    assert all(len(v) == 1 for v in gs.values()), "grup duplikat terpecah di >1 split"
    # integritas: gambar wajib-train (kelas train-only / terdeteksi augmentasi) tidak ada di val/test
    assert not any(c["forced_reason"] and c["split"] != "train" for c in cands), "gambar wajib-train bocor ke val/test"

    # verifikasi mandiri: jarak phash dan embedding minimum lintas split
    idx = {s_: np.array([i for i, x in enumerate(splits) if x == s_], int) for s_ in SPLITS + (HOLD,)}
    leak = {}
    for a_, b_ in [("val", "train"), ("test", "train"), ("test", "val")] + ([(HOLD, "train"), (HOLD, "val"), (HOLD, "test")] if len(idx[HOLD]) else []):
        d = min_dist_dihedral(var[idx[a_]], var[idx[b_]][:, 0])
        leak[f"{a_}-{b_}"] = {"pairs_leq_thr": int((d <= cfg["dup_threshold"]).sum()), "min": float(d.min())}
        if E is not None:
            S = ad.sim_matrix(E[idx[a_]], E[idx[b_]][:, 0])
            leak[f"{a_}-{b_}"]["emb_max_cos"] = round(float(S.max()), 4)
            leak[f"{a_}-{b_}"]["emb_pairs_ge_thr"] = int((S >= aug["embedding"]["cosine"]).sum())
    bn = defaultdict(set)
    for c in cands:
        st = c.get("orig_path", c["path"]).stem
        if ".rf." in st:
            bn[(c["src"], re.sub(r"\.rf\.[0-9a-f]+.*$", "", st))].add(c["split"])
    info["rf_basename_terpecah"] = sum(len(v) > 1 for v in bn.values())
    print("  verifikasi leakage:", leak, "| basename Roboflow terpecah:", info["rf_basename_terpecah"], flush=True)

    # audit independen saudara val/test vs train (SIFT k lebih besar, K4, phash, embedding)
    if grp.get("sift", {}).get("enabled"):
        info["audit"] = sibling_audit(cands, splits, E, var, cfg, grp.get("sibling_audit_knn", 50))
        print("  audit saudara:", json.dumps({k_: v_["dengan_saudara_gabungan"] for k_, v_ in info["audit"].items()}), flush=True)
        aud = info["audit"]
        n_vt = len(set(aud["val-train"]["indeks"]) | set(aud["test-train"]["indeks"]))
        n_all = sum(1 for s_ in splits if s_ in ("val", "test"))
        acc_pct = float(grp.get("audit_accept_pct", 1.0))
        info["audit_kriteria"] = {"n_dengan_saudara_di_train": n_vt, "n_val_test": n_all, "persen": round(100.0 * n_vt / n_all, 3),
                                  "ambang_persen": acc_pct, "lolos": 100.0 * n_vt / n_all < acc_pct,
                                  "test_val_tambahan": aud["test-val"]["dengan_saudara_gabungan"], "k": grp.get("sibling_audit_knn"),
                                  "tanpa_aturan_sebaran": bool(grp.get("audit_ignore_frac")),
                                  "test_new_session_dengan_saudara": {k_: v_["dengan_saudara_gabungan"] for k_, v_ in aud.items() if k_.startswith(HOLD)}}
        if tw.get("enabled"):
            ta = time_audit(cands, splits, lambda c_: dr.timestamp_of(c_, ts_map), grp.get("audit_time_windows", [tw["gap_s"]]), tuple(tw.get("sources", ["ds4"])))
            info["audit_waktu"] = ta
            gap_ = tw["gap_s"]
            n_bad = sum(v_["dengan_foto_dalam_jendela"] for k_, v_ in ta.items() if k_.startswith(f"+-{gap_:g}s|"))
            info["audit_kriteria"]["waktu_jendela_terpilih_pelanggaran"] = n_bad
            info["audit_kriteria"]["lolos_sift"] = info["audit_kriteria"]["lolos"]
            info["audit_kriteria"]["lolos_waktu"] = n_bad == 0
            info["audit_kriteria"]["lolos"] = info["audit_kriteria"]["lolos_sift"] and n_bad == 0
        print("  kriteria audit:", json.dumps(info["audit_kriteria"]), flush=True)
    for c, g, s_ in zip(cands, groups, splits):
        c["group"], c["split"] = int(g), s_
    info["test_changes"] = log_test_changes(cands, cfg, ROOT / cfg["report_dir"])
    info["ds4_tanggal"] = None
    if ts_map:
        ds_, cn_ = defaultdict(set), Counter()
        for c in cands:
            if c["src"] == "ds4":
                t_ = ts_map.get(Path(c.get("orig_path", c["path"])).name.lower())
                if t_:
                    ds_[c["split"]].add(t_.date().isoformat())
                    cn_[c["split"]] += 1
        info["ds4_tanggal"] = {"tanggal_unik": {sp_: len(v_) for sp_, v_ in ds_.items()}, "gambar_bertanggal": dict(cn_),
                               "semua_tanggal_val_test_ada_di_train": all(ds_[sp_] <= ds_["train"] for sp_ in ("val", "test"))}

    # montase 10 pasangan acak pada jarak phash 12..ambang (untuk pemeriksaan visual)
    info["montage"] = pairs_montage(cands, var, classes, 12, cfg["dup_threshold"], 10, cfg["seed"],
                                    ROOT / cfg["report_dir"] / "pairs_12_14.jpg")

    print("[5/5] menulis dataset dan laporan ...", flush=True)
    if not args.dry_run:
        materialize(cands, splits, out, classes, cfg.get("link_mode", "hardlink"))
        (out / "lists").mkdir(exist_ok=True)
        multi = [c["file"] for c in cands if c["split"] == "test" and len(c["cls_ids"]) >= 2]
        (out / "lists/test_multidisease.txt").write_text("\n".join(f"images/test/{f}" for f in multi) + ("\n" if multi else ""))
        with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["file", "split", "class", "source", "provisional", "train_only", "aug_possible", "aug_reasons",
                        "group", "n_boxes", "n_boxes_from_polygon", "orig_split", "orig_path"])
            for c in cands:
                w.writerow([c["file"], c["split"], class_label(c, classes), c["tag"],
                            c["tag"] in cfg.get("provisional_sources", []), bool(c.get("train_only")),
                            c["aug_possible"], "|".join(c["aug_reasons"]), c["group"], len(c["boxes"]),
                            sum(b[5] for b in c["boxes"]), c["orig_split"], rel(c.get("orig_path", c["path"]))])
        meta_out = write_dataset_meta(out, cfg, args.config, info, cands, classes)
        data_yaml = {
            "path": out.as_posix(), "train": "images/train", "val": "images/val", "test": "images/test",
            **({HOLD: f"images/{HOLD}"} if any(c["split"] == HOLD for c in cands) else {}),
            "names": {i: n for i, n in enumerate(classes)},
            "dataset_name": cfg["dataset_name"], "run_prefix": cfg["run_prefix"],
            "provisional": bool(cfg.get("provisional_sources")),
            "provisional_sources": cfg.get("provisional_sources", []), "seed": cfg["seed"],
            "grouping_rule_version": dr.RULE_VERSION, "manifest_sha256": meta_out["manifest_sha256"],
            "frozen": bool(grp.get("freeze")) and bool((info.get("audit_kriteria") or {}).get("lolos")),
        }
        (out / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8")
        info["meta"] = {"manifest_sha256": meta_out["manifest_sha256"], "grup_split_sha256": meta_out["grup_split_sha256"],
                        "config_sha256": meta_out["hash_berkas"]["config"]}
        if grp.get("freeze"):
            frozen = data_yaml["frozen"]
            hl = hashlib.sha256()
            for sp in [s_ for s_ in SPLITS + (HOLD,) if (out / "labels" / s_).exists()]:
                for lf in sorted((out / "labels" / sp).iterdir()):
                    hl.update(f"{sp}/{lf.name}\n".encode())
                    hl.update(lf.read_bytes())
            hi = hashlib.sha256("\n".join(f"{c['split']}/{c['file']}" for c in sorted(cands, key=lambda c: (c["split"], c["file"]))).encode()).hexdigest()
            fz = {"frozen": frozen, "frozen_at": datetime.datetime.now().isoformat(timespec="seconds") if frozen else None,
                  "alasan_tidak_dibekukan": None if frozen else "kriteria audit tidak terpenuhi; menunggu keputusan pengguna",
                  "manifest_sha256": meta_out["manifest_sha256"], "data_yaml_sha256": sha256_file(out / "data.yaml"),
                  "labels_sha256": hl.hexdigest(), "daftar_gambar_sha256": hi,
                  "aturan": "Setelah dibekukan, isi dataset ini tidak boleh diubah; perubahan lanjutan harus menjadi versi baru."}
            mj = json.loads((out / "dataset_meta.json").read_text(encoding="utf-8"))
            mj["frozen"], mj["freeze"] = frozen, fz
            (out / "dataset_meta.json").write_text(json.dumps(mj, indent=2, ensure_ascii=False), encoding="utf-8")
            (out / ("FROZEN.txt" if frozen else "NOT_FROZEN.txt")).write_text(
                ("DIBEKUKAN %s\nmanifest_sha256=%s\nPerubahan lanjutan harus menjadi versi baru.\n" % (fz["frozen_at"], fz["manifest_sha256"])) if frozen
                else "TIDAK dibekukan: kriteria audit tidak terpenuhi. Menunggu keputusan pengguna.\n", encoding="utf-8")
            info["freeze"] = fz
        # daftar per sumber agar metrik per sumber mudah dihitung
        for sp in [s_ for s_ in SPLITS + (HOLD,) if any(c["split"] == s_ for c in cands)]:
            for src_tag in sorted({c["tag"] for c in cands if c["split"] == sp}):
                fl = [c["file"] for c in cands if c["split"] == sp and c["tag"] == src_tag]
                (out / "lists" / f"{sp}_{src_tag}.txt").write_text("\n".join(f"images/{sp}/{f}" for f in fl) + "\n")
    else:
        for c in cands:
            c["file"] = ""
    text = build_report(cands, classes, cfg, excl, meta, groups, leak, args.dry_run, info)
    with open(ROOT / cfg["report_dir"] / "exclusions_detail.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sumber", "alasan", "berkas"])
        w.writerows(sorted(set(excl.items)))
    (ROOT / cfg["report_dir"] / "summary.json").write_text(json.dumps(
        {"seed": cfg["seed"], "n_images": len(cands), "leak_check": leak,
         "versions": {"numpy": np.__version__, "python": sys.version.split()[0]}}, indent=2), encoding="utf-8")
    print("\n" + text)
    print(f"\nLaporan: {cfg['report_dir']}/split_report.md")


if __name__ == "__main__":
    main()
