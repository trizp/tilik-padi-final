#!/usr/bin/env python3
"""
aug_detect.py - Mendeteksi gambar hasil augmentasi / saudara augmentasi TANPA bergantung pada satu cara.

Tiga detektor:
  (a) pola nama file : kata kunci (aug, rotat, flip, ...) dan banyak versi Roboflow (".rf.<hash>") dari
                       nama dasar yang sama.
  (b) heuristik piksel: sudut/tepi hitam (rotasi dengan expand) dan tepi "tertarik" (edge-extend/stretch).
  (c) embedding      : DINOv2-small (CLS, cosine). Gambar punya "saudara" bila ada gambar lain dengan
                       cosine >= ambang pada salah satu dari 8 varian dihedral. Juga dipakai untuk
                       menggabungkan grup duplikat (mencegah kebocoran) saat split.

Mode CLI = KALIBRASI + KONTROL POSITIF (hasil ke reports/aug_detect/):
  python scripts/aug_detect.py                 # statistik, tanpa ambang (untuk memilih ambang)
  python scripts/aug_detect.py --cosine 0.90   # evaluasi pada ambang tertentu

Kontrol positif: (1) pasangan duplikat dari phash (jarak <= 10) harus terdeteksi oleh embedding;
(2) transformasi sintetis (rotasi bebas, crop/zoom, kecerahan, blur, noise) pada gambar acak.
"""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
EMB_CACHE = ROOT / "data/interim/dino_emb.npz"
PIX_CACHE = ROOT / "data/interim/pixel_metrics.json"
MODEL_NAME = "dinov2_vits14"

NAME_RE = re.compile(r"(?:^|[^a-z])(aug(?:mented)?|rotat[a-z]*|flip[a-z]*|mirror[a-z]*|bright[a-z]*|noise[a-z]*|blur[a-z]*|"
                     r"zoom[a-z]*|crop[a-z]*|shear[a-z]*|jitter)(?:[^a-z]|$)", re.I)


def sig(p: Path):
    st = Path(p).stat()
    try:
        r = Path(p).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        r = Path(p).as_posix()
    return f"{r}|{st.st_size}|{st.st_mtime_ns}"


# ------------------------------------------------------------------ (a) nama
def name_flags(paths):
    """Kembalikan list alasan (string, '' = tidak ditandai) untuk tiap path."""
    stems = [Path(p).stem for p in paths]
    base = [re.sub(r"\.rf\.[0-9a-f]+.*$", "", s) for s in stems]
    cnt = Counter(b for s, b in zip(stems, base) if ".rf." in s)
    out = []
    for s, b in zip(stems, base):
        m = NAME_RE.search(s)
        if m:
            out.append(f"kata:{m.group(1).lower()}")
        elif ".rf." in s and cnt[b] > 1:
            out.append("versi_ganda_roboflow")
        else:
            out.append("")
    return out


# ------------------------------------------------------------------ (b) piksel
def _pixel_one(path):
    try:
        with Image.open(path) as im:
            im.draft("RGB", (512, 512))
            im = im.convert("RGB")
            s = 256 / max(im.size)
            im = im.resize((max(8, round(im.width * s)), max(8, round(im.height * s))), Image.BILINEAR)
        a = np.asarray(im).astype(np.float32)
        g = a.mean(2)
        H, W = g.shape
        blk = a.max(2) < 16
        ph, pw = max(1, int(.15 * H)), max(1, int(.15 * W))
        corner = max(blk[:ph, :pw].mean(), blk[:ph, -pw:].mean(), blk[-ph:, :pw].mean(), blk[-ph:, -pw:].mean())
        fh, fw = max(1, int(.04 * H)), max(1, int(.04 * W))
        frame = np.ones_like(blk, bool)
        frame[fh:-fh, fw:-fw] = False
        border = blk[frame].mean()
        gx, gy = np.abs(np.diff(g, axis=1)), np.abs(np.diff(g, axis=0))
        sh, sw = max(2, int(.12 * H)), max(2, int(.12 * W))
        strips = [(gx[:, :sw].mean(), gy[:, :sw].mean()), (gx[:, -sw:].mean(), gy[:, -sw:].mean()),
                  (gy[:sh, :].mean(), gx[:sh, :].mean()), (gy[-sh:, :].mean(), gx[-sh:, :].mean())]
        # rasio gradien sepanjang arah tarikan / gradien tegak lurus; kecil = piksel tepi "ditarik"
        smear = min((a_ / (b_ + 1e-6)) if b_ > 2.0 else 1.0 for a_, b_ in strips)
        return {"corner_black": float(corner), "border_black": float(border), "smear": float(smear)}
    except Exception:
        return None


def pixel_metrics(paths):
    cache = json.loads(PIX_CACHE.read_text()) if PIX_CACHE.exists() else {}
    sigs = [sig(p) for p in paths]
    todo = [i for i, s in enumerate(sigs) if s not in cache]
    if todo:
        with ProcessPoolExecutor() as ex:
            for i, r in zip(todo, ex.map(_pixel_one, [str(paths[i]) for i in todo], chunksize=32)):
                cache[sigs[i]] = r
        PIX_CACHE.parent.mkdir(parents=True, exist_ok=True)
        PIX_CACHE.write_text(json.dumps(cache))
    return [cache[s] for s in sigs]


def pixel_flag(m, th):
    """th: {corner_black, border_black, smear}. Kembalikan alasan atau ''."""
    if m is None:
        return ""
    if m["corner_black"] >= th["corner_black"]:
        return "sudut_hitam"
    if m["border_black"] >= th["border_black"]:
        return "tepi_hitam"
    if m["smear"] <= th["smear"]:
        return "tepi_tertarik"
    return ""


# ------------------------------------------------------------------ (c) embedding
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def _to_variants(im: Image.Image):
    """8 varian dihedral sebagai tensor-numpy (8,3,224,224)."""
    im = im.convert("RGB")
    out = []
    for k in range(4):
        r = im.rotate(90 * k, expand=True) if k else im
        for v in (r, r.transpose(Image.FLIP_LEFT_RIGHT)):
            a = (np.asarray(v.resize((224, 224), Image.BICUBIC), np.float32) / 255 - _MEAN) / _STD
            out.append(a.transpose(2, 0, 1))
    return np.stack(out)


class _DS:
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        if isinstance(it, Image.Image):
            im = it
        else:
            im = Image.open(it)
            im.draft("RGB", (512, 512))
        return _to_variants(im)


_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        import torch
        m = torch.hub.load("facebookresearch/dinov2", MODEL_NAME, trust_repo=True)
        _MODEL = m.eval().cuda().half()
    return _MODEL


def embed_items(items, batch=16, workers=6):
    """items: list path/PIL. Kembalikan (N,8,384) float32 ternormalisasi L2 (indeks 0 = asli)."""
    import torch
    from torch.utils.data import DataLoader
    m = _model()
    out = []
    dl = DataLoader(_DS(items), batch_size=batch, num_workers=workers if workers and not isinstance(items[0], Image.Image) else 0)
    with torch.no_grad():
        for x in dl:
            b = x.shape[0]
            e = m(x.view(-1, 3, 224, 224).cuda().half()).float()
            e = torch.nn.functional.normalize(e, dim=-1).view(b, 8, -1)
            out.append(e.cpu().numpy())
    return np.concatenate(out)


def embeddings(paths):
    """Embedding ber-cache (kunci: path|ukuran|mtime). Kembalikan (N,8,384) float32."""
    cache = {}
    if EMB_CACHE.exists():
        z = np.load(EMB_CACHE, allow_pickle=True)
        for s, e in zip(z["sig"], z["emb"]):
            cache[str(s)] = e
    sigs = [sig(p) for p in paths]
    todo = [i for i, s in enumerate(sigs) if s not in cache]
    print(f"  embedding {MODEL_NAME}: {len(paths)} gambar, {len(todo)} perlu dihitung", flush=True)
    if todo:
        E = embed_items([str(paths[i]) for i in todo])
        for i, e in zip(todo, E):
            cache[sigs[i]] = e.astype(np.float16)
        EMB_CACHE.parent.mkdir(parents=True, exist_ok=True)
        keys = list(cache)
        np.savez_compressed(EMB_CACHE, sig=np.array(keys), emb=np.array([cache[k] for k in keys]))
    return np.stack([cache[s] for s in sigs]).astype(np.float32)


def sim_matrix(Ea, Eb_orig, chunk=2048):
    """S[i,j] = max_v cos(Ea[i,v], Eb_orig[j])  -> array (len(Ea), len(Eb))."""
    out = np.empty((len(Ea), len(Eb_orig)), np.float32)
    for s in range(0, len(Ea), chunk):
        x = Ea[s:s + chunk]
        out[s:s + chunk] = np.max([x[:, v] @ Eb_orig.T for v in range(x.shape[1])], axis=0)
    return out


def sim_pairs(E, thr, chunk=1024):
    """Pasangan (i<j) dengan kemiripan embedding >= thr (maks dua arah, 8 varian dihedral)."""
    O = E[:, 0]
    pairs = []
    for s in range(0, len(E), chunk):
        S = np.max([E[s:s + chunk, v] @ O.T for v in range(E.shape[1])], axis=0)
        ii, jj = np.where(S >= thr)
        ii = ii + s
        m = ii != jj
        pairs.append(np.stack([ii[m], jj[m]], 1))
    P = np.concatenate(pairs)
    if len(P):
        P = np.unique(np.sort(P, axis=1), axis=0)
    return P


# ------------------------------------------------------------------ kalibrasi (CLI)
def _transforms():
    def rot15(im):
        return im.rotate(15, expand=True, fillcolor=(0, 0, 0))

    def rot_small(im):
        return im.rotate(-7, expand=False, resample=Image.BICUBIC, fillcolor=(0, 0, 0))

    def crop80(im):
        w, h = im.size
        return im.crop((int(.1 * w), int(.1 * h), int(.9 * w), int(.9 * h)))

    def zoom65(im):
        w, h = im.size
        return im.crop((int(.2 * w), int(.15 * h), int(.85 * w), int(.8 * h)))

    return {
        "rotasi_15_expand": rot15, "rotasi_-7": rot_small, "crop_80%": crop80, "zoom_65%": zoom65,
        "kecerahan_x1.4": lambda im: ImageEnhance.Brightness(im).enhance(1.4),
        "kecerahan_x0.6": lambda im: ImageEnhance.Brightness(im).enhance(0.6),
        "kontras_x1.5": lambda im: ImageEnhance.Contrast(im).enhance(1.5),
        "blur_r2": lambda im: im.filter(ImageFilter.GaussianBlur(2)),
        "flip_horizontal": lambda im: im.transpose(Image.FLIP_LEFT_RIGHT),
        "noise_sigma12": lambda im: Image.fromarray(np.clip(
            np.asarray(im.convert("RGB"), np.float32) + np.random.default_rng(0).normal(0, 12, (im.height, im.width, 3)),
            0, 255).astype(np.uint8)),
    }


def pct(a, q):
    return float(np.percentile(a, q))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/classes.yaml")
    ap.add_argument("--cosine", type=float, default=None, help="ambang cosine yang dievaluasi")
    ap.add_argument("--n-synth", type=int, default=300)
    args = ap.parse_args()

    import yaml
    sys.path.insert(0, str(Path(__file__).parent))
    import prepare_dataset as pd_
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    classes, excl, meta = cfg["disease_classes"], pd_.Excl(), {}
    import random
    rng = random.Random(cfg["seed"])
    cands = []
    for s in cfg["sources"]:
        if s.get("role", "pool") != "pool":
            continue
        if s["type"] == "yolo":
            cands += pd_.collect_yolo(s, classes, excl, meta)
        elif s["type"] == "yolo_folders":
            cands += pd_.collect_yolo_folders(s, classes, excl, meta)
        else:
            cands += pd_.collect_classification(s, classes, excl, meta, rng)
    pd_.resize_before_hash(cands)
    paths = [c["path"] for c in cands]
    src = np.array([c["src"] for c in cands])
    cls = np.array([pd_.class_label(c, classes) for c in cands])
    print(f"pool: {len(cands)} gambar", dict(Counter(src)))
    rep = {"model": MODEL_NAME, "n_pool": len(cands)}

    # ---- (a) dan (b)
    ds1 = np.where(src == "ds1")[0]
    nf = name_flags([paths[i] for i in ds1])
    rep["a_nama_ds1"] = {"n": len(ds1), "ditandai": sum(bool(x) for x in nf),
                         "alasan": dict(Counter(x for x in nf if x))}
    pm = pixel_metrics(paths)
    name_pos = np.array([bool(x) for x in nf])
    keys = ("corner_black", "border_black", "smear")
    arr = {k: np.array([m[k] if m else np.nan for m in pm]) for k in keys}
    ref = {}
    for k in keys:
        ref[k] = {
            "ds4_p1/p50/p99/p99.9": [round(pct(arr[k][src == "ds4"], q), 4) for q in (1, 50, 99, 99.9)],
            "ds1_nama_aug_p1/p50/p99": [round(pct(arr[k][ds1][name_pos], q), 4) for q in (1, 50, 99)],
            "ds1_tanpa_tanda_p1/p50/p99": [round(pct(arr[k][ds1][~name_pos], q), 4) for q in (1, 50, 99)],
        }
    rep["b_piksel_distribusi"] = ref

    # ---- (c) embedding
    E = embeddings(paths)
    rep["c_embedding_dim"] = int(E.shape[-1])
    # pasangan duplikat phash (kontrol positif)
    var, ok = pd_.compute_hashes(cands, cfg["hash_size"])
    assert ok.all()
    P = pd_.near_pairs(var[:, 0], var[:, 0], 10, True)
    Q = np.concatenate([P] + [pd_.near_pairs(var[:, v], var[:, 0], 10, False) for v in range(1, 8)])
    Q = np.unique(np.sort(Q[Q[:, 0] != Q[:, 1]], axis=1), axis=0)
    O = E[:, 0]
    pos = np.array([max(float(E[i, v] @ O[j]) for v in range(8)) for i, j in Q])
    pos_plain = np.array([max(float(E[i, 0] @ O[j]), float(E[i, 1] @ O[j])) for i, j in Q])   # tanpa rotasi 90
    rep["kontrol_positif_phash"] = {
        "n_pasangan_phash<=10": int(len(Q)),
        "cosine_p1/p5/p25/p50": [round(pct(pos, q), 4) for q in (1, 5, 25, 50)],
        "cosine_tanpa_rotasi_p1/p5/p50": [round(pct(pos_plain, q), 4) for q in (1, 5, 50)],
    }
    # negatif: pasangan acak yang jauh menurut phash
    g = np.random.default_rng(0)
    n = len(E)
    I, J = g.integers(0, n, 400000), g.integers(0, n, 400000)
    keep = I != J
    I, J = I[keep], J[keep]
    neg = np.empty(len(I), np.float32)
    for s in range(0, len(I), 20000):
        a, b = I[s:s + 20000], J[s:s + 20000]
        neg[s:s + 20000] = np.max([(E[a, v] * O[b]).sum(1) for v in range(8)], axis=0)
    same = (cls[I] == cls[J]) & (src[I] == src[J])
    rep["negatif_acak"] = {
        "n": int(len(neg)),
        "semua_p50/p99/p99.9/max": [round(pct(neg, q), 4) for q in (50, 99, 99.9, 100)],
        "kelas+sumber_sama_p50/p99/p99.9/max": [round(pct(neg[same], q), 4) for q in (50, 99, 99.9, 100)],
    }
    print(json.dumps(rep, indent=1, ensure_ascii=False))

    # ---- evaluasi pada ambang
    if args.cosine is not None:
        t = args.cosine
        ev = {"cosine": t,
              "recall_pasangan_phash": round(float((pos >= t).mean()), 4),
              "recall_pasangan_phash_tanpa_rotasi90": round(float((pos_plain >= t).mean()), 4),
              "fp_acak_semua": round(float((neg >= t).mean()), 6),
              "fp_acak_kelas_sumber_sama": round(float((neg[same] >= t).mean()), 6)}
        # kontrol positif sintetis
        idx = np.random.default_rng(1).choice(n, args.n_synth, replace=False)
        T = _transforms()
        syn = {}
        for name, fn in T.items():
            ims = []
            for i in idx:
                with Image.open(paths[i]) as im:
                    ims.append(fn(im.convert("RGB")))
            Et = embed_items(ims, batch=16, workers=0)
            s_emb = np.array([max(float(Et[k, v] @ O[i]) for v in range(8)) for k, i in enumerate(idx)])
            syn[name] = {"recall_embedding": round(float((s_emb >= t).mean()), 3),
                         "cosine_median": round(float(np.median(s_emb)), 3)}
        ev["sintetis"] = syn
        # recall phash untuk transformasi yang sama (pembanding)
        for name, fn in T.items():
            ok_ = 0
            for i in idx:
                with Image.open(paths[i]) as im:
                    tim = fn(im.convert("RGB"))
                import imagehash
                hs = cfg["hash_size"]
                hv = []
                for k in range(4):
                    r = tim.rotate(90 * k, expand=True) if k else tim
                    hv.append(imagehash.phash(r, hash_size=hs).hash.flatten())
                    hv.append(imagehash.phash(r.transpose(Image.FLIP_LEFT_RIGHT), hash_size=hs).hash.flatten())
                d = min(int((h != var[i, 0]).sum()) for h in hv)
                ok_ += d <= cfg["dup_threshold"]
            ev["sintetis"][name]["recall_phash(<=%d)" % cfg["dup_threshold"]] = round(ok_ / len(idx), 3)
        # ukuran grup bila embedding dipakai untuk menggabungkan
        P2 = sim_pairs(E, t)
        parent = list(range(n))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        for a, b in P2:
            ra, rb = find(int(a)), find(int(b))
            if ra != rb:
                parent[ra] = rb
        sizes = Counter(find(i) for i in range(n))
        ev["grup_embedding_saja"] = {"n_pasangan": int(len(P2)), "n_grup": len(sizes), "terbesar": max(sizes.values()),
                                     "gambar_di_grup>1": int(sum(v for v in sizes.values() if v > 1))}
        rep["evaluasi"] = ev
        print(json.dumps(ev, indent=1, ensure_ascii=False))
    out = ROOT / "reports/aug_detect"
    out.mkdir(parents=True, exist_ok=True)
    (out / ("calibration.json" if args.cosine is None else f"calibration_cos{args.cosine}.json")).write_text(
        json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
