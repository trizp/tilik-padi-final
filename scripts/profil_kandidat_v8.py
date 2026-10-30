#!/usr/bin/env python3
"""
profil_kandidat_v8.py - Profil kandidat dataset v8 (HANYA membaca berkas; tidak menjalankan model deteksi apa pun).

Per kandidat (data/raw/kandidat_v8/<nama>/): kelas, gambar dan kotak per kelas, format label (kotak 5 kolom vs poligon),
median luas kotak per kelas (dibandingkan ds4 dan ds1 di prov_v7 train), resolusi, jenis foto (heuristik: "daun petik berlatar
polos" bila porsi latar terang-netral >= 15% DAN piksel hijau < 35%; selain itu "daun di tanaman/lapangan"; dikalibrasi dengan
ds3 = daun petik, ds4 = lapangan), dan montase 12 contoh dengan kotak label.
Keluaran: reports/kandidat_v8/{profil_<nama>.json, profil_ringkas.csv, montase_<nama>.jpg}
"""
import csv
import json
import random
import statistics as S
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw/kandidat_v8"
REP = ROOT / "reports/kandidat_v8"
IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
WARNA = [(42, 120, 214), (235, 104, 52), (27, 175, 122), (237, 161, 0), (232, 123, 164), (0, 131, 0), (74, 58, 167), (227, 73, 72)]


def jenis_foto(path):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    im.thumbnail((256, 256))
    a = np.asarray(im, np.float32) / 255
    hsv = np.asarray(im.convert("HSV"), np.float32) / 255
    terang = float(((a.mean(2) > 0.75) & (hsv[..., 1] < 0.15)).mean())
    hijau = float(((hsv[..., 0] > 0.17) & (hsv[..., 0] < 0.45) & (hsv[..., 1] > 0.25) & (hsv[..., 2] > 0.2)).mean())
    return ("daun petik berlatar polos" if terang >= 0.15 and hijau < 0.35 else "daun di tanaman/lapangan"), terang, hijau, float(a.mean())


def baca_label(p):
    out, poligon = [], 0
    for ln in p.read_text().splitlines() if p.exists() else []:
        t = ln.split()
        if not t:
            continue
        c = int(float(t[0]))
        v = list(map(float, t[1:]))
        if len(v) == 4:
            out.append((c, v[2] * v[3]))
        else:                                          # poligon -> kotak pembatas
            poligon += 1
            xs, ys = v[0::2], v[1::2]
            out.append((c, (max(xs) - min(xs)) * (max(ys) - min(ys))))
    return out, poligon


def acuan():
    man = list(csv.DictReader(open(ROOT / "data/processed/prov_v7/manifest.csv", encoding="utf-8")))
    hasil = defaultdict(list)
    for r in man:
        if r["split"] != "train" or r["class"] == "background":
            continue
        lab = ROOT / "data/processed/prov_v7/labels/train" / (Path(r["file"]).stem + ".txt")
        for c, a in baca_label(lab)[0]:
            src = "ds1" if r["source"].startswith("ds1") else r["source"]
            hasil[f"{src}|{r['class']}"].append(a)
    return {k: round(100 * S.median(v), 3) for k, v in hasil.items()}


def main():
    REP.mkdir(parents=True, exist_ok=True)
    ref = acuan()
    ringkas = []
    for d in sorted(p for p in RAW.iterdir() if p.is_dir()):
        yml = next(d.rglob("data.yaml"))
        dy = yaml.safe_load(yml.read_text(encoding="utf-8"))
        names = dy["names"] if isinstance(dy["names"], list) else [dy["names"][k] for k in sorted(dy["names"])]
        imgs = sorted(p for p in d.rglob("*") if p.suffix.lower() in IMG)
        per_kelas_img, per_kelas_box, luas, poligon_total = Counter(), Counter(), defaultdict(list), 0
        jenis, terang_l, hijau_l, cerah_l, ukuran, split = Counter(), [], [], [], Counter(), Counter()
        jenis_per_kelas = defaultdict(Counter)
        tanpa_label, contoh = 0, defaultdict(list)
        for p in imgs:
            lab = p.parent.parent / "labels" / (p.stem + ".txt")
            ls, pg = baca_label(lab)
            poligon_total += pg
            split[p.parent.parent.name] += 1
            if not ls:
                tanpa_label += 1
            kel = sorted({names[c] for c, _ in ls})
            for k in kel:
                per_kelas_img[k] += 1
            for c, a in ls:
                per_kelas_box[names[c]] += 1
                luas[names[c]].append(a)
            j, t, h, c_ = jenis_foto(p)
            jenis[j] += 1
            terang_l.append(t); hijau_l.append(h); cerah_l.append(c_)
            for k in kel or ["(tanpa label)"]:
                jenis_per_kelas[k][j] += 1
                contoh[k].append(p)
            ukuran[Image.open(p).size] += 1
        prof = {"kandidat": d.name, "data_yaml": str(yml.relative_to(ROOT)), "kelas": names, "gambar": len(imgs), "per_split": dict(split),
                "gambar_tanpa_label": tanpa_label, "format_label": "YOLO kotak" if poligon_total == 0 else f"campuran: {poligon_total} baris poligon",
                "gambar_per_kelas": dict(per_kelas_img), "kotak_per_kelas": dict(per_kelas_box),
                "median_luas_kotak_persen_per_kelas": {k: round(100 * S.median(v), 3) for k, v in luas.items()},
                "acuan_median_luas_kotak_persen_prov_v7_train": ref,
                "resolusi": {f"{w}x{h}": n for (w, h), n in ukuran.most_common(5)},
                "jenis_foto": dict(jenis), "jenis_foto_per_kelas": {k: dict(v) for k, v in jenis_per_kelas.items()},
                "rata2_latar_terang_netral_persen": round(100 * S.mean(terang_l), 1), "rata2_piksel_hijau_persen": round(100 * S.mean(hijau_l), 1),
                "rata2_kecerahan": round(S.mean(cerah_l), 3)}
        (REP / f"profil_{d.name}.json").write_text(json.dumps(prof, indent=2, ensure_ascii=False), encoding="utf-8")
        # montase 12 contoh dengan kotak label, sebar antar kelas
        rng = random.Random(8)
        kels = [k for k in names if contoh.get(k)]
        pilih = []
        while len(pilih) < 12 and any(contoh[k] for k in kels):
            for k in kels:
                if contoh[k] and len(pilih) < 12:
                    p = rng.choice(contoh[k])
                    contoh[k].remove(p)
                    if p not in [x for _, x in pilih]:
                        pilih.append((k, p))
        T = 300
        cv = Image.new("RGB", (4 * T, 3 * (T + 18) + 22), "white")
        dd = ImageDraw.Draw(cv)
        dd.text((4, 4), f"{d.name}: kotak label (warna per kelas). Kelas: " + ", ".join(f"{i}={n}" for i, n in enumerate(names)), fill="black")
        for i, (k, p) in enumerate(pilih):
            im = Image.open(p).convert("RGB")
            W, H = im.size
            dr = ImageDraw.Draw(im)
            lab = p.parent.parent / "labels" / (p.stem + ".txt")
            for ln in lab.read_text().splitlines() if lab.exists() else []:
                t = ln.split()
                if not t:
                    continue
                c, v = int(float(t[0])), list(map(float, t[1:]))
                if len(v) == 4:
                    x, y, w, h = v
                    box = [(x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H]
                else:
                    box = [min(v[0::2]) * W, min(v[1::2]) * H, max(v[0::2]) * W, max(v[1::2]) * H]
                dr.rectangle(box, outline=WARNA[c % len(WARNA)], width=max(2, W // 160))
            im.thumbnail((T, T))
            X, Y = (i % 4) * T, 22 + (i // 4) * (T + 18)
            cv.paste(im, (X, Y + 18))
            dd.text((X + 3, Y + 3), f"{k} | {p.name[:30]}", fill="black")
        cv.save(REP / f"montase_{d.name}.jpg", quality=86)
        ringkas.append({"kandidat": d.name, "gambar": len(imgs), "kelas": "; ".join(names), "resolusi": next(iter(prof["resolusi"])),
                        "format": prof["format_label"], "daun_petik_berlatar_polos_persen": round(100 * jenis["daun petik berlatar polos"] / len(imgs), 1),
                        "gambar_per_kelas": json.dumps(prof["gambar_per_kelas"], ensure_ascii=False),
                        "median_luas_kotak_persen": json.dumps(prof["median_luas_kotak_persen_per_kelas"], ensure_ascii=False)})
        print(json.dumps({k: prof[k] for k in ("kandidat", "gambar", "per_split", "gambar_tanpa_label", "format_label", "gambar_per_kelas", "kotak_per_kelas",
                                                "median_luas_kotak_persen_per_kelas", "resolusi", "jenis_foto", "rata2_piksel_hijau_persen", "rata2_latar_terang_netral_persen")},
                         ensure_ascii=False))
    with open(REP / "profil_ringkas.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(ringkas[0]))
        w.writeheader()
        w.writerows(ringkas)
    print("acuan prov_v7 train (median luas kotak %):", ref)


if __name__ == "__main__":
    main()
