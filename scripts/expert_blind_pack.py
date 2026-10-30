#!/usr/bin/env python3
"""
expert_blind_pack.py - (1) Analisis asal-usul pasangan lintas kelas ds4 (Original_Filename dan tanggal-jam dari
Dataset metadata.xlsx; pola selisih nomor berkas konstan Blast vs Brown spot) dan (2) paket pakar versi BUTA.

TIDAK mengubah dataset atau label. Keluaran di reports/prov_v4/:
  ds4_pair_metadata.csv      - pasangan ds4 dari pasangan_lintas_kelas_untuk_pakar.xlsx + metadata
  blast_brown_offsets.csv    - seluruh pasangan Blast~Brown spot ds4 (SIFT diterima) + selisih nomor
  offset_runs.csv            - deret pasangan dengan selisih nomor konstan (>= 3 nomor Blast berurutan)
  expert_pack/               - pair_XX_A.jpg, pair_XX_B.jpg (1024 px, tanpa EXIF), paket_pakar.pdf, lembar_jawaban.xlsx
  answer_key.csv             - KUNCI JAWABAN (label asli + penempatan A/B); TERPISAH dari expert_pack/
  expert_blind_pack.json     - ringkasan
"""
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
RD = ROOT / "reports/prov_v4"
PACK = RD / "expert_pack"
FONT = "C:/Windows/Fonts/arial.ttf"
FONTB = "C:/Windows/Fonts/arialbd.ttf"
OPTIONS = ["leaf_blast", "brown_spot", "bacterial_leaf_blight", "sehat", "dua_penyakit", "lainnya", "tidak yakin"]


def num(p):
    m = re.findall(r"(\d+)", Path(p).stem)
    return int(m[-1]) if m else None


def load_meta():
    xl = pd.read_excel(ROOT / "data/raw/ds4/docs/Dataset metadata.xlsx", sheet_name="Image_Metadata")
    rx = xl["Original_Filename"].astype(str).str.extract(r"(20\d{2})(\d{2})(\d{2})[_-]?(\d{2})(\d{2})(\d{2})")
    ok = rx.notna().all(axis=1)
    xl["ts"] = pd.NaT
    xl.loc[ok, "ts"] = pd.to_datetime(rx[ok].astype(str).agg("".join, axis=1), format="%Y%m%d%H%M%S", errors="coerce")
    xl["ts"] = pd.to_datetime(xl["ts"])
    return {str(n).lower(): (str(o), t) for n, o, t in zip(xl["New_Filename"], xl["Original_Filename"], xl["ts"])}


def relation(ma, mb):
    """Hubungan dua foto menurut metadata: 'foto_asli_sama', 'selisih<=2s', 'selisih>2s', atau 'tanpa_jam'."""
    (oa, ta), (ob, tb) = ma, mb
    if oa == ob:
        return "foto_asli_sama", 0.0
    if pd.isna(ta) or pd.isna(tb):
        return "tanpa_jam", None
    dt = abs((ta - tb).total_seconds())
    return ("selisih<=2s" if dt <= 2 else "selisih>2s"), dt


def main():
    rep = {}
    meta = load_meta()

    def m(p):
        return meta.get(Path(p).name.lower(), ("?", pd.NaT))

    # ================= 1a. pasangan ds4 di berkas pakar
    xp = pd.read_excel(RD / "pasangan_lintas_kelas_untuk_pakar.xlsx", sheet_name="Pasangan")
    d4 = xp[(xp.sumber_a == "ds4") & (xp.sumber_b == "ds4")]
    rows = []
    for r in d4.itertuples():
        rel_, dt = relation(m(r.berkas_a), m(r.berkas_b))
        rows.append({"no_pasangan": r.no_pasangan, "berkas_a": Path(r.berkas_a).name, "berkas_b": Path(r.berkas_b).name,
                     "label_a": r.label_a_di_dataset, "label_b": r.label_b_di_dataset, "inlier": r.kecocokan_SIFT_inlier,
                     "original_a": m(r.berkas_a)[0], "original_b": m(r.berkas_b)[0],
                     "jam_a": str(m(r.berkas_a)[1]), "jam_b": str(m(r.berkas_b)[1]), "selisih_detik": dt, "hubungan": rel_,
                     "selisih_nomor_berkas": (num(r.berkas_b) - num(r.berkas_a)) if num(r.berkas_a) is not None else None})
    pd.DataFrame(rows).to_csv(RD / "ds4_pair_metadata.csv", index=False, encoding="utf-8")
    c = Counter(x["hubungan"] for x in rows)
    rep["1a_pasangan_ds4_di_xlsx"] = {"n": len(rows), "per_hubungan": dict(c),
                                       "foto_asli_sama_atau_selisih<=2s": c["foto_asli_sama"] + c["selisih<=2s"],
                                       "per_kelompok_inlier": {g: dict(Counter(x["hubungan"] for x, gg in zip(rows, d4.kelompok_kecocokan) if gg == g))
                                                               for g in sorted(set(d4.kelompok_kecocokan))}}

    # ================= 1b. seluruh pasangan Blast~Brown spot ds4: selisih nomor berkas
    sp = pd.read_csv(RD / "sift_pairs.csv")
    acc = sp[(sp.status == "diterima") & (sp.sumber_a == "ds4") & (sp.sumber_b == "ds4")]
    bb = []
    for r in acc.itertuples():
        ka, kb = {r.kelas_a, r.kelas_b} == {"leaf_blast", "brown_spot"}, None
        if not ka:
            continue
        pb, pr = (r.a, r.b) if r.kelas_a == "leaf_blast" else (r.b, r.a)
        rel_, dt = relation(m(pb), m(pr))
        bb.append({"blast": Path(pb).name, "brown_spot": Path(pr).name, "n_blast": num(pb), "n_brown": num(pr),
                   "selisih_nomor": num(pr) - num(pb), "inlier": int(r.inlier), "hubungan_metadata": rel_, "selisih_detik": dt,
                   "original_blast": m(pb)[0], "original_brown": m(pr)[0]})
    dfb = pd.DataFrame(bb)
    dfb.to_csv(RD / "blast_brown_offsets.csv", index=False, encoding="utf-8")
    # deret konstan: nomor Blast berurutan (n, n+1, ...) dengan selisih nomor yang sama
    runs = []
    for off, g in dfb.groupby("selisih_nomor"):
        ns = sorted(set(g.n_blast))
        start = prev = ns[0]
        for n in ns[1:] + [None]:
            if n is not None and n == prev + 1:
                prev = n
                continue
            if prev - start + 1 >= 3:
                sub = g[(g.n_blast >= start) & (g.n_blast <= prev)]
                runs.append({"selisih_nomor": int(off), "blast_dari": int(start), "blast_sampai": int(prev),
                             "brown_dari": int(start + off), "brown_sampai": int(prev + off), "panjang_deret": int(prev - start + 1),
                             "pasangan": int(len(sub)), "inlier_median": float(sub.inlier.median()), "inlier_min": int(sub.inlier.min()),
                             "hubungan_metadata": "; ".join(f"{k}:{v}" for k, v in Counter(sub.hubungan_metadata).most_common())})
            if n is not None:
                start = prev = n
    dfr = pd.DataFrame(runs).sort_values(["pasangan", "inlier_median"], ascending=False) if runs else pd.DataFrame()
    dfr.to_csv(RD / "offset_runs.csv", index=False, encoding="utf-8")
    in_runs = sum(int(x["pasangan"]) for x in runs)
    rep["1b_blast_brown_ds4"] = {
        "pasangan_SIFT_diterima": int(len(dfb)), "gambar_blast_unik": int(dfb.blast.nunique()), "gambar_brown_unik": int(dfb.brown_spot.nunique()),
        "selisih_nomor_unik": int(dfb.selisih_nomor.nunique()), "selisih_terbanyak": {int(k): int(v) for k, v in dfb.selisih_nomor.value_counts().head(10).items()},
        "deret_konstan_>=3": len(runs), "pasangan_dalam_deret_konstan": in_runs, "persen_pasangan_dalam_deret": round(100 * in_runs / max(1, len(dfb)), 1),
        "hubungan_metadata_semua": dict(Counter(dfb.hubungan_metadata)),
        "inlier>100_dalam_deret": int(sum(((dfb.inlier > 100) & dfb.apply(lambda r: any(x["selisih_nomor"] == r.selisih_nomor and x["blast_dari"] <= r.n_blast <= x["blast_sampai"] for x in runs), axis=1)).tolist())),
        "inlier>100_total": int((dfb.inlier > 100).sum())}

    # ================= 2. paket pakar BUTA
    rnd = random.Random(2026)
    used = set()
    ds4acc = acc.copy()

    def take(df, n):
        idx = list(df.index)
        rnd.shuffle(idx)
        out = []
        for i in idx:
            r = df.loc[i]
            if r.a in used or r.b in used:
                continue
            out.append(r)
            used.update({r.a, r.b})
            if len(out) == n:
                break
        return out
    isbb = ds4acc.apply(lambda r: {r.kelas_a, r.kelas_b} == {"leaf_blast", "brown_spot"}, axis=1)
    isbh = ds4acc.apply(lambda r: {r.kelas_a, r.kelas_b} == {"brown_spot", "background"}, axis=1)
    sel = ([("blast~brown inlier>100", r) for r in take(ds4acc[isbb & (ds4acc.inlier > 100)], 12)] +
           [("blast~brown inlier 21-100", r) for r in take(ds4acc[isbb & (ds4acc.inlier >= 21) & (ds4acc.inlier <= 100)], 6)] +
           [("brown_spot~sehat", r) for r in take(ds4acc[isbh], 6)])
    rnd.shuffle(sel)
    PACK.mkdir(parents=True, exist_ok=True)
    for f in PACK.glob("pair_*.jpg"):
        f.unlink()
    key, pages = [], []
    W, H = 2339, 1654                                     # A4 lanskap 200 dpi
    f_t, f_b, f_s = ImageFont.truetype(FONTB, 46), ImageFont.truetype(FONT, 30), ImageFont.truetype(FONTB, 34)
    # halaman petunjuk
    pg = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(pg)
    lines = ["PAKET PENILAIAN FOTO DAUN PADI (versi buta)", "",
             "Setiap halaman berisi dua foto: A (kiri) dan B (kanan).",
             "Untuk MASING-MASING foto, centang satu pilihan penyakit yang paling sesuai menurut Anda:",
             "   leaf_blast  /  brown_spot  /  bacterial_leaf_blight  /  sehat  /  dua_penyakit  /  lainnya  /  tidak yakin",
             "Lalu tulis catatan bila perlu (misalnya: bercak yang terlihat, bagian daun, keraguan).",
             "", "Label dari dataset sengaja TIDAK ditampilkan. Mohon nilai berdasarkan apa yang terlihat di foto.",
             "Jawaban juga bisa diisi di berkas lembar_jawaban.xlsx (kolom sama dengan halaman ini).",
             "", f"Jumlah pasangan: {len(sel)}."]
    y = 120
    for i, ln in enumerate(lines):
        d.text((120, y), ln, fill="black", font=f_t if i == 0 else f_b)
        y += 70 if i == 0 else 52
    pages.append(pg)
    for n, (cat, r) in enumerate(sel, 1):
        swap = rnd.random() < 0.5
        A, B = (r.b, r.a) if swap else (r.a, r.b)
        la, lb = (r.kelas_b, r.kelas_a) if swap else (r.kelas_a, r.kelas_b)
        for tag, p in (("A", A), ("B", B)):
            im = Image.open(ROOT / p).convert("RGB")                     # convert -> tanpa EXIF/metadata
            s = 1024 / max(im.size)
            if s != 1:
                im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
            im.save(PACK / f"pair_{n:02d}_{tag}.jpg", quality=95)
        rel_, dt = relation(m(A), m(B))
        key.append({"no_pasangan": f"{n:02d}", "kategori": cat, "file_A": f"pair_{n:02d}_A.jpg", "file_B": f"pair_{n:02d}_B.jpg",
                    "berkas_asli_A": A, "berkas_asli_B": B, "label_dataset_A": la, "label_dataset_B": lb, "A_B_ditukar": swap,
                    "inlier_SIFT": int(r.inlier), "original_filename_A": m(A)[0], "original_filename_B": m(B)[0],
                    "hubungan_metadata": rel_, "selisih_detik": dt})
        # halaman PDF
        pg = Image.new("RGB", (W, H), "white")
        d = ImageDraw.Draw(pg)
        d.text((60, 30), f"Pasangan {n:02d} dari {len(sel)}", fill="black", font=f_t)
        side = 1040
        for k_, tag in enumerate(("A", "B")):
            x0 = 60 + k_ * (side + 140)
            im = Image.open(PACK / f"pair_{n:02d}_{tag}.jpg")
            im.thumbnail((side, side))
            pg.paste(im, (x0, 110))
            d.text((x0, 110 + side + 10), f"Foto {tag}", fill="black", font=f_s)
            yy = 110 + side + 60
            xx = x0
            for opt in OPTIONS:
                wtxt = d.textlength(opt, font=f_b)
                if xx + 40 + wtxt > x0 + side:
                    xx, yy = x0, yy + 48
                d.rectangle([xx, yy + 4, xx + 28, yy + 32], outline="black", width=3)
                d.text((xx + 38, yy), opt, fill="black", font=f_b)
                xx += 38 + wtxt + 40
            d.text((x0, yy + 60), "Catatan: ______________________________________________", fill="black", font=f_b)
        pages.append(pg)
    pdf = PACK / "paket_pakar.pdf"
    pages[0].save(pdf, save_all=True, append_images=pages[1:], resolution=200)
    pages[1].resize((W // 2, H // 2)).save(RD / "expert_pack_preview_halaman2.png")   # pratinjau tata letak (di luar paket)
    # lembar jawaban xlsx (tanpa label)
    from openpyxl import Workbook
    from openpyxl.worksheet.datavalidation import DataValidation
    wb = Workbook()
    ws = wb.active
    ws.title = "Jawaban"
    ws.append(["no_pasangan", "file_A", "jawaban_A", "file_B", "jawaban_B", "catatan"])
    dv = DataValidation(type="list", formula1='"' + ",".join(OPTIONS) + '"', allow_blank=True)
    ws.add_data_validation(dv)
    for k_ in key:
        ws.append([k_["no_pasangan"], k_["file_A"], "", k_["file_B"], "", ""])
    dv.add(f"C2:C{len(key) + 1}")
    dv.add(f"E2:E{len(key) + 1}")
    for col, w in zip("ABCDEF", (12, 18, 24, 18, 24, 60)):
        ws.column_dimensions[col].width = w
    wb.save(PACK / "lembar_jawaban.xlsx")
    with open(RD / "answer_key.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(key[0]))
        w.writeheader()
        w.writerows(key)
    rep["2_paket_buta"] = {"n_pasangan": len(key), "per_kategori": dict(Counter(k_["kategori"] for k_ in key)),
                           "A_B_ditukar": sum(k_["A_B_ditukar"] for k_ in key), "hubungan_metadata": dict(Counter(k_["hubungan_metadata"] for k_ in key)),
                           "isi_expert_pack": sorted(p.name for p in PACK.iterdir())[:6] + ["..."], "halaman_pdf": len(pages)}
    (RD / "expert_blind_pack.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(rep, indent=1, ensure_ascii=False, default=str))
    if len(dfr):
        print("\nDERET SELISIH NOMOR KONSTAN (teratas):")
        print(dfr.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
