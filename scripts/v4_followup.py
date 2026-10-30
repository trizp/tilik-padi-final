#!/usr/bin/env python3
"""
v4_followup.py - Bahan pemeriksaan untuk prov_v4 (tidak mengubah dataset, tidak mengubah label apa pun):
  A. merged_by_threshold_5_15.jpg : 10 pasangan acak yang SEKARANG tergabung karena ambang sebaran turun (sebaran 5%-15%)
  B. rejected_below_5.jpg          : 10 pasangan acak yang tetap DITOLAK (sebaran < 5%)
  C. pasangan_lintas_kelas_untuk_pakar.xlsx : 40 pasangan lintas kelas untuk ditinjau pakar (kolom pakar KOSONG)
Keluaran tambahan: reports/prov_v4/v4_followup.json
"""
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
RD = ROOT / "reports/prov_v4"


def montage(pairs, captions, out, per_row, tile):
    W = 30
    rows = (len(pairs) + per_row - 1) // per_row
    cv = Image.new("RGB", (per_row * 2 * tile, rows * (tile + W)), "white")
    dd = ImageDraw.Draw(cv)
    for k, (a, b) in enumerate(pairs):
        x0, y0 = (k % per_row) * 2 * tile, (k // per_row) * (tile + W)
        dd.text((x0 + 3, y0 + 3), captions[k][0], fill="black")
        dd.text((x0 + 3, y0 + 16), captions[k][1], fill="black")
        for t, p in enumerate((a, b)):
            cv.paste(Image.open(ROOT / p).convert("RGB").resize((tile, tile)), (x0 + t * tile, y0 + W))
    cv.save(out, quality=88)


def pick(df, n, rnd, used):
    """Pilih n pasangan: sebisanya setengah ds1~ds1 dan setengah ds4~ds4; tiap gambar dipakai sekali (juga lintas pemanggilan)."""
    out = []
    for want, mask in (((n + 1) // 2, (df.sumber_a == "ds1") & (df.sumber_b == "ds1")),
                       (n, (df.sumber_a == "ds4") & (df.sumber_b == "ds4")),
                       (n, df.index == df.index)):
        idx = list(df[mask].index)
        rnd.shuffle(idx)
        got = 0
        for i in idx:
            r = df.loc[i]
            if r.a in used or r.b in used or i in out:
                continue
            out.append(i)
            used |= {r.a, r.b}
            got += 1
            if got >= want or len(out) >= n:
                break
        if len(out) >= n:
            break
    return out


def main():
    rep = {}
    rnd = random.Random(11)
    sp = pd.read_csv(RD / "sift_pairs.csv")
    thr_new = 0.05
    now = sp[(sp.status == "diterima") & (sp.frac_kotak_inlier >= thr_new) & (sp.frac_kotak_inlier < 0.15)]
    rej = sp[sp.status == "ditolak_sebaran_kecil"]
    rep["A_tergabung_karena_ambang_turun"] = {"n_pasangan": int(len(now)), "inlier_median": float(now.inlier.median()) if len(now) else None,
                                              "per_sumber": {f"{a}~{b}": int(n) for (a, b), n in now.groupby(["sumber_a", "sumber_b"]).size().items()}}
    rep["B_tetap_ditolak"] = {"n_pasangan": int(len(rej)), "inlier_median": float(rej.inlier.median()) if len(rej) else None,
                              "frac_median": float(rej.frac_kotak_inlier.median()) if len(rej) else None,
                              "per_sumber": {f"{a}~{b}": int(n) for (a, b), n in rej.groupby(["sumber_a", "sumber_b"]).size().items()}}
    for name, df, out, tag in (("A", now, "merged_by_threshold_5_15.jpg", "TERGABUNG (5-15%)"), ("B", rej, "rejected_below_5.jpg", "DITOLAK (<5%)")):
        idx = list(df.index)
        rnd.shuffle(idx)
        idx = idx[:10]
        montage([(df.loc[i].a, df.loc[i].b) for i in idx],
                [(f"{tag} inlier={df.loc[i].inlier} sebaran={df.loc[i].frac_kotak_inlier}", f"{df.loc[i].kelas_a}/{df.loc[i].sumber_a} ~ {df.loc[i].kelas_b}/{df.loc[i].sumber_b}") for i in idx],
                RD / out, 2, 300)
        rep[f"{name}_ditampilkan"] = [{"a": Path(df.loc[i].a).name, "b": Path(df.loc[i].b).name, "inlier": int(df.loc[i].inlier), "sebaran": float(df.loc[i].frac_kotak_inlier)} for i in idx]

    # ---- C. xlsx untuk pakar
    from openpyxl import Workbook
    from openpyxl.drawing.image import Image as XImage
    from openpyxl.styles import Alignment, Font, PatternFill
    cc = sp[(sp.status == "diterima") & (sp.kelas_a != sp.kelas_b)]
    quarantined = set()
    qf = RD / "quarantine.csv"
    if qf.exists():
        quarantined = {r["berkas"] for r in csv.DictReader(open(qf, encoding="utf-8"))}
    used = set()
    hi = pick(cc[cc.inlier > 100], 20, rnd, used)
    lo = pick(cc[(cc.inlier >= 21) & (cc.inlier <= 100)], 20, rnd, used)
    sel = [(cc.loc[i], "inlier>100") for i in hi] + [(cc.loc[i], "inlier 21-100") for i in lo]
    rnd.shuffle(sel)
    wb = Workbook()
    ws = wb.active
    ws.title = "Pasangan"
    head = ["no_pasangan", "berkas_a", "berkas_b", "label_a_di_dataset", "label_b_di_dataset", "sumber_a", "sumber_b", "kecocokan_SIFT_inlier",
            "kelompok_kecocokan", "status_karantina_prov_v4", "foto_a", "foto_b", "label_menurut_pakar", "catatan_pakar"]
    ws.append(head)
    for c in ws[1]:
        c.font = Font(bold=True)
        c.fill = PatternFill("solid", fgColor="DDEBF7")
        c.alignment = Alignment(wrap_text=True, vertical="center")
    thumbs = ROOT / "data/interim/_xlsx_thumbs"
    thumbs.mkdir(parents=True, exist_ok=True)
    tmpfiles = []
    for n, (r, grp) in enumerate(sel, 1):
        both_q = (r.a in quarantined) and (r.b in quarantined)
        ws.append([n, r.a, r.b, r.kelas_a, r.kelas_b, r.sumber_a, r.sumber_b, int(r.inlier), grp,
                   "kedua foto dikarantina" if both_q else ("sebagian dikarantina" if (r.a in quarantined or r.b in quarantined) else "tidak dikarantina"), "", "", "", ""])
        row = ws.max_row
        ws.row_dimensions[row].height = 112
        for col, path in (("K", r.a), ("L", r.b)):
            tp = thumbs / f"{n}_{col}.jpg"
            Image.open(ROOT / path).convert("RGB").resize((140, 140)).save(tp, quality=80)
            tmpfiles.append(tp)
            img = XImage(str(tp))
            img.width, img.height = 140, 140
            ws.add_image(img, f"{col}{row}")
    widths = {"A": 11, "B": 38, "C": 38, "D": 20, "E": 20, "F": 9, "G": 9, "H": 12, "I": 14, "J": 22, "K": 21, "L": 21, "M": 34, "N": 40}
    for k_, v_ in widths.items():
        ws.column_dimensions[k_].width = v_
    for rr in ws.iter_rows(min_row=2):
        for c in rr:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "B2"
    pt = wb.create_sheet("Petunjuk")
    for line in ["PETUNJUK UNTUK PAKAR (Bahasa Indonesia sederhana)",
                 "Setiap baris = dua foto yang dicurigai SAMA atau sangat mirip, tetapi di dataset diberi label berbeda.",
                 "Kolom 'foto_a' dan 'foto_b' menampilkan kedua foto. Nama berkas lengkap ada di kolom berkas_a dan berkas_b.",
                 "Kolom 'kecocokan_SIFT_inlier': makin besar angkanya, makin besar kemungkinan itu foto yang sama (di atas 100 hampir pasti sama).",
                 "Yang perlu diisi: 'label_menurut_pakar' dan 'catatan_pakar' (keduanya sengaja KOSONG).",
                 "Saran cara mengisi 'label_menurut_pakar': tulis 'A=<penyakit>; B=<penyakit>' atau 'foto sama = <penyakit>' atau 'tidak yakin'.",
                 "Pilihan penyakit: leaf_blast, brown_spot, bacterial_leaf_blight, sehat (healthy), dua_penyakit, lainnya.",
                 "Label asli di dataset TIDAK diubah oleh berkas ini; hasil penilaian Anda hanya dipakai sebagai masukan keputusan.",
                 "Kolom 'status_karantina_prov_v4': foto yang dikarantina tidak dipakai untuk train/val/test sampai ada keputusan label."]:
        pt.append([line])
    pt.column_dimensions["A"].width = 130
    pt["A1"].font = Font(bold=True, size=13)
    xp = RD / "pasangan_lintas_kelas_untuk_pakar.xlsx"
    wb.save(xp)
    for tp in tmpfiles:
        tp.unlink(missing_ok=True)
    try:
        thumbs.rmdir()
    except OSError:
        pass
    rep["C_xlsx"] = {"berkas": str(xp.relative_to(ROOT)), "n_pasangan": len(sel), "per_kelompok": dict(Counter(g for _, g in sel)),
                     "per_sumber": dict(Counter(f"{r.sumber_a}~{r.sumber_b}" for r, _ in sel)),
                     "per_pasangan_kelas": dict(Counter("~".join(sorted((r.kelas_a, r.kelas_b))) for r, _ in sel)),
                     "kolom": head}
    (RD / "v4_followup.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(rep, indent=1, ensure_ascii=False, default=str)[:6000])


if __name__ == "__main__":
    main()
