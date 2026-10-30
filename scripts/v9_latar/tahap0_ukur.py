#!/usr/bin/env python3
"""
tahap0_ukur.py - Ukur metode pemisahan daun pada sampel tahap 0 (±100 foto VAL v7) dan buat montase 20 contoh per metode.

Per metode (masker mentah -> pasca-proses bersama di umum.pasca_proses):
- retensi kotak: % kotak bercak yang >= 50% luasnya di dalam masker (label HANYA dipakai di sini, untuk mengukur);
- foto terpotong parah (proksi, tanpa masker acuan): foto berkotak dengan < 50% total luas kotak di dalam masker, ATAU masker < 5% foto;
- luas masker (% foto): makin kecil = makin banyak latar dibuang (tidak ada acuan latar, jadi dibaca bersama montase);
- waktu per foto (median), perangkat dicatat.

  .venv/Scripts/python scripts/v9_latar/tahap0_ukur.py rembg_u2net rembg_isnet sam2 warna
Keluaran: reports/v9_latar/tahap0_ringkasan.{json,md}, reports/v9_latar/tahap0_per_foto.csv, reports/v9_latar/montase_tahap0_<metode>.jpg
"""
import csv
import json
import random
import statistics as st
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import umum  # noqa: E402

NAMA_KELAS = {0: "Blast", 1: "Brown spot", 2: "BLB"}
STRATUM = {("ds4", "leaf_blast"): "ds4 Blast", ("ds4", "brown_spot"): "ds4 Brown spot", ("ds4", "background"): "ds4 sehat",
           ("ds1_provisional", "bacterial_leaf_blight"): "ds1 BLB"}


def ukur(metode, sampel):
    d = umum.KERJA / metode
    waktu = json.loads((d / "waktu.json").read_text(encoding="utf-8"))
    baris, ret_kotak = [], []
    for f, src, kls in sampel:
        bgr = cv2.imread(str(umum.VAL_IMG / f))
        h, w = bgr.shape[:2]
        mm = cv2.imread(str(d / "masker_mentah" / (Path(f).stem + ".png")), cv2.IMREAD_GRAYSCALE)
        if mm.shape != (h, w):
            mm = cv2.resize(mm, (w, h), interpolation=cv2.INTER_LINEAR)
        m = umum.pasca_proses(mm >= 128)
        kotak = umum.baca_kotak(f, w, h)
        r = umum.retensi_kotak(m, kotak)
        ret_kotak += [(STRATUM[(src, kls)], c, v) for c, v, _ in r]
        luas_total = sum(a for _, _, a in r)
        ret_luas = sum(v * a for _, v, a in r) / luas_total if luas_total else None
        luas_m = float(m.mean())
        parah = (luas_m < 0.05) or (ret_luas is not None and ret_luas < 0.5)
        baris.append({"metode": metode, "berkas": f, "stratum": STRATUM[(src, kls)], "n_kotak": len(r),
                      "kotak_ge50": sum(v >= 0.5 for _, v, _ in r), "retensi_luas_kotak": None if ret_luas is None else round(ret_luas, 4),
                      "luas_masker": round(luas_m, 4), "terpotong_parah": parah, "waktu_s": waktu.get(f)})
    return baris, ret_kotak


def ringkas(baris, ret_kotak):
    def r1(x):
        return round(100 * x, 1)
    out = {}
    for grup in ["semua"] + list(STRATUM.values()):
        b = baris if grup == "semua" else [x for x in baris if x["stratum"] == grup]
        k = ret_kotak if grup == "semua" else [x for x in ret_kotak if x[0] == grup]
        out[grup] = {"n_foto": len(b), "n_kotak": len(k),
                     "persen_kotak_ge50": r1(sum(v >= 0.5 for _, _, v in k) / len(k)) if k else None,
                     "persen_foto_terpotong_parah": r1(sum(x["terpotong_parah"] for x in b) / len(b)),
                     "median_luas_masker_persen": r1(st.median(x["luas_masker"] for x in b)),
                     "median_waktu_s": round(st.median(x["waktu_s"] for x in b if x["waktu_s"] is not None), 3)}
    return out


def ubin(img, ukuran=200):
    h, w = img.shape[:2]
    s = ukuran / max(h, w)
    kecil = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    kanvas = np.full((ukuran, ukuran, 3), 255, np.uint8)
    y0, x0 = (ukuran - kecil.shape[0]) // 2, (ukuran - kecil.shape[1]) // 2
    kanvas[y0:y0 + kecil.shape[0], x0:x0 + kecil.shape[1]] = kecil
    return kanvas


def montase(metode, contoh):
    d = umum.KERJA / metode
    sel = []
    for f, src, kls in contoh:
        bgr = cv2.imread(str(umum.VAL_IMG / f))
        h, w = bgr.shape[:2]
        mm = cv2.imread(str(d / "masker_mentah" / (Path(f).stem + ".png")), cv2.IMREAD_GRAYSCALE)
        if mm.shape != (h, w):
            mm = cv2.resize(mm, (w, h))
        m = umum.pasca_proses(mm >= 128)
        hasil = umum.terapkan(bgr[..., ::-1], m)[..., ::-1].copy()
        asli = bgr.copy()
        t = max(2, round(max(h, w) / 250))
        for c, x1, y1, x2, y2 in umum.baca_kotak(f, w, h):        # kotak digambar HANYA untuk dilihat (evaluasi)
            for im in (asli, hasil):
                cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), t)
        mk = np.repeat((m.astype(np.uint8) * 255)[..., None], 3, axis=2)
        trip = np.hstack([ubin(asli), ubin(mk), ubin(hasil)])
        cv2.putText(trip, f"{STRATUM[(src, kls)]}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(trip, f"{STRATUM[(src, kls)]}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        sel.append(np.pad(trip, ((4, 4), (4, 4), (0, 0)), constant_values=255))
    baris = [np.hstack(sel[i:i + 2]) for i in range(0, len(sel), 2)]
    judul = np.full((40, baris[0].shape[1], 3), 255, np.uint8)
    cv2.putText(judul, f"{metode}: asli (kotak merah = label, hanya untuk dilihat) | masker | hasil (latar abu 114)", (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.imwrite(str(umum.LAPORAN / f"montase_tahap0_{metode}.jpg"), np.vstack([judul] + baris), [cv2.IMWRITE_JPEG_QUALITY, 88])


def main():
    metode_list = sys.argv[1:]
    sampel = [tuple(x) for x in json.loads((umum.LAPORAN / "sampel_tahap0.json").read_text(encoding="utf-8"))]
    rng = random.Random(umum.SEED + 1)
    contoh = []
    for s in STRATUM:
        contoh += rng.sample([x for x in sampel if (x[1], x[2]) == s], 5)       # 20 contoh tetap untuk semua metode
    semua, ring = [], {}
    for mt in metode_list:
        b, k = ukur(mt, sampel)
        semua += b
        ring[mt] = ringkas(b, k)
        ring[mt]["perangkat"] = json.loads((umum.KERJA / mt / "waktu.json").read_text(encoding="utf-8")).get("perangkat")
        montase(mt, contoh)
        print(mt, ring[mt]["semua"], flush=True)
    with open(umum.LAPORAN / "tahap0_per_foto.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(semua[0]))
        w.writeheader()
        w.writerows(semua)
    (umum.LAPORAN / "tahap0_ringkasan.json").write_text(json.dumps({"LABEL": "uji coba VAL v7 (tanpa training)", "contoh_montase": [c[0] for c in contoh],
                                                                     "metode": ring}, indent=2, ensure_ascii=False), encoding="utf-8")
    md = ["# Tahap 0: uji coba pemisahan daun (VAL v7, ±100 foto, tanpa training)", "",
          "Masker dibuat TANPA label; kotak bercak hanya dipakai untuk mengukur. Latar diganti abu-abu (114,114,114); kotak tidak diubah.",
          "Terpotong parah (proksi): foto berkotak dengan < 50% total luas kotak di dalam masker, atau masker < 5% foto.", "",
          "| Metode | Kelompok | Foto | Kotak | Kotak >= 50% di daun | Foto terpotong parah | Median luas masker | Median waktu/foto |",
          "|---|---|---|---|---|---|---|---|"]
    for mt in metode_list:
        for g, v in ring[mt].items():
            if g == "perangkat":
                continue
            md.append(f"| {mt} | {g} | {v['n_foto']} | {v['n_kotak']} | {v['persen_kotak_ge50']}% | {v['persen_foto_terpotong_parah']}% | "
                      f"{v['median_luas_masker_persen']}% | {v['median_waktu_s']} s ({ring[mt]['perangkat']}) |")
    (umum.LAPORAN / "tahap0_ringkasan.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
