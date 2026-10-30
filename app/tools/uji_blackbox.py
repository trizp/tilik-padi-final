#!/usr/bin/env python3
"""
uji_blackbox.py - Uji black-box lewat jalur yang sama dengan tampilan (core.alur.periksa_foto).
10 foto VAL prov_v7 dipilih ACAK per kelas (seed 20261014), bukan dipilih berdasarkan hasil: 3 Blast, 3 Brown spot, 2 BLB,
2 daun sehat (ds4). Ditambah 4 skenario masukan (turunan foto pertama, dibuat di memori): PNG, EXIF diputar, bukan gambar, >10 MB.
Ditambah skenario penyampaian, foto ds4 VAL yang SENGAJA dipilih dari docs/uji/pindai_val.csv karena memicu fiturnya: kemungkinan lain
+ cara membedakan, tingkat kemungkinan, tidak terdeteksi tepat di bawah ambang rendah, daun sehat, dan foto ulang (alur v9).
Alur v9 (2026-10-20): foto yang lama dipertahankan bila masih memicu fiturnya; bila tidak, dipakai foto ds4 pertama (urutan nama) yang
memenuhi. Untuk semua foto dicatat juga fitur-fitur itu.
Catatan: model aplikasi dilatih dengan train+val+test (v9: prov_v9_deploy), jadi foto VAL termasuk data latihnya. Uji ini memeriksa
PERILAKU SISTEM (2026-10-20): hasil tampil, tingkat dan saran sesuai aturan, foto ulang diminta saat daun tidak jelas, masukan buruk
ditolak. Kebenaran diagnosis TIDAK dinilai di sini; rujukannya hasil evaluasi model (README, "Hasil evaluasi alur aplikasi").

  app/.venv/Scripts/python.exe app/tools/uji_blackbox.py
Keluaran: app/docs/uji/blackbox.md dan blackbox.json
"""
import csv
import io
import json
import random
import sys
from pathlib import Path

from PIL import Image

APP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP))
from core import alur, gambar, inferensi, konfigurasi, pemisah_daun  # noqa: E402
from rbs import mesin  # noqa: E402
from ui.komponen import LABEL_TINGKAT, PENYAKIT  # noqa: E402

PROYEK = APP.parent
VAL = PROYEK / "data/processed/prov_v7/images/val"
SEED = 20261014
JUMLAH = {"leaf_blast": 3, "brown_spot": 3, "bacterial_leaf_blight": 2, "background": 2}
ATURAN_KHUSUS = {"leaf_blast": "R001", "brown_spot": "R003", "bacterial_leaf_blight": "R002"}
HARAPAN_UMUM = ("Hasil tampil dengan tingkat dan saran sesuai aturan: terdeteksi = kotak + D01 dan aturan penyakit yang tampil "
                "(R001/R002/R003); kemungkinan = kotak + K01, K02; tidak terdeteksi = tanpa kotak + N01, N02. Bila daun tidak jelas: "
                "foto ulang diminta, tanpa diagnosis dan tanpa aturan")
FITUR = [  # (foto ds4 VAL, label, harapan perilaku, fungsi pemeriksa) - dipilih dari docs/uji/pindai_val.csv karena memicu fitur, bukan acak
    ("ds4_blast-321_7c914ca8.jpg", "leaf_blast",
     "Terdeteksi dengan kemungkinan lain: penyakit lain (>= ambang rendah, berbeda dari diagnosis) tampil; cara membedakan TERBUKA "
     "karena pasangan Blast dan Brown spot; saran tetap mengikuti diagnosis utama (D01 + aturan penyakitnya)",
     lambda h: h["tingkat"] == "terdeteksi" and h["kemungkinan_lain"] not in (None, h["diagnosis"])
     and {h["diagnosis"], h["kemungkinan_lain"]} == {"leaf_blast", "brown_spot"} and h["cara_membedakan"] == "terbuka"
     and h["aturan"] == f"D01+{ATURAN_KHUSUS[h['diagnosis']]}"),
    ("ds4_blast-104_67def77f.jpg", "leaf_blast",
     "Terdeteksi Blast atau Brown spot tanpa kemungkinan lain: cara membedakan TERLIPAT; kotak digambar; D01 + aturan penyakitnya",
     lambda h: h["tingkat"] == "terdeteksi" and h["diagnosis"] in ("leaf_blast", "brown_spot") and h["kemungkinan_lain"] is None
     and h["cara_membedakan"] == "terlipat" and h["kotak_digambar"] > 0 and h["aturan"] == f"D01+{ATURAN_KHUSUS[h['diagnosis']]}"),
    ("ds4_blast-414_94feb6e6.jpg", "leaf_blast",
     "Tingkat KEMUNGKINAN (keyakinan antara ambang rendah dan ambang terdeteksi): kotak digambar; K01, K02 (konfirmasi ke penyuluh, "
     "tanpa saran obat)",
     lambda h: h["tingkat"] == "kemungkinan" and h["kotak_digambar"] > 0 and h["aturan"] == "K01+K02"),
    ("ds4_brown-spot-2173_3657de07.jpg", "brown_spot",
     "Tidak terdeteksi dengan keyakinan teratas tepat di bawah ambang rendah: tanpa kotak dan tanpa kotak putus-putus (tampilan bercak "
     "berkeyakinan rendah dinonaktifkan di config); N01, N02; tanpa cara membedakan",
     lambda h: h["tingkat"] == "tidak_terdeteksi" and h["bercak_rendah"] is None and h["kotak_digambar"] == 0
     and h["aturan"] == "N01+N02" and h["cara_membedakan"] is None),
    ("ds4_Healthy-1014_c5aa8fd9.jpg", "background",
     "Tidak terdeteksi: tanpa kotak, tanpa bercak berkeyakinan rendah, tanpa cara membedakan; N01, N02",
     lambda h: h["tingkat"] == "tidak_terdeteksi" and h["bercak_rendah"] is None and h["kotak_digambar"] == 0
     and h["aturan"] == "N01+N02" and h["cara_membedakan"] is None),
    ("ds4_Healthy-1432_5998a115.jpg", "background",
     "Daun tidak jelas (pemisah daun tidak menemukan satu helai daun): foto ulang diminta; tidak didiagnosis, tanpa aturan dan kotak",
     lambda h: h["foto_ulang"] and h["aturan"] == "" and h["kotak_digambar"] == 0),
]


def nama(p):
    return PENYAKIT[p]["nama"] if p in PENYAKIT else "tidak ada"


def pilih_foto():
    man = [r for r in csv.DictReader(open(PROYEK / "data/processed/prov_v7/manifest.csv", encoding="utf-8")) if r["split"] == "val"]
    rng = random.Random(SEED)
    out = []
    for kelas, n in JUMLAH.items():
        calon = sorted(r["file"] for r in man if r["class"] == kelas and (kelas != "background" or r["source"] == "ds4"))
        out += [(kelas, f) for f in rng.sample(calon, n)]
    return out


def ringkas(h):
    d = h.diagnosis
    return {"foto_ulang": False, "diagnosis": d.penyakit or "tidak_ada", "tingkat": d.tingkat, "keyakinan": round(d.keyakinan, 4),
            "aturan": "+".join(L.aturan.id for L in h.inferensi.jejak), "kotak_digambar": h.kotak_digambar,
            "kemungkinan_lain": h.kemungkinan_lain.penyakit if h.kemungkinan_lain else None,
            "keyakinan_lain": round(h.kemungkinan_lain.keyakinan, 4) if h.kemungkinan_lain else None,
            "bercak_rendah": round(h.bercak_rendah.keyakinan, 4) if h.bercak_rendah else None,
            "cara_membedakan": h.cara_membedakan}


def periksa(data, cfg, det, aturan, pem):
    """Ringkasan hasil; foto yang diminta foto ulang (alur v9) dicatat foto_ulang=True. Foto yang ditolak tetap melempar FotoDitolak."""
    try:
        return ringkas(alur.periksa_foto(data, cfg, det, aturan, pem))
    except gambar.FotoPerluDiulang:
        return {"foto_ulang": True, "diagnosis": None, "tingkat": None, "keyakinan": None, "aturan": "", "kotak_digambar": 0,
                "kemungkinan_lain": None, "keyakinan_lain": None, "bercak_rendah": None, "cara_membedakan": None}


def perilaku_sesuai(h, batas_tampil_rendah):
    """Perilaku sistem konsisten (bukan kebenaran diagnosis): tingkat -> kotak dan rantai aturan, penyampaian tambahan konsisten."""
    if h["foto_ulang"]:
        return h["aturan"] == "" and h["kotak_digambar"] == 0 and h["tingkat"] is None
    t = h["tingkat"]
    pola = {"terdeteksi": f"D01+{ATURAN_KHUSUS.get(h['diagnosis'], '?')}", "kemungkinan": "K01+K02", "tidak_terdeteksi": "N01+N02"}[t]
    relevan = t != "tidak_terdeteksi" and bool({h["diagnosis"], h["kemungkinan_lain"]} & {"leaf_blast", "brown_spot"})
    pasangan = {h["diagnosis"], h["kemungkinan_lain"]} == {"leaf_blast", "brown_spot"}
    return (h["aturan"] == pola and (h["kotak_digambar"] > 0) == (t != "tidak_terdeteksi")
            and (h["kemungkinan_lain"] is None or (t != "tidak_terdeteksi" and h["kemungkinan_lain"] != h["diagnosis"]))
            and (h["bercak_rendah"] is None if batas_tampil_rendah is None else True)
            and h["cara_membedakan"] == (("terbuka" if pasangan else "terlipat") if relevan else None))


def teks_aktual(h):
    if h["foto_ulang"]:
        return "Diminta foto ulang (tidak didiagnosis)"
    if h["tingkat"] == "tidak_terdeteksi":
        return f"Tidak terdeteksi (keyakinan tertinggi {100 * h['keyakinan']:.1f}%); aturan {h['aturan'].replace('+', ', ')}".replace(".", ",", 1)
    return f"{nama(h['diagnosis'])}, {LABEL_TINGKAT[h['tingkat']].lower()} ({100 * h['keyakinan']:.0f}%); aturan {h['aturan'].replace('+', ', ')}"


def tambahan(h):
    t = []
    if h.get("kemungkinan_lain"):
        t.append(f"kemungkinan lain {nama(h['kemungkinan_lain'])} ({100 * h['keyakinan_lain']:.0f}%)")
    if h.get("cara_membedakan"):
        t.append(f"cara membedakan {h['cara_membedakan']}")
    if h.get("bercak_rendah") is not None:
        t.append(f"bercak berkeyakinan rendah {nama(h['diagnosis'])} ({100 * h['bercak_rendah']:.1f}%)".replace(".", ",", 1))
    return ("; " + "; ".join(t)) if t else ""


def main():
    cfg = konfigurasi.muat()
    det = inferensi.Detektor(cfg.model_path, cfg.imgsz, cfg.conf_min, cfg.iou, cfg.max_det)
    pem = pemisah_daun.PemisahDaun(cfg.pemisah_path, cfg.pemisah_tau) if cfg.pemisah_path else None
    aturan = mesin.muat_aturan()
    baris = []
    foto = pilih_foto()
    for no, (kelas, f) in enumerate(foto, 1):
        h = periksa((VAL / f).read_bytes(), cfg, det, aturan, pem)
        harap = HARAPAN_UMUM
        sesuai = perilaku_sesuai(h, cfg.batas_tampil_rendah)
        aktual = teks_aktual(h) + tambahan(h)
        baris.append({"no": no, "masukan": f"VAL {f} (label: {nama(kelas) if kelas != 'background' else 'daun sehat'})",
                      "harapan": harap, "aktual": aktual, "sesuai": sesuai, **h})

    # skenario masukan khusus (turunan foto pertama, di memori)
    asal_f = foto[0][1]
    asal = (VAL / asal_f).read_bytes()
    ref = periksa(asal, cfg, det, aturan, pem)
    im = Image.open(io.BytesIO(asal)).convert("RGB")
    png = io.BytesIO()
    im.save(png, "PNG")
    rot = io.BytesIO()
    ex = Image.Exif()
    ex[0x0112] = 6                                   # penampil harus memutar 90 derajat searah jarum jam
    im.transpose(Image.Transpose.ROTATE_90).save(rot, "JPEG", quality=95, exif=ex.tobytes())
    khusus = [("Foto no. 1 disimpan ulang sebagai PNG", png.getvalue(), "Hasil sama dengan foto no. 1 (kelas dan tingkat)"),
              ("Foto no. 1 diputar 90 derajat dengan tag EXIF orientasi 6", rot.getvalue(),
               "Orientasi dikoreksi; kelas dan tingkat sama dengan foto no. 1"),
              ("Berkas teks berakhiran .jpg", b"bukan foto", "Ditolak dengan pesan cara memperbaiki"),
              ("Berkas 10,5 MB", b"\xff\xd8" + b"\x00" * int(10.5 * 1024 * 1024), "Ditolak karena melebihi 10 MB")]
    for i, (masuk, data, harap) in enumerate(khusus, len(baris) + 1):
        try:
            h = periksa(data, cfg, det, aturan, pem)
            aktual = ("Diminta foto ulang" if h["foto_ulang"] else
                      f"{nama(h['diagnosis'])}, {LABEL_TINGKAT[h['tingkat']].lower()} ({100 * h['keyakinan']:.0f}%)")
            sesuai = (h["foto_ulang"], h["diagnosis"], h["tingkat"]) == (ref["foto_ulang"], ref["diagnosis"], ref["tingkat"]) \
                and "Ditolak" not in harap
        except gambar.FotoDitolak as e:
            h, aktual = {}, f"Ditolak: \"{e}\""
            sesuai = "Ditolak" in harap
        baris.append({"no": i, "masukan": masuk, "harapan": harap, "aktual": aktual, "sesuai": sesuai, **h})

    for i, (f, kelas, harap, cek) in enumerate(FITUR, len(baris) + 1):
        h = periksa((VAL / f).read_bytes(), cfg, det, aturan, pem)
        aktual = teks_aktual(h)
        lab = "daun sehat" if kelas == "background" else nama(kelas)
        baris.append({"no": i, "masukan": f"VAL {f} (label: {lab}; dipilih karena memicu fitur)", "harapan": harap,
                      "aktual": aktual + tambahan(h), "sesuai": bool(cek(h)) and perilaku_sesuai(h, cfg.batas_tampil_rendah), **h})

    n_foto = sum(1 for b in baris[:10] if b["sesuai"])
    n_khusus = sum(1 for b in baris[10:14] if b["sesuai"])
    n_fitur = sum(1 for b in baris[14:] if b["sesuai"])
    (APP / "docs/uji/blackbox.json").write_text(json.dumps({"seed": SEED, "model": cfg.nama_model, "foto_sesuai": n_foto, "khusus_sesuai": n_khusus,
                                                            "fitur_sesuai": n_fitur, "fitur_n": len(FITUR), "baris": baris},
                                                           indent=2, ensure_ascii=False), encoding="utf-8")
    md = ["# Uji black-box", "",
          f"Jalur: `core.alur.periksa_foto` (sama dengan tampilan). 10 foto VAL prov_v7 dipilih acak per kelas (seed {SEED}),",
          "bukan berdasarkan hasil.", "",
          "Yang diuji adalah **perilaku sistem**: hasil tampil, tingkat dan saran sesuai aturan, kotak hanya pada hasil yang bukan \"tidak",
          "terdeteksi\", penyampaian tambahan konsisten, foto ulang diminta saat daun tidak jelas, dan masukan buruk ditolak dengan pesan.",
          "**Kebenaran diagnosis tidak dinilai di sini**: label foto ditulis di kolom masukan hanya sebagai informasi. Ketepatan diagnosis",
          "dilaporkan pada hasil evaluasi model (README, bagian \"Hasil evaluasi alur aplikasi\"), yang memakai foto uji yang tidak dipakai",
          "untuk melatih model. Model aplikasi dilatih dengan train+val+test, jadi foto VAL di sini termasuk data latihnya.", "",
          f"Model: {cfg.nama_model} (`{cfg.model_path.name}`)"
          + (f", pemisah daun `{cfg.pemisah_path.name}` (tau {f'{cfg.pemisah_tau:.2f}'.replace('.', ',')}). Foto yang tidak "
             "memperlihatkan satu helai daun yang jelas diminta foto ulang dan tidak didiagnosis." if cfg.pemisah_path else "."),
          f"Skenario 15-{14 + len(FITUR)} memakai foto ds4 VAL yang sengaja dipilih dari hasil pemindaian seluruh foto VAL (pindai_val.csv,",
          "hanya di folder proyek skripsi) karena memicu fitur penyampaian (kemungkinan lain, cara membedakan, tingkat kemungkinan, tidak",
          "terdeteksi, foto ulang).", "",
          f"**Foto VAL: perilaku sesuai {n_foto}/10. Skenario masukan khusus: {n_khusus}/4 sesuai. Skenario fitur penyampaian: "
          f"{n_fitur}/{len(FITUR)} sesuai.**", "",
          f"**Perilaku umum (P)**, harapan untuk foto no. 1-10: {HARAPAN_UMUM}. Penyampaian tambahan konsisten: kemungkinan lain hanya "
          "pada hasil yang bukan tidak terdeteksi dan berbeda dari diagnosis; cara membedakan terbuka untuk pasangan Blast dan Brown spot, "
          "terlipat bila Blast atau Brown spot muncul tanpa pasangan itu.", "",
          "| No | Masukan | Hasil diharapkan (perilaku sistem) | Hasil aktual | Sesuai |", "|---|---|---|---|---|"]
    md += [f"| {b['no']} | {b['masukan']} | {'Perilaku umum (P)' if b['harapan'] == HARAPAN_UMUM else b['harapan']} | {b['aktual']} | "
           f"{'ya' if b['sesuai'] else 'TIDAK'} |" for b in baris]
    (APP / "docs/uji/blackbox.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return baris


if __name__ == "__main__":
    main()
