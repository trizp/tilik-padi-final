#!/usr/bin/env python3
"""
bekukan_verifikasi_ds3.py - Bekukan hasil verifikasi ds3 putaran 2 dan susun DAFTAR UJI UTAMA FINAL (sebelum model dijalankan
pada ds3). Aturan (KEPUTUSAN 2026-10-08 dan 2026-10-09): uji utama = foto yang dinilai "sesuai" oleh KEDUA penilai pada hasil
gabungan (putaran 1 untuk foto non-ragu, putaran 2 untuk foto yang dinilai ulang); penilai 2: "sesuai" = diagnosis buta sama
dengan label asli. Foto yang belum dinilai ulang tidak berstatus "sesuai".

Nilai gabungan dihitung ULANG dari kolom mentah (penilai 1: E dan L; penilai 2: jawaban di berkas penilai 2, kolom C dan F)
lalu dicocokkan dengan kolom rumus Q dan R di berkas utama.
Keluaran:
  data/annotation/verifikasi_ds3/putaran2/  salinan hanya-baca kedua berkas + SHA256.txt + laporan_putaran2.{json,md}
  data/annotation/verifikasi_ds3/uji_utama_final/  ds3_uji_utama.txt (jalur<TAB>group_id), ds3_tambahan_160.txt,
      ds3_tungro_luar_cakupan.txt, uji_utama_rinci.csv, SHA256.txt
"""
import csv
import hashlib
import json
import os
import shutil
import stat
from collections import Counter
from datetime import datetime
from pathlib import Path

from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data/annotation/verifikasi_ds3"
P2D, UD = D / "putaran2", D / "uji_utama_final"
N = ["sesuai", "tidak sesuai", "ragu"]
HARAP = {"blight": "BLB", "blast": "Blast"}
KODE = ["BURAM", "TIDAK_COCOK", "DUA_PENYAKIT", "MIRIP_BROWN_SPOT", "KRITERIA_KURANG"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ubah(diag, kelas):
    if not diag:
        return ""
    return "ragu" if diag == "ragu" else ("sesuai" if diag == HARAP[kelas] else "tidak sesuai")


def kappa(pairs):
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    c1, c2 = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(c1[k] * c2[k] for k in N) / n / n
    m = {a: {b: sum(1 for x, y in pairs if x == a and y == b) for b in N} for a in N}
    return {"n": n, "po": round(po, 4), "pe": round(pe, 4), "kappa": round((po - pe) / (1 - pe), 4) if pe < 1 else None, "matriks": m}


def main():
    for d in (P2D, UD):
        if d.exists():
            raise SystemExit(f"[DITOLAK] {d} sudah ada (sudah dibekukan)")
    main_f, p2_f = D / "verifikasi_ds3.xlsx", D / "verifikasi_ds3_penilai2.xlsx"
    v = load_workbook(main_f, data_only=True, read_only=True)["Verifikasi"]
    rows = list(v.iter_rows(min_row=2, max_row=161, max_col=18, values_only=True))
    pn = load_workbook(p2_f, read_only=True)["Penilaian"]
    p2 = {r[1]: r for r in pn.iter_rows(min_row=2, max_row=161, max_col=7, values_only=True)}
    kunci = {r["kode"]: r for r in csv.DictReader(open(D / "kunci_kode_penilai2.csv", encoding="utf-8"))}
    data, beda_rumus = [], 0
    for r in rows:
        no, berkas, kelas, _, e, _, g_rumus, _, grup, kode, _, l, m, _, o, _, q_rumus, r_rumus = r
        pr = p2[kode]
        d1, d2, kode2 = pr[2], pr[5], pr[6]
        p1_r1, p1_r2 = e, (l or "")
        p2_r1, p2_r2 = ubah(d1, kelas), ubah(d2, kelas)
        g1 = p1_r2 if p1_r1 == "ragu" else p1_r1
        g2 = p2_r2 if p2_r1 == "ragu" else p2_r1
        beda_rumus += (g1 != (q_rumus or "")) or (g2 != (r_rumus or "")) or (p2_r1 != (g_rumus or ""))
        assert kunci[kode]["no"] == str(no) and kunci[kode]["label_asli_ds3"] == kelas, kode
        data.append({"no": no, "berkas": kunci[kode]["berkas"], "kelas": kelas, "group_id": grup, "kode_p2": kode,
                     "p1_putaran1": p1_r1, "p1_putaran2": p1_r2, "p1_kode": m or "", "p1_gabungan": g1,
                     "p2_diag_putaran1": d1, "p2_diag_putaran2": d2 or "", "p2_kode": kode2 or "", "p2_putaran1": p2_r1, "p2_gabungan": g2,
                     "belum_dinilai_ulang_p1": p1_r1 == "ragu" and not p1_r2, "belum_dinilai_ulang_p2": p2_r1 == "ragu" and not d2,
                     "uji_utama": g1 == "sesuai" and g2 == "sesuai"})
    if beda_rumus:
        raise SystemExit(f"[GAGAL] {beda_rumus} baris: hitungan ulang berbeda dengan kolom rumus Excel")

    lap = {"dibuat": datetime.now().isoformat(timespec="seconds"), "status": "BELUM ada model yang dijalankan pada ds3",
           "aturan": "uji utama = sesuai menurut kedua penilai pada gabungan putaran 1 (non-ragu) dan putaran 2 (yang dinilai ulang)",
           "hitungan_ulang_cocok_dengan_rumus_excel": True,
           "belum_dinilai_ulang": {"penilai1": [x["no"] for x in data if x["belum_dinilai_ulang_p1"]],
                                   "penilai2": [x["no"] for x in data if x["belum_dinilai_ulang_p2"]]},
           "per_kelas": {}, "kesepakatan_gabungan": {}, "kesepakatan_putaran1": {}, "kode_alasan_putaran2": {}}
    for kel in ("blight", "blast", None):
        rs = [x for x in data if kel is None or x["kelas"] == kel]
        nm = kel or "semua"
        lap["per_kelas"][nm] = {"penilai1_gabungan": dict(Counter(x["p1_gabungan"] or "belum dinilai ulang" for x in rs)),
                                "penilai2_gabungan": dict(Counter(x["p2_gabungan"] or "belum dinilai ulang" for x in rs)),
                                "diagnosis_buta_p2_putaran2": dict(Counter(x["p2_diag_putaran2"] for x in rs if x["p2_putaran1"] == "ragu")),
                                "uji_utama_foto": sum(x["uji_utama"] for x in rs),
                                "uji_utama_kelompok": len({x["group_id"] for x in rs if x["uji_utama"]})}
        lap["kesepakatan_gabungan"][nm] = kappa([(x["p1_gabungan"], x["p2_gabungan"]) for x in rs if x["p1_gabungan"] in N and x["p2_gabungan"] in N])
        lap["kesepakatan_putaran1"][nm] = kappa([(x["p1_putaran1"], x["p2_putaran1"]) for x in rs])
    for pen, kk, nil in (("penilai1", "p1_kode", "p1_putaran2"), ("penilai2", "p2_kode", "p2_diag_putaran2")):
        lap["kode_alasan_putaran2"][pen] = {kel: {k: sum(1 for x in data if x["kelas"] == kel and x[kk] == k) for k in KODE} for kel in ("blight", "blast")}
        lap["kode_alasan_putaran2"][pen]["kode_wajib_kosong"] = sum(1 for x in data if x[nil] in ("ragu", "tidak sesuai", "bukan keduanya") and not x[kk])
    lap["DUA_PENYAKIT"] = {"penilai1": sum(1 for x in data if x["p1_kode"] == "DUA_PENYAKIT"), "penilai2": sum(1 for x in data if x["p2_kode"] == "DUA_PENYAKIT"),
                           "keduanya": sum(1 for x in data if x["p1_kode"] == "DUA_PENYAKIT" and x["p2_kode"] == "DUA_PENYAKIT")}

    # ---- bekukan berkas verifikasi
    P2D.mkdir(parents=True)
    tujuan = [P2D / "verifikasi_ds3_putaran2_penilai1.xlsx", P2D / "verifikasi_ds3_putaran2_penilai2.xlsx"]
    shutil.copy2(main_f, tujuan[0])
    shutil.copy2(p2_f, tujuan[1])
    (P2D / "laporan_putaran2.json").write_text(json.dumps(lap, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(P2D / "verifikasi_putaran2_per_foto.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)
    # ---- daftar uji utama final + analisis tambahan + tungro
    UD.mkdir(parents=True)
    utama = [x for x in data if x["uji_utama"]]
    (UD / "ds3_uji_utama.txt").write_text("".join(f"{x['berkas']}\t{x['group_id']}\n" for x in sorted(utama, key=lambda x: x["no"])), encoding="utf-8")
    (UD / "ds3_tambahan_160.txt").write_text("".join(f"{x['berkas']}\t{x['group_id']}\n" for x in sorted(data, key=lambda x: x["no"])), encoding="utf-8")
    shutil.copy2(ROOT / "reports/prov_v6/ds3_eksternal/ds3_bersih_tungro.txt", UD / "ds3_tungro_luar_cakupan.txt")
    with open(UD / "uji_utama_rinci.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["no", "berkas", "kelas", "group_id", "p1_gabungan", "p2_gabungan", "sha256_foto"])
        w.writeheader()
        for x in sorted(utama, key=lambda x: x["no"]):
            w.writerow({k: x[k] for k in ("no", "berkas", "kelas", "group_id", "p1_gabungan", "p2_gabungan")} | {"sha256_foto": sha(ROOT / x["berkas"])})
    for d in (P2D, UD):
        lines = [f"{sha(p)}  {p.name}" for p in sorted(d.iterdir()) if p.is_file() and p.name != "SHA256.txt"]
        (d / "SHA256.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        for p in d.iterdir():
            os.chmod(p, stat.S_IREAD)
    print(json.dumps({k: lap[k] for k in ("belum_dinilai_ulang", "per_kelas", "DUA_PENYAKIT", "kode_alasan_putaran2")}, indent=1, ensure_ascii=False))
    print("kesepakatan gabungan:", {k: (v["n"], v["po"], v["kappa"]) for k, v in lap["kesepakatan_gabungan"].items()})
    print("kesepakatan putaran1:", {k: (v["n"], v["po"], v["kappa"]) for k, v in lap["kesepakatan_putaran1"].items()})
    print("matriks gabungan semua:", lap["kesepakatan_gabungan"]["semua"]["matriks"])


if __name__ == "__main__":
    main()
