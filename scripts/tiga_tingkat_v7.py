#!/usr/bin/env python3
"""
tiga_tingkat_v7.py - Kunci parameter tiga tingkat jawaban aplikasi dari VAL prov_v7 (berkas per citra yang sudah ada; tanpa
menjalankan model, tanpa test/ds3). Prosedur sama dengan keputusan 2026-10-13 (reports/final_eval_v7/aplikasi/tiga_tingkat_val.json):

- ambang_terdeteksi = ambang diagnosis terkunci model final (runs/<run>/ambang_diagnosis.json);
- t_rendah = ambang terbesar (grid 0,01) sehingga >= 90% citra val berpenyakit (termasuk BLB provisional) punya confidence
  kotak teratas >= t_rendah.

  python scripts/tiga_tingkat_v7.py --run v7_final_prov_yolo11s_s2 --cek reports/final_eval_v7/aplikasi/tiga_tingkat_val.json
  python scripts/tiga_tingkat_v7.py --run v7_final_prov_yolov8s_s2 --out reports/final_eval_v7/aplikasi/tiga_tingkat_val_yolov8s.json
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PENYAKIT = ("leaf_blast", "brown_spot", "bacterial_leaf_blight")


def tingkat(c, terdeteksi, rendah):
    return "terdeteksi" if c >= terdeteksi else ("kemungkinan" if c >= rendah else "tidak terdeteksi")


def hitung(run):
    amb = json.loads((ROOT / f"runs/{run}/ambang_diagnosis.json").read_text(encoding="utf-8"))
    terdeteksi = float(amb["ambang"])
    rows = list(csv.DictReader(open(ROOT / f"runs/{run}/eval_val/diagnosis_dengan_background_per_citra.csv", encoding="utf-8")))
    sakit = [r for r in rows if r["label"] in PENYAKIT]
    sehat = [r for r in rows if r["label"] not in PENYAKIT]
    conf = lambda r: float(r["conf_teratas"]) if r["conf_teratas"] not in ("", None) else 0.0  # noqa: E731
    grid = [round(0.01 * i, 2) for i in range(1, 100)]
    kurva = {t: round(100 * sum(conf(r) >= t for r in sakit) / len(sakit), 1) for t in grid}
    kurva_tepat = {t: 100 * sum(conf(r) >= t for r in sakit) / len(sakit) for t in grid}
    rendah = max(t for t in grid if kurva_tepat[t] >= 90.0 and t <= terdeteksi)
    persen = lambda xs, k: round(100 * sum(1 for r in xs if tingkat(conf(r), terdeteksi, rendah) == k) / len(xs), 2)  # noqa: E731
    pk = {}
    for kel in PENYAKIT:
        xs = [r for r in sakit if r["label"] == kel]
        pk[kel] = {k: round(100 * sum(1 for r in xs if tingkat(conf(r), terdeteksi, rendah) == k) / len(xs), 1)
                   for k in ("terdeteksi", "kemungkinan", "tidak terdeteksi")}
    lolos = [r for r in sakit if conf(r) >= rendah]
    benar = sum(1 for r in lolos if r["kelas_teratas"] == r["label"])
    return {
        "model": run, "ambang_terdeteksi": terdeteksi, "t_rendah": rendah,
        "aturan": "t_rendah = ambang terbesar (grid 0,01) sehingga >=90% citra val berpenyakit (termasuk BLB provisional) punya confidence kotak teratas >= t_rendah",
        "n_val": len(rows), "n_berpenyakit": len(sakit), "n_sehat": len(sehat),
        "persen_berpenyakit_terdeteksi_atau_kemungkinan_pada_t_rendah": round(kurva_tepat[rendah], 2),
        "persen_berpenyakit_pada_t_rendah_plus_0.01": round(kurva_tepat[round(rendah + 0.01, 2)], 2),
        "berpenyakit_per_tingkat_persen": {k: persen(sakit, k) for k in ("terdeteksi", "kemungkinan", "tidak terdeteksi")},
        "sehat_per_tingkat_persen": {k: persen(sehat, k) for k in ("terdeteksi", "kemungkinan", "tidak terdeteksi")},
        "berpenyakit_per_kelas_per_tingkat_persen": pk,
        "kelas_teratas_benar_pada_berpenyakit_terdeteksi_atau_kemungkinan_persen": round(100 * benar / len(lolos), 1),
        "kurva_persen_berpenyakit_ge_t": {str(t): v for t, v in kurva.items()},
        "sumber": [f"runs/{run}/ambang_diagnosis.json", f"runs/{run}/eval_val/diagnosis_dengan_background_per_citra.csv"],
        "catatan": "dihitung dari berkas per citra VAL yang sudah ada; test, test_new_session, dan ds3 tidak dipakai",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--cek", default=None, help="berkas acuan yang harus sama (uji reproduksi)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    h = hitung(a.run)
    if a.cek:
        ref = json.loads((ROOT / a.cek).read_text(encoding="utf-8"))
        kunci = ["ambang_terdeteksi", "t_rendah", "n_val", "n_berpenyakit", "n_sehat", "persen_berpenyakit_terdeteksi_atau_kemungkinan_pada_t_rendah",
                 "persen_berpenyakit_pada_t_rendah_plus_0.01", "berpenyakit_per_tingkat_persen", "sehat_per_tingkat_persen",
                 "berpenyakit_per_kelas_per_tingkat_persen", "kelas_teratas_benar_pada_berpenyakit_terdeteksi_atau_kemungkinan_persen"]
        beda = {k: (ref.get(k), h.get(k)) for k in kunci if ref.get(k) != h.get(k)}
        print("REPRODUKSI", "SAMA" if not beda else f"BEDA: {beda}")
        sys.exit(0 if not beda else 1)
    out = ROOT / a.out
    if out.exists():
        sys.exit(f"[DITOLAK] {out} sudah ada")
    out.write_text(json.dumps(h, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in h.items() if k != "kurva_persen_berpenyakit_ge_t"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
