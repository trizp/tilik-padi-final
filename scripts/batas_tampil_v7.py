#!/usr/bin/env python3
"""
batas_tampil_v7.py - Pilih batas TAMPILAN "bercak berkeyakinan rendah" aplikasi dari VAL prov_v7 (KEPUTUSAN_PROYEK.md 2026-10-16).
Parameter tampilan dari val, BUKAN ambang diagnosis. Tanpa menjalankan model, tanpa test/test_new_session/ds3.

- Sumber: runs/<run>/eval_val/diagnosis_dengan_background_per_citra.csv (kotak teratas per foto) dan ambang terkunci run itu.
- Fitur tampil bila foto "tidak terdeteksi" (kotak teratas < ambang) tetapi kotak teratas >= batas.
- Grid batas 0,10 sampai di bawah ambang (langkah 0,01). Untuk tiap batas: jumlah foto sehat dan sakit yang menampilkan fitur, dan
  proporsi sehat = sehat / (sehat + sakit).
- Batas terpilih = batas TERENDAH dengan proporsi sehat <= 25%. Bila tidak ada: fitur dimatikan.

  python scripts/batas_tampil_v7.py --run v7_final_prov_yolov8s_s2 --out reports/final_eval_v7/aplikasi/batas_tampil_val_yolov8s.json
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PENYAKIT = ("leaf_blast", "brown_spot", "bacterial_leaf_blight")
MAKS_SEHAT = 25.0


def hitung(run):
    amb = float(json.loads((ROOT / f"runs/{run}/ambang_diagnosis.json").read_text(encoding="utf-8"))["ambang"])
    rows = list(csv.DictReader(open(ROOT / f"runs/{run}/eval_val/diagnosis_dengan_background_per_citra.csv", encoding="utf-8")))
    conf = lambda r: float(r["conf_teratas"]) if r["conf_teratas"] not in ("", None) else 0.0  # noqa: E731
    tak = [r for r in rows if conf(r) < amb]
    grid = [round(0.01 * i, 2) for i in range(10, 100) if round(0.01 * i, 2) < amb]
    tabel = []
    for b in grid:
        tampil = [r for r in tak if conf(r) >= b]
        sehat = sum(r["label"] not in PENYAKIT for r in tampil)
        sakit = len(tampil) - sehat
        benar = sum(r["label"] in PENYAKIT and r["kelas_teratas"] == r["label"] for r in tampil)
        tabel.append({"batas": b, "sehat": sehat, "sakit": sakit, "sakit_kelas_bercak_benar": benar,
                      "proporsi_sehat_persen": round(100 * sehat / len(tampil), 1) if tampil else None})
    ok = [t for t in tabel if t["proporsi_sehat_persen"] is not None and t["proporsi_sehat_persen"] <= MAKS_SEHAT]
    pilih = ok[0]["batas"] if ok else None
    return {"run": run, "ambang_terdeteksi": amb, "aturan": f"batas terendah dalam [0,10; ambang) dengan proporsi sehat <= {MAKS_SEHAT:.0f}% "
            "di antara foto tidak terdeteksi yang menampilkan bercak berkeyakinan rendah", "batas_terpilih": pilih,
            "fitur_aktif": pilih is not None, "n_val": len(rows), "n_sehat_val": sum(r["label"] not in PENYAKIT for r in rows),
            "n_tidak_terdeteksi": len(tak), "tabel": tabel,
            "sumber": [f"runs/{run}/ambang_diagnosis.json", f"runs/{run}/eval_val/diagnosis_dengan_background_per_citra.csv"],
            "catatan": "parameter TAMPILAN dari val, bukan ambang diagnosis; test, test_new_session, dan ds3 tidak dipakai"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    h = hitung(a.run)
    out = ROOT / a.out
    if out.exists():
        sys.exit(f"[DITOLAK] {out} sudah ada")
    out.write_text(json.dumps(h, indent=2, ensure_ascii=False), encoding="utf-8")
    md = [f"# Batas tampilan bercak berkeyakinan rendah ({a.run})", "", "**Parameter tampilan dari VAL v7, bukan ambang diagnosis.** "
          f"Ambang terdeteksi {h['ambang_terdeteksi']}; {h['n_tidak_terdeteksi']} dari {h['n_val']} foto val tidak terdeteksi.", "",
          f"Aturan: {h['aturan']}.", "", f"**Batas terpilih: {h['batas_terpilih'] if h['fitur_aktif'] else 'TIDAK ADA (fitur dimatikan)'}**", "",
          "| Batas | Sehat | Sakit (kelas bercak benar) | Proporsi sehat |", "|---|---|---|---|"]
    md += [f"| {t['batas']:.2f} | {t['sehat']} | {t['sakit']} ({t['sakit_kelas_bercak_benar']}) | "
           f"{'-' if t['proporsi_sehat_persen'] is None else str(t['proporsi_sehat_persen']) + '%'}"
           f"{' <- terpilih' if t['batas'] == h['batas_terpilih'] else ''} |" for t in h["tabel"]]
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
