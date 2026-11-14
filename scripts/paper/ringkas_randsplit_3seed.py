#!/usr/bin/env python3
"""Ringkasan split acak 3 seed (revisi paper SOFTT, Tabel II baris "Random").

Definisi angka SAMA dengan reports/final_eval_v7/tabel/tabel10_ketepatan_perkiraan(_per_run).csv:
  mAP50-95 test acak tanpa BLB = eval_test/metrik_per_kelas.csv, subset 'bersih', kelas 'SEMUA'
  mAP50-95 sesi baru            = eval_test_new_session/metrik_per_kelas.csv, subset 'dengan_background', kelas 'SEMUA'
  akurasi                       = proporsi label == diagnosis di diagnosis_bersih_per_citra.csv / diagnosis_dengan_background_per_citra.csv
  selisih = test acak - sesi baru; kenaikan relatif = selisih / sesi baru
Rata-rata ± simpangan baku sampel (ddof=1) atas seed yang tersedia; kenaikan relatif dihitung per seed lalu dirata-rata,
dan juga dari rata-rata (selisih rata-rata / sesi baru rata-rata).
"""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/revisi_paper/randsplit_3seed"
MODELS = {"yolov8n": "YOLOv8n", "yolo11n": "YOLO11n"}
SEEDS = (0, 1, 2)


def semua(p, subset):
    return float(next(r for r in csv.DictReader(open(p, encoding="utf-8")) if r["subset"] == subset and r["kelas"] == "SEMUA")["mAP50_95"])


def akurasi(p):
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    return sum(r["label"] == r["diagnosis"] for r in rows) / len(rows)


def ms(v, d=3):
    v = np.asarray(v, float)
    return f"{v.mean():.{d}f} ± {v.std(ddof=1):.{d}f}" if len(v) > 1 else f"{v.mean():.{d}f}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    per_run, ring = [], []
    for m, lab in MODELS.items():
        for s in SEEDS:
            rd = ROOT / "runs" / f"v7_randsplit_prov_{m}_s{s}"
            if not (rd / "eval_test_new_session" / "ringkasan.json").exists() or not (rd / "eval_test" / "ringkasan.json").exists():
                continue
            r = {"run": rd.name, "model": lab, "seed": s,
                 "ambang": json.loads((rd / "ambang_diagnosis.json").read_text(encoding="utf-8"))["ambang"],
                 "map_test_acak_tanpa_blb": semua(rd / "eval_test/metrik_per_kelas.csv", "bersih"),
                 "map_sesi_baru": semua(rd / "eval_test_new_session/metrik_per_kelas.csv", "dengan_background"),
                 "akurasi_test_acak_tanpa_blb": akurasi(rd / "eval_test/diagnosis_bersih_per_citra.csv"),
                 "akurasi_sesi_baru": akurasi(rd / "eval_test_new_session/diagnosis_dengan_background_per_citra.csv")}
            r["selisih_map"] = r["map_test_acak_tanpa_blb"] - r["map_sesi_baru"]
            r["selisih_akurasi"] = r["akurasi_test_acak_tanpa_blb"] - r["akurasi_sesi_baru"]
            r["kenaikan_relatif_map_persen"] = 100 * r["selisih_map"] / r["map_sesi_baru"]
            per_run.append(r)
        rr = [r for r in per_run if r["model"] == lab]
        if not rr:
            continue
        g = lambda k: [r[k] for r in rr]
        ring.append({"Model": lab, "Seed": ",".join(str(r["seed"]) for r in rr), "n_seed": len(rr),
                     "mAP50-95 test acak tanpa BLB": ms(g("map_test_acak_tanpa_blb")), "mAP50-95 sesi baru": ms(g("map_sesi_baru")),
                     "Selisih mAP50-95": ms(g("selisih_map")), "Akurasi test acak tanpa BLB": ms(g("akurasi_test_acak_tanpa_blb")),
                     "Akurasi sesi baru": ms(g("akurasi_sesi_baru")), "Selisih akurasi": ms(g("selisih_akurasi")),
                     "Kenaikan relatif mAP50-95 (rata-rata per seed)": ms(g("kenaikan_relatif_map_persen"), 1) + "%",
                     "Kenaikan relatif mAP50-95 (dari rata-rata)": f"{100 * np.mean(g('selisih_map')) / np.mean(g('map_sesi_baru')):.1f}%"})
    with open(OUT / "per_run.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(per_run[0]))
        w.writeheader()
        w.writerows([{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in per_run])
    with open(OUT / "ringkasan.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(ring[0]))
        w.writeheader()
        w.writerows(ring)
    tex = [f"Random & {r['Model']} & {r['Selisih mAP50-95'].replace('±', '$\\pm$').replace('-', '$-$')} & "
           f"{r['Selisih akurasi'].replace('±', '$\\pm$').replace('-', '$-$')} \\\\" for r in ring]
    md = ["# Split acak multi-seed (revisi paper SOFTT, Tabel II baris Random)", "",
          "Definisi sama dengan `reports/final_eval_v7/tabel/tabel10_ketepatan_perkiraan.csv`: test acak tanpa BLB (subset bersih) vs "
          "test sesi baru (dengan background). Rata-rata ± simpangan baku sampel atas seed.", "",
          "## Per run", "", "| Run | Ambang | mAP test acak | mAP sesi baru | Selisih mAP | Akurasi test acak | Akurasi sesi baru | Selisih akurasi | Kenaikan relatif |",
          "|---|---|---|---|---|---|---|---|---|"]
    md += [f"| {r['run']} | {r['ambang']} | {r['map_test_acak_tanpa_blb']:.4f} | {r['map_sesi_baru']:.4f} | {r['selisih_map']:.4f} | "
           f"{r['akurasi_test_acak_tanpa_blb']:.4f} | {r['akurasi_sesi_baru']:.4f} | {r['selisih_akurasi']:.4f} | {r['kenaikan_relatif_map_persen']:.1f}% |"
           for r in per_run]
    md += ["", "## Ringkasan per model", "", "| " + " | ".join(ring[0]) + " |", "|" + "---|" * len(ring[0])]
    md += ["| " + " | ".join(str(v) for v in r.values()) + " |" for r in ring]
    md += ["", "## Baris LaTeX (Tabel II)", "", "```", *tex, "```", ""]
    (OUT / "RINGKASAN.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
