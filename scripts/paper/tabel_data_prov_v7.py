#!/usr/bin/env python3
"""Tabel I paper: jumlah foto dan jumlah kelompok per kelas di setiap split prov_v7 (hanya membaca manifest; tidak menyentuh model).
Kelompok = kolom `group` manifest (kembaran dekat + rentetan foto digabung sebelum split). Satu kelompok bisa memuat lebih dari satu
kelas, sehingga jumlah kelompok per kelas bisa lebih besar daripada jumlah kelompok berbeda per split (dilaporkan terpisah).
Keluaran: paper/figures/tabel_data_prov_v7.csv dan paper/figures/tabel_data_prov_v7.tex"""
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAN = ROOT / "data/processed/prov_v7/manifest.csv"
OUT = ROOT / "paper/figures"
SPLITS = [("train", "Train"), ("val", "Val"), ("test", "Test"), ("test_new_session", "New session")]
KELAS = [("leaf_blast", "Blast"), ("brown_spot", "Brown spot"), ("bacterial_leaf_blight", "BLB$^{\\mathrm{a}}$"),
         ("background", "Healthy$^{\\mathrm{b}}$")]


def main():
    rows = list(csv.DictReader(open(MAN, encoding="utf-8")))
    foto, grup, grup_split, foto_split = defaultdict(int), defaultdict(set), defaultdict(set), defaultdict(int)
    for r in rows:
        foto[(r["split"], r["class"])] += 1
        grup[(r["split"], r["class"])].add(r["group"])
        grup_split[r["split"]].add(r["group"])
        foto_split[r["split"]] += 1
    # kelompok yang muncul di lebih dari satu split = kebocoran (harus 0)
    split_per_grup = defaultdict(set)
    for r in rows:
        split_per_grup[r["group"]].add(r["split"])
    lintas = sum(len(v) > 1 for v in split_per_grup.values())

    out = []
    for k, nk in KELAS:
        row = {"Kelas": k}
        for sp, _ in SPLITS:
            row[f"{sp} foto"] = foto[(sp, k)]
            row[f"{sp} kelompok"] = len(grup[(sp, k)])
        out.append(row)
    tot = {"Kelas": "TOTAL (kelompok berbeda per split)"}
    for sp, _ in SPLITS:
        tot[f"{sp} foto"] = foto_split[sp]
        tot[f"{sp} kelompok"] = len(grup_split[sp])
    out.append(tot)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "tabel_data_prov_v7.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)

    # LaTeX satu kolom IEEE: sel = foto/kelompok
    sel = lambda sp, k: f"{foto[(sp, k)]}/{len(grup[(sp, k)])}"  # noqa: E731
    t = ["\\begin{table}[htbp]",
         "\\caption{Dataset Splits: Images/Groups per Class}",
         "\\begin{center}",
         "\\setlength{\\tabcolsep}{3pt}",
         "\\begin{tabular}{|l|c|c|c|c|}",
         "\\hline",
         "\\textbf{Class} & " + " & ".join(f"\\textbf{{{n}}}" for _, n in SPLITS) + " \\\\",
         "\\hline"]
    for k, nk in KELAS:
        t.append(f"{nk} & " + " & ".join(sel(sp, k) for sp, _ in SPLITS) + " \\\\")
    t += ["\\hline",
          "Total & " + " & ".join(f"{foto_split[sp]}/{len(grup_split[sp])}" for sp, _ in SPLITS) + " \\\\",
          "\\hline",
          "\\end{tabular}",
          "\\par\\vspace{2pt}{\\scriptsize Cells: images/groups. Groups contain near-duplicate and burst images and never cross splits. "
          "$^{\\mathrm{a}}$Provisional, single source. $^{\\mathrm{b}}$Background images without boxes. "
          "A group may contain images of more than one class, so class groups do not sum to the total, which counts distinct groups.}",
          "\\label{tab:data}",
          "\\end{center}",
          "\\end{table}"]
    (OUT / "tabel_data_prov_v7.tex").write_text("\n".join(t) + "\n", encoding="utf-8")
    for r in out:
        print(r)
    print("kelompok lintas split:", lintas)


if __name__ == "__main__":
    main()
