#!/usr/bin/env python3
"""
data_flow.py - Tabel alur data: untuk tiap sumber dan KELAS ASLI, jumlah gambar mentah, yang dibuang (beserta alasan),
yang dikarantina, dan jumlah akhir per split. Juga daftar gambar ds4 di luar cakupan (Rice Tungro, Sheath blight, Leaf smut).

Setiap gambar mentah dicocokkan ke satu hasil: split (train/val/test) > karantina > dibuang (alasan) > dicadangkan/di luar cakupan.
Selisih (gambar mentah yang tidak punya catatan) dihitung dan ditampilkan; seharusnya 0.

Pemakaian: python scripts/data_flow.py [--config configs/classes_v4.yaml]
Keluaran (reports/<dataset>/): data_flow.csv, data_flow.md, out_of_scope_ds4_test_candidates.csv
"""
import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from inventory import IMG_EXT, find_yolo_label, load_yaml_names  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TARGET_NOTE = {"leaf_blast": "target", "brown_spot": "target", "bacterial_leaf_blight": "target", "background": "target (background)"}

REASON_TEXT = {
    "memuat_kelas_di_luar_target": "kelas di luar 3 kelas target",
    "tanpa_label": "tanpa file label",
    "tanpa_box_valid": "tidak punya box valid",
    "duplikat_label_bertentangan": "foto sama berlabel bertentangan (duplikat sungguhan)",
    "dipangkas_max_background_frac_train": "dipangkas agar background train 10%",
    "melebihi_max_images": "melebihi batas jumlah gambar sumber",
    "label_kosong": "file label kosong pada folder penyakit",
    "label_tidak_ditemukan": "file label tidak ditemukan",
    "gambar_tidak_terbaca": "gambar tidak terbaca",
    "kelompok_duplikat_lintas_sesi": "kelompok duplikat lintas sesi (anggota kelompok sesi test_new_session dari sesi lain)",
}


def reason_text(r):
    if r.startswith("folder_tidak_dipakai"):
        return "folder/kelas di luar cakupan (tidak dipakai)"
    if r.startswith("path_dikecualikan"):
        return "cabang Augmented tidak dipakai (augmentasi hanya boleh saat training)"
    if r.startswith("karantina"):
        return "karantina (foto sama/hampir sama berlabel kelas berbeda)"
    return REASON_TEXT.get(r, r)


def rel(p):
    return Path(p).resolve().relative_to(ROOT).as_posix()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/classes_v4.yaml")
    a = ap.parse_args()
    cfg = yaml.safe_load((ROOT / a.config).read_text(encoding="utf-8"))
    rd = ROOT / cfg["report_dir"]
    man = {r["orig_path"]: r for r in csv.DictReader(open(ROOT / cfg["output_dir"] / "manifest.csv", encoding="utf-8"))}
    quar = {r["berkas"]: r for r in csv.DictReader(open(rd / "quarantine.csv", encoding="utf-8"))} if (rd / "quarantine.csv").exists() else {}
    exd = defaultdict(list)
    for r in csv.DictReader(open(rd / "exclusions_detail.csv", encoding="utf-8")):
        exd[r["berkas"]].append(r["alasan"])

    # ---- enumerasi gambar mentah + kelas asli
    raw = []     # (sumber, kelas_asli, cabang, path_rel)
    names1, _ = load_yaml_names(ROOT / "data/raw/ds1")
    for p in sorted((ROOT / "data/raw/ds1").rglob("*")):
        if p.suffix.lower() in IMG_EXT:
            lab = find_yolo_label(p)
            cls = "(tanpa label)"
            if lab is not None:
                ids = [int(float(ln.split()[0])) for ln in lab.read_text(errors="ignore").splitlines() if ln.split()]
                if ids:
                    cls = names1[Counter(ids).most_common(1)[0][0]].replace("Rice__", "")
            raw.append(("ds1", cls, "-", rel(p)))
    for p in sorted((ROOT / "data/raw/ds2").rglob("*")):
        if p.suffix.lower() in IMG_EXT:
            br = "Augmented" if "/augmented/" in p.as_posix().lower() else ("Original" if "/original/" in p.as_posix().lower() else "?")
            raw.append(("ds2", p.parent.name, br, rel(p)))
    for p in sorted((ROOT / "data/raw/ds3").rglob("*")):
        if p.suffix.lower() in IMG_EXT:
            raw.append(("ds3", p.parent.name, "-", rel(p)))
    ds4_img = ROOT / "data/raw/ds4/images_labels/RiceLeafDiseaseBD/Original images"
    for p in sorted(ds4_img.rglob("*")):
        if p.suffix.lower() in IMG_EXT:
            raw.append(("ds4", p.parent.name, "-", rel(p)))

    # ---- hasil per gambar
    rows = defaultdict(lambda: Counter())
    reasons = defaultdict(Counter)
    unaccounted = Counter()
    for src, cls, br, path in raw:
        key = (src, cls, br)
        rows[key]["mentah"] += 1
        if path in man:
            rows[key][man[path]["split"]] += 1
        elif path in quar:
            rows[key]["karantina"] += 1
        elif src == "ds3":
            rows[key]["dicadangkan"] += 1
        elif path in exd:
            rows[key]["dibuang"] += 1
            reasons[key][reason_text(exd[path][0])] += 1
        else:
            rows[key]["tidak_tercatat"] += 1
            unaccounted[src] += 1
    # ---- keluaran
    out_rows = []
    for (src, cls, br), c in sorted(rows.items()):
        scope = "dicadangkan: external test (ds3, belum diproses)" if src == "ds3" else (
            "target" if (cls.lower().replace(" ", "_").replace("-", "_") in ("rice__bacterialleafblight", "bacterialleafblight", "bacterial_leaf_blight", "brown_spot", "brownspot",
                                                                         "blast", "leafblast", "leaf_blast", "healthy", "healthy_rice_leaf") and src != "ds2") else "")
        out_rows.append({"sumber": src, "kelas_asli": cls, "cabang": br, "mentah": c["mentah"], "train": c["train"], "val": c["val"], "test": c["test"], "test_new_session": c["test_new_session"],
                         "karantina": c["karantina"], "dibuang": c["dibuang"], "dicadangkan": c["dicadangkan"], "tidak_tercatat": c["tidak_tercatat"],
                         "alasan_dibuang": "; ".join(f"{k}: {v}" for k, v in reasons[(src, cls, br)].most_common()), "cakupan": scope})
    df = pd.DataFrame(out_rows)
    df.to_csv(rd / "data_flow.csv", index=False, encoding="utf-8")

    # ---- kandidat di luar cakupan ds4
    xl = pd.read_excel(ROOT / "data/raw/ds4/docs/Dataset metadata.xlsx", sheet_name="Image_Metadata")
    dupname = xl.groupby("Original_Filename")["New_Filename"].apply(list).to_dict()
    new2orig = dict(zip(xl["New_Filename"].astype(str).str.lower(), xl["Original_Filename"].astype(str)))
    lab_root = ROOT / "data/raw/ds4/images_labels/RiceLeafDiseaseBD/Annotated images ( visual with labels)"
    labdirs = {re.sub(r"[^a-z0-9]+", "_", d.name.lower()).strip("_"): d for d in lab_root.iterdir() if d.is_dir()}
    oos = []
    for src, cls, br, path in raw:
        if src == "ds4" and cls in ("Rice Tungro", "Sheath blight", "Leaf smut"):
            key = re.sub(r"[^a-z0-9]+", "_", cls.lower()).strip("_")
            lf = labdirs[key] / "labels" / (Path(path).stem + ".txt") if key in labdirs else None
            nbox = len([ln for ln in lf.read_text().splitlines() if ln.strip()]) if lf is not None and lf.exists() else 0
            orig = new2orig.get(Path(path).name.lower(), "")
            others = [n for n in dupname.get(orig, []) if n.lower() != Path(path).name.lower()]
            oos.append({"berkas": path, "kelas_asli": cls, "n_box_label": nbox, "original_filename": orig,
                        "foto_sama_di_folder_lain": "|".join(others), "status": "di luar cakupan: bukan train/val/test; kandidat test terpisah"})
    pd.DataFrame(oos).to_csv(rd / "out_of_scope_ds4_test_candidates.csv", index=False, encoding="utf-8")

    # ---- markdown
    md = [f"# Alur data `{cfg['dataset_name']}`\n",
          "Setiap foto mentah dicatat tepat satu hasil: masuk train/val/test, dikarantina, dibuang (dengan alasan), atau dicadangkan. "
          "Kolom `tidak_tercatat` harus 0 (artinya tidak ada foto yang hilang tanpa catatan).\n"]
    for src in ("ds1", "ds2", "ds3", "ds4"):
        sub = df[df.sumber == src]
        tot = sub[["mentah", "train", "val", "test", "test_new_session", "karantina", "dibuang", "dicadangkan", "tidak_tercatat"]].sum()
        md.append(f"## {src}: {int(tot.mentah)} foto mentah\n")
        md.append("| kelas asli | cabang | mentah | train | val | test | test_new_session | karantina | dibuang | dicadangkan | tidak tercatat | alasan dibuang |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in sub.itertuples():
            md.append(f"| {r.kelas_asli} | {r.cabang} | {r.mentah} | {r.train} | {r.val} | {r.test} | {r.test_new_session} | {r.karantina} | {r.dibuang} | {r.dicadangkan} | {r.tidak_tercatat} | {r.alasan_dibuang} |")
        md.append(f"| **jumlah** | | **{int(tot.mentah)}** | **{int(tot.train)}** | **{int(tot.val)}** | **{int(tot.test)}** | **{int(tot.test_new_session)}** | **{int(tot.karantina)}** | **{int(tot.dibuang)}** | **{int(tot.dicadangkan)}** | **{int(tot.tidak_tercatat)}** | |\n")
    allt = df[["mentah", "train", "val", "test", "test_new_session", "karantina", "dibuang", "dicadangkan", "tidak_tercatat"]].sum()
    md.append(f"## Jumlah seluruh sumber\n\nMentah {int(allt.mentah)} = train {int(allt.train)} + val {int(allt.val)} + test {int(allt.test)} + test_new_session {int(allt.test_new_session)} + karantina {int(allt.karantina)} "
              f"+ dibuang {int(allt.dibuang)} + dicadangkan {int(allt.dicadangkan)} + tidak tercatat {int(allt.tidak_tercatat)}.\n")
    oc = Counter(r["kelas_asli"] for r in oos)
    md.append("## Di luar cakupan: kandidat test terpisah (ds4)\n")
    md.append(f"Foto ds4 kelas Rice Tungro ({oc['Rice Tungro']}), Sheath blight ({oc['Sheath blight']}), dan Leaf smut ({oc['Leaf smut']}) TIDAK dimasukkan ke train/val/test. "
              f"Daftar lengkap (dengan jumlah box label dan foto ganda di folder lain): `out_of_scope_ds4_test_candidates.csv` "
              f"({len(oos)} foto; {sum(1 for r in oos if r['foto_sama_di_folder_lain'])} di antaranya fotonya juga ada di folder kelas lain).\n")
    (rd / "data_flow.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
