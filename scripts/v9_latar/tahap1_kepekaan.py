#!/usr/bin/env python3
"""
tahap1_kepekaan.py - ANALISIS VAL (tanpa training): kepekaan model final v7 terhadap penghapusan latar.

1. bangun : semua foto VAL prov_v7 -> latar diganti abu-abu (114,114,114) dengan masker SAM2 aturan v2 (data/derived/v9_latar/sam2_v2),
            disimpan JPEG q95. Label: (a) sama persis; (b) "buang-kotak": kotak dengan < 50% luas di dalam daun dibuang. Masker dibuat
            tanpa label; label hanya dipakai untuk aturan buang-kotak dan pengukuran.
2. evaluasi: 12 model final (4 varian x 3 seed, best.pt terkunci, sha dicek), pengaturan sama dengan evaluate.py (conf 0,001, iou 0,7,
            imgsz latihan, batch 8). mAP pada subset "bersih" (tanpa ds1_provisional, seperti tabel utama) dan "dengan BLB prov."
            (semua VAL); akurasi diagnosis per foto pada subset bersih dengan ambang terkunci. Label citra untuk diagnosis = label asli.
   Uji reproduksi: VAL asli harus sama dengan eval_val tersimpan (mAP bersih dan akurasi_val_saat_dipilih).

  .venv/Scripts/python scripts/v9_latar/tahap1_kepekaan.py bangun
  .venv/Scripts/python scripts/v9_latar/tahap1_kepekaan.py evaluasi
Keluaran: reports/v9_latar/tahap1_{per_run.csv, ringkasan.md, ringkasan.json, buang_kotak_val.json}
"""
import csv
import hashlib
import json
import os
import shutil
import statistics as S
import sys
from collections import Counter
from pathlib import Path

os.environ["YOLO_AUTOINSTALL"] = "false"
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import umum  # noqa: E402

ROOT = umum.ROOT
MASKER = umum.KERJA / "sam2_v2"
TL = umum.KERJA / "val_tanpa_latar"
EV = umum.KERJA / "eval_tahap1"
NAMES = {0: "leaf_blast", 1: "brown_spot", 2: "bacterial_leaf_blight"}
MODEL = ["yolov8n", "yolo11n", "yolov8s", "yolo11s"]
RUNS = [f"v7_final_prov_{m}_s{s}" for m in MODEL for s in (0, 1, 2)]
TAK = "sehat_atau_tidak_terdeteksi"


def bangun():
    rows = umum.manifest_val()
    (TL / "images").mkdir(parents=True, exist_ok=True)
    (TL / "labels").mkdir(parents=True, exist_ok=True)
    (TL / "labels_buang").mkdir(parents=True, exist_ok=True)
    info = json.loads((MASKER / "info.json").read_text(encoding="utf-8"))
    stat = {"per_kelas": {}, "per_sumber_kelas": {}, "cadangan_tidak_dipisah": 0, "n_foto": 0, "foto_kehilangan_semua_kotak": Counter()}
    for r in rows:
        f = r["file"]
        bgr = cv2.imread(str(umum.VAL_IMG / f))
        h, w = bgr.shape[:2]
        mm = cv2.imread(str(MASKER / "masker_mentah" / (Path(f).stem + ".png")), cv2.IMREAD_GRAYSCALE)
        m = umum.pasca_proses(mm >= 128)
        out = bgr.copy()
        out[~m] = umum.ABU[::-1]
        cv2.imwrite(str(TL / "images" / f), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
        lbl = umum.VAL_LBL / (Path(f).stem + ".txt")
        baris = [b for b in (lbl.read_text().split("\n") if lbl.exists() else []) if b.strip()]
        (TL / "labels" / lbl.name).write_text("\n".join(baris) + ("\n" if baris else ""))
        kotak = umum.baca_kotak(f, w, h)
        ret = umum.retensi_kotak(m, kotak)
        simpan = [b for b, (_, v, _) in zip(baris, ret) if v >= 0.5]
        (TL / "labels_buang" / lbl.name).write_text("\n".join(simpan) + ("\n" if simpan else ""))
        for c, v, _ in ret:
            for kunci, nama in (("per_kelas", NAMES[c]), ("per_sumber_kelas", f"{r['source']}|{NAMES[c]}")):
                d = stat[kunci].setdefault(nama, {"kotak": 0, "dibuang": 0})
                d["kotak"] += 1
                d["dibuang"] += int(v < 0.5)
        if baris and not simpan:
            stat["foto_kehilangan_semua_kotak"][r["class"]] += 1
        stat["cadangan_tidak_dipisah"] += int(info.get(f, {}).get("cadangan", False))
        stat["n_foto"] += 1
    for kunci in ("per_kelas", "per_sumber_kelas"):
        for d in stat[kunci].values():
            d["persen_dibuang"] = round(100 * d["dibuang"] / d["kotak"], 1)
    stat["foto_kehilangan_semua_kotak"] = dict(stat["foto_kehilangan_semua_kotak"])
    stat["catatan"] = "VAL prov_v7, SAM2 aturan v2; buang-kotak = kotak dengan < 50% luas di dalam daun dibuang"
    (umum.LAPORAN / "buang_kotak_val.json").write_text(json.dumps(stat, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stat, indent=1, ensure_ascii=False))


def subset(nama, img_dir, lbl_dir, pilih):
    d = EV / nama
    if d.exists():
        shutil.rmtree(d)
    (d / "images").mkdir(parents=True)
    (d / "labels").mkdir(parents=True)
    for r in umum.manifest_val():
        if not pilih(r):
            continue
        f = r["file"]
        os.link(img_dir / f, d / "images" / f)
        lbl = lbl_dir / (Path(f).stem + ".txt")
        if lbl.exists():
            shutil.copyfile(lbl, d / "labels" / lbl.name)
    y = EV / f"_{nama}.yaml"
    y.write_text(yaml.safe_dump({"path": str(d), "train": "images", "val": "images", "names": NAMES}), encoding="utf-8")
    return y, d


def label_citra(lbl):
    cl = {int(b.split()[0]) for b in lbl.read_text().split("\n") if b.strip()} if lbl.exists() else set()
    if not cl:
        return TAK
    return NAMES[cl.pop()] if len(cl) == 1 else None


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def evaluasi():
    from ultralytics import YOLO
    bersih = lambda r: r["source"] != "ds1_provisional"  # noqa: E731
    semua = lambda r: True  # noqa: E731
    sub = {"asli_bersih": subset("asli_bersih", umum.VAL_IMG, umum.VAL_LBL, bersih),
           "tanpa_bersih": subset("tanpa_bersih", TL / "images", TL / "labels", bersih),
           "tanpa_bersih_buang": subset("tanpa_bersih_buang", TL / "images", TL / "labels_buang", bersih),
           "asli_semua": subset("asli_semua", umum.VAL_IMG, umum.VAL_LBL, semua),
           "tanpa_semua": subset("tanpa_semua", TL / "images", TL / "labels", semua)}
    lab = {r["file"]: label_citra(umum.VAL_LBL / (Path(r["file"]).stem + ".txt")) for r in umum.manifest_val() if bersih(r)}
    hasil = []
    for run in RUNS:
        amb = json.loads((ROOT / f"runs/{run}/ambang_diagnosis.json").read_text(encoding="utf-8"))
        w = ROOT / f"runs/{run}/weights/best.pt"
        assert sha(w) == amb["weights_sha256"], run
        model = YOLO(str(w))
        imgsz = int(model.ckpt.get("train_args", {}).get("imgsz") or 640)
        baris = {"run": run, "model": run.split("_")[3], "seed": run[-1], "ambang": amb["ambang"],
                 "akurasi_val_tersimpan": amb["akurasi_val_saat_dipilih"]}
        for nama, (y, d) in sub.items():
            r = model.val(data=str(y), split="val", imgsz=imgsz, batch=8, device=0, conf=0.001, iou=0.7, plots=False,
                          project=str(EV / "_val"), name=f"{run}_{nama}", exist_ok=True, verbose=False)
            baris[f"{nama}_mAP50"] = round(float(r.box.map50), 4)
            baris[f"{nama}_mAP50_95"] = round(float(r.box.map), 4)
        for nama in ("asli_bersih", "tanpa_bersih"):
            d = sub[nama][1]
            files = [f for f, l in lab.items() if l is not None]
            benar = 0
            for i in range(0, len(files), 16):
                part = files[i:i + 16]
                for f, res in zip(part, model.predict([str(d / "images" / f) for f in part], imgsz=imgsz, conf=0.001, iou=0.7,
                                                      device=0, verbose=False)):
                    cf = res.boxes.conf.cpu().numpy()
                    if len(cf) and cf.max() >= amb["ambang"]:
                        dia = NAMES[int(res.boxes.cls[int(cf.argmax())].item())]
                    else:
                        dia = TAK
                    benar += int(dia == lab[f])
            baris[f"{nama}_akurasi"] = round(benar / len(files), 4)
            baris["n_diagnosis"] = len(files)
        tersimpan = json.loads((ROOT / f"runs/{run}/eval_val/ringkasan.json").read_text(encoding="utf-8"))
        ref = next(m for m in tersimpan["metrik"] if m["subset"] == "bersih" and m["kelas"] == "SEMUA")
        baris["reproduksi_mAP50_95_bersih_tersimpan"] = ref["mAP50_95"]
        hasil.append(baris)
        print({k: v for k, v in baris.items() if "mAP50_95" in k or "akurasi" in k}, flush=True)
    with open(umum.LAPORAN / "tahap1_per_run.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(hasil[0]))
        wr.writeheader()
        wr.writerows(hasil)
    ringkas(hasil)


def ms(xs):
    return f"{S.mean(xs):.3f} ± {S.stdev(xs):.3f}".replace(".", ",")


def ringkas(hasil):
    rep_map = max(abs(h["asli_bersih_mAP50_95"] - h["reproduksi_mAP50_95_bersih_tersimpan"]) for h in hasil)
    rep_acc = max(abs(h["asli_bersih_akurasi"] - h["akurasi_val_tersimpan"]) for h in hasil)
    md = ["# Tahap 1: kepekaan model final v7 terhadap penghapusan latar (ANALISIS VAL, tanpa training)", "",
          "VAL prov_v7; latar diganti abu-abu (114,114,114) dengan SAM2 aturan v2 (tanpa label). 4 model x 3 seed, best.pt dan ambang",
          "diagnosis terkunci. Bersih = tanpa ds1_provisional (BLB), seperti tabel utama. Rata-rata ± simpangan baku 3 seed.", "",
          f"Uji reproduksi VAL asli vs eval_val tersimpan: selisih maks. mAP50-95 bersih {rep_map:.4f}; akurasi diagnosis {rep_acc:.4f}.", "",
          "| Model | mAP50 bersih: asli / tanpa latar / tanpa latar + buang-kotak | mAP50-95 bersih: asli / tanpa / tanpa + buang | "
          "mAP50-95 dengan BLB prov.: asli / tanpa | Akurasi diagnosis bersih: asli / tanpa latar |", "|---|---|---|---|---|"]
    ring = {}
    for m in MODEL:
        hs = [h for h in hasil if h["model"] == m]
        g = lambda k: [h[k] for h in hs]  # noqa: E731
        ring[m] = {k: {"rata2": round(S.mean(g(k)), 4), "std": round(S.stdev(g(k)), 4)} for k in hs[0] if k.endswith(("mAP50", "mAP50_95", "akurasi"))}
        ring[m]["selisih_akurasi_tanpa_minus_asli"] = round(S.mean(g("tanpa_bersih_akurasi")) - S.mean(g("asli_bersih_akurasi")), 4)
        ring[m]["selisih_mAP50_95_tanpa_minus_asli"] = round(S.mean(g("tanpa_bersih_mAP50_95")) - S.mean(g("asli_bersih_mAP50_95")), 4)
        md.append(f"| {m} | {ms(g('asli_bersih_mAP50'))} / {ms(g('tanpa_bersih_mAP50'))} / {ms(g('tanpa_bersih_buang_mAP50'))} | "
                  f"{ms(g('asli_bersih_mAP50_95'))} / {ms(g('tanpa_bersih_mAP50_95'))} / {ms(g('tanpa_bersih_buang_mAP50_95'))} | "
                  f"{ms(g('asli_semua_mAP50_95'))} / {ms(g('tanpa_semua_mAP50_95'))} | {ms(g('asli_bersih_akurasi'))} / {ms(g('tanpa_bersih_akurasi'))} |")
    md += ["", "Selisih rata-rata (tanpa latar - asli), subset bersih:", ""]
    md += [f"- {m}: mAP50-95 {ring[m]['selisih_mAP50_95_tanpa_minus_asli']:+.4f}; akurasi diagnosis {ring[m]['selisih_akurasi_tanpa_minus_asli']:+.4f}"
           .replace(".", ",") for m in MODEL]
    md += ["", "Catatan: model final dilatih dengan latar asli; foto tanpa latar disimpan ulang sebagai JPEG q95. Label sama persis kecuali kolom",
           "\"buang-kotak\". Sumber: tahap1_per_run.csv."]
    (umum.LAPORAN / "tahap1_ringkasan.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (umum.LAPORAN / "tahap1_ringkasan.json").write_text(json.dumps({"LABEL": "ANALISIS VAL", "reproduksi_selisih_maks": {"mAP50_95": rep_map,
                                                                    "akurasi": rep_acc}, "per_model": ring}, indent=2), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    {"bangun": bangun, "evaluasi": evaluasi}[sys.argv[1]]()
