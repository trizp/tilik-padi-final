#!/usr/bin/env python3
"""
kunci_ambang_aplikasi_v9.py - Fase 4 butir 2 (KEPUTUSAN_PROYEK.md 2026-10-20): ambang aplikasi v9 dikunci di VAL prov_v9 memakai ALUR APLIKASI
(masker YOLOv8n-seg tau 0,35 + model final v9_final_prov_yolov8s_s2); foto cadangan (diminta foto ulang) tidak dihitung.

- ambang terdeteksi : evaluate.pilih_ambang (akurasi per citra maksimum pada val bersih; seri -> F1 makro, lalu ambang terkecil; grid 0,01-0,95)
- t_rendah          : ambang terbesar (grid 0,01) <= ambang terdeteksi dengan >= 90% foto val berpenyakit (termasuk BLB) punya keyakinan
                      teratas >= t_rendah (prosedur tiga_tingkat_v7.py)
- batas tampilan    : batas terendah dalam [0,10; ambang) dengan proporsi sehat <= 25% di antara foto tidak terdeteksi yang menampilkan bercak
                      berkeyakinan rendah (prosedur batas_tampil_v7.py); tidak ada -> fitur nonaktif
Tidak menyentuh test, test sesi baru, atau ds3.

  .venv/Scripts/python scripts/v9_latar/kunci_ambang_aplikasi_v9.py
Keluaran: reports/v9_latar/aplikasi_v9/fase4_ambang_aplikasi.json, fase4_ambang_per_citra.csv (menolak menimpa)
"""
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ["YOLO_AUTOINSTALL"] = "false"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate import GRID_AMBANG, TAK_TERDETEKSI, pilih_ambang  # noqa: E402

V9 = ROOT / "data/processed/prov_v9"
ALUR = ROOT / "data/derived/v9_latar/alur_seg_val/images"
SEG = ROOT / "data/derived/v9_latar/seg_n_val/info.json"
RUN = "v9_final_prov_yolov8s_s2"
REP = ROOT / "reports/v9_latar/aplikasi_v9"
PENYAKIT = ("leaf_blast", "brown_spot", "bacterial_leaf_blight")
NAMES = {0: "leaf_blast", 1: "brown_spot", 2: "bacterial_leaf_blight"}


def label_citra(lbl):
    cl = {int(b.split()[0]) for b in lbl.read_text().split("\n") if b.strip()} if lbl.exists() else set()
    if not cl:
        return TAK_TERDETEKSI
    return NAMES[cl.pop()] if len(cl) == 1 else None


def main():
    out = REP / "fase4_ambang_aplikasi.json"
    if out.exists():
        sys.exit(f"[DITOLAK] {out} sudah ada (ambang sudah dikunci)")
    from ultralytics import YOLO
    seg = json.loads(SEG.read_text(encoding="utf-8"))
    cad = seg["cadangan"]
    rows = [r for r in csv.DictReader(open(V9 / "manifest.csv", encoding="utf-8")) if r["split"] == "val"]
    pakai = [r for r in rows if not cad[r["file"]]]
    model = YOLO(str(ROOT / "runs" / RUN / "weights/best.pt"))
    top = {}
    fs = [r["file"] for r in pakai]
    for i in range(0, len(fs), 16):
        part = fs[i:i + 16]
        for f, res in zip(part, model.predict([str(ALUR / f) for f in part], imgsz=640, conf=0.001, iou=0.7, device=0, verbose=False)):
            cf = res.boxes.conf.cpu().numpy()
            top[f] = (float(cf.max()), int(res.boxes.cls[int(cf.argmax())].item())) if len(cf) else (0.0, -1)
    lab = {r["file"]: label_citra(V9 / "labels/val" / (Path(r["file"]).stem + ".txt")) for r in pakai}
    # ambang terdeteksi: val bersih (tanpa ds1_provisional), tanpa foto multi-kelas
    bersih = [r["file"] for r in pakai if r["source"] != "ds1_provisional" and lab[r["file"]] is not None]
    kelas = list(NAMES.values()) + [TAK_TERDETEKSI]
    best, kurva = pilih_ambang([lab[f] for f in bersih], [top[f] for f in bersih], NAMES, kelas)
    amb = best["ambang"]
    # t_rendah: semua foto berpenyakit (termasuk BLB) yang tidak cadangan
    sakit = [f for f in fs if lab[f] in PENYAKIT]
    grid = [round(0.01 * i, 2) for i in range(1, 100)]
    persen = {t: 100 * sum(top[f][0] >= t for f in sakit) / len(sakit) for t in grid}
    rendah = max(t for t in grid if persen[t] >= 90.0 and t <= amb)
    # batas tampilan bercak berkeyakinan rendah
    tak = [f for f in fs if top[f][0] < amb and lab[f] is not None]
    tabel = []
    for b in [round(0.01 * i, 2) for i in range(10, 100) if round(0.01 * i, 2) < amb]:
        tampil = [f for f in tak if top[f][0] >= b]
        sehat = sum(lab[f] == TAK_TERDETEKSI for f in tampil)
        tabel.append({"batas": b, "sehat": sehat, "sakit": len(tampil) - sehat,
                      "proporsi_sehat_persen": round(100 * sehat / len(tampil), 1) if tampil else None})
    ok = [t for t in tabel if t["proporsi_sehat_persen"] is not None and t["proporsi_sehat_persen"] <= 25.0]
    batas = ok[0]["batas"] if ok else None
    hasil = {"LABEL": "ambang aplikasi v9 (VAL, alur aplikasi); dikunci SEBELUM fase 5",
             "alur": "masker YOLOv8n-seg tau " + str(seg["tau"]) + " + " + RUN, "dikunci": datetime.now().isoformat(timespec="seconds"),
             "n_val": len(rows), "n_cadangan_tidak_dihitung": len(rows) - len(pakai), "n_dipakai": len(pakai),
             "ambang_terdeteksi": amb, "akurasi_val_bersih_saat_dipilih": best["akurasi"], "f1_makro_val_bersih": best["f1_makro"],
             "n_val_bersih": len(bersih), "grid_ambang": f"{GRID_AMBANG[0]}..{GRID_AMBANG[-1]}",
             "t_rendah": rendah, "tingkat_kemungkinan_kosong": rendah >= amb,
             "persen_berpenyakit_ge_t_rendah": round(persen[rendah], 2), "persen_berpenyakit_ge_t_rendah_plus_0.01": round(persen[round(rendah + 0.01, 2)], 2),
             "n_berpenyakit": len(sakit),
             "batas_tampil_bercak_rendah": batas, "bercak_rendah_aktif": batas is not None, "tabel_batas_tampil": tabel,
             "ambang_final_s2_alur_SAM2_pembanding": json.loads((ROOT / "runs" / RUN / "ambang_diagnosis.json").read_text(encoding="utf-8"))["ambang"],
             "catatan": "foto cadangan = diminta foto ulang (tidak didiagnosis), jadi tidak ikut pemilihan ambang; test/tns/ds3 tidak disentuh"}
    REP.mkdir(parents=True, exist_ok=True)
    with open(REP / "fase4_ambang_per_citra.csv", "w", newline="", encoding="utf-8") as fo:
        w = csv.writer(fo)
        w.writerow(["berkas", "sumber", "label", "conf_teratas", "kelas_teratas"])
        for r in pakai:
            f = r["file"]
            w.writerow([f, r["source"], lab[f], round(top[f][0], 4), NAMES.get(top[f][1], "")])
    out.write_text(json.dumps(hasil, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in hasil.items() if k != "tabel_batas_tampil"}, indent=1, ensure_ascii=False))
    print("tabel batas:", [(t["batas"], t["sehat"], t["sakit"], t["proporsi_sehat_persen"]) for t in tabel])


if __name__ == "__main__":
    main()
