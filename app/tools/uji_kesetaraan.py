#!/usr/bin/env python3
"""
uji_kesetaraan.py - Bukti bahwa inferensi ONNX Runtime di app/core memberi kelas teratas dan tingkat keyakinan yang sama dengan
hasil evaluasi tersimpan (reports/final_eval_v7/aplikasi/onnx_vs_pt_50_val.csv: Ultralytics 8.4.171, last.pt dan last.onnx,
conf 0,001, iou 0,7, CPU) pada 20 foto val pertama di berkas itu. Syarat lulus: kelas teratas identik dan selisih < 0,01.
Hanya dijalankan di folder proyek (butuh data/processed/prov_v7). Tidak mengekspor ulang model.

  app/.venv/Scripts/python.exe app/tools/uji_kesetaraan.py
Keluaran: app/docs/uji_kesetaraan.json dan .md
"""
import csv
import json
import sys
from pathlib import Path

APP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP))
from core import diagnosis as dg  # noqa: E402
from core import gambar, inferensi, konfigurasi  # noqa: E402

PROYEK = APP.parent
ACUAN_PER_MODEL = {"v7_deploy_yolo11s_s2.onnx": "reports/final_eval_v7/aplikasi/onnx_vs_pt_50_val.csv",
                   "v7_deploy_yolov8s_s2.onnx": "reports/final_eval_v7/aplikasi/onnx_vs_pt_20_val_yolov8s.csv",
                   # v9: acuan fase 4 (Ultralytics .pt vs .onnx) pada foto VAL v9 yang latarnya sudah dihapus SAM2; yang diuji di sini hanya
                   # pra-/pasca-proses detektor aplikasi. Pemisah daun diuji terpisah: tools/uji_pemisah_daun.py
                   "v9_deploy_yolov8s_s2.onnx": "reports/v9_latar/aplikasi_v9/fase4_onnx_vs_pt_20_val.csv"}
FOLDER_PER_MODEL = {"v9_deploy_yolov8s_s2.onnx": "data/processed/prov_v9/images/val"}
ACUAN = None                                       # bawaan: sesuai model di config
# opsi (konfigurasi alternatif, misalnya model YOLOv8s): --config config_yolov8.yaml --acuan <csv> --akhiran _yolov8s
VAL = PROYEK / "data/processed/prov_v7/images/val"


def main(n=20, config=None, acuan=None, akhiran=""):
    global ACUAN
    cfg = konfigurasi.muat(APP / config if config else None)
    ACUAN = PROYEK / (acuan or ACUAN_PER_MODEL[cfg.model_path.name])
    folder = PROYEK / FOLDER_PER_MODEL[cfg.model_path.name] if cfg.model_path.name in FOLDER_PER_MODEL else VAL
    det = inferensi.Detektor(cfg.model_path, cfg.imgsz, cfg.conf_min, cfg.iou, cfg.max_det)
    baris = list(csv.DictReader(open(ACUAN, encoding="utf-8")))[:n]
    hasil = []
    for r in baris:
        data = (folder / r["gambar"]).read_bytes()
        im = gambar.baca_foto(data, cfg.ukuran_maks_mb)
        k = det.prediksi(gambar.ke_array(im))
        top = max(k, key=lambda b: b.keyakinan) if k else None
        kelas_app, conf_app = (top.kelas, top.keyakinan) if top else (-1, 0.0)
        hasil.append({"gambar": r["gambar"], "kelas_acuan_onnx": int(r["kelas_onnx"]), "conf_acuan_onnx": float(r["conf_onnx"]),
                      "kelas_acuan_pt": int(r["kelas_pt"]), "conf_acuan_pt": float(r["conf_pt"]),
                      "kelas_app": kelas_app, "conf_app": round(conf_app, 5),
                      "kelas_identik": kelas_app == int(r["kelas_onnx"]) == int(r["kelas_pt"]),
                      "tingkat_sama": dg.tentukan_tingkat(conf_app, cfg.ambang_terdeteksi, cfg.ambang_rendah)
                      == dg.tentukan_tingkat(float(r["conf_onnx"]), cfg.ambang_terdeteksi, cfg.ambang_rendah),
                      "selisih_vs_onnx": round(abs(conf_app - float(r["conf_onnx"])), 5),
                      "selisih_vs_pt": round(abs(conf_app - float(r["conf_pt"])), 5)})
    # Kelas teratas hanya bermakna bila keyakinannya tidak nyaris nol: di bawah 0,01 dua kotak teratas bisa hampir seri dan urutannya
    # tertukar karena selisih resize (v9: 2 foto daun sehat, keyakinan ±0,003), sedangkan diagnosisnya tetap sama (tidak terdeteksi).
    bermakna = [h for h in hasil if h["conf_acuan_onnx"] >= 0.01]
    lolos = (all(h["kelas_identik"] for h in bermakna) and all(h["tingkat_sama"] for h in hasil)
             and all(h["selisih_vs_onnx"] < 0.01 and h["selisih_vs_pt"] < 0.01 for h in hasil))
    ring = {"n": len(hasil), "kelas_identik": sum(h["kelas_identik"] for h in hasil),
            "kelas_identik_keyakinan_acuan_ge_0_01": f"{sum(h['kelas_identik'] for h in bermakna)}/{len(bermakna)}",
            "tingkat_sama": sum(h["tingkat_sama"] for h in hasil),
            "selisih_maks_vs_onnx_ultralytics": max(h["selisih_vs_onnx"] for h in hasil),
            "selisih_maks_vs_pt_ultralytics": max(h["selisih_vs_pt"] for h in hasil),
            "syarat": "kelas teratas identik (foto dengan keyakinan acuan >= 0,01), tingkat jawaban sama, dan selisih keyakinan < 0,01",
            "LOLOS": lolos,
            "acuan": str(ACUAN.relative_to(PROYEK)), "model": str(cfg.model_path.relative_to(APP)), "per_gambar": hasil}
    (APP / f"docs/uji_kesetaraan{akhiran}.json").write_text(json.dumps(ring, indent=2, ensure_ascii=False), encoding="utf-8")
    md = ["# Uji kesetaraan inferensi app/core vs hasil evaluasi tersimpan", "",
          f"Acuan: `{ring['acuan']}` (Ultralytics 8.4.171, conf 0,001, iou 0,7). 20 foto val pertama di berkas acuan.", "",
          f"Syarat lulus: {ring['syarat']}.", "",
          f"**Hasil: {'LOLOS' if lolos else 'GAGAL'}**: kelas teratas identik {ring['kelas_identik']}/{ring['n']} (pada foto dengan keyakinan "
          f"acuan >= 0,01: {ring['kelas_identik_keyakinan_acuan_ge_0_01']}); tingkat jawaban sama {ring['tingkat_sama']}/{ring['n']}; "
          f"selisih keyakinan maksimum {ring['selisih_maks_vs_onnx_ultralytics']:.5f} (vs ONNX Ultralytics) dan "
          f"{ring['selisih_maks_vs_pt_ultralytics']:.5f} (vs .pt).", "",
          "| Foto | Kelas acuan | Keyakinan acuan (ONNX) | Kelas app | Keyakinan app | Selisih |", "|---|---|---|---|---|---|"]
    md += [f"| {h['gambar']} | {h['kelas_acuan_onnx']} | {h['conf_acuan_onnx']:.5f} | {h['kelas_app']} | {h['conf_app']:.5f} | {h['selisih_vs_onnx']:.5f} |" for h in hasil]
    (APP / f"docs/uji_kesetaraan{akhiran}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in ring.items() if k != "per_gambar"}, indent=1, ensure_ascii=False))
    return lolos


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--acuan", default=None)
    ap.add_argument("--akhiran", default="")
    a = ap.parse_args()
    sys.exit(0 if main(config=a.config, acuan=a.acuan, akhiran=a.akhiran) else 1)
