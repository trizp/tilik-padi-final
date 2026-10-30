#!/usr/bin/env python3
"""pindai_val.py - Jalankan alur aplikasi pada semua foto VAL prov_v7 untuk memilih foto uji black-box dan tangkapan layar.
Hanya VAL (bukan test, test_new_session, atau ds3). Catatan: model aplikasi dilatih dengan train+val+test (v9: prov_v9_deploy), jadi
foto val termasuk data latihnya; hasil di sini bukan ukuran generalisasi. Alur v9: foto yang diminta foto ulang dicatat foto_ulang=1.
Keluaran: docs/uji/pindai_val.csv"""
import csv
import sys
from pathlib import Path

APP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP))
from core import alur, gambar, inferensi, konfigurasi, pemisah_daun  # noqa: E402
from rbs import mesin  # noqa: E402

PROYEK = APP.parent


def main():
    cfg = konfigurasi.muat()
    det = inferensi.Detektor(cfg.model_path, cfg.imgsz, cfg.conf_min, cfg.iou, cfg.max_det)
    pem = pemisah_daun.PemisahDaun(cfg.pemisah_path, cfg.pemisah_tau) if cfg.pemisah_path else None
    aturan = mesin.muat_aturan()
    man = [r for r in csv.DictReader(open(PROYEK / "data/processed/prov_v7/manifest.csv", encoding="utf-8")) if r["split"] == "val"]
    out = []
    for r in man:
        p = PROYEK / "data/processed/prov_v7/images/val" / r["file"]
        try:
            h = alur.periksa_foto(p.read_bytes(), cfg, det, aturan, pem)
        except gambar.FotoPerluDiulang:
            out.append({"file": r["file"], "label": r["class"], "sumber": r["source"], "foto_ulang": 1, "diagnosis": "", "tingkat": "",
                        "keyakinan": "", "aturan": "", "kemungkinan_lain": "", "keyakinan_lain": "", "bercak_rendah": "",
                        "cara_membedakan": ""})
            continue
        out.append({"file": r["file"], "label": r["class"], "sumber": r["source"], "foto_ulang": 0,
                    "diagnosis": h.diagnosis.penyakit or "tidak_ada",
                    "tingkat": h.diagnosis.tingkat, "keyakinan": round(h.diagnosis.keyakinan, 4),
                    "aturan": "+".join(L.aturan.id for L in h.inferensi.jejak),
                    "kemungkinan_lain": h.kemungkinan_lain.penyakit if h.kemungkinan_lain else "",
                    "keyakinan_lain": round(h.kemungkinan_lain.keyakinan, 4) if h.kemungkinan_lain else "",
                    "bercak_rendah": round(h.bercak_rendah.keyakinan, 4) if h.bercak_rendah else "",
                    "cara_membedakan": h.cara_membedakan})
    with open(APP / "docs/uji/pindai_val.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    from collections import Counter
    print(len(out), "foto ulang:", sum(o["foto_ulang"] for o in out), Counter(o["tingkat"] for o in out), "kemungkinan lain:", sum(1 for o in out if o["kemungkinan_lain"]),
          "bercak rendah:", sum(1 for o in out if o["bercak_rendah"] != ""))


if __name__ == "__main__":
    main()
