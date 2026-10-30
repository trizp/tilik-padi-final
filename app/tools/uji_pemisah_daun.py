"""uji_pemisah_daun.py - Kesetaraan pemisah daun di aplikasi (core.pemisah_daun lewat gambar.baca_foto, seperti saat foto diunggah)
dengan masker yang dipakai di evaluasi skripsi (fase 1-5; scripts/v9_latar, foto dibaca cv2.imread) pada SEMUA foto VAL prov_v7 asli.
Yang dibandingkan: keputusan cadangan (foto ulang) dan IoU masker. Hanya VAL; butuh dataset proyek.
Keluaran: docs/uji/pemisah_daun.json dan docs/uji/pemisah_daun.md.

Jalankan dari folder app/:  .venv\\Scripts\\python tools\\uji_pemisah_daun.py"""
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

APP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP))
from core import gambar, konfigurasi, pemisah_daun  # noqa: E402

PROYEK = APP.parent
VAL = PROYEK / "data/processed/prov_v7/images/val"
ACUAN = PROYEK / "data/derived/v9_latar/seg_n_val"
OUT = APP / "docs/uji"


def main():
    cfg = konfigurasi.muat()
    info = json.loads((ACUAN / "info.json").read_text(encoding="utf-8"))
    assert info["tau"] == cfg.pemisah_tau
    pem = pemisah_daun.PemisahDaun(cfg.pemisah_path, cfg.pemisah_tau)
    iou, beda, waktu = [], [], []
    n_cad_app = n_cad_acuan = 0
    for f, cad_acuan in sorted(info["cadangan"].items()):
        arr = gambar.ke_array(gambar.baca_foto((VAL / f).read_bytes(), cfg.ukuran_maks_mb, cfg.sisi_maks_proses))
        t0 = time.perf_counter()
        m, cad = pem.pisahkan(arr)
        waktu.append(1000 * (time.perf_counter() - t0))
        n_cad_app += cad
        n_cad_acuan += cad_acuan
        if cad != cad_acuan:
            beda.append({"foto": f, "cadangan_aplikasi": cad, "cadangan_acuan": cad_acuan})
            continue
        if not cad:
            a = np.asarray(Image.open(ACUAN / "masker" / (Path(f).stem + ".png"))) > 0
            if a.shape != m.shape:
                beda.append({"foto": f, "ukuran_beda": [list(a.shape), list(m.shape)]})
                continue
            iou.append(float((m & a).sum() / max((m | a).sum(), 1)))
    res = {
        "foto_val": len(info["cadangan"]),
        "cadangan_aplikasi": n_cad_app, "cadangan_acuan": n_cad_acuan,
        "keputusan_cadangan_sama": len(info["cadangan"]) - sum("cadangan_aplikasi" in b for b in beda),
        "beda": beda,
        "iou_masker": {"n": len(iou), "median": round(statistics.median(iou), 5), "minimum": round(min(iou), 5),
                       "persen_iou_ge_0_99": round(100 * sum(x >= 0.99 for x in iou) / len(iou), 2),
                       "persen_iou_ge_0_98": round(100 * sum(x >= 0.98 for x in iou) / len(iou), 2)},
        "waktu_pisahkan_ms_median": round(statistics.median(waktu), 1),
        "catatan": "Acuan = masker fase 1 (data/derived/v9_latar/seg_n_val; tau 0,35). Aplikasi membaca foto lewat PIL + koreksi EXIF + "
                   "batas sisi 2048 px; acuan lewat cv2.imread pada resolusi asli.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pemisah_daun.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    i = res["iou_masker"]
    (OUT / "pemisah_daun.md").write_text("\n".join([
        "# Uji kesetaraan pemisah daun (aplikasi vs evaluasi skripsi)", "",
        f"Semua {res['foto_val']} foto VAL prov_v7 asli. Acuan: masker YOLOv8n-seg fase 1 yang dipakai di evaluasi alur aplikasi (tau 0,35).",
        "Jalur aplikasi: `gambar.baca_foto` (PIL, koreksi EXIF, sisi maks. 2048 px) lalu `core.pemisah_daun.PemisahDaun.pisahkan`.", "",
        "| Ukuran | Hasil |", "|---|---|",
        f"| Keputusan cadangan (foto ulang) sama | {res['keputusan_cadangan_sama']}/{res['foto_val']} |",
        f"| Foto cadangan: aplikasi / acuan | {res['cadangan_aplikasi']} / {res['cadangan_acuan']} |",
        f"| IoU masker (foto bukan cadangan, n = {i['n']}) | median {i['median']}; minimum {i['minimum']}; "
        f"{i['persen_iou_ge_0_99']}% >= 0,99; {i['persen_iou_ge_0_98']}% >= 0,98 |",
        f"| Waktu pisahkan per foto (median, CPU) | {res['waktu_pisahkan_ms_median']} ms |", "",
        "Foto yang berbeda: " + (", ".join(f"`{b['foto']}`" for b in beda) if beda else "tidak ada") + ".", ""]), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "beda"}, indent=1, ensure_ascii=False), "\nbeda:", beda[:10])
    return res


if __name__ == "__main__":
    main()
