#!/usr/bin/env python3
"""
buat_alternatif_yolov8.py - Siapkan versi YOLOv8 dari README dan docs aplikasi TANPA mengaktifkannya (KEPUTUSAN_PROYEK.md 2026-10-15).
Berkas aktif (README.md, docs/*.md, config.yaml) tidak diubah. Keluaran: app/alternatif_yolov8/ (README.md, docs/*.md, AKTIVASI.md).
Angka diambil dari hasil yang sudah ada: docs/uji/sumber_daya_yolov8s.json, docs/uji_kesetaraan_yolov8s.json, config_yolov8.yaml.
"""
import json
from pathlib import Path

import yaml

APP = Path(__file__).resolve().parent.parent
OUT = APP / "alternatif_yolov8"


def koma(x, d=0):
    return f"{x:.{d}f}".replace(".", ",")


def ganti(teks, pasangan, nama):
    for a, b in pasangan:
        if a not in teks:
            raise SystemExit(f"[GAGAL] {nama}: teks tidak ditemukan: {a[:70]!r}")
        teks = teks.replace(a, b)
    return teks


def main():
    cfg = yaml.safe_load((APP / "config_yolov8.yaml").read_text(encoding="utf-8"))
    t = cfg["ambang"]["terdeteksi"]
    sd = json.loads((APP / "docs/uji/sumber_daya_yolov8s.json").read_text(encoding="utf-8"))
    ks = json.loads((APP / "docs/uji_kesetaraan_yolov8s.json").read_text(encoding="utf-8"))
    (OUT / "docs").mkdir(parents=True, exist_ok=True)
    tp = round(100 * t)

    r = (APP / "README.md").read_text(encoding="utf-8")
    i0, i1 = r.index("## Tangkapan layar"), r.index("## Hasil pengujian")
    tangkapan = f"""## Tangkapan layar

Tampilan HP lebar 390 px dengan model YOLOv8s (dua tingkat jawaban, karena tingkat "kemungkinan" tidak dipakai):

| Halaman awal | Terdeteksi |
|---|---|
| ![Halaman awal](docs/tangkapan_layar/awal_390.png) | ![Hasil terdeteksi](docs/tangkapan_layar/terdeteksi_390.png) |

Foto di tangkapan layar berasal dari **RiceLeafDiseaseBD** (Munna dkk., 2026, Mendeley Data V3, doi:10.17632/86s4jzj2m4.3), lisensi
**CC BY 4.0**: `Blast/blast_104.jpg`. Gambar daun dan petunjuk memotret di halaman awal adalah ilustrasi buatan sendiri.

"""
    r = r[:i0] + tangkapan + r[i1:]
    r = ganti(r, [
        ("- Model: YOLO11s (`models/v7_deploy_yolo11s_s2.onnx`, model final aplikasi), dijalankan dengan ONNX Runtime di CPU.",
         "- Model: YOLOv8s (`models/v7_deploy_yolov8s_s2.onnx`, model aplikasi), dijalankan dengan ONNX Runtime di CPU."),
        ("""- Diagnosis: kotak dengan tingkat keyakinan tertinggi. Tingkatnya: **terdeteksi** (≥ 35%), **kemungkinan** (29–35%), dan
  **tidak terdeteksi** (< 29%).""",
         f"""- Diagnosis: kotak dengan tingkat keyakinan tertinggi. Tingkatnya: **terdeteksi** (≥ {tp}%) dan **tidak terdeteksi** (< {tp}%).
  Tingkat "kemungkinan" tidak dipakai: dengan prosedur yang sama di val, batas bawahnya jatuh tepat pada ambang terdeteksi."""),
        ("| Kesetaraan inferensi vs hasil evaluasi tersimpan (20 foto VAL) | kelas teratas identik 20/20; selisih keyakinan maks. 0,002 | `docs/uji_kesetaraan.md` |",
         f"| Kesetaraan inferensi vs hasil Ultralytics (20 foto VAL) | kelas teratas identik {ks['kelas_identik']}/{ks['n']}; selisih keyakinan maks. {koma(ks['selisih_maks_vs_onnx_ultralytics'], 3)} | `docs/uji_kesetaraan_yolov8s.md` |"),
        ("| Black-box (10 foto VAL acak per kelas + 4 skenario masukan + 2 skenario penyampaian) | 10/10, 4/4, dan 2/2 sesuai harapan | `docs/uji/blackbox.md` |",
         "| Black-box (10 foto VAL acak per kelas + 4 skenario masukan + 2 skenario penyampaian) | belum dijalankan untuk YOLOv8s | - |"),
        ("  - **Kemungkinan lain**: bila ada bercak penyakit lain dengan keyakinan ≥ batas bawah (29%), ditampilkan satu di bawah nama penyakit;",
         f"  - **Kemungkinan lain**: bila ada bercak penyakit lain dengan keyakinan ≥ {tp}%, ditampilkan satu di bawah nama penyakit;"),
        ("| RAM | diam 145 MB; puncak 250 MB; 158 MB setelah 11 foto | `docs/uji/sumber_daya.md` |",
         f"| RAM | diam {koma(sd['rss_diam_setelah_model_dimuat_MB'])} MB; puncak {koma(sd['rss_puncak_saat_memproses_MB'])} MB; {koma(sd['rss_diam_setelah_11_foto_MB'])} MB setelah 11 foto | `docs/uji/sumber_daya_yolov8s.md` |"),
        ("| Waktu per foto (CPU laptop) | inti ±0,11 detik; ujung ke ujung di browser ±0,21 detik; foto 4000x3000 ±0,39 detik | `docs/uji/sumber_daya.md` |",
         f"| Waktu per foto (CPU laptop) | inti ±{koma(sd['waktu_inti_ms_median_20_val'] / 1000, 2)} detik; ujung ke ujung di browser ±{koma(sd['waktu_ujung_ke_ujung_ms_median_val'] / 1000, 2)} detik; foto 4000x3000 ±{koma(sd['waktu_ujung_ke_ujung_ms_median_4000x3000'] / 1000, 2)} detik | `docs/uji/sumber_daya_yolov8s.md` |"),
        ("- Model: Ultralytics YOLO11 (AGPL-3.0).", "- Model: Ultralytics YOLOv8 (AGPL-3.0)."),
        ("mengikuti lisensi Ultralytics YOLO11 yang", "mengikuti lisensi Ultralytics YOLOv8 yang"),
    ], "README")
    (OUT / "README.md").write_text(r, encoding="utf-8")

    a = (APP / "docs/arsitektur.md").read_text(encoding="utf-8")
    a = ganti(a, [
        ("M[(models/v7_deploy_yolo11s_s2.onnx)]", "M[(models/v7_deploy_yolov8s_s2.onnx)]"),
        ("C[(config.yaml<br/>ambang 0,35 dan 0,29,<br/>batas tampilan 0,10, teks)]", f"C[(config.yaml<br/>ambang {koma(t, 2)},<br/>batas tampilan 0,10, teks)]"),
        ("| kemungkinan | K01 Diagnosis belum pasti | `dugaan`, `perlu_konfirmasi = ya` | K02 | konfirmasi ke penyuluh + pencegahan umum |",
         "| kemungkinan | (tidak terjadi dengan YOLOv8s: batas bawah = ambang terdeteksi) | - | - | - |"),
        ("- RAM diam 145 MB, puncak 250 MB, dan 158 MB setelah 11 foto;",
         f"- RAM diam {koma(sd['rss_diam_setelah_model_dimuat_MB'])} MB, puncak {koma(sd['rss_puncak_saat_memproses_MB'])} MB, dan {koma(sd['rss_diam_setelah_11_foto_MB'])} MB setelah 11 foto;"),
        ("hasil: `docs/uji/sumber_daya.md`", "hasil: `docs/uji/sumber_daya_yolov8s.md`"),
    ], "arsitektur")
    (OUT / "docs/arsitektur.md").write_text(a, encoding="utf-8")

    d = (APP / "docs/deploy.md").read_text(encoding="utf-8")
    d = ganti(d, [("mengikuti model YOLO11 dari Ultralytics", "mengikuti model YOLOv8 dari Ultralytics")], "deploy")
    (OUT / "docs/deploy.md").write_text(d, encoding="utf-8")

    s = (APP / "docs/desain.md").read_text(encoding="utf-8")
    s = ganti(s, [('| Baris info "YOLO11s · ONNX · 61 ms"', '| Baris info "YOLOv8s · ONNX · 69 ms"')], "desain")
    (OUT / "docs/desain.md").write_text(s, encoding="utf-8")

    (OUT / "AKTIVASI.md").write_text(f"""# Mengaktifkan model YOLOv8s di aplikasi (BELUM dilakukan)

Disiapkan 2026-10-15 karena judul dan rumusan masalah proposal menyebut YOLOv8 (KEPUTUSAN_PROYEK.md). Alasannya kesesuaian judul,
BUKAN kinerja: di test, YOLO11s sedikit lebih baik. Aktifkan hanya bila dospem memilih YOLOv8.

## Mencoba tanpa mengaktifkan

```powershell
cd <folder app>
$env:TILIK_CONFIG = "config_yolov8.yaml"
.venv\\Scripts\\streamlit run streamlit_app.py
```

## Mengaktifkan

1. Arsipkan konfigurasi YOLO11s: salin `config.yaml` menjadi `config_yolo11s_arsip.yaml`.
2. Salin `config_yolov8.yaml` menjadi `config.yaml`.
3. Salin `alternatif_yolov8/README.md` ke `README.md` dan `alternatif_yolov8/docs/*.md` ke `docs/`.
4. Salin hasil uji YOLOv8s menjadi nama aktif: `docs/uji_kesetaraan_yolov8s.md` dan `docs/uji/sumber_daya_yolov8s.md` (atau sesuaikan tautan).
5. Ambil ulang tangkapan layar (`tools/tangkap_layar.py`). Tingkat "kemungkinan" tidak ada, dan contoh "tidak terdeteksi" perlu foto
   lain, karena foto `Healthy_1002` didiagnosis Brown spot 49% oleh YOLOv8s.
6. Ubah test `test_config_aktif_tetap_yolo11s_dan_tiga_tingkat` di `tests/test_inti.py` agar sesuai model aktif. Jalankan `pytest`.
7. Ganti `models/v7_deploy_yolo11s_s2.onnx` dengan `models/v7_deploy_yolov8s_s2.onnx` di daftar izin `tools/siapkan_unggah.py`, lalu
   bangun ulang folder unggah. Model YOLO11s tetap disimpan di `app/models` sebagai arsip.

Ambang: terdeteksi = {koma(t, 2)}, rendah = {koma(cfg['ambang']['rendah'], 2)} (`reports/final_eval_v7/aplikasi/tiga_tingkat_val_yolov8s.json`).
""", encoding="utf-8")
    print("ok:", sorted(p.relative_to(APP).as_posix() for p in OUT.rglob("*.md")))


if __name__ == "__main__":
    main()
