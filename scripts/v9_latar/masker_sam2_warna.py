#!/usr/bin/env python3
"""
masker_sam2_warna.py - Masker daun TANPA label dengan (a) SAM2 mode otomatis ("segment everything") + aturan memilih daun, dan
(b) ambang warna HSV sebagai pembanding.

- SAM2 lewat ultralytics yang SUDAH ada di venv proyek; tidak ada paket yang dipasang (YOLO_AUTOINSTALL=false; pip freeze dibandingkan
  sebelum/sesudah). Bobot sam2.1_b.pt diunduh ke tools/_model_latar/sam/.
- Grid 16x16 titik, batch 16 (bawaan 32x32/64 memenuhi GPU 4 GB).
- Aturan memilih daun dari semua masker SAM2 (tanpa label): luas 3-95% foto; >= 50% pikselnya berwarna vegetasi (HSV H 8-100,
  S >= 40, V >= 40); skor = sqrt(luas) x (0,5 + 0,5 x kedekatan ke pusat) x ketajaman relatif (rerata |Laplacian| di masker / di
  seluruh foto). Masker berskor tertinggi = daun utama.
- Ambang warna: piksel vegetasi (aturan HSV yang sama) -> buka/tutup morfologi; komponen besar dipilih oleh pasca-proses bersama.

  .venv/Scripts/python scripts/v9_latar/masker_sam2_warna.py --metode sam2|warna --daftar <berkas> --nama <folder>
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["YOLO_AUTOINSTALL"] = "false"
import cv2  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
V7_IMG = ROOT / "data/processed/prov_v7/images"
SAM_DIR = ROOT / "tools/_model_latar/sam"


def vegetasi(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (h >= 8) & (h <= 100) & (s >= 40) & (v >= 40)


def masker_warna(bgr):
    m = vegetasi(bgr).astype(np.uint8)
    k = max(3, round(min(m.shape) / 120)) | 1
    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, el), cv2.MORPH_CLOSE, el)
    return m.astype(bool), {}


def pilih_daun(bgr, masks):
    """Aturan tanpa label: lihat docstring modul. Kembalikan (masker terpilih, info)."""
    h, w = bgr.shape[:2]
    veg = vegetasi(bgr)
    lap = np.abs(cv2.Laplacian(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.CV_32F, ksize=3))
    lap_rata = float(lap.mean()) + 1e-6
    yy, xx = np.mgrid[0:h, 0:w]
    terbaik, info = None, {"kandidat": 0}
    for m in masks:
        luas = float(m.mean())
        if not (0.03 <= luas <= 0.95):
            continue
        if veg[m].mean() < 0.5:
            continue
        info["kandidat"] += 1
        cy, cx = yy[m].mean() / h, xx[m].mean() / w
        pusat = 1 - min(1.0, np.hypot(cx - 0.5, cy - 0.5) / np.hypot(0.5, 0.5))
        tajam = float(lap[m].mean()) / lap_rata
        skor = np.sqrt(luas) * (0.5 + 0.5 * pusat) * tajam
        if terbaik is None or skor > terbaik[0]:
            terbaik = (skor, m, {"luas": round(luas, 4), "pusat": round(float(pusat), 3), "tajam": round(tajam, 3)})
    if terbaik is None:
        return np.zeros((h, w), bool), info
    info.update(terbaik[2])
    return terbaik[1], info


def pilih_daun_v2(bgr, masks):
    """Aturan v2 (tanpa label), revisi setelah melihat montase v1: skor ketajaman memilih latar bertekstur (gulma) dan foto tanpa
    kandidat menjadi abu-abu seluruhnya. Kandidat sama (luas 3-95%, >= 50% piksel vegetasi); skor = sqrt(luas) x (0,2 + cakupan
    kotak tengah 40% x 40% foto). Bila tidak ada kandidat atau terpilih < 5% foto: CADANGAN = foto tidak dipisah (masker penuh)."""
    h, w = bgr.shape[:2]
    veg = vegetasi(bgr)
    y0, y1, x0, x1 = int(0.3 * h), int(0.7 * h), int(0.3 * w), int(0.7 * w)
    terbaik, info = None, {"kandidat": 0, "cadangan": False}
    for m in masks:
        luas = float(m.mean())
        if not (0.03 <= luas <= 0.95) or veg[m].mean() < 0.5:
            continue
        info["kandidat"] += 1
        tengah = float(m[y0:y1, x0:x1].mean())
        skor = np.sqrt(luas) * (0.2 + tengah)
        if terbaik is None or skor > terbaik[0]:
            terbaik = (skor, m, {"luas": round(luas, 4), "tengah": round(tengah, 3)})
    if terbaik is None or terbaik[2]["luas"] < 0.05:
        info["cadangan"] = True
        return np.ones((h, w), bool), info
    info.update(terbaik[2])
    return terbaik[1], info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metode", choices=["sam2", "warna"], required=True)
    ap.add_argument("--daftar", required=True)
    ap.add_argument("--nama", required=True)
    ap.add_argument("--bobot", default="sam2.1_b.pt")
    ap.add_argument("--stride", type=int, default=16, help="titik grid per sisi (bawaan ultralytics 32)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--aturan", choices=["v1", "v2"], default="v2", help="aturan memilih daun dari masker SAM2")
    ap.add_argument("--split", default="val", choices=["train", "val", "test", "test_new_session"],
                    help="split prov_v7 sumber foto (masker dibuat TANPA label di semua split)")
    ap.add_argument("--folder-sumber", default=None,
                    help="folder sumber lain (mis. data/raw/ds3); daftar berisi jalur relatif terhadap folder ini; nama masker = jalur dengan '/' -> '__'")
    a = ap.parse_args()
    out = ROOT / "data/derived/v9_latar" / a.nama / "masker_mentah"
    out.mkdir(parents=True, exist_ok=True)
    berkas = [b for b in Path(a.daftar).read_text(encoding="utf-8").split("\n") if b.strip()]
    model = None
    if a.metode == "sam2":
        import torch
        from ultralytics import SAM
        SAM_DIR.mkdir(parents=True, exist_ok=True)
        cwd = os.getcwd()
        os.chdir(SAM_DIR)                       # unduhan bobot ke tools/_model_latar/sam
        model = SAM(a.bobot)
        os.chdir(cwd)
        # grid titik lebih jarang dan batch kecil: grid bawaan 32x32/batch 64 memenuhi 4 GB GPU dan melambat (>60 s/foto)
        from ultralytics.models.sam.predict import Predictor
        asli = Predictor.generate

        def generate(self, im, *x, **k):
            k.setdefault("points_stride", a.stride)
            k.setdefault("points_batch_size", a.batch)
            return asli(self, im, *x, **k)
        Predictor.generate = generate
    waktu, info_semua = {}, {}
    for i, f in enumerate(berkas):
        tujuan = out / ((f.replace("/", "__").rsplit(".", 1)[0] if a.folder_sumber else Path(f).stem) + ".png")
        if tujuan.exists():
            continue
        bgr = cv2.imread(str((ROOT / a.folder_sumber / f) if a.folder_sumber else (V7_IMG / a.split / f)))
        t0 = time.perf_counter()
        if a.metode == "sam2":
            r = model(bgr, verbose=False, device=0)[0]
            masks = r.masks.data.cpu().numpy().astype(bool) if r.masks is not None else np.zeros((0,) + bgr.shape[:2], bool)
            if masks.shape[1:] != bgr.shape[:2]:
                masks = np.stack([cv2.resize(m.astype(np.uint8), (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
                                  for m in masks]) if len(masks) else np.zeros((0,) + bgr.shape[:2], bool)
            m, info = (pilih_daun_v2 if a.aturan == "v2" else pilih_daun)(bgr, masks)
            info["n_masker_sam"] = int(len(masks))
            torch.cuda.synchronize()
        else:
            m, info = masker_warna(bgr)
        waktu[f] = time.perf_counter() - t0
        info_semua[f] = info
        cv2.imwrite(str(tujuan), (m.astype(np.uint8) * 255))
        if i % 50 == 0:
            print(i, len(berkas), f"{waktu[f]:.2f}s", info, flush=True)
    for nama, isi in (("waktu.json", waktu), ("info.json", info_semua)):
        p = out.parent / nama
        lama = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        lama.update({k: (round(v, 4) if isinstance(v, float) else v) for k, v in isi.items()})
        if nama == "waktu.json":
            lama["perangkat"] = "GPU RTX 3050 Ti (SAM2)" if a.metode == "sam2" else "CPU"
        p.write_text(json.dumps(lama, indent=0), encoding="utf-8")
    print("selesai", a.nama, len(berkas))


if __name__ == "__main__":
    sys.exit(main())
