"""Fase 6 aplikasi v9 (KEPUTUSAN_PROYEK.md 2026-10-20, pilihan a): salin kedua model ONNX ke app/models.
- detektor: runs/v9_deploy_yolov8s_s2/weights/last.onnx (metadata sudah dibersihkan di fase 4; sha diverifikasi);
- pemisah daun: runs/v9_pemisah_daun_yolov8n_seg/weights/best.onnx; metadata "description" berisi jalur lokal -> diganti teks netral pada
  SALINAN (berkas di runs/ tidak diubah); graf diverifikasi identik byte demi byte.
Jalankan dengan venv proyek: .venv\\Scripts\\python.exe scripts\\v9_latar\\fase6_salin_model_aplikasi.py"""
import hashlib
import json
import shutil
from pathlib import Path

import onnx as onnxlib

ROOT = Path(__file__).resolve().parents[2]
APP_MODELS = ROOT / "app" / "models"
OUT = ROOT / "reports" / "v9_latar" / "aplikasi_v9" / "fase6_model_aplikasi.json"
DET_SRC = ROOT / "runs" / "v9_deploy_yolov8s_s2" / "weights" / "last.onnx"
DET_SHA = "bdcbc52941c767ace4fd932e6512d6930d48b5713ca5416b0ddad19e8716d1d5"
SEG_SRC = ROOT / "runs" / "v9_pemisah_daun_yolov8n_seg" / "weights" / "best.onnx"
SEG_DESC = "Ultralytics YOLOv8n-seg model trained on v9_daun_seg"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def berisi_jalur(mo):
    return [p.key for p in mo.metadata_props if ":\\" in p.value or "TELKOM" in p.value or "TRISTAN" in p.value or "Tilik" in p.value]


def main():
    assert sha(DET_SRC) == DET_SHA, "sha detektor tidak sama dengan fase 4"
    det_dst = APP_MODELS / "v9_deploy_yolov8s_s2.onnx"
    shutil.copy2(DET_SRC, det_dst)
    assert sha(det_dst) == DET_SHA

    sha_seg_asal = sha(SEG_SRC)
    mo = onnxlib.load(str(SEG_SRC))
    g0 = mo.graph.SerializeToString()
    for p in mo.metadata_props:
        if p.key == "description":
            p.value = SEG_DESC
    seg_dst = APP_MODELS / "v9_pemisah_daun_yolov8n_seg.onnx"
    onnxlib.save(mo, str(seg_dst))
    m2 = onnxlib.load(str(seg_dst))
    assert m2.graph.SerializeToString() == g0, "graf pemisah daun berubah"
    assert sha(SEG_SRC) == sha_seg_asal, "berkas asal berubah"

    res = {
        "detektor": {"asal": str(DET_SRC.relative_to(ROOT)), "tujuan": str(det_dst.relative_to(ROOT)), "sha256": sha(det_dst),
                     "metadata_berisi_jalur": berisi_jalur(onnxlib.load(str(det_dst))), "ukuran_MB": round(det_dst.stat().st_size / 2 ** 20, 2)},
        "pemisah_daun": {"asal": str(SEG_SRC.relative_to(ROOT)), "sha256_asal": sha_seg_asal, "tujuan": str(seg_dst.relative_to(ROOT)),
                         "sha256": sha(seg_dst), "metadata_description": SEG_DESC, "graf_identik": True,
                         "metadata_berisi_jalur": berisi_jalur(m2), "ukuran_MB": round(seg_dst.stat().st_size / 2 ** 20, 2)},
    }
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
