#!/usr/bin/env python3
"""
masker_rembg.py - Masker daun dengan rembg (penghapus latar umum, salient object). Dijalankan dengan venv TERPISAH tools/_venv_latar.
Tidak memakai label. Keluaran: masker mentah (alfa 0-255, PNG) di data/derived/v9_latar/<metode>/masker_mentah/ + waktu per foto.

  tools/_venv_latar/Scripts/python scripts/v9_latar/masker_rembg.py --model u2net --daftar <berkas teks nama foto> --nama rembg_u2net
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("U2NET_HOME", str(ROOT / "tools/_model_latar/u2net"))
from PIL import Image  # noqa: E402
from rembg import new_session, remove  # noqa: E402

VAL_IMG = ROOT / "data/processed/prov_v7/images/val"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="u2net")
    ap.add_argument("--daftar", required=True)
    ap.add_argument("--nama", required=True)
    a = ap.parse_args()
    out = ROOT / "data/derived/v9_latar" / a.nama / "masker_mentah"
    out.mkdir(parents=True, exist_ok=True)
    sesi = new_session(a.model)
    berkas = [b for b in Path(a.daftar).read_text(encoding="utf-8").split("\n") if b.strip()]
    waktu = {}
    for i, f in enumerate(berkas):
        tujuan = out / (Path(f).stem + ".png")
        if tujuan.exists():
            continue
        im = Image.open(VAL_IMG / f).convert("RGB")
        t0 = time.perf_counter()
        m = remove(im, session=sesi, only_mask=True)
        waktu[f] = time.perf_counter() - t0
        m.save(tujuan)
        if i % 100 == 0:
            print(i, len(berkas), f"{waktu[f]:.2f}s", flush=True)
    wp = out.parent / "waktu.json"
    lama = json.loads(wp.read_text(encoding="utf-8")) if wp.exists() else {}
    lama.update(waktu)
    wp.write_text(json.dumps({"perangkat": "CPU (onnxruntime)", **{k: round(v, 4) for k, v in lama.items() if k != "perangkat"}}, indent=0),
                  encoding="utf-8")
    print("selesai", a.nama, len(berkas))


if __name__ == "__main__":
    sys.exit(main())
