#!/usr/bin/env python3
"""Tangkapan layar aplikasi Tilik Padi dengan device scale factor 2 (HP 390 px dan desktop 1280 px, ukuran CSS) untuk gambar 14 bahan
presentasi. Memakai ulang CDP dan daftar foto dari app/tools/tangkap_layar.py tanpa mengubah alat itu maupun app/docs/tangkapan_layar/.
Aplikasi harus SUDAH berjalan (streamlit run streamlit_app.py --server.port 8599 dari folder app/).
Keluaran: bahan_presentasi/skripsi_id/14_tangkapan_layar_asli/<keadaan>_<lebar>.png (piksel = 2 x ukuran CSS)

  .venv\\Scripts\\python.exe scripts\\presentasi\\tangkap_layar_2x.py [--url http://localhost:8599]"""
import argparse
import base64
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "app/tools"))
import tangkap_layar as tl  # noqa: E402

OUT = ROOT / "bahan_presentasi/skripsi_id/14_tangkapan_layar_asli"
SKALA = 2
KEADAAN = ["terdeteksi", "kemungkinan", "kemungkinan_lain", "foto_ulang"]     # + awal dan tentang


def atur(c, lebar, tinggi):
    c("Emulation.setDeviceMetricsOverride", width=lebar, height=tinggi, deviceScaleFactor=SKALA, mobile=lebar < 600)


def buka(c, url, lebar):
    atur(c, lebar, 900)
    c("Page.navigate", url=url)
    c.tunggu("!!document.querySelector('h1.judul')")
    c.tunggu("document.fonts.status === 'loaded'")
    time.sleep(1.0)


def potret(c, path, lebar):
    tinggi = c.js("""Math.max(document.body.scrollHeight,
        ...[...document.querySelectorAll('[data-testid="stAppViewContainer"],[data-testid="stMain"],section.main')].map(e => e.scrollHeight))""")
    tinggi = int(min(max(tinggi or 900, 700), 8000))
    atur(c, lebar, tinggi)
    time.sleep(1.2)
    data = c("Page.captureScreenshot", format="png", captureBeyondViewport=True)["data"]
    path.write_bytes(base64.b64decode(data))
    return tinggi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8599")
    a = ap.parse_args()
    if OUT.exists():
        shutil.rmtree(OUT)                       # salinan 1x lama (dibuat oleh gambar_skripsi.py) diganti versi 2x
    OUT.mkdir(parents=True)
    exe = next(p for p in tl.CHROME if Path(p).exists())
    prof = tempfile.mkdtemp(prefix="cdp2x_")
    pr = subprocess.Popen([exe, "--headless=new", "--remote-debugging-port=9235", f"--user-data-dir={prof}", "--hide-scrollbars",
                           "--no-first-run", "--no-default-browser-check", "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    hasil = {}
    try:
        for _ in range(50):
            try:
                tabs = json.loads(urllib.request.urlopen("http://127.0.0.1:9235/json").read())
                ws = next(t["webSocketDebuggerUrl"] for t in tabs if t["type"] == "page")
                break
            except Exception:
                time.sleep(0.2)
        c = tl.CDP(ws)
        c("Page.enable")
        c("Runtime.enable")
        for lebar in (390, 1280):
            buka(c, a.url, lebar)
            hasil[f"awal_{lebar}"] = potret(c, OUT / f"awal_{lebar}.png", lebar)
            for k in KEADAAN:
                f = tl.FOTO[k]
                buka(c, a.url, lebar)
                c.tunggu("!!document.querySelector('input[type=file]')")
                obj = c("Runtime.evaluate", expression="document.querySelector('input[type=file]')")["result"]["objectId"]
                c("DOM.setFileInputFiles", files=[str(tl.VAL / f)], objectId=obj)
                c.tunggu("!!document.querySelector('.pita') || !!document.querySelector('.pesan-salah') || "
                         "!!document.querySelector('[data-testid=\"stException\"]')", batas=90)
                c.tunggu("[...document.querySelectorAll('[data-testid=\"stImage\"] img')].every(i => i.complete && i.naturalWidth > 0)")
                galat = c.js("document.querySelector('[data-testid=\"stException\"]')?.innerText || ''")
                if galat:
                    raise RuntimeError(f"Streamlit galat ({k}): {galat[:300]}")
                time.sleep(0.8)
                hasil[f"{k}_{lebar}"] = potret(c, OUT / f"{k}_{lebar}.png", lebar)
            atur(c, lebar, 900)
            c("Page.navigate", url=a.url.rstrip("/") + "/tentang")
            c.tunggu("!!document.querySelector('.tentang')")
            time.sleep(1.0)
            hasil[f"tentang_{lebar}"] = potret(c, OUT / f"tentang_{lebar}.png", lebar)
    finally:
        pr.terminate()
        shutil.rmtree(prof, ignore_errors=True)
    print(json.dumps(hasil, indent=1))


if __name__ == "__main__":
    main()
