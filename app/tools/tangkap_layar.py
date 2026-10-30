#!/usr/bin/env python3
"""
tangkap_layar.py - Tangkapan layar aplikasi lewat Chrome headless + DevTools Protocol (websocket-client; tanpa Playwright/Selenium).
Membuka aplikasi yang SUDAH berjalan, mengisi input berkas (tombol "Unggah foto") dengan foto VAL, menunggu hasil, lalu memotret
seluruh halaman pada lebar 390 px dan 1280 px. Kamera memakai perangkat palsu Chrome (--use-fake-device-for-media-stream).
Foto uji tidak disalin ke folder app/.

  (terminal 1)  cd app && .venv/Scripts/streamlit run streamlit_app.py --server.headless true --server.port 8599
  (terminal 2)  app/.venv/Scripts/python.exe app/tools/tangkap_layar.py [--url http://localhost:8599] [--akhiran _v1]
Keluaran: app/docs/tangkapan_layar/<kondisi>_<lebar>.png
"""
import argparse
import base64
import json
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import websocket

APP = Path(__file__).resolve().parent.parent
VAL = APP.parent / "data/processed/prov_v7/images/val"
# Alur v9 (2026-10-20): foto yang sama dengan v7 bila masih memperlihatkan kondisinya (pindai_val.csv), ditambah tingkat kemungkinan dan
# foto ulang (foto yang sama dengan skenario uji black-box). Foto v7 YOLOv8s: docs/arsip_v7_yolov8s/; YOLO11s: docs/arsip_yolo11s/
FOTO = {"terdeteksi": "ds4_blast-104_67def77f.jpg", "tidak_terdeteksi": "ds4_Healthy-1014_c5aa8fd9.jpg",
        "kemungkinan_lain": "ds4_blast-321_7c914ca8.jpg",
        "bercak_kecil": "ds4_brown-spot-1148_4f78bf6e.jpg",
        "kemungkinan": "ds4_blast-414_94feb6e6.jpg",
        "foto_ulang": "ds4_Healthy-1432_5998a115.jpg"}
CHROME = [r"C:\Program Files\Google\Chrome\Application\chrome.exe", r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"]


class CDP:
    def __init__(self, ws_url):
        self.ws = websocket.create_connection(ws_url, timeout=60, suppress_origin=True)
        self.n = 0

    def __call__(self, method, **params):
        self.n += 1
        self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            m = json.loads(self.ws.recv())
            if m.get("id") == self.n:
                if "error" in m:
                    raise RuntimeError(f"{method}: {m['error']}")
                return m.get("result", {})

    def js(self, expr):
        r = self("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
        return r.get("result", {}).get("value")

    def tunggu(self, expr, batas=60, jeda=0.3):
        t0 = time.time()
        while time.time() - t0 < batas:
            if self.js(expr):
                return True
            time.sleep(jeda)
        raise TimeoutError(expr)


def potret(c, path, lebar):
    tinggi = c.js("""Math.max(document.body.scrollHeight,
        ...[...document.querySelectorAll('[data-testid="stAppViewContainer"],[data-testid="stMain"],section.main')].map(e => e.scrollHeight))""")
    tinggi = int(min(max(tinggi or 900, 700), 8000))
    c("Emulation.setDeviceMetricsOverride", width=lebar, height=tinggi, deviceScaleFactor=1, mobile=lebar < 600)
    time.sleep(1.0)
    data = c("Page.captureScreenshot", format="png", captureBeyondViewport=True)["data"]
    path.write_bytes(base64.b64decode(data))
    return tinggi


def buka(c, url, lebar):
    c("Emulation.setDeviceMetricsOverride", width=lebar, height=900, deviceScaleFactor=1, mobile=lebar < 600)
    c("Page.navigate", url=url)
    c.tunggu("!!document.querySelector('h1.judul')")
    c.tunggu("document.fonts.status === 'loaded'")
    time.sleep(1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8599")
    ap.add_argument("--akhiran", default="")
    a = ap.parse_args()
    out = APP / "docs/tangkapan_layar"
    out.mkdir(parents=True, exist_ok=True)
    exe = next(p for p in CHROME if Path(p).exists())
    prof = tempfile.mkdtemp(prefix="cdp_")
    pr = subprocess.Popen([exe, "--headless=new", "--remote-debugging-port=9233", f"--user-data-dir={prof}", "--hide-scrollbars",
                           "--no-first-run", "--no-default-browser-check", "--use-fake-device-for-media-stream",
                           "--use-fake-ui-for-media-stream", "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):
            try:
                tabs = json.loads(urllib.request.urlopen("http://127.0.0.1:9233/json").read())
                ws = next(t["webSocketDebuggerUrl"] for t in tabs if t["type"] == "page")
                break
            except Exception:
                time.sleep(0.3)
        c = CDP(ws)
        c("Page.enable")
        c("Runtime.enable")
        hasil = {}
        for lebar in (390, 1280):
            buka(c, a.url, lebar)
            hasil[f"awal_{lebar}"] = potret(c, out / f"awal_{lebar}{a.akhiran}.png", lebar)
            buka(c, a.url, lebar)
            c.js("[...document.querySelectorAll('button')].find(b => b.innerText.trim() === 'Ambil foto daun').click()")
            c.tunggu("!!document.querySelector('[data-testid=stCameraInput] video') && "
                     "document.querySelector('[data-testid=stCameraInput] video').readyState >= 2", batas=30)
            time.sleep(1.5)
            hasil[f"kamera_{lebar}"] = potret(c, out / f"kamera_{lebar}{a.akhiran}.png", lebar)
            for kondisi, f in FOTO.items():
                buka(c, a.url, lebar)
                c.tunggu("!!document.querySelector('input[type=file]')")
                obj = c("Runtime.evaluate", expression="document.querySelector('input[type=file]')")["result"]["objectId"]
                c("DOM.setFileInputFiles", files=[str(VAL / f)], objectId=obj)
                c.tunggu("!!document.querySelector('.pita') || !!document.querySelector('.pesan-salah') || "
                         "!!document.querySelector('[data-testid=\"stException\"]')", batas=90)
                c.tunggu("[...document.querySelectorAll('[data-testid=\"stImage\"] img')].every(i => i.complete && i.naturalWidth > 0)")
                galat = c.js("document.querySelector('[data-testid=\"stException\"]')?.innerText || ''")
                if galat:
                    raise RuntimeError(f"Streamlit galat ({kondisi}): {galat[:500]}")
                time.sleep(0.8)
                hasil[f"{kondisi}_{lebar}"] = potret(c, out / f"{kondisi}_{lebar}{a.akhiran}.png", lebar)
            c("Emulation.setDeviceMetricsOverride", width=lebar, height=900, deviceScaleFactor=1, mobile=lebar < 600)
            c("Page.navigate", url=a.url.rstrip("/") + "/tentang")
            c.tunggu("!!document.querySelector('.tentang')")
            time.sleep(1.0)
            hasil[f"tentang_{lebar}"] = potret(c, out / f"tentang_{lebar}{a.akhiran}.png", lebar)
        print(json.dumps(hasil, indent=1))
    finally:
        pr.terminate()
        try:
            pr.wait(10)
        except subprocess.TimeoutExpired:
            pr.kill()
        shutil.rmtree(prof, ignore_errors=True)


if __name__ == "__main__":
    main()
