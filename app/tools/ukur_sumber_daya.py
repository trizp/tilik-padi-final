#!/usr/bin/env python3
"""
ukur_sumber_daya.py - Ukur RAM proses Streamlit (diam dan saat memproses) dan waktu respons per foto di CPU.

1. Menjalankan server Streamlit baru (port 8611) dan mengukur RSS prosesnya (psutil, termasuk proses anak):
   - setelah server siap (model belum dimuat),
   - diam setelah halaman pertama dibuka (model sudah dimuat lewat st.cache_resource),
   - puncak selama memproses foto (sampel tiap 20 ms), lewat browser headless (CDP) yang mengunggah foto sungguhan.
2. Waktu respons di CPU:
   - inti: core.alur.periksa_foto di proses ini (median 20 foto VAL + foto besar 4000x3000), tanpa jaringan;
   - ujung ke ujung di browser: dari berkas dipilih sampai pita keyakinan (atau pesan foto ulang) tampil.
Alur v9: termasuk pemisah daun YOLOv8n-seg (dua sesi ONNX Runtime). Jumlah foto yang diminta foto ulang ikut dicatat.
Foto besar dibuat di folder sementara sistem (bukan di app/) dan dihapus setelahnya.

  app/.venv/Scripts/python.exe app/tools/ukur_sumber_daya.py
Keluaran: app/docs/uji/sumber_daya.json dan .md
"""
import json
import os
import platform
import shutil
import statistics as S
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

import psutil
from PIL import Image

APP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP))
sys.path.insert(0, str(APP / "tools"))
from core import alur, gambar, inferensi, konfigurasi, pemisah_daun  # noqa: E402
from rbs import mesin  # noqa: E402
import tangkap_layar as tl  # noqa: E402

VAL = APP.parent / "data/processed/prov_v7/images/val"
PORT = 8611


def rss(p):
    try:
        return p.memory_info().rss + sum(c.memory_info().rss for c in p.children(recursive=True))
    except psutil.Error:
        return 0


def mb(x):
    return round(x / 2 ** 20, 1)


def main(config=None, akhiran=""):
    import os
    if config:
        os.environ["TILIK_CONFIG"] = config        # dipakai proses ini dan diwariskan ke server Streamlit
    import csv
    acuan = [r["gambar"] for r in csv.DictReader(open(APP.parent / "reports/final_eval_v7/aplikasi/onnx_vs_pt_50_val.csv", encoding="utf-8"))][:20]
    tmp = Path(tempfile.mkdtemp(prefix="ukur_"))
    besar = tmp / "foto_besar_4000x3000.jpg"
    Image.open(VAL / acuan[0]).convert("RGB").resize((4000, 3000), Image.BICUBIC).save(besar, quality=92)

    # ---- 2a. waktu inti (proses ini)
    cfg = konfigurasi.muat()
    det = inferensi.Detektor(cfg.model_path, cfg.imgsz, cfg.conf_min, cfg.iou, cfg.max_det)
    pem = pemisah_daun.PemisahDaun(cfg.pemisah_path, cfg.pemisah_tau) if cfg.pemisah_path else None
    aturan = mesin.muat_aturan()
    n_ulang = [0]

    def periksa(d):                                    # foto cadangan -> FotoPerluDiulang (pengguna diminta foto ulang)
        try:
            alur.periksa_foto(d, cfg, det, aturan, pem)
        except gambar.FotoPerluDiulang:
            n_ulang[0] += 1

    for f in acuan[:3]:
        periksa((VAL / f).read_bytes())                                          # pemanasan
    n_ulang[0] = 0
    t_inti = []
    for f in acuan:
        d = (VAL / f).read_bytes()
        t0 = time.perf_counter()
        periksa(d)
        t_inti.append(1000 * (time.perf_counter() - t0))
    db = besar.read_bytes()
    t_besar = []
    for _ in range(5):
        t0 = time.perf_counter()
        periksa(db)
        t_besar.append(1000 * (time.perf_counter() - t0))
    del det, pem

    # ---- 1 + 2b. server Streamlit
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    srv = subprocess.Popen([str(APP / ".venv/Scripts/streamlit.exe"), "run", "streamlit_app.py", "--server.headless", "true",
                            "--server.port", str(PORT)], cwd=APP, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    proses = psutil.Process(srv.pid)
    hasil = {}
    try:
        for _ in range(100):
            try:
                if urllib.request.urlopen(f"http://localhost:{PORT}/_stcore/health").read() == b"ok":
                    break
            except Exception:
                time.sleep(0.3)
        time.sleep(2)
        hasil["rss_server_siap_MB"] = mb(rss(proses))
        exe = next(p for p in tl.CHROME if Path(p).exists())
        prof = tempfile.mkdtemp(prefix="cdp_")
        chrome = subprocess.Popen([exe, "--headless=new", "--remote-debugging-port=9244", f"--user-data-dir={prof}", "about:blank"],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            for _ in range(50):
                try:
                    tabs = json.loads(urllib.request.urlopen("http://127.0.0.1:9244/json").read())
                    ws = next(t["webSocketDebuggerUrl"] for t in tabs if t["type"] == "page")
                    break
                except Exception:
                    time.sleep(0.3)
            c = tl.CDP(ws)
            c("Page.enable")
            url = f"http://localhost:{PORT}"
            tl.buka(c, url, 390)
            time.sleep(5)
            hasil["rss_diam_setelah_model_dimuat_MB"] = mb(rss(proses))
            puncak, berhenti = [0], threading.Event()

            def sampel():
                while not berhenti.is_set():
                    puncak[0] = max(puncak[0], rss(proses))
                    time.sleep(0.02)
            th = threading.Thread(target=sampel, daemon=True)
            th.start()
            t_ui = {}
            for label, path in [("val", VAL / f) for f in acuan[:8]] + [("besar_4000x3000", besar)] * 3:
                tl.buka(c, url, 390)
                c.js("[...document.querySelectorAll('button')].find(b => b.innerText.trim() === 'Unggah foto').click()")
                c.tunggu("!!document.querySelector('input[type=file]')")
                obj = c("Runtime.evaluate", expression="document.querySelector('input[type=file]')")["result"]["objectId"]
                t0 = time.perf_counter()
                c("DOM.setFileInputFiles", files=[str(path)], objectId=obj)
                c.tunggu("!!document.querySelector('.pita') || !!document.querySelector('.pesan-salah')", batas=120, jeda=0.02)
                t_ui.setdefault(label, []).append(1000 * (time.perf_counter() - t0))
            berhenti.set()
            th.join()
            time.sleep(5)
            hasil["rss_puncak_saat_memproses_MB"] = mb(puncak[0])
            hasil["rss_diam_setelah_11_foto_MB"] = mb(rss(proses))
        finally:
            chrome.terminate()
            shutil.rmtree(prof, ignore_errors=True)
    finally:
        srv.terminate()
        try:
            srv.wait(10)
        except subprocess.TimeoutExpired:
            srv.kill()
        shutil.rmtree(tmp, ignore_errors=True)

    hasil.update({
        "foto_ulang_saat_ukur_inti": n_ulang[0],
        "waktu_inti_ms_median_20_val": round(S.median(t_inti), 1), "waktu_inti_ms_maks_20_val": round(max(t_inti), 1),
        "waktu_inti_ms_median_foto_4000x3000": round(S.median(t_besar), 1),
        "waktu_ujung_ke_ujung_ms_median_val": round(S.median(t_ui["val"]), 1),
        "waktu_ujung_ke_ujung_ms_median_4000x3000": round(S.median(t_ui["besar_4000x3000"]), 1),
        "mesin": {"cpu": platform.processor(), "inti_logis": psutil.cpu_count(), "os": platform.platform(), "python": platform.python_version()},
        "catatan": "RSS = memori fisik proses streamlit + anaknya. Waktu inti = baca foto + "
                   + ("pemisah daun YOLOv8n-seg + inferensi ONNX detektor" if cfg.pemisah_path else "inferensi ONNX")
                   + " + diagnosis + forward chaining + gambar kotak."})
    (APP / f"docs/uji/sumber_daya{akhiran}.json").write_text(json.dumps(hasil, indent=2, ensure_ascii=False), encoding="utf-8")
    m = hasil["mesin"]
    md = ["# Pengukuran RAM dan waktu respons", "", f"Mesin: {m['cpu']} ({m['inti_logis']} inti logis), {m['os']}, Python {m['python']}. CPU saja.", "",
          "| Ukuran | Nilai |", "|---|---|",
          f"| RAM server siap (model belum dimuat) | {hasil['rss_server_siap_MB']} MB |",
          f"| RAM diam (model sudah dimuat) | {hasil['rss_diam_setelah_model_dimuat_MB']} MB |",
          f"| RAM puncak saat memproses (8 foto VAL + 3 foto 4000x3000) | {hasil['rss_puncak_saat_memproses_MB']} MB |",
          f"| RAM diam setelah 11 foto | {hasil['rss_diam_setelah_11_foto_MB']} MB |",
          f"| Waktu inti per foto VAL (median / maks, 20 foto) | {hasil['waktu_inti_ms_median_20_val']} / {hasil['waktu_inti_ms_maks_20_val']} ms |",
          f"| Waktu inti foto 4000x3000 (median 5x) | {hasil['waktu_inti_ms_median_foto_4000x3000']} ms |",
          f"| Waktu ujung ke ujung di browser, foto VAL (median 8) | {hasil['waktu_ujung_ke_ujung_ms_median_val']} ms |",
          f"| Waktu ujung ke ujung di browser, foto 4000x3000 (median 3) | {hasil['waktu_ujung_ke_ujung_ms_median_4000x3000']} ms |", "",
          hasil["catatan"]]
    (APP / f"docs/uji/sumber_daya{akhiran}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="konfigurasi alternatif, mis. config_yolov8.yaml (tanpa mengaktifkannya)")
    ap.add_argument("--akhiran", default="")
    a = ap.parse_args()
    main(config=a.config, akhiran=a.akhiran)
