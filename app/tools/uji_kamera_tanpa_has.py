#!/usr/bin/env python3
"""
uji_kamera_tanpa_has.py - Uji teks tombol kamera di browser (Chrome headless, kamera palsu) dalam tiga keadaan:
  A. normal;
  B. semua aturan CSS yang memakai :has() dihapus dari halaman (meniru browser lama);
  C. seperti B, ditambah skrip label dimatikan (iframe skrip dihapus sebelum kamera dibuka) -> teks bawaan harus tetap terlihat.
Juga memeriksa teks setelah memotret ("Hapus foto dan ambil ulang"). Aplikasi harus sudah berjalan (port 8599).
Keluaran: docs/uji/kamera_tanpa_has.md dan docs/tangkapan_layar/kamera_tanpa_has_390.png
"""
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

APP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP / "tools"))
import tangkap_layar as tl  # noqa: E402

HAPUS_HAS = """(function(){var n=0; for (const sh of document.styleSheets){ let r; try { r = sh.cssRules; } catch(e) { continue; }
  for (let i = r.length - 1; i >= 0; i--) { if (r[i].cssText.includes(':has(')) { sh.deleteRule(i); n++; } } } return n; })()"""
INFO = """(function(){var b=document.querySelector('[data-testid=stCameraInputButton]'); if(!b) return null;
  var cs=getComputedStyle(b); return {teks:b.innerText.trim(), font:cs.fontSize, tinggi:b.getBoundingClientRect().height};})()"""


def main(url="http://localhost:8599"):
    exe = next(p for p in tl.CHROME if Path(p).exists())
    prof = tempfile.mkdtemp(prefix="cdp_")
    pr = subprocess.Popen([exe, "--headless=new", "--remote-debugging-port=9288", f"--user-data-dir={prof}", "--hide-scrollbars",
                           "--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream", "about:blank"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    hasil = {}
    try:
        for _ in range(50):
            try:
                ws = next(t["webSocketDebuggerUrl"] for t in json.loads(urllib.request.urlopen("http://127.0.0.1:9288/json").read()) if t["type"] == "page")
                break
            except Exception:
                time.sleep(0.3)
        c = tl.CDP(ws)
        c("Page.enable")
        for nama, hapus_has, matikan_skrip in (("A_normal", False, False), ("B_tanpa_has", True, False), ("C_tanpa_has_tanpa_skrip", True, True)):
            tl.buka(c, url, 390)
            n_has = c.js(HAPUS_HAS) if hapus_has else 0
            if matikan_skrip:
                c.js("document.querySelector('.st-key-label_id')?.remove(); window.__pengamatLabel?.disconnect(); true")
            c.js("[...document.querySelectorAll('button')].find(b => b.innerText.trim() === 'Ambil foto daun').click()")
            c.tunggu("!!document.querySelector('[data-testid=stCameraInput] video') && document.querySelector('[data-testid=stCameraInput] video').readyState >= 2", batas=30)
            time.sleep(1.2)
            if hapus_has:
                n_has += c.js(HAPUS_HAS)                      # aturan yang dimuat setelah kamera terbuka juga dihapus
            sebelum = c.js(INFO)
            if nama == "B_tanpa_has":
                tl.potret(c, APP / "docs/tangkapan_layar/kamera_tanpa_has_390.png", 390)
            c.js("document.querySelector('[data-testid=stCameraInputButton]').click()")
            c.tunggu("!!document.querySelector('[data-testid=stCameraInput] img')", batas=30)
            time.sleep(1.0)
            sesudah = c.js(INFO)
            hasil[nama] = {"aturan_has_dihapus": n_has, "sebelum_memotret": sebelum, "sesudah_memotret": sesudah}
            print(nama, hasil[nama], flush=True)
    finally:
        pr.terminate()
        shutil.rmtree(prof, ignore_errors=True)
    ok = (hasil["A_normal"]["sebelum_memotret"]["teks"] == "Ambil foto" and hasil["B_tanpa_has"]["sebelum_memotret"]["teks"] == "Ambil foto"
          and hasil["B_tanpa_has"]["sesudah_memotret"]["teks"] == "Hapus foto dan ambil ulang"
          and all(v["sebelum_memotret"]["teks"] and v["sesudah_memotret"]["teks"] for v in hasil.values())
          and all(float(v["sebelum_memotret"]["font"].rstrip("px")) >= 14 for v in hasil.values()))
    md = ["# Uji tombol kamera tanpa CSS :has()", "", f"**Hasil: {'LOLOS' if ok else 'GAGAL'}**", "",
          "| Keadaan | Aturan :has() dihapus | Teks sebelum memotret | Teks sesudah memotret | Ukuran huruf |", "|---|---|---|---|---|"]
    for k, v in hasil.items():
        md.append(f"| {k} | {v['aturan_has_dihapus']} | {v['sebelum_memotret']['teks']} | {v['sesudah_memotret']['teks']} | {v['sebelum_memotret']['font']} |")
    md += ["", "C = skrip label dimatikan: teks bawaan Streamlit (bahasa Inggris) tetap terlihat, tidak pernah tombol kosong."]
    (APP / "docs/uji/kamera_tanpa_has.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
