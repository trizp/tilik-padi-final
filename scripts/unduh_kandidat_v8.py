#!/usr/bin/env python3
"""
unduh_kandidat_v8.py - Unduh kandidat dataset v8 (HANYA unduh; tidak menjalankan model apa pun). Dijalankan dengan venv alat
terpisah (tools/_venv_alat), bukan venv proyek.

  python scripts/unduh_kandidat_v8.py info      # daftar versi Roboflow (augmentasi, jumlah gambar, lisensi) -> reports/kandidat_v8/roboflow_info.json
  python scripts/unduh_kandidat_v8.py roboflow  # unduh versi terpilih (tanpa augmentasi bila ada) format YOLOv8
  python scripts/unduh_kandidat_v8.py kaggle    # unduh dataset Kaggle (CLI kaggle, token di ~/.kaggle)

Kunci API Roboflow dibaca dari variabel lingkungan ROBOFLOW_API_KEY dan TIDAK pernah ditulis ke berkas/log (pesan galat disensor).
Keluaran: data/raw/kandidat_v8/<nama>/
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw/kandidat_v8"
REP = ROOT / "reports/kandidat_v8"
RF = {"rf_gymvi_5class": ("dataset-gymvi", "deteksi-penyakit-pada-daun-padi---5-class"),
      "rf_aditia_4ued0": ("aditia-putra", "deteksi-penyakit-daun-padi-4ued0"),
      "rf_daunpadi_b1cjw": ("daun-padi-8ckhf", "penyakit-daun-padi-b1cjw")}
KAGGLE = ("kg_hadiur_ricespot", "hadiurrahmannabil/rice-leaf-spot-disease-annotated-dataset")


def kunci():
    k = os.environ.get("ROBOFLOW_API_KEY")
    if not k:
        sys.exit("[GAGAL] ROBOFLOW_API_KEY tidak ada")
    return k


def sensor(s):
    k = os.environ.get("ROBOFLOW_API_KEY", "")
    s = str(s)
    return s.replace(k, "***") if k else s


def get_json(url):
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as e:
        sys.exit(f"[GAGAL] permintaan API: {sensor(e)}")


def info():
    REP.mkdir(parents=True, exist_ok=True)
    k, out = kunci(), {}
    for nama, (ws, pj) in RF.items():
        d = get_json(f"https://api.roboflow.com/{ws}/{pj}?api_key={k}")
        p = d.get("project", {})
        vers = []
        for v in d.get("versions", []):
            vers.append({"id": v.get("id", "").split("/")[-1], "nama": v.get("name"), "gambar": v.get("images"), "split": v.get("splits"),
                         "augmentasi": v.get("augmentation"), "praproses": v.get("preprocessing"), "dibuat": v.get("created")})
        out[nama] = {"workspace": ws, "proyek": pj, "judul": p.get("name"), "lisensi": p.get("license"), "jenis": p.get("type"),
                     "kelas": p.get("classes"), "jumlah_gambar_proyek": p.get("images"), "versi": vers,
                     "url": f"https://universe.roboflow.com/{ws}/{pj}"}
    (REP / "roboflow_info.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    for nama, o in out.items():
        print(nama, o["judul"], "| lisensi", o["lisensi"], "| kelas", o["kelas"], "| gambar", o["jumlah_gambar_proyek"])
        for v in o["versi"]:
            print("   versi", v["id"], v["gambar"], "| augmentasi", json.dumps(v["augmentasi"])[:120], "| split", v["split"])


def pilih_versi(o):
    """Versi tanpa augmentasi (augmentation kosong); bila tidak ada, versi dengan gambar paling sedikit per gambar sumber (dicatat)."""
    tanpa = [v for v in o["versi"] if not v["augmentasi"]]
    if tanpa:
        return max(tanpa, key=lambda v: int(v["id"])), "tanpa augmentasi"
    return min(o["versi"], key=lambda v: v["gambar"] or 1e9), "SEMUA versi beraugmentasi; dipilih yang gambarnya paling sedikit"


def roboflow():
    k = kunci()
    info_ = json.loads((REP / "roboflow_info.json").read_text(encoding="utf-8"))
    catatan = {}
    for nama, o in info_.items():
        tujuan = RAW / nama
        if tujuan.exists() and any(tujuan.iterdir()):
            print(f"{nama}: sudah ada, dilewati")
            continue
        v, alasan = pilih_versi(o)
        url = f"https://api.roboflow.com/{o['workspace']}/{o['proyek']}/{v['id']}/yolov8?api_key={k}"
        link = None
        for _ in range(30):                                    # ekspor bisa perlu dibuat dulu di server
            d = get_json(url)
            link = (d.get("export") or {}).get("link")
            if link:
                break
            time.sleep(10)
        if not link:
            sys.exit(f"[GAGAL] {nama}: tautan ekspor tidak tersedia")
        tujuan.mkdir(parents=True, exist_ok=True)
        zp = tujuan / "_unduhan.zip"
        try:
            urllib.request.urlretrieve(link, zp)
        except Exception as e:
            sys.exit(f"[GAGAL] unduh {nama}: {sensor(e)}")
        with zipfile.ZipFile(zp) as z:
            z.extractall(tujuan)
        zp.unlink()
        catatan[nama] = {"versi": v["id"], "alasan": alasan, "gambar_versi": v["gambar"], "augmentasi": v["augmentasi"], "praproses": v["praproses"],
                         "format": "yolov8 (ekspor Roboflow)"}
        print(f"{nama}: versi {v['id']} ({alasan}) diunduh")
    p = REP / "unduhan_roboflow.json"
    lama = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    lama.update(catatan)
    p.write_text(json.dumps(lama, indent=2, ensure_ascii=False), encoding="utf-8")


def kaggle():
    nama, ref = KAGGLE
    tujuan = RAW / nama
    if tujuan.exists() and any(tujuan.iterdir()):
        print(f"{nama}: sudah ada, dilewati")
        return
    tujuan.mkdir(parents=True, exist_ok=True)
    exe = Path(sys.executable).parent / ("kaggle.exe" if os.name == "nt" else "kaggle")
    r = subprocess.run([str(exe), "datasets", "download", "-d", ref, "-p", str(tujuan), "--unzip"], capture_output=True, text=True)
    print(sensor(r.stdout[-800:]), sensor(r.stderr[-800:]))
    if r.returncode:
        sys.exit("[GAGAL] unduh Kaggle")
    m = subprocess.run([str(exe), "datasets", "metadata", "-d", ref, "-p", str(REP)], capture_output=True, text=True)
    print(sensor(m.stdout[-300:]), sensor(m.stderr[-300:]))


if __name__ == "__main__":
    {"info": info, "roboflow": roboflow, "kaggle": kaggle}[sys.argv[1]]()
