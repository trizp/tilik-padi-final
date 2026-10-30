#!/usr/bin/env python3
"""
evaluasi_akhir_v9.py - Kunci ambang diagnosis 6 model v9 pada VAL v9 (tanpa membuka test; KEPUTUSAN_PROYEK.md 2026-10-16).

  python scripts/evaluasi_akhir_v9.py kunci   # dipanggil otomatis oleh antrean_v9.py setelah keenam run selesai
  python scripts/evaluasi_akhir_v9.py uji     # rencana evaluasi yang dicatat SEBELUM training (atas perintah 2026-10-17)

* Ambang dipilih pada val BERSIH v9 (tanpa ds1_provisional), prosedur sama dengan v7 (evaluate.py --split val).
* Menulis reports/final_eval_v9/kunci_model.{csv,json}: sha256 best.pt, ambang, versi library.
* Fase uji (rencana yang dicatat sebelum training, KEPUTUSAN_PROYEK.md 2026-10-16; dijalankan atas perintah 2026-10-17), tiap set SEKALI
  (evaluate.py menolak evaluasi ulang pada folder keluaran yang sama):
  1. utama  : model v9 pada test dan test sesi baru TANPA latar (prov_v9)           -> runs/v9_*/eval_test, eval_test_new_session
  2. silang : model v7 (s) pada test dan test sesi baru tanpa latar (prov_v9)       -> runs/v7_final_prov_*/eval_v9tl_<split>
              model v9 pada test dan test sesi baru ASLI (prov_v7)                  -> runs/v9_*/eval_v7asli_<split>
  3. ds3    : model v9 pada ds3 tanpa latar (utama 70, tambahan 160, Tungro 80)     -> runs/v9_*/eval_ds3tl_<himpunan>
  Ambang = ambang terkunci MASING-MASING model (v7 di VAL v7, v9 di VAL v9). Tanpa benchmark kecepatan.
"""
import csv
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
OUT = ROOT / "reports" / "final_eval_v9"
MODELS = ["yolov8s", "yolo11s"]
FINAL = [f"v9_final_prov_{m}_s{s}" for m in MODELS for s in (0, 1, 2)]
DATA_FINAL = "data/processed/prov_v9/data.yaml"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def log(msg):
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with open(OUT / "evaluasi_akhir.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def jalan(args, logf):
    with open(logf, "w", encoding="utf-8") as f:
        f.write("# " + " ".join(args) + "\n")
        f.flush()
        return subprocess.run([PY, "scripts/evaluate.py"] + args, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode


def kunci():
    rows = []
    OUT.mkdir(parents=True, exist_ok=True)
    for run in FINAL:
        rd = ROOT / "runs" / run
        w = rd / "weights" / "best.pt"
        if not (rd / "run_stats.json").exists() or not w.exists():
            sys.exit(f"[GAGAL] {run} belum selesai")
        args = ["--weights", str(w.relative_to(ROOT)), "--split", "val", "--tanpa-benchmark", "--data", DATA_FINAL]
        if not (rd / "ambang_diagnosis.json").exists():
            log(f"{run}: evaluasi VAL + kunci ambang")
            rc = jalan(args, rd / "eval_val.log")
            if rc != 0:
                sys.exit(f"[GAGAL] {run} kode {rc}; lihat {rd / 'eval_val.log'}")
        k = json.loads((rd / "ambang_diagnosis.json").read_text(encoding="utf-8"))
        rs = json.loads((rd / "run_stats.json").read_text(encoding="utf-8"))
        rows.append({"run": run, "best_pt_sha256": sha256(w), "ambang_diagnosis": k["ambang"], "dipilih_pada": k["dipilih_pada"],
                     "akurasi_val_saat_dipilih": k["akurasi_val_saat_dipilih"], "dikunci": k["dikunci"],
                     "torch": rs["versi"]["torch"], "ultralytics": rs["versi"]["ultralytics"], "python": rs["versi"]["python"],
                     "cuda": rs["versi"]["cuda"]})
        assert rows[-1]["best_pt_sha256"] == k["weights_sha256"], run
    with open(OUT / "kunci_model.csv", "w", newline="", encoding="utf-8") as f:
        w_ = csv.DictWriter(f, fieldnames=list(rows[0]))
        w_.writeheader()
        w_.writerows(rows)
    (OUT / "kunci_model.json").write_text(json.dumps({"dibuat": datetime.now().isoformat(timespec="seconds"),
                                                      "platform": platform.platform(), "model": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"kunci selesai: {len(rows)} model -> reports/final_eval_v9/kunci_model.csv")


V7RUNS = [f"v7_final_prov_{m}_s{s}" for m in MODELS for s in (0, 1, 2)]
DATA_V7 = "data/processed/prov_v7/data.yaml"
DS3TL = "data/derived/v9_latar/ds3_tanpa_latar"
HIMP_DS3 = {"utama": "ds3_uji_utama.txt", "tambahan": "ds3_tambahan_160.txt", "tungro": "ds3_tungro_luar_cakupan.txt"}
PETA_DS3 = ["blight=bacterial_leaf_blight", "blast=leaf_blast", "tungro=luar_cakupan"]


def uji():
    kp = OUT / "kunci_model.json"
    if not kp.exists():
        sys.exit("[DITOLAK] jalankan fase kunci dulu")
    kunci_ = {r["run"]: r for r in json.loads(kp.read_text(encoding="utf-8"))["model"]}
    for run in FINAL:
        if sha256(ROOT / "runs" / run / "weights" / "best.pt") != kunci_[run]["best_pt_sha256"]:
            sys.exit(f"[DITOLAK] best.pt {run} berubah sejak dikunci")
    for run in V7RUNS:
        k = json.loads((ROOT / "runs" / run / "ambang_diagnosis.json").read_text(encoding="utf-8"))
        if sha256(ROOT / "runs" / run / "weights" / "best.pt") != k["weights_sha256"]:
            sys.exit(f"[DITOLAK] best.pt {run} tidak sama dengan kunci ambang v7")
    tugas = []
    for run in FINAL:
        for sp in ("test", "test_new_session"):
            tugas.append((run, f"eval_{sp}", ["--data", DATA_FINAL, "--split", sp, "--izinkan-test"]))
    for run in V7RUNS:
        for sp in ("test", "test_new_session"):
            tugas.append((run, f"eval_v9tl_{sp}", ["--data", DATA_FINAL, "--split", sp, "--izinkan-test"]))
    for run in FINAL:
        for sp in ("test", "test_new_session"):
            tugas.append((run, f"eval_v7asli_{sp}", ["--data", DATA_V7, "--split", sp, "--izinkan-test"]))
    for run in FINAL:
        for nama, f in HIMP_DS3.items():
            tugas.append((run, f"eval_ds3tl_{nama}", ["--data", DATA_FINAL, "--set-folder", DS3TL, "--izinkan-uji-eksternal",
                                                     "--peta-folder"] + PETA_DS3 + ["--daftar", f"{DS3TL}/daftar/{f}"]))
    log(f"fase uji: {len(tugas)} evaluasi")
    for run, nama, extra in tugas:
        out = ROOT / "runs" / run / nama
        if (out / "ringkasan.json").exists() or (out / "ringkasan_diagnosis.json").exists():
            log(f"{run} {nama}: sudah ada, dilewati (tidak dievaluasi ulang)")
            continue
        args = ["--weights", f"runs/{run}/weights/best.pt", "--tanpa-benchmark", "--out", f"runs/{run}/{nama}"] + extra
        rc = jalan(args, ROOT / "runs" / run / f"{nama}.log")
        log(f"{run} {nama}: kode {rc}")
        if rc != 0:
            sys.exit(f"[GAGAL] {run} {nama}; lihat runs/{run}/{nama}.log")
    log("fase uji selesai")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    {"kunci": kunci, "uji": uji}[sys.argv[1]]()
