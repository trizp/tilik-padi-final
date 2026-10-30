#!/usr/bin/env python3
"""
evaluasi_akhir_v7.py - Pengendali evaluasi akhir prov_v7 (salinan evaluasi_akhir.py; v6 tidak diubah) (memanggil evaluate.py; setiap set uji dievaluasi SEKALI).

  python scripts/evaluasi_akhir_v7.py kunci   # Langkah 0: kunci ambang diagnosis 14 model pada VAL (tanpa membuka test)
  python scripts/evaluasi_akhir_v7.py uji     # Langkah 1+3: test, test_new_session, di luar cakupan ds4 (12 final), randsplit

* Final (12): ambang dipilih pada val BERSIH; randsplit (2): pada val acak SEMUA kelas.
* Fase kunci menulis reports/final_eval/kunci_model.{csv,json}: sha256 best.pt, ambang, versi library.
* Fase uji menolak berjalan bila ada model tanpa kunci ambang atau sha256 best.pt berubah sejak dikunci.
* ds3 TIDAK dievaluasi di sini.
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
OUT = ROOT / "reports" / "final_eval_v7"
MODELS = ["yolov8n", "yolo11n", "yolov8s", "yolo11s"]
FINAL = [f"v7_final_prov_{m}_s{s}" for m in MODELS for s in (0, 1, 2)]
RAND = [f"v7_randsplit_prov_{m}_s0" for m in ("yolov8n", "yolo11n")]
DATA_RAND = "data/processed/prov_v7_randsplit/data.yaml"
DATA_FINAL = "data/processed/prov_v7/data.yaml"
DS4_ORIG = "data/raw/ds4/images_labels/RiceLeafDiseaseBD/Original images"
LUAR = ["Rice Tungro=luar_cakupan", "Sheath blight=luar_cakupan", "Leaf smut=luar_cakupan"]


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
    for run in FINAL + RAND:
        rd = ROOT / "runs" / run
        w = rd / "weights" / "best.pt"
        if not (rd / "run_stats.json").exists() or not w.exists():
            sys.exit(f"[GAGAL] {run} belum selesai")
        args = ["--weights", str(w.relative_to(ROOT)), "--split", "val", "--tanpa-benchmark"]
        if not run.startswith("v7_randsplit"):
            args += ["--data", DATA_FINAL]
        if run.startswith("v7_randsplit"):
            args += ["--data", DATA_RAND, "--ambang-pada", "semua"]
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
    log(f"kunci selesai: {len(rows)} model -> reports/final_eval/kunci_model.csv")


def uji():
    kp = OUT / "kunci_model.json"
    if not kp.exists():
        sys.exit("[DITOLAK] jalankan fase kunci dulu")
    kunci_ = {r["run"]: r for r in json.loads(kp.read_text(encoding="utf-8"))["model"]}
    for run in FINAL + RAND:
        if sha256(ROOT / "runs" / run / "weights" / "best.pt") != kunci_[run]["best_pt_sha256"]:
            sys.exit(f"[DITOLAK] best.pt {run} berubah sejak dikunci")
    tugas = []
    for run in FINAL:
        w = f"runs/{run}/weights/best.pt"
        tugas += [(run, "test", ["--weights", w, "--data", DATA_FINAL, "--split", "test", "--izinkan-test"], f"runs/{run}/eval_test/ringkasan.json"),
                  (run, "test_new_session", ["--weights", w, "--data", DATA_FINAL, "--split", "test_new_session", "--izinkan-test", "--tanpa-benchmark"],
                   f"runs/{run}/eval_test_new_session/ringkasan.json"),
                  (run, "luar_cakupan_ds4", ["--weights", w, "--data", DATA_FINAL, "--set-folder", DS4_ORIG, "--izinkan-uji-eksternal", "--peta-folder"] + LUAR +
                   ["--out", f"runs/{run}/eval_luar_cakupan_ds4"], f"runs/{run}/eval_luar_cakupan_ds4/ringkasan_diagnosis.json")]
    for run in RAND:
        w = f"runs/{run}/weights/best.pt"
        tugas += [(run, "test_acak", ["--weights", w, "--data", DATA_RAND, "--split", "test", "--izinkan-test", "--tanpa-benchmark"],
                   f"runs/{run}/eval_test/ringkasan.json"),
                  (run, "test_new_session", ["--weights", w, "--data", DATA_RAND, "--split", "test_new_session", "--izinkan-test",
                                             "--tanpa-benchmark"], f"runs/{run}/eval_test_new_session/ringkasan.json")]
    for run, nama, args, jejak in tugas:
        if (ROOT / jejak).exists():
            log(f"{run} {nama}: sudah dievaluasi, dilewati")
            continue
        log(f"{run} {nama}: mulai")
        rc = jalan(args, ROOT / "runs" / run / f"eval_{nama}.log")
        log(f"{run} {nama}: selesai kode {rc}")
        if rc != 0:
            sys.exit(f"[GAGAL] {run} {nama}; lihat runs/{run}/eval_{nama}.log")
    log("fase uji selesai")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    {"kunci": kunci, "uji": uji}[sys.argv[1]]()
