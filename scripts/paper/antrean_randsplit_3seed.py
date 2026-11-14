#!/usr/bin/env python3
"""Split acak multi-seed untuk revisi paper SOFTT (Tabel II baris "Random"): seed 1 dan 2 untuk YOLOv8n dan YOLO11n.

Perintah latih IDENTIK dengan antrean_v7.py (run randsplit s0):
  train.py --data data/processed/prov_v7_randsplit/data.yaml --model <m>.pt --seed <s>
           --epochs 100 --batch 8 --workers 2 --patience 0 --imgsz 640 --val-after --nama v7_randsplit_prov_<m>_s<s>
Evaluasi IDENTIK dengan evaluasi_akhir_v7.py:
  kunci  : evaluate.py --weights <best.pt> --split val --tanpa-benchmark --data <randsplit> --ambang-pada semua
  uji    : evaluate.py ... --split test --izinkan-test --tanpa-benchmark        (log eval_test_acak.log)
           evaluate.py ... --split test_new_session --izinkan-test --tanpa-benchmark
Ambang dikunci di val acak SEBELUM test dibuka; sha256 best.pt dicatat dan dicek ulang sebelum uji.
Dataset, split, dan run s0 tidak diubah. Run yang sudah selesai dilewati (aman dijalankan ulang).
Berhenti halus: buat berkas reports/revisi_paper/randsplit_3seed/BERHENTI.
"""
import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
OUT = ROOT / "reports/revisi_paper/randsplit_3seed"
STOP, STATUS, KUNCI = OUT / "BERHENTI", OUT / "status.json", OUT / "kunci.json"
DATA_RAND = "data/processed/prov_v7_randsplit/data.yaml"
UMUM = ["--epochs", "100", "--batch", "8", "--workers", "2", "--patience", "0", "--imgsz", "640", "--val-after"]
JOBS = [(f"v7_randsplit_prov_{m}_s{s}", m, s) for s in (1, 2) for m in ("yolov8n", "yolo11n")]
MIN_DISK_GB = 0.8
MAKS_TUNGGU_MEMORI, JEDA_MEMORI_S, MAKS_RESUME = 36, 600, 5


def sekarang():
    return datetime.now().isoformat(timespec="seconds")


def log(msg):
    line = f"[{sekarang()}] {msg}"
    print(line, flush=True)
    with open(OUT / "antrean.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def baca_json(p, bawaan):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else bawaan


def tulis_json(p, d):
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def n_epoch(rd):
    p = rd / "epoch_stats.jsonl"
    if not p.exists():
        return 0
    return len({json.loads(ln)["epoch"] for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()})


def selesai_latih(rd):
    return (rd / "run_stats.json").exists() and n_epoch(rd) >= 100 and (rd / "weights" / "best.pt").exists()


def jalankan(cmd, logf):
    with open(logf, "w", encoding="utf-8") as f:
        f.write(f"# {sekarang()} {' '.join(cmd)}\n")
        f.flush()
        r = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    return r.returncode, logf.read_text(encoding="utf-8", errors="replace")


def cek_disk():
    free = shutil.disk_usage(ROOT).free / 1024 ** 3
    if free < MIN_DISK_GB:
        log(f"[BERHENTI] sisa disk {free:.2f} GB < {MIN_DISK_GB} GB")
        sys.exit(2)
    return round(free, 2)


def latih(n, m, s, st):
    rd = ROOT / "runs" / n
    if selesai_latih(rd):
        return True
    tunggu = resume = 0
    while True:
        if STOP.exists():
            log("berkas BERHENTI ditemukan; keluar")
            sys.exit(0)
        last = rd / "weights" / "last.pt"
        if rd.exists() and not last.exists():
            tujuan = rd.with_name(f"{n}_gagal_{datetime.now():%Y%m%d_%H%M%S}")
            rd.rename(tujuan)
            log(f"{n}: folder tanpa last.pt dipindah ke {tujuan.name}")
        if last.exists():
            cmd, jenis = [PY, "scripts/train.py", "--resume", str(last.relative_to(ROOT))], "lanjutan"
        else:
            cmd, jenis = [PY, "scripts/train.py", "--data", DATA_RAND, "--model", f"{m}.pt", "--seed", str(s)] + UMUM + ["--nama", n], "baru"
        disk = cek_disk()
        logf = OUT / f"{n}_{jenis}_{datetime.now():%Y%m%d_%H%M%S}.log"
        st[n] = {"status": "latih", "jenis": jenis, "mulai": sekarang(), "disk_gb": disk, "log": logf.name}
        tulis_json(STATUS, st)
        log(f"{n}: mulai latih ({jenis}); sisa disk {disk} GB")
        t0 = time.time()
        rc, teks = jalankan(cmd, logf)
        if selesai_latih(rd):
            rs = json.loads((rd / "run_stats.json").read_text(encoding="utf-8"))
            log(f"{n}: latih selesai (kode {rc}, {(time.time() - t0) / 3600:.2f} jam; jam GPU {rs.get('jam_gpu_seluruh_epoch')}; "
                f"batch sesuai {rs.get('batch_sesuai_permintaan')})")
            return True
        if "[DITOLAK] memori virtual" in teks:
            tunggu += 1
            if tunggu > MAKS_TUNGGU_MEMORI:
                log(f"{n}: memori tidak cukup setelah {MAKS_TUNGGU_MEMORI} kali menunggu; keluar")
                sys.exit(3)
            log(f"{n}: ditolak cek memori; tunggu {JEDA_MEMORI_S} s ({tunggu}/{MAKS_TUNGGU_MEMORI})")
            time.sleep(JEDA_MEMORI_S)
            continue
        resume += 1
        if resume > MAKS_RESUME:
            log(f"{n}: gagal {MAKS_RESUME} kali (kode terakhir {rc}); keluar. Lihat {logf.name}")
            sys.exit(4)
        log(f"{n}: berhenti sebelum selesai (kode {rc}, epoch {n_epoch(rd)}); dilanjutkan ({resume}/{MAKS_RESUME})")
        time.sleep(30)


def evaluasi(n, st):
    rd = ROOT / "runs" / n
    w = f"runs/{n}/weights/best.pt"
    kunci = baca_json(KUNCI, {})
    if not (rd / "ambang_diagnosis.json").exists():
        log(f"{n}: evaluasi VAL + kunci ambang (val acak semua kelas)")
        rc, _ = jalankan([PY, "scripts/evaluate.py", "--weights", w, "--split", "val", "--tanpa-benchmark", "--data", DATA_RAND,
                          "--ambang-pada", "semua"], rd / "eval_val.log")
        if rc != 0:
            log(f"{n}: evaluasi val gagal kode {rc}")
            sys.exit(5)
    k = json.loads((rd / "ambang_diagnosis.json").read_text(encoding="utf-8"))
    h = sha256(ROOT / w)
    if n not in kunci:
        assert h == k["weights_sha256"], f"{n}: sha256 best.pt tidak sama dengan saat kunci ambang"
        kunci[n] = {"best_pt_sha256": h, "ambang_diagnosis": k["ambang"], "dipilih_pada": k["dipilih_pada"],
                    "akurasi_val_saat_dipilih": k["akurasi_val_saat_dipilih"], "dikunci": k["dikunci"]}
        tulis_json(KUNCI, kunci)
        log(f"{n}: ambang dikunci {k['ambang']} (akurasi val {k['akurasi_val_saat_dipilih']})")
    if h != kunci[n]["best_pt_sha256"]:
        log(f"[DITOLAK] {n}: best.pt berubah sejak dikunci")
        sys.exit(6)
    for nama, split, jejak in (("test_acak", "test", "eval_test/ringkasan.json"),
                               ("test_new_session", "test_new_session", "eval_test_new_session/ringkasan.json")):
        if (rd / jejak).exists():
            continue
        st[n] = {"status": f"uji {nama}", "mulai": sekarang()}
        tulis_json(STATUS, st)
        log(f"{n} {nama}: mulai (sekali)")
        rc, _ = jalankan([PY, "scripts/evaluate.py", "--weights", w, "--data", DATA_RAND, "--split", split, "--izinkan-test",
                          "--tanpa-benchmark"], rd / f"eval_{nama}.log")
        log(f"{n} {nama}: selesai kode {rc}")
        if rc != 0:
            sys.exit(7)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    st = baca_json(STATUS, {})
    log(f"antrean mulai: {[j[0] for j in JOBS]}")
    for n, m, s in JOBS:
        latih(n, m, s, st)
        evaluasi(n, st)
        st[n] = {"status": "selesai", "selesai": sekarang()}
        tulis_json(STATUS, st)
    log("antrean selesai; jalankan scripts/paper/ringkas_randsplit_3seed.py")


if __name__ == "__main__":
    main()
