#!/usr/bin/env python3
"""
antrean_v7.py - Antrean latih ulang pada prov_v7 (orientasi ds4 diperbaiki). Konfigurasi, seed, dan aturan pemilihan IDENTIK
dengan run final v6 (lihat KEPUTUSAN_PROYEK.md 2026-10-10).

Urutan: 12 run final (seed 0: yolov8n, yolo11n, yolov8s, yolo11s; lalu seed 1; seed 2) -> 2 run randsplit (yolov8n, yolo11n, s0).
  final    : --data data/derived/prov_v7_valsel/data_valsel.yaml  (best.pt dari val bersih)  nama v7_final_prov_<model>_s<seed>
  randsplit: --data data/processed/prov_v7_randsplit/data.yaml     (best.pt dari val acak)   nama v7_randsplit_prov_<model>_s0
  semua    : 100 epoch, imgsz 640, batch 8, workers 2, patience 0, --val-after
* Cek memori oleh train.py sebelum tiap run/lanjutan; ditolak -> tunggu 10 menit (maks 36x).
* Resume otomatis bila last.pt ada (maks 5x per run); folder tanpa last.pt diganti nama *_gagal_<waktu> lalu diulang (maks 2x).
* Run yang sudah selesai dilewati -> aman dijalankan ulang. Berhenti halus: buat runs/_antrean_v7/BERHENTI.
* Evaluasi akhir TIDAK dijalankan di sini.
Status: runs/_antrean_v7/status.json (python scripts/status_final.py --versi v7).
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
DIR = ROOT / "runs" / "_antrean_v7"
STATUS, PID, STOP = DIR / "status.json", DIR / "antrean.pid", DIR / "BERHENTI"
EPOCHS = 100
MAKS_TUNGGU_MEMORI, JEDA_MEMORI_S, MAKS_RESUME, MAKS_ULANG_BARU = 36, 600, 5, 2
DATA_FINAL = "data/derived/prov_v7_valsel/data_valsel.yaml"
DATA_RAND = "data/processed/prov_v7_randsplit/data.yaml"
UMUM = ["--epochs", str(EPOCHS), "--batch", "8", "--workers", "2", "--patience", "0", "--imgsz", "640", "--val-after"]
JOBS = ([(f"v7_final_prov_{m}_s{s}", m, ["--data", DATA_FINAL, "--model", f"{m}.pt", "--seed", str(s)])
         for s in (0, 1, 2) for m in ("yolov8n", "yolo11n", "yolov8s", "yolo11s")] +
        [(f"v7_randsplit_prov_{m}_s0", m, ["--data", DATA_RAND, "--model", f"{m}.pt", "--seed", "0"]) for m in ("yolov8n", "yolo11n")])


def sekarang():
    return datetime.now().isoformat(timespec="seconds")


def log(msg):
    line = f"[{sekarang()}] {msg}"
    print(line, flush=True)
    with open(DIR / "antrean.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def n_epoch(rd):
    p = rd / "epoch_stats.jsonl"
    if not p.exists():
        return 0
    return len({json.loads(ln)["epoch"] for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()})


def selesai(rd):
    return (rd / "run_stats.json").exists() and n_epoch(rd) >= EPOCHS and (rd / "weights" / "best.pt").exists()


def proses_hidup(pid):
    try:
        import psutil
        return psutil.pid_exists(pid) and "python" in psutil.Process(pid).name().lower()
    except Exception:
        return False


def simpan(st):
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(STATUS)


def jalankan(cmd, logf):
    with open(logf, "w", encoding="utf-8") as f:
        f.write(f"# {sekarang()} {' '.join(cmd)}\n")
        f.flush()
        r = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    return r.returncode, logf.read_text(encoding="utf-8", errors="replace")


def latih(n, argumen, st):
    rd = ROOT / "runs" / n
    info = st["run"].setdefault(n, {"status": "menunggu", "percobaan": []})
    if selesai(rd):
        info["status"] = "selesai"
        simpan(st)
        log(f"{n}: sudah selesai, dilewati")
        return True
    tunggu = resume = ulang = 0
    while True:
        last = rd / "weights" / "last.pt"
        if rd.exists() and not last.exists():
            tujuan = rd.with_name(f"{n}_gagal_{datetime.now():%Y%m%d_%H%M%S}")
            rd.rename(tujuan)
            log(f"{n}: folder tanpa last.pt dipindah ke {tujuan.name}")
        if last.exists():
            cmd, jenis = [PY, "scripts/train.py", "--resume", str(last.relative_to(ROOT))], "lanjutan"
        else:
            cmd, jenis = [PY, "scripts/train.py"] + argumen + UMUM + ["--nama", n], "baru"
        k = len(info["percobaan"]) + 1
        logf = DIR / f"{n}_{k:02d}_{jenis}.log"
        info.update(status="berjalan", epoch_selesai=n_epoch(rd))
        info["percobaan"].append({"ke": k, "jenis": jenis, "mulai": sekarang(), "log": logf.name})
        st["aktif"] = n
        simpan(st)
        log(f"{n}: mulai percobaan {k} ({jenis})")
        rc, teks = jalankan(cmd, logf)
        info["percobaan"][-1].update(selesai=sekarang(), kode_keluar=rc, epoch_selesai=n_epoch(rd))
        info["epoch_selesai"] = n_epoch(rd)
        if selesai(rd):
            rs = json.loads((rd / "run_stats.json").read_text(encoding="utf-8"))
            info["status"] = "selesai" if rs.get("batch_sesuai_permintaan", True) else "selesai_batch_berbeda"
            info["jam_gpu"] = rs.get("jam_gpu_seluruh_epoch")
            simpan(st)
            log(f"{n}: {info['status']} (kode {rc}, jam GPU {info.get('jam_gpu')})")
            return True
        if n_epoch(rd) >= EPOCHS and (rd / "weights" / "best.pt").exists():
            jalankan([PY, "scripts/train.py", "--perbarui-stats", str(rd.relative_to(ROOT))], DIR / f"{n}_perbarui_stats.log")
            info["status"] = "selesai_tanpa_val_after"
            simpan(st)
            log(f"{n}: 100 epoch lengkap tetapi kode keluar {rc}; run_stats dibangun ulang")
            return True
        if "[DITOLAK] memori virtual" in teks:
            tunggu += 1
            info["status"] = f"menunggu_memori ({tunggu}/{MAKS_TUNGGU_MEMORI})"
            simpan(st)
            log(f"{n}: ditolak cek memori ({tunggu}/{MAKS_TUNGGU_MEMORI}); tunggu {JEDA_MEMORI_S // 60} menit")
            if tunggu >= MAKS_TUNGGU_MEMORI:
                info["status"] = "gagal_memori"
                simpan(st)
                return False
            time.sleep(JEDA_MEMORI_S)
            continue
        if (rd / "weights" / "last.pt").exists():
            resume += 1
            if resume > MAKS_RESUME:
                info["status"] = "gagal_resume_berulang"
                simpan(st)
                log(f"{n}: gagal setelah {MAKS_RESUME} lanjutan")
                return False
            log(f"{n}: berhenti tidak normal (kode {rc}) di epoch {n_epoch(rd)}; lanjut otomatis {resume}/{MAKS_RESUME} dalam 60 s")
            time.sleep(60)
            continue
        ulang += 1
        if ulang > MAKS_ULANG_BARU:
            info["status"] = "gagal_sebelum_epoch_1"
            simpan(st)
            log(f"{n}: gagal sebelum epoch 1 sebanyak {ulang} kali, lihat {logf.name}")
            return False
        log(f"{n}: gagal sebelum ada last.pt (kode {rc}); ulang {ulang}/{MAKS_ULANG_BARU} dalam 60 s")
        time.sleep(60)


def main():
    DIR.mkdir(parents=True, exist_ok=True)
    if PID.exists():
        try:
            lama = int(PID.read_text().strip())
        except ValueError:
            lama = -1
        if lama != os.getpid() and proses_hidup(lama):
            sys.exit(f"[DITOLAK] antrean v7 lain masih berjalan (PID {lama}).")
    PID.write_text(str(os.getpid()))
    st = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}
    st.update(pid=os.getpid(), mulai_antrean=sekarang(), urutan=[n for n, _, _ in JOBS], aktif=None)
    st.setdefault("run", {})
    simpan(st)
    log(f"antrean v7 mulai (PID {os.getpid()}): {len(JOBS)} run")
    try:
        for n, _, argumen in JOBS:
            if STOP.exists():
                log("berkas BERHENTI ditemukan -> antrean berhenti")
                break
            latih(n, argumen, st)              # run gagal dicatat; antrean lanjut ke run berikutnya
            st["aktif"] = None
            simpan(st)
    finally:
        st["aktif"] = None
        st["berhenti_antrean"] = sekarang()
        simpan(st)
        if PID.exists() and PID.read_text().strip() == str(os.getpid()):
            PID.unlink()
        log("antrean v7 berhenti; evaluasi akhir TIDAK dijalankan otomatis")


if __name__ == "__main__":
    main()
