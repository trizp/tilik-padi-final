#!/usr/bin/env python3
"""
antrean_final.py - Antrean run FINAL: 4 model x 3 seed, berurutan selang-seling model per seed:
  seed 0: yolov8n, yolo11n, yolov8s, yolo11s ; seed 1: ... ; seed 2: ...

* Konfigurasi identik (HP baku train.py): 100 epoch, imgsz 640, batch 8, workers 2, patience 0;
  best.pt dipilih dengan data_valsel.yaml (val bersih tanpa BLB provisional). Nama run: final_prov_v6_<model>_s<seed>.
* Cek memori dilakukan train.py sebelum TIAP run dan tiap lanjutan; bila ditolak karena memori, antrean menunggu
  10 menit lalu mencoba lagi (maks 36 kali = 6 jam), tanpa --abaikan-cek-memori.
* Resume otomatis: bila train.py berhenti tidak normal dan weights/last.pt ada -> lanjut dengan --resume (maks 5 kali per run).
  Folder run tanpa last.pt (gagal sebelum epoch 1 selesai) diganti nama menjadi <run>_gagal_<waktu> lalu run diulang (maks 2 kali).
* Run yang sudah selesai (100 epoch tercatat + run_stats.json) dilewati -> antrean aman dijalankan ulang kapan saja.
* Status: runs/_antrean_final/status.json (dibaca scripts/status_final.py); log tiap percobaan di runs/_antrean_final/.
* Berhenti halus: buat berkas runs/_antrean_final/BERHENTI -> antrean berhenti setelah run yang sedang berjalan selesai.

Jalankan terlepas dari terminal (PowerShell):
  Start-Process -FilePath .venv\\Scripts\\python.exe -ArgumentList scripts\\antrean_final.py -WindowStyle Hidden
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
DATA = "data/derived/prov_v6_valsel/data_valsel.yaml"
MODELS = ["yolov8n", "yolo11n", "yolov8s", "yolo11s"]
SEEDS = [0, 1, 2]
EPOCHS = 100
DIR = ROOT / "runs" / "_antrean_final"
STATUS = DIR / "status.json"
PID = DIR / "antrean.pid"
STOP = DIR / "BERHENTI"
MAKS_TUNGGU_MEMORI, JEDA_MEMORI_S, MAKS_RESUME, MAKS_ULANG_BARU = 36, 600, 5, 2


def nama(m, s):
    return f"final_prov_v6_{m}_s{s}"


def sekarang():
    return datetime.now().isoformat(timespec="seconds")


def log(msg):
    line = f"[{sekarang()}] {msg}"
    print(line, flush=True)
    with open(DIR / "antrean.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def n_epoch(run_dir):
    p = run_dir / "epoch_stats.jsonl"
    if not p.exists():
        return 0
    return len({json.loads(ln)["epoch"] for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()})


def selesai(run_dir):
    return (run_dir / "run_stats.json").exists() and n_epoch(run_dir) >= EPOCHS and (run_dir / "weights" / "best.pt").exists()


def simpan_status(st):
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(STATUS)


def proses_hidup(pid):
    try:
        import psutil
        return psutil.pid_exists(pid) and "python" in psutil.Process(pid).name().lower()
    except Exception:
        return False


def jalankan(cmd, logf):
    with open(logf, "w", encoding="utf-8") as f:
        f.write(f"# {sekarang()} {' '.join(cmd)}\n")
        f.flush()
        r = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    teks = logf.read_text(encoding="utf-8", errors="replace")
    return r.returncode, teks


def main():
    DIR.mkdir(parents=True, exist_ok=True)
    if PID.exists():
        try:
            lama = int(PID.read_text().strip())
        except ValueError:
            lama = -1
        if lama != os.getpid() and proses_hidup(lama):
            sys.exit(f"[DITOLAK] antrean lain masih berjalan (PID {lama}).")
    PID.write_text(str(os.getpid()))
    antrean = [(m, s) for s in SEEDS for m in MODELS]
    st = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}
    st.update({"pid": os.getpid(), "mulai_antrean": sekarang(), "urutan": [nama(m, s) for m, s in antrean], "aktif": None})
    st.setdefault("run", {})
    simpan_status(st)
    log(f"antrean mulai (PID {os.getpid()}): {len(antrean)} run")
    try:
        for m, s in antrean:
            if STOP.exists():
                log("berkas BERHENTI ditemukan -> antrean berhenti")
                break
            n = nama(m, s)
            rd = ROOT / "runs" / n
            info = st["run"].setdefault(n, {"status": "menunggu", "percobaan": []})
            if selesai(rd):
                if info.get("status") != "selesai":
                    info["status"] = "selesai"
                    simpan_status(st)
                log(f"{n}: sudah selesai, dilewati")
                continue
            tunggu, resume, ulang = 0, 0, 0
            while True:
                last = rd / "weights" / "last.pt"
                if rd.exists() and not last.exists():
                    tujuan = rd.with_name(f"{n}_gagal_{datetime.now():%Y%m%d_%H%M%S}")
                    rd.rename(tujuan)
                    log(f"{n}: folder tanpa last.pt dipindah ke {tujuan.name}")
                if last.exists():
                    cmd = [PY, "scripts/train.py", "--resume", str(last.relative_to(ROOT))]
                    jenis = "lanjutan"
                else:
                    cmd = [PY, "scripts/train.py", "--data", DATA, "--model", f"{m}.pt", "--epochs", str(EPOCHS), "--seed", str(s),
                           "--batch", "8", "--workers", "2", "--patience", "0", "--imgsz", "640", "--nama", n, "--val-after"]
                    jenis = "baru"
                k = len(info["percobaan"]) + 1
                logf = DIR / f"{n}_{k:02d}_{jenis}.log"
                info.update(status="berjalan", epoch_selesai=n_epoch(rd))
                info["percobaan"].append({"ke": k, "jenis": jenis, "mulai": sekarang(), "log": logf.name})
                st["aktif"] = n
                simpan_status(st)
                log(f"{n}: mulai percobaan {k} ({jenis})")
                rc, teks = jalankan(cmd, logf)
                info["percobaan"][-1].update(selesai=sekarang(), kode_keluar=rc, epoch_selesai=n_epoch(rd))
                info["epoch_selesai"] = n_epoch(rd)
                if selesai(rd):
                    rs = json.loads((rd / "run_stats.json").read_text(encoding="utf-8"))
                    info["status"] = "selesai" if rs.get("batch_sesuai_permintaan", True) else "selesai_batch_berbeda"
                    info["jam_gpu"] = rs.get("jam_gpu_seluruh_epoch")
                    info["jumlah_lanjutan"] = rs.get("jumlah_lanjutan_resume")
                    simpan_status(st)
                    log(f"{n}: {info['status']} (kode {rc}, jam GPU {info.get('jam_gpu')})")
                    break
                if n_epoch(rd) >= EPOCHS and (rd / "weights" / "best.pt").exists():
                    # epoch lengkap tetapi langkah sesudahnya (validasi best.pt / ringkasan) gagal: jangan di-resume
                    jalankan([PY, "scripts/train.py", "--perbarui-stats", str(rd.relative_to(ROOT))], DIR / f"{n}_perbarui_stats.log")
                    info["status"] = "selesai_tanpa_val_after"
                    simpan_status(st)
                    log(f"{n}: 100 epoch lengkap tetapi train.py keluar dengan kode {rc}; run_stats dibangun ulang, lihat {logf.name}")
                    break
                if "[DITOLAK] memori virtual" in teks:
                    tunggu += 1
                    info["status"] = f"menunggu_memori ({tunggu}/{MAKS_TUNGGU_MEMORI})"
                    simpan_status(st)
                    log(f"{n}: ditolak cek memori, tunggu {JEDA_MEMORI_S // 60} menit ({tunggu}/{MAKS_TUNGGU_MEMORI})")
                    if tunggu >= MAKS_TUNGGU_MEMORI:
                        info["status"] = "gagal_memori"
                        break
                    time.sleep(JEDA_MEMORI_S)
                    continue
                if (rd / "weights" / "last.pt").exists():
                    resume += 1
                    if resume > MAKS_RESUME:
                        info["status"] = "gagal_resume_berulang"
                        log(f"{n}: gagal setelah {MAKS_RESUME} lanjutan, lihat {logf.name}")
                        break
                    log(f"{n}: berhenti tidak normal (kode {rc}) di epoch {n_epoch(rd)}; lanjut otomatis {resume}/{MAKS_RESUME} dalam 60 s")
                    time.sleep(60)
                    continue
                ulang += 1
                if ulang > MAKS_ULANG_BARU:
                    info["status"] = "gagal_sebelum_epoch_1"
                    log(f"{n}: gagal sebelum epoch 1 sebanyak {ulang} kali, lihat {logf.name}")
                    break
                log(f"{n}: gagal sebelum ada last.pt (kode {rc}); ulang {ulang}/{MAKS_ULANG_BARU} dalam 60 s")
                time.sleep(60)
            st["aktif"] = None
            simpan_status(st)
    finally:
        st["aktif"] = None
        st["berhenti_antrean"] = sekarang()
        simpan_status(st)
        if PID.exists() and PID.read_text().strip() == str(os.getpid()):
            PID.unlink()
        log("antrean berhenti")


if __name__ == "__main__":
    main()
