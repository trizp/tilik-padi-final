#!/usr/bin/env python3
"""
antrean_lanjutan.py - Pekerjaan SESUDAH antrean run final (urutan dicatat di KEPUTUSAN_PROYEK.md):

  0. TUNGGU sampai proses antrean_final.py berhenti DAN ke-12 run final selesai (cek tiap 10 menit; ringan).
     Bila antrean final berhenti sebelum semua run selesai, antrean ini tetap menunggu (tidak lanjut) dan mencatatnya di log.
  1. Ukuran kebocoran ketat split acak: scripts/kebocoran_ketat_randsplit.py (bila gagal, antrean BERHENTI; training tidak dimulai).
  2. Training randsplit: yolov8n lalu yolo11n, seed 0, konfigurasi identik run final (100 epoch, imgsz 640, batch 8, workers 2,
     patience 0); best.pt dipilih dengan val acak semua kelas (data.yaml prov_v6_randsplit). Nama run randsplit_prov_v6_<model>_s0.
     Cek memori oleh train.py sebelum tiap run/lanjutan; resume otomatis seperti antrean final.
  Evaluasi akhir TIDAK dijalankan di sini (dilakukan bersamaan dengan evaluasi akhir run final, atas perintah pemilik skripsi).

Status: runs/_antrean_lanjutan/status.json dan antrean.log (juga ditampilkan scripts/status_final.py).
Berhenti halus: buat berkas runs/_antrean_lanjutan/BERHENTI.
Jalankan terlepas (PowerShell):
  Start-Process -FilePath .venv\\Scripts\\python.exe -ArgumentList scripts\\antrean_lanjutan.py -WindowStyle Hidden
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import antrean_final as af  # noqa: E402

ROOT = af.ROOT
PY = sys.executable
DIR = ROOT / "runs" / "_antrean_lanjutan"
STATUS, PID, STOP = DIR / "status.json", DIR / "antrean.pid", DIR / "BERHENTI"
DATA = "data/processed/prov_v6_randsplit/data.yaml"
MODELS = ["yolov8n", "yolo11n"]
EPOCHS = 100
JEDA_TUNGGU_S = 600


def sekarang():
    return datetime.now().isoformat(timespec="seconds")


def log(msg):
    line = f"[{sekarang()}] {msg}"
    print(line, flush=True)
    with open(DIR / "antrean.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def simpan(st):
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(STATUS)


def final_selesai():
    """(antrean final masih hidup?, jumlah run final selesai)."""
    pid = None
    if af.PID.exists():
        try:
            pid = int(af.PID.read_text().strip())
        except ValueError:
            pid = None
    hidup = bool(pid and af.proses_hidup(pid))
    n = sum(af.selesai(ROOT / "runs" / af.nama(m, s)) for s in af.SEEDS for m in af.MODELS)
    return hidup, n


def jalankan(cmd, logf):
    with open(logf, "w", encoding="utf-8") as f:
        f.write(f"# {sekarang()} {' '.join(cmd)}\n")
        f.flush()
        r = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    return r.returncode, logf.read_text(encoding="utf-8", errors="replace")


def latih(m, st):
    """Satu run randsplit dengan cek memori (train.py), resume otomatis, dan pengulangan seperti antrean final."""
    n = f"randsplit_prov_v6_{m}_s0"
    rd = ROOT / "runs" / n
    info = st["run"].setdefault(n, {"status": "menunggu", "percobaan": []})
    if af.selesai(rd):
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
            cmd = [PY, "scripts/train.py", "--data", DATA, "--model", f"{m}.pt", "--epochs", str(EPOCHS), "--seed", "0",
                   "--batch", "8", "--workers", "2", "--patience", "0", "--imgsz", "640", "--val-after"]
            jenis = "baru"
        k = len(info["percobaan"]) + 1
        logf = DIR / f"{n}_{k:02d}_{jenis}.log"
        info["status"] = "berjalan"
        info["percobaan"].append({"ke": k, "jenis": jenis, "mulai": sekarang(), "log": logf.name})
        st["aktif"] = n
        simpan(st)
        log(f"{n}: mulai percobaan {k} ({jenis})")
        rc, teks = jalankan(cmd, logf)
        info["percobaan"][-1].update(selesai=sekarang(), kode_keluar=rc, epoch_selesai=af.n_epoch(rd))
        if af.selesai(rd):
            info["status"] = "selesai"
            simpan(st)
            log(f"{n}: selesai (kode {rc})")
            return True
        if af.n_epoch(rd) >= EPOCHS and (rd / "weights" / "best.pt").exists():
            jalankan([PY, "scripts/train.py", "--perbarui-stats", str(rd.relative_to(ROOT))], DIR / f"{n}_perbarui_stats.log")
            info["status"] = "selesai_tanpa_val_after"
            simpan(st)
            log(f"{n}: 100 epoch lengkap tetapi kode keluar {rc}; run_stats dibangun ulang")
            return True
        if "[DITOLAK] memori virtual" in teks:
            tunggu += 1
            info["status"] = f"menunggu_memori ({tunggu}/{af.MAKS_TUNGGU_MEMORI})"
            simpan(st)
            log(f"{n}: ditolak cek memori ({tunggu}/{af.MAKS_TUNGGU_MEMORI}), tunggu {af.JEDA_MEMORI_S // 60} menit")
            if tunggu >= af.MAKS_TUNGGU_MEMORI:
                info["status"] = "gagal_memori"
                simpan(st)
                return False
            time.sleep(af.JEDA_MEMORI_S)
            continue
        if (rd / "weights" / "last.pt").exists():
            resume += 1
            if resume > af.MAKS_RESUME:
                info["status"] = "gagal_resume_berulang"
                simpan(st)
                log(f"{n}: gagal setelah {af.MAKS_RESUME} lanjutan")
                return False
            log(f"{n}: berhenti tidak normal (kode {rc}) di epoch {af.n_epoch(rd)}; lanjut otomatis {resume}/{af.MAKS_RESUME}")
            time.sleep(60)
            continue
        ulang += 1
        if ulang > af.MAKS_ULANG_BARU:
            info["status"] = "gagal_sebelum_epoch_1"
            simpan(st)
            log(f"{n}: gagal sebelum epoch 1 sebanyak {ulang} kali")
            return False
        log(f"{n}: gagal sebelum ada last.pt (kode {rc}); ulang {ulang}/{af.MAKS_ULANG_BARU}")
        time.sleep(60)


def main():
    DIR.mkdir(parents=True, exist_ok=True)
    if PID.exists():
        try:
            lama = int(PID.read_text().strip())
        except ValueError:
            lama = -1
        if lama != os.getpid() and af.proses_hidup(lama):
            sys.exit(f"[DITOLAK] antrean lanjutan lain masih berjalan (PID {lama}).")
    PID.write_text(str(os.getpid()))
    st = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}
    st.update(pid=os.getpid(), mulai=sekarang(), aktif=None)
    st.setdefault("run", {})
    st.setdefault("tahap", "0_menunggu_run_final")
    simpan(st)
    log(f"antrean lanjutan mulai (PID {os.getpid()})")
    try:
        # ---- 0. tunggu run final
        catat_terakhir = 0.0
        while True:
            if STOP.exists():
                log("berkas BERHENTI ditemukan -> berhenti")
                return
            hidup, n_sel = final_selesai()
            total = len(af.SEEDS) * len(af.MODELS)
            st["run_final_selesai"] = f"{n_sel}/{total}"
            if not hidup and n_sel == total:
                log(f"run final selesai ({n_sel}/{total}) dan antrean final sudah berhenti -> lanjut")
                break
            st["tahap"] = "0_menunggu_run_final" if hidup else "0_menunggu: antrean final BERHENTI sebelum selesai"
            simpan(st)
            if time.time() - catat_terakhir > 3600:
                log(f"menunggu run final: {n_sel}/{total} selesai; antrean final {'berjalan' if hidup else 'TIDAK berjalan (perlu dilanjutkan pemilik)'}")
                catat_terakhir = time.time()
            time.sleep(JEDA_TUNGGU_S)
        # ---- 1. kebocoran ketat
        if st.get("kebocoran_ketat") != "selesai":
            st["tahap"] = "1_kebocoran_ketat"
            st["aktif"] = "kebocoran_ketat_randsplit"
            simpan(st)
            log("mulai ukuran kebocoran ketat split acak")
            rc, _ = jalankan([PY, "scripts/kebocoran_ketat_randsplit.py"], DIR / "kebocoran_ketat.log")
            if rc != 0 or not (ROOT / "reports/prov_v6_randsplit/kebocoran_ketat.json").exists():
                st["kebocoran_ketat"] = f"gagal (kode {rc})"
                st["tahap"] = "BERHENTI: kebocoran ketat gagal, training randsplit TIDAK dimulai"
                simpan(st)
                log(f"kebocoran ketat GAGAL (kode {rc}); lihat kebocoran_ketat.log. Training randsplit tidak dimulai.")
                return
            st["kebocoran_ketat"] = "selesai"
            simpan(st)
            log("kebocoran ketat selesai -> reports/prov_v6_randsplit/kebocoran_ketat.md")
        # ---- 2. training randsplit
        st["tahap"] = "2_training_randsplit"
        simpan(st)
        for m in MODELS:
            if STOP.exists():
                log("berkas BERHENTI ditemukan -> berhenti")
                return
            if not latih(m, st):
                st["tahap"] = f"BERHENTI: run randsplit {m} gagal"
                simpan(st)
                return
        st["tahap"] = "selesai (evaluasi akhir menunggu perintah pemilik skripsi)"
        simpan(st)
        log("antrean lanjutan selesai; evaluasi akhir TIDAK dijalankan otomatis")
    finally:
        st["aktif"] = None
        st["berhenti"] = sekarang()
        simpan(st)
        if PID.exists() and PID.read_text().strip() == str(os.getpid()):
            PID.unlink()


if __name__ == "__main__":
    main()
