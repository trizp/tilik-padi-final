#!/usr/bin/env python3
"""
status_final.py - Tampilkan status antrean training (hanya membaca berkas; aman dijalankan kapan saja).

  python scripts/status_final.py              # antrean v9 (prov_v9 latar dihapus: 6 run final)
  python scripts/status_final.py --versi v7   # antrean v7 (prov_v7: 12 final + 2 randsplit), selesai
  python scripts/status_final.py --versi v6   # antrean v6 lama (final + antrean lanjutan), arsip

Menampilkan: apakah antrean hidup, status tiap run, epoch selesai, mAP50-95 val terakhir/terbaik, perkiraan waktu selesai
(median detik/epoch run yang sudah berjalan; sebelum ada data dipakai angka run v6), memori virtual tersedia, log terakhir.
"""
import argparse
import json
import statistics
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from train import memori_virtual  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PERKIRAAN_S = {"yolov8n": 68, "yolo11n": 70, "yolov8s": 121, "yolo11s": 120}   # median detik/epoch run final v6 (batch 8)
OVERHEAD_S = 120                                                                 # persiapan + validasi akhir per run
EPOCHS = 100


def epochs(rd):
    p = rd / "epoch_stats.jsonl"
    if not p.exists():
        return []
    rows = {}
    for ln in p.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            e = json.loads(ln)
            rows[e["epoch"]] = e
    return [rows[k] for k in sorted(rows)]


def proses_hidup(pid):
    try:
        import psutil
        return psutil.pid_exists(pid) and "python" in psutil.Process(pid).name().lower()
    except Exception:
        return False


def tampil(judul, jobs, d):
    """jobs: [(nama_run, model)]; d: folder antrean (status.json, antrean.pid, antrean.log)."""
    st = json.loads((d / "status.json").read_text(encoding="utf-8")) if (d / "status.json").exists() else {}
    pid = int((d / "antrean.pid").read_text().strip()) if (d / "antrean.pid").exists() else None
    hidup = bool(pid and proses_hidup(pid))
    print(f"{judul}: {'BERJALAN (PID %d)' % pid if hidup else 'TIDAK berjalan'}; aktif: {st.get('aktif')}; mulai {st.get('mulai_antrean')}")
    data = {n: epochs(ROOT / "runs" / n) for n, _ in jobs}
    det = {}
    for n, m in jobs:
        det.setdefault(m, []).extend(e["detik_total"] for e in data[n][1:])
    spe = {m: (statistics.median(v) if v else PERKIRAAN_S[m]) for m, v in det.items()}
    sisa = 0.0
    print(f"\n{'run':<32}{'status':<26}{'epoch':>8}{'mAP50-95 akhir':>16}{'terbaik (ep)':>16}")
    for n, m in jobs:
        info = st.get("run", {}).get(n, {})
        ep = data[n]
        k = len(ep)
        stt = info.get("status", "menunggu")
        if not stt.startswith("selesai") and not stt.startswith("gagal"):
            sisa += (EPOCHS - k) * spe[m] + OVERHEAD_S
        akhir = ep[-1]["metrik_val"].get("metrics/mAP50-95(B)") if ep else None
        best = max(ep, key=lambda e: e["metrik_val"].get("metrics/mAP50-95(B)", 0)) if ep else None
        tb = f"{best['metrik_val']['metrics/mAP50-95(B)']:.4f} ({best['epoch']})" if best else "-"
        print(f"{n:<32}{stt:<26}{k:>5}/{EPOCHS}{(f'{akhir:.4f}' if akhir is not None else '-'):>16}{tb:>16}")
    print("\nDetik/epoch dipakai untuk perkiraan: " + ", ".join(f"{m} {spe[m]:.0f}" + ("" if det[m] else " (angka v6)") for m in spe))
    if sisa:
        print(f"Sisa waktu perkiraan: {sisa / 3600:.1f} jam -> selesai sekitar {(datetime.now() + timedelta(seconds=sisa)):%a %d-%m %H:%M}")
    else:
        print("Tidak ada run tersisa.")
    lg = d / "antrean.log"
    if lg.exists():
        print("\nLog antrean (5 baris terakhir):")
        for ln in lg.read_text(encoding="utf-8").splitlines()[-5:]:
            print("  " + ln)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--versi", choices=["v9", "v7", "v6"], default="v9")
    a = ap.parse_args()
    if a.versi == "v9":
        import antrean_v9 as av
        tampil("Antrean v9 (latar dihapus)", [(n, m) for n, m, _ in av.JOBS], av.DIR)
        stp = av.DIR / "status.json"
        if stp.exists():
            print("Kunci ambang VAL v9:", json.loads(stp.read_text(encoding="utf-8")).get("kunci_ambang", "belum (setelah 6 run selesai)"))
    elif a.versi == "v7":
        import antrean_v7 as av
        tampil("Antrean v7", [(n, m) for n, m, _ in av.JOBS], av.DIR)
    else:
        import antrean_final as af
        tampil("Antrean final v6", [(af.nama(m, s), m) for s in af.SEEDS for m in af.MODELS], af.DIR)
        dl = ROOT / "runs" / "_antrean_lanjutan"
        if (dl / "status.json").exists():
            print()
            tampil("Antrean lanjutan v6 (randsplit)", [(f"randsplit_prov_v6_{m}_s0", m) for m in ("yolov8n", "yolo11n")], dl)
    mv = memori_virtual()
    print(f"\nMemori virtual tersedia {mv.get('virtual_tersedia_gb')} GB dari {mv.get('virtual_batas_gb')} GB; RAM tersedia {mv.get('ram_tersedia_gb')} GB")


if __name__ == "__main__":
    main()
