#!/usr/bin/env python3
"""
train.py - Training YOLO (Ultralytics) dengan hyperparameter IDENTIK untuk semua model (yolov8n/s, yolo11n/s).

* Hyperparameter tetap ada di HP (disimpan ke run dir) agar perbandingan YOLOv8 vs YOLO11 adil.
* Nama run = <run_prefix dari data.yaml><model>_s<seed>[_<tag>]  -> run dari dataset provisional otomatis berawalan prov_.
* Hanya split TRAIN dan VAL yang dipakai. Skrip ini TIDAK PERNAH mengevaluasi split test / test_new_session.
* Tiap epoch ditulis BERTAHAP ke <run>/epoch_stats.jsonl: waktu, batch aktual, VRAM puncak, metrik val semua kelas
  DAN per kelas (P, R, mAP50, mAP50-95). Aman bila run terputus; --resume melanjutkan pencatatan.
* Awal tiap segmen (run baru / lanjutan) ditulis ke <run>/run_segments.jsonl: memori GPU terpakai SEBELUM run
  (gpu_terpakai_sebelum_run_gb; proses lain seperti tampilan Windows), versi library.
* Versi torch/ultralytics dicek terhadap requirements.lock; run ditolak bila berbeda (kecuali --abaikan-lock).
* Ultralytics bisa MENGECILKAN batch diam-diam bila memori habis di epoch 1; batch aktual dicatat dan
  run dinyatakan GAGAL bila berbeda dari yang diminta.
* Memori virtual Windows (commit) dicatat per fase (persiapan/train/val/simpan/val_akhir) per epoch dan per segmen.
  Run DITOLAK mulai bila memori virtual tersedia < puncak terukur (configs/ambang_memori.json) + 5 GB,
  kecuali --abaikan-cek-memori.
* Pemilihan best.pt: pakai data_valsel.yaml (scripts/buat_valsel.py) agar skor pemilihan hanya dari val bersih
  (tanpa BLB provisional). Aturannya dicatat di run_config.json (aturan_pemilihan_checkpoint).

Contoh:
  python scripts/train.py --data data/processed/prov_v6/data.yaml --model yolov8n.pt --epochs 3 --seed 0 --tag smoke --val-after
  python scripts/train.py --data ... --model yolo11s.pt --epochs 1 --batch 16 --close-mosaic 0 --tag bench_b16   (pengukuran batch)
  python scripts/train.py --data ... --model yolov8n.pt --epochs 150 --seed 0 --patience 0                     (kurva latihan)
  python scripts/train.py --resume runs/prov_v6_yolov8n_s0/weights/last.pt
"""
import argparse
import csv
import json
import platform
import statistics
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import yaml
from jalur import jalur_keluaran  # noqa: E402  (pengaman jalur keluaran)

ROOT = Path(__file__).resolve().parent.parent

# Hyperparameter identik untuk semua model (jangan diubah per model).
# 2026-10-04 (keputusan run final): patience 50->0, workers 4->2 (cek memori: workers 2 tidak lebih lambat, runs/prov_v6_cekmemori_w2).
HP = dict(imgsz=640, batch=8, optimizer="SGD", lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005,
          warmup_epochs=3.0, cos_lr=False, close_mosaic=10, patience=0, amp=True, workers=2,
          hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, fliplr=0.5, mosaic=1.0, mixup=0.0)
LOCKED = ("torch", "torchvision", "ultralytics")


def cek_lock():
    """Bandingkan versi terpasang dengan requirements.lock. Kembalikan daftar selisih."""
    lock = ROOT / "requirements.lock"
    if not lock.exists():
        return ["requirements.lock tidak ada"]
    want = {}
    for ln in lock.read_text(encoding="utf-8").splitlines():
        if "==" in ln and not ln.startswith("#"):
            k, v = ln.strip().split("==", 1)
            want[k.lower()] = v
    from importlib.metadata import version
    return [f"{p}: terpasang {version(p)} != lock {want.get(p)}" for p in LOCKED if version(p) != want.get(p)]


AMBANG_VIRTUAL_GB = 10.0


def memori_virtual():
    """Memori virtual Windows (commit = RAM + pagefile): batas, tersedia, RAM fisik tersedia (GB)."""
    try:
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MS()
        m.dwLength = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        g = 1024 ** 3
        return {"virtual_batas_gb": round(m.ullTotalPageFile / g, 2), "virtual_tersedia_gb": round(m.ullAvailPageFile / g, 2),
                "ram_total_gb": round(m.ullTotalPhys / g, 2), "ram_tersedia_gb": round(m.ullAvailPhys / g, 2)}
    except Exception as e:  # bukan Windows / gagal membaca
        return {"galat": str(e)}


AMBANG_MEMORI_P = ROOT / "configs" / "ambang_memori.json"   # puncak terukur (kenaikan commit oleh training), lihat berkasnya
MARGIN_MEMORI_GB = 5.0


def cek_memori_virtual(abaikan=False):
    """Tolak run bila memori virtual tersedia < puncak terukur + 5 GB (kecuali --abaikan-cek-memori).
    Tanpa configs/ambang_memori.json: hanya peringatan dengan ambang lama 10 GB."""
    m = memori_virtual()
    v = m.get("virtual_tersedia_gb")
    print(f"[memori] virtual tersedia {v} GB dari batas {m.get('virtual_batas_gb')} GB (RAM tersedia {m.get('ram_tersedia_gb')} GB)", flush=True)
    if AMBANG_MEMORI_P.exists():
        amb = json.loads(AMBANG_MEMORI_P.read_text(encoding="utf-8"))
        perlu = round(amb["puncak_terukur_gb"] + MARGIN_MEMORI_GB, 2)
        m.update(ambang_perlu_gb=perlu, ambang_sumber=amb.get("sumber"), lolos_cek=v is not None and v >= perlu,
                 cek_diabaikan=bool(abaikan))
        if not m["lolos_cek"]:
            pesan = (f"memori virtual tersedia {v} GB < {perlu} GB (puncak terukur {amb['puncak_terukur_gb']} GB + {MARGIN_MEMORI_GB} GB; "
                     f"sumber {amb.get('sumber')}). Tutup aplikasi lain atau perbesar pagefile.")
            if not abaikan:
                sys.exit("[DITOLAK] " + pesan + " Pakai --abaikan-cek-memori untuk tetap menjalankan.")
            print("[PERINGATAN, diabaikan] " + pesan, flush=True)
        else:
            print(f"[memori] lolos cek: {v} GB >= {perlu} GB", flush=True)
    else:
        m["peringatan"] = v is not None and v < AMBANG_VIRTUAL_GB
        if m["peringatan"]:
            print(f"[PERINGATAN] memori virtual tersedia {v} GB < {AMBANG_VIRTUAL_GB} GB (belum ada {AMBANG_MEMORI_P.name}).", flush=True)
    return m


class MemoriPoller:
    """Memori virtual (commit) tiap 0,5 s, dikelompokkan per FASE (persiapan / train / val / simpan / val_akhir):
    * sistem: commit terpakai (batas - tersedia) dan sisa terendah;
    * proses: jumlah private bytes (commit) proses training + seluruh anak (worker dataloader) dan jumlah proses.
    Kenaikan oleh training = commit sistem puncak - commit sistem sebelum run."""

    def __init__(self, interval=0.5):
        self.interval, self.fase, self._stop = interval, "persiapan", threading.Event()
        m = memori_virtual()
        self.ok = "virtual_batas_gb" in m
        self.awal_terpakai = round(m["virtual_batas_gb"] - m["virtual_tersedia_gb"], 2) if self.ok else None
        self.total, self.epoch = {}, {}
        try:
            import psutil
            self.ps = psutil.Process()
        except Exception:
            self.ps = None

    def _proses(self):
        if not self.ps:
            return None, None
        tot, n = 0, 0
        for p in [self.ps] + self.ps.children(recursive=True):
            try:
                mi = p.memory_info()
                tot += getattr(mi, "private", mi.vms)
                n += 1
            except Exception:       # proses worker bisa berhenti di tengah pembacaan
                pass
        return tot / 1024 ** 3, n

    def _catat(self, wadah, fase, terpakai, sisa, proc, n):
        d = wadah.setdefault(fase, {"puncak_terpakai_gb": 0.0, "sisa_terendah_gb": 1e9, "puncak_proses_gb": 0.0, "proses_maks": 0, "n_sampel": 0})
        d["puncak_terpakai_gb"] = max(d["puncak_terpakai_gb"], terpakai)
        d["sisa_terendah_gb"] = min(d["sisa_terendah_gb"], sisa)
        if proc is not None:
            d["puncak_proses_gb"] = max(d["puncak_proses_gb"], proc)
            d["proses_maks"] = max(d["proses_maks"], n)
        d["n_sampel"] += 1

    def _run(self):
        while not self._stop.is_set():
            try:
                m = memori_virtual()
                terpakai, sisa = m["virtual_batas_gb"] - m["virtual_tersedia_gb"], m["virtual_tersedia_gb"]
                proc, n = self._proses()
                f = self.fase
                self._catat(self.total, f, terpakai, sisa, proc, n)
                self._catat(self.epoch, f, terpakai, sisa, proc, n)
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        if self.ok:
            threading.Thread(target=self._run, daemon=True).start()

    def _bulat(self, w):
        out = {}
        for f, d in w.items():
            out[f] = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in d.items()}
            if self.awal_terpakai is not None:
                out[f]["kenaikan_puncak_gb"] = round(d["puncak_terpakai_gb"] - self.awal_terpakai, 2)
        return out

    def ambil_epoch(self):
        """Ringkasan sejak awal epoch, lalu reset akumulator epoch."""
        r, self.epoch = self._bulat(dict(self.epoch)), {}
        return r

    def stop(self):
        self._stop.set()
        if not self.ok:
            return {}
        per_fase = self._bulat(self.total)
        semua = [d for d in per_fase.values()]
        return {"commit_terpakai_sebelum_run_gb": self.awal_terpakai, "per_fase": per_fase,
                "puncak_terpakai_gb": max((d["puncak_terpakai_gb"] for d in semua), default=None),
                "kenaikan_puncak_gb": max((d.get("kenaikan_puncak_gb", 0) for d in semua), default=None),
                "sisa_terendah_gb": min((d["sisa_terendah_gb"] for d in semua), default=None),
                "puncak_proses_gb": max((d["puncak_proses_gb"] for d in semua), default=None)}


def info_lanjutan(segmen):
    """Jumlah lanjutan dan epoch saat error: run terputus di epoch tempat segmen berikutnya mulai."""
    lanj = [s for s in segmen if str(s.get("jenis", "")).startswith("lanjutan")]
    return {"jumlah_lanjutan_resume": len(lanj), "epoch_saat_error": [s.get("epoch_awal") for s in lanj],
            "waktu_lanjutan": [s.get("mulai") for s in lanj]}


class GpuPoller:
    """Membaca memori terpakai SELURUH GPU (semua proses, termasuk konteks CUDA dan tampilan Windows) tiap 0,5 s."""

    def __init__(self, interval=0.5):
        self.interval, self.peak_mb, self.util, self.baseline_mb, self.total_mb = interval, 0, [], None, None
        self._stop = threading.Event()
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nv, self.h = pynvml, pynvml.nvmlDeviceGetHandleByIndex(0)
            m = pynvml.nvmlDeviceGetMemoryInfo(self.h)
            self.baseline_mb, self.total_mb = m.used / 2 ** 20, m.total / 2 ** 20
        except Exception:
            self.nv = None

    def _run(self):
        while not self._stop.is_set():
            m = self.nv.nvmlDeviceGetMemoryInfo(self.h)
            self.peak_mb = max(self.peak_mb, m.used / 2 ** 20)
            self.util.append(self.nv.nvmlDeviceGetUtilizationRates(self.h).gpu)
            time.sleep(self.interval)

    def start(self):
        if self.nv:
            threading.Thread(target=self._run, daemon=True).start()

    def baseline_gb(self):
        return round(self.baseline_mb / 1024, 3) if self.nv else None

    def stop(self):
        self._stop.set()
        if not self.nv:
            return {}
        return {"gpu_total_gb": round(self.total_mb / 1024, 3), "gpu_terpakai_sebelum_run_gb": self.baseline_gb(),
                "gpu_terpakai_puncak_gb": round(self.peak_mb / 1024, 3), "gpu_sisa_minimum_gb": round((self.total_mb - self.peak_mb) / 1024, 3),
                "gpu_util_median_persen": statistics.median(self.util) if self.util else None,
                "gpu_util_rata2_persen": round(statistics.mean(self.util), 1) if self.util else None}

    def util_rata2_sejak(self, i):
        u = self.util[i:]
        return round(statistics.mean(u), 1) if u else None


def baca_jsonl(p):
    """Baca jsonl; bila satu epoch tercatat lebih dari sekali (setelah resume), ambil catatan terakhir."""
    if not p.exists():
        return []
    rows = [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if rows and "epoch" in rows[0]:
        rows = list({r["epoch"]: r for r in rows}.values())
        rows.sort(key=lambda r: r["epoch"])
    return rows


def pasang_pencatat(model, torch, poller, info_segmen, mpoll=None):
    """Pasang callback pencatat per epoch. Data ditulis bertahap ke run dir (aman untuk resume)."""
    t = {"iter": [], "n_epoch_dicatat": 0}

    def on_train_start(tr):
        d = Path(tr.save_dir)
        t["dir"] = d
        t["n_epoch_dicatat"] = len(baca_jsonl(d / "epoch_stats.jsonl"))
        seg = dict(info_segmen, mulai=datetime.now().isoformat(timespec="seconds"), epoch_awal=int(tr.start_epoch) + 1,
                   gpu_terpakai_sebelum_run_gb=poller.baseline_gb())
        with open(d / "run_segments.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(seg, ensure_ascii=False) + "\n")
        print(f"[stats] segmen mulai epoch {seg['epoch_awal']}; gpu_terpakai_sebelum_run_gb = {seg['gpu_terpakai_sebelum_run_gb']}", flush=True)

    def on_train_epoch_start(tr):
        t["start"] = time.time()
        t["iter"] = []
        t["util_i"] = len(poller.util)
        if mpoll:
            mpoll.fase = "train"
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_train_batch_end(tr):
        t["iter"].append(time.time())

    def on_train_epoch_end(tr):
        t["train_end"] = time.time()
        if mpoll:
            mpoll.fase = "val"

    def on_fit_epoch_end(tr):
        now = time.time()
        if int(tr.epoch) + 1 > tr.epochs or "train_end" not in t or t.get("epoch_terakhir") == int(tr.epoch):
            # Ultralytics memanggil callback ini sekali lagi saat validasi akhir best.pt -> bukan epoch, catat terpisah
            t["validasi_akhir_best"] = {"detik": round(now - t.get("train_end", now), 1),
                                        "metrik_val": {k: round(float(v), 5) for k, v in (tr.metrics or {}).items()},
                                        "memori_virtual": mpoll.ambil_epoch() if mpoll else None}
            return
        t["epoch_terakhir"] = int(tr.epoch)
        it = t["iter"]
        dts = [b - a_ for a_, b in zip(it, it[1:])]
        skip = max(20, len(dts) // 10)              # buang iterasi awal (pemanasan dataloader/cuDNN)
        steady = statistics.median(dts[skip:]) if len(dts) > skip + 10 else None
        per_kelas = {}
        try:
            box = tr.validator.metrics.box
            for k, ci in enumerate(box.ap_class_index):
                p_, r_, ap50, ap = box.class_result(k)
                per_kelas[tr.data["names"][int(ci)]] = {"precision": round(float(p_), 5), "recall": round(float(r_), 5),
                                                        "mAP50": round(float(ap50), 5), "mAP50_95": round(float(ap), 5)}
        except Exception as e:  # pencatatan tidak boleh menghentikan training
            per_kelas = {"_galat": str(e)}
        n_tr = len(getattr(getattr(tr, "train_loader", None), "dataset", []) or [])
        n_va = len(getattr(getattr(tr, "test_loader", None), "dataset", []) or [])
        e = {"epoch": int(tr.epoch) + 1, "batch_aktual": int(tr.batch_size), "n_iterasi": len(it),
             "n_gambar_train": n_tr, "n_gambar_val": n_va,
             "detik_per_gambar_train": round((t["train_end"] - t["start"]) / n_tr, 5) if n_tr else None,
             "detik_per_gambar_val": round((now - t["train_end"]) / n_va, 5) if n_va else None,
             "gpu_util_rata2_persen": poller.util_rata2_sejak(t.get("util_i", 0)),
             "memori_virtual": mpoll.ambil_epoch() if mpoll else None,
             "detik_train": round(t["train_end"] - t["start"], 1), "detik_val": round(now - t["train_end"], 1),
             "detik_total": round(now - t["start"], 1),
             "detik_per_iterasi_median_stabil": round(steady, 4) if steady else None,
             "detik_train_stabil_perkiraan": round(steady * len(it), 1) if steady else None,
             "vram_puncak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3) if torch.cuda.is_available() else None,
             "vram_puncak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024 ** 3, 3) if torch.cuda.is_available() else None,
             "lr": {k: float(v) for k, v in (tr.lr or {}).items()},
             "metrik_val": {k: round(float(v), 5) for k, v in (tr.metrics or {}).items()}, "per_kelas_val": per_kelas}
        with open(t["dir"] / "epoch_stats.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
        if mpoll:   # sesudah epoch terakhir Ultralytics memvalidasi ulang best.pt
            mpoll.fase = "val_akhir" if int(tr.epoch) + 1 >= tr.epochs else "simpan"
        print(f"[stats] epoch {e['epoch']}/{tr.epochs}: {e['detik_total']} s (train {e['detik_train']}, val {e['detik_val']}), "
              f"batch {e['batch_aktual']}, VRAM puncak {e['vram_puncak_reserved_gb']} GB, "
              f"mAP50-95 {e['metrik_val'].get('metrics/mAP50-95(B)')}", flush=True)

    for ev, fn in (("on_train_start", on_train_start), ("on_train_epoch_start", on_train_epoch_start),
                   ("on_train_batch_end", on_train_batch_end), ("on_train_epoch_end", on_train_epoch_end),
                   ("on_fit_epoch_end", on_fit_epoch_end)):
        model.add_callback(ev, fn)
    return t


def ringkas(run_dir, stats, batch_diminta):
    """Bangun ringkasan dari epoch_stats.jsonl (seluruh segmen)."""
    ep = baca_jsonl(run_dir / "epoch_stats.jsonl")
    stats["epoch"] = ep
    stats["segmen"] = baca_jsonl(run_dir / "run_segments.jsonl")
    stats.update(info_lanjutan(stats["segmen"]))
    if ep:
        nomor = [e["epoch"] for e in ep]
        stats["epoch_tidak_tercatat"] = sorted(set(range(1, max(nomor) + 1)) - set(nomor))
        steady = ep[1:] if len(ep) > 1 else ep          # epoch pertama memuat overhead awal
        mean = sum(e["detik_total"] for e in steady) / len(steady)
        stats["detik_per_epoch_rata2_(tanpa_epoch_1)" if len(ep) > 1 else "detik_per_epoch_(hanya_epoch_1)"] = round(mean, 1)
        stats["perkiraan_100_epoch_jam"] = round(mean * 100 / 3600, 2)
        stats["jam_gpu_seluruh_epoch"] = round(sum(e["detik_total"] for e in ep) / 3600, 2)
        if ep[-1]["detik_train_stabil_perkiraan"]:
            stats["perkiraan_100_epoch_jam_(train_stabil+val)"] = round((ep[-1]["detik_train_stabil_perkiraan"] + ep[-1]["detik_val"]) * 100 / 3600, 2)
        stats["vram_puncak_reserved_gb"] = max(e["vram_puncak_reserved_gb"] or 0 for e in ep)
        stats["batch_aktual"] = sorted({e["batch_aktual"] for e in ep})
        stats["batch_sesuai_permintaan"] = stats["batch_aktual"] == [batch_diminta]
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=HP["batch"])
    ap.add_argument("--close-mosaic", type=int, default=None,
                    help="HANYA untuk pengukuran: run 1 epoch dengan close_mosaic=10 mematikan mosaic sejak awal (tidak representatif)")
    ap.add_argument("--patience", type=int, default=None, help="0 = tanpa early stopping")
    ap.add_argument("--workers", type=int, default=None, help="jumlah worker dataloader train (val = 2x); baku HP['workers']")
    ap.add_argument("--imgsz", type=int, default=None, help="HANYA uji coba ukuran input; baku HP['imgsz'] (identik semua model)")
    ap.add_argument("--nama", default=None, help="nama run lengkap (wajib berawalan run_prefix dataset), menggantikan <model>_s<seed>_<tag>")
    ap.add_argument("--abaikan-cek-memori", action="store_true",
                    help="tetap jalan walau memori virtual tersedia < puncak terukur + 5 GB (configs/ambang_memori.json)")
    ap.add_argument("--tanpa-val", action="store_true",
                    help="model deploy: tanpa validasi per epoch (Ultralytics tetap memvalidasi epoch terakhir); checkpoint yang dipakai = last.pt")
    ap.add_argument("--tag", default="")
    ap.add_argument("--project", default="runs")
    ap.add_argument("--resume", default=None, help="path last.pt untuk melanjutkan")
    ap.add_argument("--val-after", action="store_true", help="evaluasi best.pt pada split VAL (per kelas) setelah training")
    ap.add_argument("--abaikan-lock", action="store_true", help="jalankan walau versi tidak sama dengan requirements.lock")
    ap.add_argument("--perbarui-stats", default=None, metavar="RUN_DIR",
                    help="TANPA training: bangun ulang run_stats.json dari epoch_stats.jsonl + run_segments.jsonl")
    a = ap.parse_args()

    if a.perbarui_stats:
        run_dir = jalur_keluaran(a.perbarui_stats)
        sp = run_dir / "run_stats.json"
        base = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else             json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        st = ringkas(run_dir, base, base.get("hp", {}).get("batch", HP["batch"]))
        sp.write_text(json.dumps(st, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({k: st.get(k) for k in ("run", "jumlah_lanjutan_resume", "epoch_saat_error", "epoch_tidak_tercatat",
                                                 "batch_sesuai_permintaan", "jam_gpu_seluruh_epoch")}, ensure_ascii=False))
        return

    beda = cek_lock()
    if beda and not a.abaikan_lock:
        sys.exit("[GAGAL] versi library tidak sama dengan requirements.lock:\n  " + "\n  ".join(beda))

    mem = cek_memori_virtual(a.abaikan_cek_memori)     # dicatat di awal tiap segmen (run baru / lanjutan)
    mpoll = MemoriPoller()         # memori virtual per fase selama run
    poller = GpuPoller()           # dibaca SEBELUM model dimuat -> memori yang dipakai proses lain
    import torch
    from ultralytics import YOLO, __version__ as ul_ver

    versi = {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda, "ultralytics": ul_ver,
             "platform": platform.platform(), "sesuai_lock": not beda}
    if a.resume:
        last = Path(a.resume) if Path(a.resume).is_absolute() else ROOT / a.resume
        run_dir = last.parent.parent
        prev = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8")) if (run_dir / "run_config.json").exists() else {}
        stats = dict(prev, versi=versi)
        model = YOLO(str(last))
        pasang_pencatat(model, torch, poller, {"jenis": "lanjutan (resume)", "versi": versi, "memori_virtual_awal": mem}, mpoll)
        poller.start()
        mpoll.start()
        try:
            model.train(resume=True)
        finally:
            stats["gpu_nvml_segmen_terakhir"] = poller.stop()
            stats["memori_virtual_segmen_terakhir"] = mpoll.stop()
        batch_diminta = prev.get("hp", {}).get("batch", HP["batch"])
    else:
        data_p = (ROOT / a.data) if not Path(a.data).is_absolute() else Path(a.data)
        dy = yaml.safe_load(data_p.read_text(encoding="utf-8"))
        # berkas turunan (data_valsel.yaml) membawa induk_data -> metadata dataset beku dibaca dari induknya
        induk = Path(dy["induk_data"]) if dy.get("induk_data") else data_p
        meta_p = induk.parent / "dataset_meta.json"
        meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
        if not meta.get("frozen"):
            print("[PERINGATAN] dataset belum dibekukan (frozen=false).")
        stem = Path(a.model).stem
        name = f"{dy.get('run_prefix', '')}{stem}_s{a.seed}" + (f"_{a.tag}" if a.tag else "")
        if a.nama:
            rp = dy.get("run_prefix", "")
            # run final: "final_" + run_prefix (penanda dataset provisional tetap ada di nama)
            if not (a.nama.startswith(rp) or a.nama.startswith("final_" + rp)):
                sys.exit(f"[GAGAL] --nama harus berawalan '{rp}' atau 'final_{rp}'")
            name = a.nama
        hp = dict(HP, batch=a.batch)
        override = {}
        if a.workers is not None and a.workers != HP["workers"]:
            hp["workers"] = override["workers"] = a.workers
        if a.imgsz is not None and a.imgsz != HP["imgsz"]:
            hp["imgsz"] = override["imgsz"] = a.imgsz
        val_tiap_epoch = not a.tanpa_val
        if a.tanpa_val:
            override["val_tiap_epoch"] = False
        if a.close_mosaic is not None:
            hp["close_mosaic"] = override["close_mosaic"] = a.close_mosaic
        if a.patience is not None:
            hp["patience"] = override["patience"] = a.patience
        stats = {"run": name, "model": a.model, "epochs": a.epochs, "seed": a.seed, "hp": hp, "hp_override_dari_HP_baku": override,
                 "data": str(data_p), "dataset": dy.get("dataset_name"), "provisional": dy.get("provisional"),
                 "manifest_sha256": dy.get("manifest_sha256"), "frozen": meta.get("frozen"),
                 "induk_data": str(induk) if induk != data_p else None,
                 "aturan_pemilihan_checkpoint": dy.get("aturan_pemilihan_checkpoint") or
                 "best.pt = fitness Ultralytics (mAP50-95) tertinggi pada val penuh data.yaml (TERMASUK gambar provisional bila ada)",
                 "val_dikecualikan_sumber": dy.get("val_dikecualikan_sumber", []),
                 "val_kelas_berlabel": dy.get("val_kelas_berlabel"),
                 "cek_memori_awal": mem,
                 "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                 "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2) if torch.cuda.is_available() else None,
                 "versi": versi}
        model = YOLO(a.model)
        pasang_pencatat(model, torch, poller, {"jenis": "baru", "versi": versi, "memori_virtual_awal": mem}, mpoll)

        def simpan_config(tr):         # konfigurasi run disimpan di awal agar resume bisa membacanya
            (Path(tr.save_dir) / "run_config.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
        model.add_callback("on_pretrain_routine_end", simpan_config)
        poller.start()
        mpoll.start()
        t0 = time.time()
        try:
            model.train(data=str(data_p), epochs=a.epochs, seed=a.seed, deterministic=True, device=0, project=str(jalur_keluaran(a.project)),
                        name=name, exist_ok=False, plots=True, val=val_tiap_epoch, **hp)
        finally:
            stats["gpu_nvml_segmen_terakhir"] = poller.stop()
            stats["memori_virtual_segmen_terakhir"] = mpoll.stop()
        stats["detik_total_training_segmen_ini"] = round(time.time() - t0, 1)
        run_dir = Path(model.trainer.save_dir)
        batch_diminta = a.batch
    stats = ringkas(run_dir, stats, batch_diminta)
    if a.val_after and not a.resume:
        best = YOLO(str(run_dir / "weights" / "best.pt"))
        r = best.val(data=stats["data"], split="val", imgsz=stats["hp"]["imgsz"], batch=stats["hp"]["batch"], device=0, plots=False,
                     project=str(run_dir), name="val_best")
        names = r.names
        rows = []
        for k, ci in enumerate(r.box.ap_class_index):
            p_, r_, ap50, ap = r.box.class_result(k)
            rows.append({"kelas": names[int(ci)], "precision": round(float(p_), 4), "recall": round(float(r_), 4),
                         "f1": round(float(2 * p_ * r_ / (p_ + r_)) if (p_ + r_) else 0.0, 4), "mAP50": round(float(ap50), 4), "mAP50_95": round(float(ap), 4)})
        rows.append({"kelas": "SEMUA", "precision": round(float(r.box.mp), 4), "recall": round(float(r.box.mr), 4),
                     "f1": round(float(2 * r.box.mp * r.box.mr / (r.box.mp + r.box.mr)) if (r.box.mp + r.box.mr) else 0.0, 4),
                     "mAP50": round(float(r.box.map50), 4), "mAP50_95": round(float(r.box.map), 4)})
        with open(run_dir / "val_per_class.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        stats["val_per_class"] = rows
        stats["val_speed_ms"] = {k: round(float(v), 3) for k, v in r.speed.items()}
    (run_dir / "run_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in stats.items() if k != "epoch"}, indent=1, ensure_ascii=False)[:4000])
    if stats.get("epoch") and not stats["batch_sesuai_permintaan"]:
        sys.exit(f"[GAGAL] batch aktual {stats['batch_aktual']} != diminta {batch_diminta} (Ultralytics mengecilkan batch karena memori habis)")


if __name__ == "__main__":
    main()
