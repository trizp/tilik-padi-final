#!/usr/bin/env python3
"""
bench_batch.py - Tahap A: ukur VRAM puncak dan detik per epoch untuk tiap model x batch (1 epoch, imgsz 640, AMP).

Tiap kombinasi dijalankan lewat scripts/train.py --epochs 1 --close-mosaic 0 (mosaic MENYALA, seperti 90% epoch run
sungguhan; dengan close_mosaic=10 run 1 epoch akan berjalan tanpa mosaic). Run berurutan, satu per proses, agar memori
GPU kosong di awal tiap run. Run yang gagal (mis. memori habis) tetap dicatat.
Hanya split train dan val yang dipakai.

Keluaran: reports/<dataset>/batch_benchmark.csv, batch_benchmark.md, bench_logs/<run>.log
Pemakaian: python scripts/bench_batch.py --data data/processed/prov_v6/data.yaml
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/prov_v6/data.yaml")
    ap.add_argument("--models", nargs="+", default=["yolov8n.pt", "yolo11n.pt", "yolov8s.pt", "yolo11s.pt"])
    ap.add_argument("--batches", nargs="+", type=int, default=[8, 16])
    a = ap.parse_args()
    dy = yaml.safe_load((ROOT / a.data).read_text(encoding="utf-8"))
    rd = ROOT / "reports" / dy["dataset_name"]
    (rd / "bench_logs").mkdir(parents=True, exist_ok=True)
    rows = []
    for b in a.batches:
        for m in a.models:
            tag = f"bench_b{b}"
            name = f"{dy.get('run_prefix', '')}{Path(m).stem}_s0_{tag}"
            log = rd / "bench_logs" / f"{name}.log"
            print(f"== {name}", flush=True)
            with open(log, "w", encoding="utf-8") as f:
                rc = subprocess.call([sys.executable, str(ROOT / "scripts/train.py"), "--data", a.data, "--model", m, "--epochs", "1",
                                      "--batch", str(b), "--close-mosaic", "0", "--seed", "0", "--tag", tag],
                                     stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
            st_p = ROOT / "runs" / name / "run_stats.json"
            row = {"model": Path(m).stem, "batch_diminta": b, "exit_code": rc}
            if st_p.exists():
                st = json.loads(st_p.read_text(encoding="utf-8"))
                e = st["epoch"][0]
                g = st.get("gpu_nvml", {})
                row.update({"batch_aktual": e["batch_aktual"], "n_iterasi": e["n_iterasi"],
                            "vram_puncak_reserved_gb": e["vram_puncak_reserved_gb"], "vram_puncak_allocated_gb": e["vram_puncak_allocated_gb"],
                            "gpu_terpakai_sebelum_run_gb": g.get("gpu_terpakai_sebelum_run_gb"), "gpu_terpakai_puncak_gb": g.get("gpu_terpakai_puncak_gb"),
                            "gpu_sisa_minimum_gb": g.get("gpu_sisa_minimum_gb"), "gpu_util_median_persen": g.get("gpu_util_median_persen"),
                            "detik_epoch1_total": e["detik_total"], "detik_epoch1_train": e["detik_train"], "detik_epoch1_val": e["detik_val"],
                            "detik_per_iterasi_stabil": e["detik_per_iterasi_median_stabil"], "detik_train_stabil": e["detik_train_stabil_perkiraan"],
                            "perkiraan_100_epoch_jam_dari_epoch1": st.get("perkiraan_100_epoch_jam"),
                            "perkiraan_100_epoch_jam_train_stabil_plus_val": st.get("perkiraan_100_epoch_jam_(train_stabil+val)")})
            else:
                row["catatan"] = "run_stats.json tidak ada (lihat log)"
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(rd / "batch_benchmark.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    md = ["# Pengukuran batch (Tahap A)\n", "1 epoch, imgsz 640, AMP, mosaic menyala (close_mosaic=0 hanya untuk pengukuran). "
          "`gpu_terpakai_puncak` = memori SELURUH GPU (termasuk proses lain dan konteks CUDA) dari NVML.\n",
          "| model | batch | batch aktual | VRAM PyTorch (GB) | GPU terpakai puncak (GB) | sisa min (GB) | detik epoch 1 | detik train stabil + val | 100 epoch (jam) | exit |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        tv = (r["detik_train_stabil"] + r["detik_epoch1_val"]) if r.get("detik_train_stabil") else None
        md.append(f"| {r['model']} | {r['batch_diminta']} | {r.get('batch_aktual', '-')} | {r.get('vram_puncak_reserved_gb', '-')} | "
                  f"{r.get('gpu_terpakai_puncak_gb', '-')} | {r.get('gpu_sisa_minimum_gb', '-')} | {r.get('detik_epoch1_total', '-')} | "
                  f"{round(tv, 1) if tv else '-'} | {r.get('perkiraan_100_epoch_jam_train_stabil_plus_val', '-')} | {r['exit_code']} |")
    (rd / "batch_benchmark.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
