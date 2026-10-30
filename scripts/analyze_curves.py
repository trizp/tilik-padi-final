#!/usr/bin/env python3
"""
analyze_curves.py - Analisis kurva latihan (Tahap B) dari results.csv dan epoch_stats.jsonl tiap run.

Hanya memakai metrik VAL yang dicatat saat training (tidak ada inferensi baru, tidak menyentuh test).

Keluaran (--out, baku reports/prov_v6/kurva_tahap_b/):
  <run>_mAP50-95.png, <run>_mAP50.png   : semua kelas + per kelas per epoch
  <run>_loss.png                        : loss train vs val (box, cls, dfl, jumlah) + learning rate
  banding_mAP50-95.png                  : semua kelas, semua run dalam satu grafik
  ringkasan_kurva.csv, ringkasan_kurva.md, per_kelas_per_epoch_<run>.csv

Definisi:
  * "epoch capai X%"  = epoch PERTAMA dengan mAP50-95 val >= X% dari maksimum run itu.
  * "epoch stabil X%" = epoch pertama yang SETELAHNYA tidak pernah turun di bawah X% maksimum.
  * Tanda overfitting = loss val (dihaluskan rata-rata bergerak 5 epoch) naik > ambang dari titik minimumnya
    SEMENTARA loss train (dihaluskan) pada rentang yang sama turun; dilaporkan per komponen loss.
Pemakaian: python scripts/analyze_curves.py --runs runs/prov_v6_yolov8n_s0 runs/prov_v6_yolo11n_s0
"""
import argparse
import json
from pathlib import Path

import matplotlib
from jalur import jalur_keluaran  # noqa: E402  (pengaman jalur keluaran)
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
KOMP = ("box", "cls", "dfl")


def md_tabel(df, index=False):
    """Tabel markdown tanpa paket tambahan (tabulate tidak dipasang; versi paket dikunci)."""
    if index:
        df = df.reset_index()
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if v is None or (isinstance(v, float) and v != v) else str(v) for v in r) + " |")
    return "\n".join(lines)


def baca_run(d):
    res = pd.read_csv(d / "results.csv")
    res.columns = [c.strip() for c in res.columns]
    rows = {}
    for ln in (d / "epoch_stats.jsonl").read_text(encoding="utf-8").splitlines():
        if ln.strip():
            e = json.loads(ln)
            rows[e["epoch"]] = e                  # catatan terakhir per epoch (setelah resume)
    pk = []
    for ep, e in sorted(rows.items()):
        for kelas, m in e.get("per_kelas_val", {}).items():
            if not kelas.startswith("_"):
                pk.append({"epoch": ep, "kelas": kelas, **m})
    return res, pd.DataFrame(pk), rows


def epoch_capai(s, ep, frac):
    mx = s.max()
    hit = ep[s >= frac * mx]
    stabil = None
    for i in range(len(s)):
        if (s.iloc[i:] >= frac * mx).all():
            stabil = int(ep.iloc[i])
            break
    return (int(hit.iloc[0]) if len(hit) else None), stabil


def nilai_di(s, ep, e):
    m = ep == e
    return round(float(s[m].iloc[0]), 4) if m.any() else None


def overfit(res, win=5, ambang=0.03):
    out = {}
    for k in KOMP + ("jumlah",):
        if k == "jumlah":
            tr = sum(res[f"train/{c}_loss"] for c in KOMP)
            va = sum(res[f"val/{c}_loss"] for c in KOMP)
        else:
            tr, va = res[f"train/{k}_loss"], res[f"val/{k}_loss"]
        trs, vas = tr.rolling(win, min_periods=1).mean(), va.rolling(win, min_periods=1).mean()
        i_min = int(vas.idxmin())
        naik = float(vas.iloc[-1] / vas.iloc[i_min] - 1)
        turun_train = float(trs.iloc[-1] / trs.iloc[i_min] - 1)
        out[k] = {"epoch_val_loss_min": int(res["epoch"].iloc[i_min]), "val_loss_min_halus": round(float(vas.iloc[i_min]), 4),
                  "val_loss_akhir_halus": round(float(vas.iloc[-1]), 4), "val_naik_dari_min_persen": round(100 * naik, 1),
                  "train_berubah_di_rentang_sama_persen": round(100 * turun_train, 1),
                  "tanda_overfitting": bool(naik > ambang and turun_train < 0)}
    return out


def plot_map(res, pk, run, kol, judul, out):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(res["epoch"], res[f"metrics/{kol}(B)"], color="black", lw=2.2, label="semua kelas")
    key = "mAP50_95" if kol == "mAP50-95" else "mAP50"
    for kelas, g in pk.groupby("kelas"):
        lab = kelas + (" (provisional)" if kelas == "bacterial_leaf_blight" else "")
        ax.plot(g["epoch"], g[key], lw=1.2, alpha=0.85, label=lab)
    for x, t in ((100, "epoch 100"), (res["epoch"].max() - 9, "mosaic mati")):
        ax.axvline(x, color="grey", ls="--", lw=0.8)
        ax.text(x, ax.get_ylim()[0], " " + t, rotation=90, va="bottom", fontsize=8, color="grey")
    ax.set_xlabel("epoch")
    ax.set_ylabel(f"{kol} val")
    ax.set_title(f"{run}: {judul}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_loss(res, run, out):
    fig, axs = plt.subplots(1, 5, figsize=(20, 4))
    for ax, k in zip(axs, KOMP + ("jumlah",)):
        tr = sum(res[f"train/{c}_loss"] for c in KOMP) if k == "jumlah" else res[f"train/{k}_loss"]
        va = sum(res[f"val/{c}_loss"] for c in KOMP) if k == "jumlah" else res[f"val/{k}_loss"]
        ax.plot(res["epoch"], tr, label="train")
        ax.plot(res["epoch"], va, label="val")
        if k in ("cls", "jumlah") and len(res) > 6:
            ax.set_ylim(top=float(max(tr.iloc[3:].max(), va.iloc[3:].max())) * 1.1)   # epoch awal val cls sangat besar
        ax.set_title(f"{k} loss")
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    lr = [c for c in res.columns if c.startswith("lr/")]
    axs[4].plot(res["epoch"], res[lr[0]], color="tab:green")
    axs[4].set_title("learning rate (pg0)")
    axs[4].set_xlabel("epoch")
    axs[4].grid(alpha=0.3)
    fig.suptitle(run)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="reports/prov_v6/kurva_tahap_b")
    ap.add_argument("--bench", default="reports/prov_v6/batch_benchmark.csv")
    a = ap.parse_args()
    out = jalur_keluaran(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ring, detail, waktu = [], {}, {}
    fig, axc = plt.subplots(figsize=(9, 5))
    for r in a.runs:
        d = ROOT / r
        run = d.name
        res, pk, rows = baca_run(d)
        pk.to_csv(out / f"per_kelas_per_epoch_{run}.csv", index=False)
        plot_map(res, pk, run, "mAP50-95", "mAP50-95 val per epoch", out / f"{run}_mAP50-95.png")
        plot_map(res, pk, run, "mAP50", "mAP50 val per epoch", out / f"{run}_mAP50.png")
        plot_loss(res, run, out / f"{run}_loss.png")
        axc.plot(res["epoch"], res["metrics/mAP50-95(B)"], label=run)
        ep = res["epoch"]
        seri = {"SEMUA": res["metrics/mAP50-95(B)"].reset_index(drop=True)}
        seri50 = {"SEMUA": res["metrics/mAP50(B)"].reset_index(drop=True)}
        for kelas, g in pk.groupby("kelas"):
            g = g.set_index("epoch").reindex(ep)
            seri[kelas] = g["mAP50_95"].reset_index(drop=True)
            seri50[kelas] = g["mAP50"].reset_index(drop=True)
        epr = ep.reset_index(drop=True)
        for kelas, s in seri.items():
            c95, s95 = epoch_capai(s, epr, 0.95)
            c99, s99 = epoch_capai(s, epr, 0.99)
            ring.append({"run": run, "kelas": kelas + (" (provisional)" if kelas == "bacterial_leaf_blight" else ""),
                         "n_epoch": int(epr.max()), "mAP50_95_maks": round(float(s.max()), 4), "epoch_maks": int(epr[s.idxmax()]),
                         "epoch_capai_95": c95, "epoch_stabil_95": s95, "epoch_capai_99": c99, "epoch_stabil_99": s99,
                         "mAP50_95_ep100": nilai_di(s, epr, 100), "mAP50_95_maks_s.d._ep100": round(float(s[epr <= 100].max()), 4),
                         "mAP50_95_ep150": nilai_di(s, epr, 150),
                         "mAP50_ep100": nilai_di(seri50[kelas], epr, 100), "mAP50_ep150": nilai_di(seri50[kelas], epr, 150)})
        detail[run] = {"overfitting": overfit(res)}
        ev = list(rows.values())
        waktu[run] = {"epoch_tercatat": len(ev), "jam_gpu": round(sum(e["detik_total"] for e in ev) / 3600, 2),
                      "detik_per_epoch_rata2": round(sum(e["detik_total"] for e in ev[1:]) / max(1, len(ev) - 1), 1),
                      "detik_per_epoch_mosaic_mati_rata2": round(sum(e["detik_total"] for e in ev[-10:]) / 10, 1) if len(ev) >= 20 else None}
        segp = d / "run_segments.jsonl"
        if segp.exists():
            waktu[run]["segmen"] = [{"jenis": s["jenis"], "mulai": s["mulai"], "epoch_awal": s["epoch_awal"],
                                     "gpu_terpakai_sebelum_run_gb": s["gpu_terpakai_sebelum_run_gb"]}
                                    for s in map(json.loads, segp.read_text(encoding="utf-8").splitlines())]
    axc.axvline(100, color="grey", ls="--", lw=0.8)
    axc.set_xlabel("epoch")
    axc.set_ylabel("mAP50-95 val (semua kelas)")
    axc.grid(alpha=0.3)
    axc.legend()
    fig.tight_layout()
    fig.savefig(out / "banding_mAP50-95.png", dpi=130)
    plt.close(fig)
    df = pd.DataFrame(ring)
    df.to_csv(out / "ringkasan_kurva.csv", index=False)

    # proyeksi jam GPU run final: model n dari Tahap B; model s dari Tahap A dikalikan rasio (terukur / perkiraan Tahap A) model n
    proj = {}
    try:
        bb = pd.read_csv(ROOT / a.bench)
        bb = bb[bb.batch_diminta == 8].set_index("model")
        rasio = []
        for run, w in waktu.items():
            m = run.split("_")[2]
            if m in bb.index:
                est = (bb.loc[m, "detik_train_stabil"] + bb.loc[m, "detik_epoch1_val"])
                rasio.append(w["detik_per_epoch_rata2"] / est)
                proj[m] = w["detik_per_epoch_rata2"]
        k = sum(rasio) / len(rasio) if rasio else 1.0
        for m in bb.index:
            if m not in proj:
                proj[m] = round(float((bb.loc[m, "detik_train_stabil"] + bb.loc[m, "detik_epoch1_val"]) * k), 1)
        proj = {"detik_per_epoch": proj, "rasio_koreksi_tahap_A": round(k, 3),
                **{f"jam_gpu_3seed_4model_{E}_epoch": round(3 * sum(proj.values()) * E / 3600, 1) for E in (100, 150)}}
    except Exception as e:  # noqa: BLE001
        proj = {"galat": str(e)}
    (out / "ringkasan_kurva.json").write_text(json.dumps({"ringkasan": ring, "detail": detail, "waktu": waktu, "proyeksi_run_final": proj},
                                                        indent=2, ensure_ascii=False), encoding="utf-8")
    md = ["# Kurva latihan Tahap B (VAL saja)\n", md_tabel(df), "\n## Tanda overfitting (loss dihaluskan 5 epoch)\n"]
    for run, dd in detail.items():
        md.append(f"### {run}\n")
        md.append(md_tabel(pd.DataFrame(dd["overfitting"]).T, index=True))
        md.append("")
    md.append("## Waktu\n")
    md.append("```json\n" + json.dumps(waktu, indent=1, ensure_ascii=False) + "\n```")
    md.append("## Proyeksi jam GPU run final (3 seed x 4 model, batch 8)\n")
    md.append("```json\n" + json.dumps(proj, indent=1, ensure_ascii=False) + "\n```")
    (out / "ringkasan_kurva.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
