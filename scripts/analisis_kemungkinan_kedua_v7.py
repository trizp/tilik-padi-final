#!/usr/bin/env python3
"""
analisis_kemungkinan_kedua_v7.py - ANALISIS VAL, BUKAN HASIL UTAMA (KEPUTUSAN_PROYEK.md 2026-10-15).

Pertanyaan: pada foto Blast yang didiagnosis salah (terutama sebagai Brown spot) dan foto Brown spot yang didiagnosis Blast,
seberapa sering kelas benarnya tetap muncul sebagai kotak dengan keyakinan >= t_rendah ("kemungkinan kedua")?

- Hanya VAL prov_v7 (1259 citra, sama dengan eval_val); test, test_new_session, dan ds3 TIDAK dipakai.
- Model final 3 seed YOLO11s dan YOLOv8s (best.pt yang dikunci di runs/<run>/ambang_diagnosis.json, sha dicek).
- Prediksi sama dengan evaluate.py: conf 0,001, iou 0,7, imgsz latihan, NMS per kelas (bawaan Ultralytics).
- Ambang terdeteksi = ambang terkunci tiap run; t_rendah = prosedur tiga tingkat yang sama (scripts/tiga_tingkat_v7.py) per run.
- Uji reproduksi: kotak teratas tiap citra harus sama dengan eval_val/diagnosis_dengan_background_per_citra.csv.
Tidak ada model, ambang, atau angka penelitian yang diubah.

  python scripts/analisis_kemungkinan_kedua_v7.py             # prediksi VAL (GPU) lalu ringkas
  python scripts/analisis_kemungkinan_kedua_v7.py --dari-csv  # ringkas ulang dari per_kasus.csv (tanpa menjalankan model)
Keluaran: reports/final_eval_v7/aplikasi/analisis_val_kemungkinan_kedua/{per_kasus.csv, ringkasan.json, RINGKASAN.md}
"""
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import tiga_tingkat_v7  # noqa: E402

OUT = ROOT / "reports/final_eval_v7/aplikasi/analisis_val_kemungkinan_kedua"
VAL = ROOT / "data/processed/prov_v7/images/val"
RUNS = [f"v7_final_prov_{m}_s{s}" for m in ("yolo11s", "yolov8s") for s in (0, 1, 2)]
TAK = "sehat_atau_tidak_terdeteksi"
KASUS = [("leaf_blast", "brown_spot"), ("leaf_blast", "bacterial_leaf_blight"), ("leaf_blast", TAK), ("brown_spot", "leaf_blast")]
BIN = [0.0, 0.05, 0.10, 0.20, "t_rendah", "ambang", 1.0001]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def prediksi(run):
    from ultralytics import YOLO
    amb = json.loads((ROOT / f"runs/{run}/ambang_diagnosis.json").read_text(encoding="utf-8"))
    w = ROOT / f"runs/{run}/weights/best.pt"
    if sha256(w) != amb["weights_sha256"]:
        sys.exit(f"[GAGAL] sha best.pt {run} tidak sama dengan ambang_diagnosis.json")
    rows = list(csv.DictReader(open(ROOT / f"runs/{run}/eval_val/diagnosis_dengan_background_per_citra.csv", encoding="utf-8")))
    model = YOLO(str(w))
    names = model.names
    imgsz = int(model.ckpt.get("train_args", {}).get("imgsz") or 640) if getattr(model, "ckpt", None) else 640
    hasil = []
    for i in range(0, len(rows), 16):
        part = rows[i:i + 16]
        for r, res in zip(part, model.predict([str(VAL / x["berkas"]) for x in part], imgsz=imgsz, conf=0.001, iou=0.7,
                                              device=0, verbose=False)):
            cf = res.boxes.conf.cpu().numpy().tolist()
            cl = [names[int(c)] for c in res.boxes.cls.cpu().numpy().tolist()]
            maks = {}
            for c, k in zip(cf, cl):
                maks[k] = max(maks.get(k, 0.0), c)
            hasil.append({"berkas": r["berkas"], "label": r["label"], "maks_per_kelas": maks, "acuan": r})
    return float(amb["ambang"]), hasil, list(names.values())


def ringkas(xs, t_rendah, ambang):
    if not xs:
        return {"n": 0}
    s = sorted(xs)
    q = statistics.quantiles(s, n=4, method="inclusive") if len(s) > 1 else [s[0]] * 3
    batas = [t_rendah if b == "t_rendah" else ambang if b == "ambang" else b for b in BIN]
    return {"n": len(s), "persen_ge_t_rendah": round(100 * sum(x >= t_rendah for x in s) / len(s), 1),
            "persen_ge_0.10": round(100 * sum(x >= 0.10 for x in s) / len(s), 1),
            "persen_tanpa_kotak_kelas_benar": round(100 * sum(x == 0.0 for x in s) / len(s), 1),
            "min": round(s[0], 4), "q1": round(q[0], 4), "median": round(q[1], 4), "q3": round(q[2], 4), "maks": round(s[-1], 4),
            "sebaran": {f"[{batas[j]:.2f}, {min(batas[j + 1], 1.0):.2f})": sum(batas[j] <= x < batas[j + 1] for x in s)
                        for j in range(len(batas) - 1) if batas[j] < batas[j + 1]}}


def baca_csv():
    lama = json.loads((OUT / "ringkasan.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader(open(OUT / "per_kasus.csv", encoding="utf-8")))
    for r in rows:
        for k in ("conf_teratas", "lain_conf", "ambang", "t_rendah"):
            r[k] = float(r[k])
    return rows, lama["ambang_per_run"], lama["uji_reproduksi_vs_eval_val"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if "--dari-csv" in sys.argv:
        ringkas_semua(*baca_csv())
        return
    per_kasus, per_run, cek = [], {}, {}
    for run in RUNS:
        ambang, hasil, kelas = prediksi(run)
        t_rendah = tiga_tingkat_v7.hitung(run)["t_rendah"]
        penyakit = [k for k in kelas]
        beda = 0
        for h in hasil:
            m = h["maks_per_kelas"]
            top_k = max(m, key=m.get) if m else None
            top_c = m[top_k] if m else 0.0
            a = h["acuan"]
            if (top_k or "") != (a["kelas_teratas"] or "") or abs(top_c - float(a["conf_teratas"] or 0)) > 0.002:
                beda += 1
            h["diagnosis"] = top_k if (top_k and top_c >= ambang) else TAK
            h["top_c"], h["top_k"] = top_c, top_k
        cek[run] = {"citra": len(hasil), "kotak_teratas_beda_dari_eval_val": beda}
        per_run[run] = {"ambang_terdeteksi": ambang, "t_rendah": t_rendah}
        for h in hasil:
            m = h["maks_per_kelas"]
            # kemungkinan lain versi aplikasi: penyakit LAIN dari diagnosis utama dengan keyakinan >= t_rendah
            lain = max(((k, c) for k, c in m.items() if k != h["top_k"] and k in penyakit), key=lambda x: x[1], default=(None, 0.0))
            per_kasus.append({"run": run, "model": run.split("_")[3], "seed": run[-1], "berkas": h["berkas"], "label": h["label"],
                              "diagnosis": h["diagnosis"], "conf_teratas": round(h["top_c"], 4), "kelas_teratas": h["top_k"] or "",
                              "conf_kelas_benar": round(m.get(h["label"], 0.0), 4) if h["label"] in penyakit else "",
                              "lain_kelas": lain[0] or "", "lain_conf": round(lain[1], 4), "ambang": ambang, "t_rendah": t_rendah})
        print(run, cek[run], per_run[run], flush=True)
    if any(v["kotak_teratas_beda_dari_eval_val"] for v in cek.values()):
        print("[PERINGATAN] kotak teratas tidak sama dengan eval_val untuk sebagian citra; lihat ringkasan.json", flush=True)

    with open(OUT / "per_kasus.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_kasus[0]))
        w.writeheader()
        w.writerows(per_kasus)
    ringkas_semua(per_kasus, per_run, cek)


def ringkas_semua(per_kasus, per_run, cek):
    ring = {"LABEL": "ANALISIS VAL, BUKAN HASIL UTAMA", "data": "VAL prov_v7 (1259 citra); test/test_new_session/ds3 tidak dipakai",
            "ambang_per_run": per_run, "uji_reproduksi_vs_eval_val": cek, "kasus": {}, "kemungkinan_lain_pada_diagnosis_benar": {}}
    for model in ("yolo11s", "yolov8s"):
        rows = [r for r in per_kasus if r["model"] == model]
        for lab, dia in KASUS:
            xs = [r for r in rows if r["label"] == lab and r["diagnosis"] == dia]
            semua = [r for r in rows if r["label"] == lab]
            key = f"{model}|{lab} -> {dia}"
            # gabungan 3 seed: persen dihitung per seed dengan t_rendah seed itu, lalu dijumlah
            ge = sum(float(r["conf_kelas_benar"]) >= r["t_rendah"] for r in xs)
            ring["kasus"][key] = {
                "n_kasus_3_seed": len(xs), "n_foto_kelas_3_seed": len(semua),
                "persen_kasus_dari_foto_kelas": round(100 * len(xs) / len(semua), 1) if semua else None,
                "kelas_benar_ge_t_rendah": ge, "persen_kelas_benar_ge_t_rendah": round(100 * ge / len(xs), 1) if xs else None,
                "per_seed": {s: ringkas([float(r["conf_kelas_benar"]) for r in xs if r["seed"] == s], per_run[f"v7_final_prov_{model}_s{s}"]["t_rendah"],
                                        per_run[f"v7_final_prov_{model}_s{s}"]["ambang_terdeteksi"]) for s in "012"},
                "keyakinan_kelas_benar_gabungan": ringkas([float(r["conf_kelas_benar"]) for r in xs], 0, 0) | {}}
            ring["kasus"][key]["keyakinan_kelas_benar_gabungan"].pop("sebaran", None)
            ring["kasus"][key]["keyakinan_kelas_benar_gabungan"].pop("persen_ge_t_rendah", None)
        # biaya fitur: seberapa sering "kemungkinan lain" muncul pada diagnosis yang BENAR (terdeteksi) dan pada foto sehat
        for lab in ("leaf_blast", "brown_spot", "bacterial_leaf_blight", "background"):
            if lab == "background":
                xs = [r for r in rows if r["label"] not in ("leaf_blast", "brown_spot", "bacterial_leaf_blight") and r["diagnosis"] != TAK]
                nama = f"{model}|sehat salah terdeteksi"
            else:
                xs = [r for r in rows if r["label"] == lab and r["diagnosis"] == lab]
                nama = f"{model}|{lab} benar"
            n = sum(r["lain_conf"] >= r["t_rendah"] for r in xs)
            ring["kemungkinan_lain_pada_diagnosis_benar"][nama] = {"n": len(xs), "muncul_kemungkinan_lain": n,
                                                                   "persen": round(100 * n / len(xs), 1) if xs else None}
        # yang benar-benar akan tampil di aplikasi: kemungkinan lain (satu penyakit lain tertinggi >= t_rendah) = kelas benar?
        for lab, dia in KASUS:
            if dia == TAK:
                continue
            xs = [r for r in rows if r["label"] == lab and r["diagnosis"] == dia]
            n = sum(r["lain_kelas"] == lab and r["lain_conf"] >= r["t_rendah"] for r in xs)
            ring.setdefault("kemungkinan_lain_tampil_kelas_benar", {})[f"{model}|{lab} -> {dia}"] = {
                "n_kasus": len(xs), "kemungkinan_lain_tampil_kelas_benar": n, "persen": round(100 * n / len(xs), 1) if xs else None}
        # bercak berkeyakinan rendah: tidak terdeteksi tetapi kotak teratas >= 0,10
        sakit = ("leaf_blast", "brown_spot", "bacterial_leaf_blight")
        for kel, cocok in (("daun sehat", lambda r: r["label"] not in sakit), ("daun sakit", lambda r: r["label"] in sakit)):
            xs = [r for r in rows if cocok(r) and r["diagnosis"] == TAK]
            br = [r for r in xs if r["conf_teratas"] >= 0.10]
            semua = [r for r in rows if cocok(r)]
            d = {"n_foto_3_seed": len(semua), "n_tidak_terdeteksi": len(xs), "bercak_rendah_tampil": len(br),
                 "persen_dari_tidak_terdeteksi": round(100 * len(br) / len(xs), 1) if xs else None,
                 "persen_dari_semua_foto_kelompok": round(100 * len(br) / len(semua), 1) if semua else None}
            if kel == "daun sakit":
                d["kelas_bercak_rendah_benar"] = sum(r["kelas_teratas"] == r["label"] for r in br)
            ring.setdefault("bercak_rendah", {})[f"{model}|{kel}"] = d
    (OUT / "ringkasan.json").write_text(json.dumps(ring, indent=2, ensure_ascii=False), encoding="utf-8")

    nm = {"leaf_blast": "Blast", "brown_spot": "Brown spot", "bacterial_leaf_blight": "BLB", TAK: "tidak terdeteksi"}
    md = ["# Analisis val: kelas benar sebagai kemungkinan kedua", "", "**ANALISIS VAL, BUKAN HASIL UTAMA.** VAL prov_v7 (1259 citra), model final 3 seed,",
          "ambang terkunci tiap run; t_rendah dengan prosedur tiga tingkat yang sama per run. Test, test sesi baru, dan ds3 tidak dipakai.", "",
          "| Run | Ambang terdeteksi | t_rendah | Kotak teratas beda dari eval_val |", "|---|---|---|---|"]
    md += [f"| {r} | {per_run[r]['ambang_terdeteksi']} | {per_run[r]['t_rendah']} | {cek[r]['kotak_teratas_beda_dari_eval_val']}/{cek[r]['citra']} |" for r in RUNS]
    md += ["", "## Kelas benar pada kasus salah diagnosis (gabungan 3 seed)", "",
           "| Model | Kasus | Kasus / foto kelas | Kelas benar >= t_rendah | Keyakinan kelas benar: median [Q1; Q3] (min-maks) | Tanpa kotak kelas benar |",
           "|---|---|---|---|---|---|"]
    for k, v in ring["kasus"].items():
        model, kas = k.split("|")
        lab, dia = kas.split(" -> ")
        g = v["keyakinan_kelas_benar_gabungan"]
        if not v["n_kasus_3_seed"]:
            md.append(f"| {model} | {nm[lab]} -> {nm[dia]} | 0 / {v['n_foto_kelas_3_seed']} | - | - | - |")
            continue
        md.append(f"| {model} | {nm[lab]} -> {nm[dia]} | {v['n_kasus_3_seed']} / {v['n_foto_kelas_3_seed']} | {v['kelas_benar_ge_t_rendah']}/{v['n_kasus_3_seed']} "
                  f"({v['persen_kelas_benar_ge_t_rendah']}%) | {g['median']} [{g['q1']}; {g['q3']}] ({g['min']}-{g['maks']}) | {g['persen_tanpa_kotak_kelas_benar']}% |")
    md += ["", "Catatan: pada kasus Blast -> tidak terdeteksi, kotak teratas < ambang; kelas benar >= t_rendah berarti foto itu sudah berada di tingkat",
           "\"kemungkinan\" (bukan kemungkinan kedua).", "", "## Sebaran keyakinan kelas benar per seed", ""]
    for k, v in ring["kasus"].items():
        for s, d in v["per_seed"].items():
            if d.get("n"):
                md.append(f"- {k}, seed {s} (n={d['n']}): " + "; ".join(f"{b}: {c}" for b, c in d["sebaran"].items()))
    md += ["", "## Seberapa sering \"kemungkinan lain\" muncul pada diagnosis yang benar (biaya tampilan)", "", "| Model | Kelompok | n (3 seed) | Muncul kemungkinan lain |", "|---|---|---|---|"]
    for k, v in ring["kemungkinan_lain_pada_diagnosis_benar"].items():
        model, kel = k.split("|")
        md.append(f"| {model} | {kel} | {v['n']} | {v['muncul_kemungkinan_lain']} ({v['persen']}%) |")
    md += ["", "## Yang akan tampil di aplikasi pada kasus salah diagnosis", "",
           "\"Kemungkinan lain\" menampilkan SATU penyakit lain dengan keyakinan tertinggi (>= t_rendah); bisa saja bukan kelas benar.", "",
           "| Model | Kasus | Kemungkinan lain = kelas benar |", "|---|---|---|"]
    for k, v in ring["kemungkinan_lain_tampil_kelas_benar"].items():
        model, kas = k.split("|")
        lab, dia = kas.split(" -> ")
        md.append(f"| {model} | {nm[lab]} -> {nm[dia]} | {v['kemungkinan_lain_tampil_kelas_benar']}/{v['n_kasus']}"
                  + (f" ({v['persen']}%)" if v["n_kasus"] else "") + " |")
    md += ["", "## Bercak berkeyakinan rendah (tidak terdeteksi, kotak teratas >= 0,10)", "",
           "| Model | Kelompok | Foto (3 seed) | Tidak terdeteksi | Bercak rendah tampil | % dari tidak terdeteksi | % dari semua foto kelompok |",
           "|---|---|---|---|---|---|---|"]
    for k, v in ring["bercak_rendah"].items():
        model, kel = k.split("|")
        ext = f" (kelas bercak = kelas benar: {v['kelas_bercak_rendah_benar']})" if "kelas_bercak_rendah_benar" in v else ""
        md.append(f"| {model} | {kel} | {v['n_foto_3_seed']} | {v['n_tidak_terdeteksi']} | {v['bercak_rendah_tampil']}{ext} | "
                  f"{v['persen_dari_tidak_terdeteksi']}% | {v['persen_dari_semua_foto_kelompok']}% |")
    md += ["", "Sumber: per_kasus.csv (satu baris per citra per run); ringkasan.json. Skrip: scripts/analisis_kemungkinan_kedua_v7.py."]
    (OUT / "RINGKASAN.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("selesai:", OUT)


if __name__ == "__main__":
    main()
