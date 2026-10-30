#!/usr/bin/env python3
"""
fase5_alur_aplikasi.py - Fase 4 (sisa) dan 5 rencana aplikasi v9 (KEPUTUSAN_PROYEK.md 2026-10-20). Label: "evaluasi alur aplikasi".

  masker   : masker YOLOv8n-seg (tau 0,35) untuk foto test dan test sesi baru prov_v9 (foto asli prov_v7) dan ds3 (utama, Tungro)
  bangun   : dataset turunan data/derived/v9_alur_aplikasi (images/labels test, test_new_session; foto cadangan = foto asli, ditandai di manifest)
             + data/derived/v9_latar/ds3_alur_aplikasi (gambar + daftar)
  kunci    : ambang aplikasi (fase 4) -> runs/v9_deploy_yolov8s_s2/ambang_diagnosis.json
  onnx     : ekspor ONNX model aplikasi (metadata dibersihkan) + kesetaraan ONNX vs .pt pada 20 foto val
  evaluasi : evaluate.py, setiap set SEKALI:
             model final YOLOv8s v9 s0-2 pada alur aplikasi test/test sesi baru (ambang terkunci masing-masing seed) -> eval_alurapp_<split>
             model aplikasi v9_deploy pada alur aplikasi test sesi baru, ds3 utama, ds3 Tungro                        -> eval_alurapp_*
  Foto cadangan = diminta foto ulang: TIDAK didiagnosis dan TANPA prediksi kotak (diterapkan di analisis_alur_aplikasi_v9.py).
"""
import csv
import hashlib
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ["YOLO_AUTOINSTALL"] = "false"
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import pemisah_daun_seg as pd  # noqa: E402
import umum  # noqa: E402

ROOT = umum.ROOT
PY = sys.executable
V7, V9 = ROOT / "data/processed/prov_v7", ROOT / "data/processed/prov_v9"
SEGW = ROOT / "runs/v9_pemisah_daun_yolov8n_seg/weights/best.pt"
TAU = 0.35
KER = umum.KERJA
ALUR = ROOT / "data/derived/v9_alur_aplikasi"
DS3A = KER / "ds3_alur_aplikasi"
UD = ROOT / "data/annotation/verifikasi_ds3/uji_utama_final"
DS3H = {"utama": "ds3_uji_utama.txt", "tungro": "ds3_tungro_luar_cakupan.txt"}
DEP = "v9_deploy_yolov8s_s2"
W_DEP = ROOT / "runs" / DEP / "weights/last.pt"
FINAL = [f"v9_final_prov_yolov8s_s{s}" for s in (0, 1, 2)]
REP = ROOT / "reports/v9_latar/aplikasi_v9"
SPL = ("test", "test_new_session")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def masker():
    from ultralytics import YOLO
    model = YOLO(str(SEGW))
    man = [r for r in csv.DictReader(open(V9 / "manifest.csv", encoding="utf-8")) if r["split"] in SPL]
    tugas = [(r["file"], V7 / "images" / r["split"] / r["file"], KER / f"seg_n_{r['split']}") for r in man]
    ds3 = sorted({b.split("\t")[0] for n in DS3H.values() for b in (UD / n).read_text(encoding="utf-8").splitlines() if b.strip()})
    tugas += [(d.replace("data/raw/ds3/", ""), ROOT / d, KER / "seg_n_ds3") for d in ds3]
    info = {}
    for i, (kunci, src, d) in enumerate(tugas):
        (d / "masker").mkdir(parents=True, exist_ok=True)
        nama = kunci.replace("/", "__").rsplit(".", 1)[0] + ".png"
        bgr = cv2.imread(str(src))
        M, c = pd.pilih_daun(bgr, *pd.prediksi_pt(model, bgr), TAU)
        cv2.imwrite(str(d / "masker" / nama), M.astype(np.uint8) * 255)
        info.setdefault(d.name, {})[kunci] = c
        if i % 200 == 0:
            print(i, len(tugas), flush=True)
    for dn, isi in info.items():
        (KER / dn / "info.json").write_text(json.dumps({"tau": TAU, "cadangan": isi}, indent=0), encoding="utf-8")
        print(dn, len(isi), "cadangan:", sum(isi.values()))


def terapkan(src, mpath, dst, cad):
    if cad:
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        return
    bgr = cv2.imread(str(src))
    M = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE) > 0
    bgr[~M] = umum.ABU[::-1]
    cv2.imwrite(str(dst), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])


def bangun():
    if ALUR.exists() or DS3A.exists():
        sys.exit("[DITOLAK] dataset alur aplikasi sudah ada")
    man = [r for r in csv.DictReader(open(V9 / "manifest.csv", encoding="utf-8")) if r["split"] in SPL]
    cadinfo = {sp: json.loads((KER / f"seg_n_{sp}/info.json").read_text(encoding="utf-8"))["cadangan"] for sp in SPL}
    for sp in SPL:
        (ALUR / "images" / sp).mkdir(parents=True)
        (ALUR / "labels" / sp).mkdir(parents=True)
    rows = []
    for r in man:
        sp, f = r["split"], r["file"]
        c = cadinfo[sp][f]
        terapkan(V7 / "images" / sp / f, KER / f"seg_n_{sp}/masker" / (Path(f).stem + ".png"), ALUR / "images" / sp / f, c)
        lab = V9 / "labels" / sp / (Path(f).stem + ".txt")
        if lab.exists():
            shutil.copyfile(lab, ALUR / "labels" / sp / lab.name)
        rows.append(dict(r, seg_cadangan="ya" if c else "tidak"))
    with open(ALUR / "manifest.csv", "w", newline="", encoding="utf-8") as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    dy9 = yaml.safe_load((V9 / "data.yaml").read_text(encoding="utf-8"))
    yml = {"path": str(ALUR).replace("\\", "/"), "test": "images/test", "test_new_session": "images/test_new_session", "val": "images/test",
           "names": dy9["names"], "dataset_name": "v9_alur_aplikasi", "provisional": True, "provisional_sources": dy9["provisional_sources"],
           "manifest_sha256": sha(ALUR / "manifest.csv"), "induk_dataset": "prov_v9", "induk_manifest_sha256": dy9["manifest_sha256"],
           "catatan": "evaluasi alur aplikasi v9: foto test/test sesi baru prov_v9, latar dihapus dengan YOLOv8n-seg tau 0,35 (cadangan = foto asli, "
                      "ditandai seg_cadangan; diperlakukan sebagai foto ulang di analisis); label prov_v9"}
    (ALUR / "data.yaml").write_text(yaml.safe_dump(yml, allow_unicode=True, sort_keys=False), encoding="utf-8")
    # ds3
    cad3 = json.loads((KER / "seg_n_ds3/info.json").read_text(encoding="utf-8"))["cadangan"]
    (DS3A / "daftar").mkdir(parents=True)
    for n, fn in DS3H.items():
        baris = []
        for b in (UD / fn).read_text(encoding="utf-8").splitlines():
            if not b.strip():
                continue
            d, g = b.split("\t")
            rel = d.replace("data/raw/ds3/", "")
            dst = DS3A / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                terapkan(ROOT / d, KER / "seg_n_ds3/masker" / (rel.replace("/", "__").rsplit(".", 1)[0] + ".png"), dst, cad3[rel])
            baris.append(f"{dst.relative_to(ROOT).as_posix()}\t{g}")
        (DS3A / "daftar" / fn).write_text("\n".join(baris) + "\n", encoding="utf-8")
    ring = {sp: {"foto": sum(r["split"] == sp for r in rows), "cadangan": sum(r["split"] == sp and r["seg_cadangan"] == "ya" for r in rows)} for sp in SPL}
    for n, fn in DS3H.items():
        fs = [b.split("\t")[0].replace("data/raw/ds3/", "") for b in (UD / fn).read_text(encoding="utf-8").splitlines() if b.strip()]
        ring[f"ds3_{n}"] = {"foto": len(fs), "cadangan": sum(cad3[f] for f in fs)}
    (REP / "fase5_cadangan.json").write_text(json.dumps(ring, indent=1), encoding="utf-8")
    print(ring)


def kunci():
    a = json.loads((REP / "fase4_ambang_aplikasi.json").read_text(encoding="utf-8"))
    rs = json.loads((ROOT / "runs" / DEP / "run_stats.json").read_text(encoding="utf-8"))
    if not W_DEP.exists() or len(rs.get("epoch", [])) < 50:
        sys.exit("[GAGAL] model aplikasi belum 50 epoch")
    p = ROOT / "runs" / DEP / "ambang_diagnosis.json"
    if p.exists():
        sys.exit(f"[DITOLAK] {p} sudah ada")
    k = {"ambang": a["ambang_terdeteksi"], "akurasi_val_saat_dipilih": None, "t_rendah_aplikasi": a["t_rendah"],
         "batas_tampil_bercak_rendah": a["batas_tampil_bercak_rendah"],
         "dipilih_pada": "DITETAPKAN: ambang aplikasi v9 dikunci di VAL dengan alur aplikasi (YOLOv8n-seg + v9_final_prov_yolov8s_s2), 2026-10-20",
         "kriteria": "reports/v9_latar/aplikasi_v9/fase4_ambang_aplikasi.json", "weights": str(W_DEP), "weights_sha256": sha(W_DEP),
         "dikunci": datetime.now().isoformat(timespec="seconds"), "catatan": "checkpoint = last.pt (50 epoch); hasil = pembanding, bukan hasil utama"}
    p.write_text(json.dumps(k, indent=2, ensure_ascii=False), encoding="utf-8")
    print(k)


def onnx():
    import onnx as onnxlib
    from ultralytics import YOLO
    m = YOLO(str(W_DEP))
    f_onnx = Path(m.export(format="onnx", imgsz=640, simplify=True, dynamic=False, device="cpu"))
    mo = onnxlib.load(str(f_onnx))
    g0 = mo.graph.SerializeToString()
    for p in mo.metadata_props:
        if p.key == "description":
            p.value = "Ultralytics YOLOv8s model trained on prov_v9_deploy"
    onnxlib.save(mo, str(f_onnx))
    assert onnxlib.load(str(f_onnx)).graph.SerializeToString() == g0
    sisa = [p.key for p in onnxlib.load(str(f_onnx)).metadata_props if ":\\" in p.value or "TELKOM" in p.value or "Tilik" in p.value]
    val = sorted((V9 / "images/val").iterdir())
    pilih = random.Random(0).sample(val, 20)
    mpt, mox = YOLO(str(W_DEP)), YOLO(str(f_onnx), task="detect")
    hasil, waktu = {}, {"pt": [], "onnx": []}
    for nama, mod in (("pt", mpt), ("onnx", mox)):
        for p in pilih[:3]:
            mod.predict(str(p), imgsz=640, conf=0.001, iou=0.7, device="cpu", verbose=False)
        hasil[nama] = []
        for p in pilih:
            t0 = time.perf_counter()
            r = mod.predict(str(p), imgsz=640, conf=0.001, iou=0.7, device="cpu", verbose=False)[0]
            waktu[nama].append(1000 * (time.perf_counter() - t0))
            c = r.boxes.conf.cpu().numpy()
            hasil[nama].append((int(r.boxes.cls[int(c.argmax())].item()), float(c.max())) if len(c) else (-1, 0.0))
    rows = [{"gambar": p.name, "kelas_pt": a[0], "conf_pt": round(a[1], 5), "kelas_onnx": b[0], "conf_onnx": round(b[1], 5),
             "kelas_identik": a[0] == b[0], "selisih_conf": round(abs(a[1] - b[1]), 5)} for p, a, b in zip(pilih, hasil["pt"], hasil["onnx"])]
    with open(REP / "fase4_onnx_vs_pt_20_val.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    res = {"model": DEP, "checkpoint": "last.pt", "sha256_pt": sha(W_DEP), "onnx": str(f_onnx.relative_to(ROOT)), "sha256_onnx": sha(f_onnx),
           "metadata_description": "Ultralytics YOLOv8s model trained on prov_v9_deploy", "metadata_berisi_jalur_atau_nama_aplikasi": sisa,
           "ukuran_MB": {"pt": round(W_DEP.stat().st_size / 2 ** 20, 2), "onnx": round(f_onnx.stat().st_size / 2 ** 20, 2)},
           "n_gambar": len(rows), "kelas_teratas_identik": sum(r["kelas_identik"] for r in rows), "selisih_conf_maks": max(r["selisih_conf"] for r in rows),
           "LOLOS": all(r["kelas_identik"] for r in rows) and max(r["selisih_conf"] for r in rows) < 0.01,
           "cpu_ms_median": {k: round(statistics.median(v), 1) for k, v in waktu.items()},
           "catatan": "20 foto val prov_v9 (tanpa latar SAM2) acak seed 0; CPU batch 1 imgsz 640 conf 0,001 iou 0,7; graf ONNX identik sebelum/sesudah ganti metadata"}
    (REP / "fase4_onnx_model_aplikasi.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False))


def jalan(args, logf):
    with open(logf, "w", encoding="utf-8") as f:
        f.write("# " + " ".join(args) + "\n")
        f.flush()
        return subprocess.run([PY, "scripts/evaluate.py"] + args, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode


def evaluasi():
    data = str((ALUR / "data.yaml").relative_to(ROOT))
    tugas = []
    for run in FINAL:
        for sp in SPL:
            tugas.append((run, "best.pt", f"eval_alurapp_{sp}", ["--data", data, "--split", sp, "--izinkan-test"]))
    tugas.append((DEP, "last.pt", "eval_alurapp_test_new_session", ["--data", data, "--split", "test_new_session", "--izinkan-test"]))
    for n, fn in DS3H.items():
        peta = ["blight=bacterial_leaf_blight", "blast=leaf_blast"] + (["tungro=luar_cakupan"] if n == "tungro" else [])
        tugas.append((DEP, "last.pt", f"eval_alurapp_ds3_{n}", ["--data", data, "--set-folder", str(DS3A.relative_to(ROOT)), "--izinkan-uji-eksternal",
                                                                "--peta-folder"] + peta + ["--daftar", str((DS3A / "daftar" / fn).relative_to(ROOT))]))
    for run, w, nama, extra in tugas:
        out = ROOT / "runs" / run / nama
        if (out / "ringkasan.json").exists() or (out / "ringkasan_diagnosis.json").exists():
            print(f"{run} {nama}: sudah ada, dilewati")
            continue
        rc = jalan(["--weights", f"runs/{run}/weights/{w}", "--tanpa-benchmark", "--out", f"runs/{run}/{nama}"] + extra, ROOT / "runs" / run / f"{nama}.log")
        print(f"{run} {nama}: kode {rc}", flush=True)
        if rc:
            sys.exit(f"[GAGAL] {run} {nama}")


if __name__ == "__main__":
    REP.mkdir(parents=True, exist_ok=True)
    {"masker": masker, "bangun": bangun, "kunci": kunci, "onnx": onnx, "evaluasi": evaluasi}[sys.argv[1]]()
