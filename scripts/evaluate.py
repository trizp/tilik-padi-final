#!/usr/bin/env python3
"""
evaluate.py - Evaluasi satu checkpoint (baku: weights/last.pt) pada SATU split.

Keluaran per checkpoint:
  * P, R, F1, mAP50, mAP50-95 per kelas dan keseluruhan, untuk dua himpunan:
      - dengan_background : seluruh gambar split (gambar background hanya bisa menambah false positive)
      - tanpa_background  : hanya gambar yang punya minimal satu box
      - bila split memuat gambar sumber provisional (data.yaml: provisional_sources), DUA subset tambahan:
        bersih          : tanpa gambar provisional (= val pemilihan checkpoint pada data_valsel.yaml)
        blb_provisional : hanya gambar provisional (BLB ds1_provisional), dilaporkan TERPISAH
  * Laju false positive pada gambar background: % gambar background dengan >= 1 deteksi dan rata-rata box per gambar,
    pada beberapa ambang confidence, beserta rincian per kelas.
  * Jumlah parameter dan GFLOPs (imgsz 640), sebelum dan sesudah fuse Conv+BN.
  * Benchmark kecepatan inferensi di mesin yang sama: batch 1, warmup, GPU dan CPU, waktu ujung ke ujung
    (praproses + inferensi + pascaproses) dari gambar yang sudah dimuat ke memori (baca disk tidak dihitung).
    Benchmark SELALU memakai gambar VAL (bukan split yang dievaluasi) agar test tidak tersentuh untuk keperluan ini.

  * DIAGNOSIS PER CITRA (ruang lingkup: satu penyakit per citra): diagnosis = kelas kotak dengan confidence tertinggi;
    tanpa kotak >= ambang -> "sehat_atau_tidak_terdeteksi". Akurasi per citra, P/R/F1 per kelas level citra, confusion
    matrix 4x4. Ambang dipilih SEKALI pada val bersih (akurasi maksimum) dan dikunci di <run>/ambang_diagnosis.json;
    split lain dan set folder memakai ambang terkunci itu. Citra berlabel >1 kelas dikeluarkan (dihitung jumlahnya).
  * --set-folder: set berlabel per folder tanpa box (ds3): hanya diagnosis per citra; kelas luar_cakupan (tungro)
    dilaporkan sebagai sebaran diagnosis. Wajib --izinkan-uji-eksternal dan ambang terkunci.

Pengaman:
  * split test dan test_new_session DITOLAK kecuali diberi --izinkan-test (keputusan pemilik skripsi).
  * hash manifest dataset dicek terhadap data.yaml (dataset beku).
  * subset gambar dibuat sebagai hardlink di folder keluaran, sehingga folder dataset tidak ditulisi (cache Ultralytics
    ikut tersimpan di folder keluaran).

Pemakaian (uji coba hanya VAL):
  python scripts/evaluate.py --weights runs/prov_v6_yolov8n_s0_smoke/weights/last.pt --split val
Keluaran: <run>/eval_<split>/ : metrik_per_kelas.csv, metrik_per_kelas.tex, fp_background.csv, kecepatan.csv,
          ringkasan.json, plot Ultralytics (confusion matrix, kurva PR) per subset.
"""
import argparse
import copy
import csv
import hashlib
import json
import os
import platform
import shutil
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import yaml
from jalur import jalur_keluaran  # noqa: E402  (pengaman jalur keluaran)

ROOT = Path(__file__).resolve().parent.parent
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLIT_TERLINDUNG = {"test", "test_new_session"}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def daftar_split(root, split):
    """Kembalikan [(gambar, label_atau_None, kelas_box[])] untuk split."""
    out = []
    for p in sorted((root / "images" / split).iterdir()):
        if p.suffix.lower() not in IMG_EXT:
            continue
        lab = root / "labels" / split / (p.stem + ".txt")
        cls = [int(float(ln.split()[0])) for ln in lab.read_text().splitlines() if ln.split()] if lab.exists() else []
        out.append((p, lab if lab.exists() else None, cls))
    return out


def tautkan(items, dst):
    """Buat salinan ringan (hardlink; salin bila gagal) gambar+label ke dst/images dan dst/labels."""
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "images").mkdir(parents=True)
    (dst / "labels").mkdir(parents=True)
    for img, lab, _ in items:
        for s, d in ((img, dst / "images" / img.name), (lab, dst / "labels" / (img.stem + ".txt")) if lab else (None, None)):
            if s is None:
                continue
            try:
                os.link(s, d)
            except OSError:
                shutil.copy2(s, d)


def f1(p, r):
    return float(2 * p * r / (p + r)) if (p + r) else 0.0


_REKAM = {"aktif": None}


def pasang_perekam():
    """Bungkus DetMetrics.update_stats agar statistik PER GAMBAR (tp per IoU, conf, kelas prediksi, kelas target) ikut
    disimpan; perhitungan Ultralytics tidak berubah. Dipakai untuk bootstrap per kelompok (analisis_akhir.py)."""
    from ultralytics.utils import metrics as um
    if getattr(um.DetMetrics.update_stats, "_dibungkus", False):
        return
    asli = um.DetMetrics.update_stats

    def rekam(self, stat):
        if _REKAM["aktif"] is not None:
            _REKAM["aktif"].append({k: stat[k] for k in ("tp", "conf", "pred_cls", "target_cls", "im_name")})
        return asli(self, stat)
    rekam._dibungkus = True
    um.DetMetrics.update_stats = rekam


def evaluasi_subset(model, items, names, out, nama, batch, imgsz):
    import pickle
    import numpy as np
    from ultralytics.utils.metrics import smooth
    pasang_perekam()
    _REKAM["aktif"] = []
    sub = out / "_subset" / nama
    tautkan(items, sub)
    yml = out / f"_data_{nama}.yaml"
    yml.write_text(yaml.safe_dump({"path": str(sub), "train": "images", "val": "images", "names": names}, allow_unicode=True), encoding="utf-8")
    r = model.val(data=str(yml), split="val", imgsz=imgsz, batch=batch, device=0, conf=0.001, iou=0.7, plots=True,
                  project=str(out), name=f"val_{nama}", exist_ok=True, verbose=False)
    with open(out / f"stats_per_citra_{nama}.pkl", "wb") as f:
        pickle.dump({"names": names, "per_citra": _REKAM["aktif"]}, f)
    _REKAM["aktif"] = None
    b = r.box
    inst = Counter(c for _, _, cl in items for c in cl)
    nimg = Counter(c for _, _, cl in items for c in set(cl))
    rows = []
    for k, ci in enumerate(b.ap_class_index):
        p_, r_, ap50, ap = b.class_result(k)
        rows.append({"subset": nama, "kelas": names[int(ci)], "n_gambar": nimg[int(ci)], "n_instance": inst[int(ci)],
                     "precision": round(float(p_), 4), "recall": round(float(r_), 4), "f1": round(f1(p_, r_), 4),
                     "mAP50": round(float(ap50), 4), "mAP50_95": round(float(ap), 4)})
    # ambang confidence tempat P/R dilaporkan (Ultralytics: argmax F1 rata-rata kelas)
    conf_f1 = float(b.px[int(smooth(np.asarray(b.f1_curve).mean(0), 0.1).argmax())]) if len(b.f1_curve) else None
    rows.append({"subset": nama, "kelas": "SEMUA", "n_gambar": len(items), "n_instance": sum(inst.values()),
                 "precision": round(float(b.mp), 4), "recall": round(float(b.mr), 4), "f1": round(f1(b.mp, b.mr), 4),
                 "mAP50": round(float(b.map50), 4), "mAP50_95": round(float(b.map), 4)})
    return rows, {"conf_titik_P_R": round(conf_f1, 4) if conf_f1 is not None else None,
                  "kecepatan_val_ms": {k: round(float(v), 3) for k, v in r.speed.items()}}


def fp_background(model, items, names, confs, imgsz, batch):
    bg = [img for img, _, cl in items if not cl]
    per_conf = {c: {"gambar_dengan_deteksi": 0, "box": 0, "per_kelas": Counter()} for c in confs}
    for i in range(0, len(bg), batch):
        for res in model.predict([str(p) for p in bg[i:i + batch]], imgsz=imgsz, conf=min(confs), iou=0.7, device=0, verbose=False):
            cf = res.boxes.conf.cpu().numpy()
            cl = res.boxes.cls.cpu().numpy().astype(int)
            for c in confs:
                m = cf >= c
                per_conf[c]["box"] += int(m.sum())
                per_conf[c]["gambar_dengan_deteksi"] += int(m.any())
                per_conf[c]["per_kelas"].update(names[int(k)] for k in cl[m])
    rows = []
    for c in confs:
        d = per_conf[c]
        rows.append({"conf": c, "n_gambar_background": len(bg), "gambar_dengan_deteksi": d["gambar_dengan_deteksi"],
                     "persen_gambar_fp": round(100 * d["gambar_dengan_deteksi"] / len(bg), 2) if bg else None,
                     "box_fp_per_gambar": round(d["box"] / len(bg), 4) if bg else None,
                     **{f"box_fp_{n}": d["per_kelas"][n] for n in names.values()}})
    return rows


def info_model(weights, imgsz):
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops
    m = YOLO(weights).model
    res = {"parameter": sum(p.numel() for p in m.parameters()), "GFLOPs": round(get_flops(m, imgsz), 3)}
    mf = copy.deepcopy(m).fuse(verbose=False)
    res.update({"parameter_fused": sum(p.numel() for p in mf.parameters()), "GFLOPs_fused": round(get_flops(mf, imgsz), 3)})
    return res


def benchmark(weights, imgs, devices, warmup, imgsz):
    import cv2
    import torch
    from ultralytics import YOLO
    arr = [cv2.imread(str(p)) for p in imgs]
    rows = []
    for dev in devices:
        if dev != "cpu" and not torch.cuda.is_available():
            continue
        m = YOLO(weights)
        sync = (lambda: torch.cuda.synchronize()) if dev != "cpu" else (lambda: None)
        for k in range(warmup):
            m.predict(arr[k % len(arr)], imgsz=imgsz, device=dev, conf=0.25, iou=0.7, half=False, verbose=False)
        tot, pre, inf, post = [], [], [], []
        for a in arr:
            sync()
            t0 = time.perf_counter()
            res = m.predict(a, imgsz=imgsz, device=dev, conf=0.25, iou=0.7, half=False, verbose=False)
            sync()
            tot.append(1000 * (time.perf_counter() - t0))
            s = res[0].speed
            pre.append(s["preprocess"]); inf.append(s["inference"]); post.append(s["postprocess"])
        q = statistics.quantiles(tot, n=10)
        rows.append({"perangkat": "GPU " + torch.cuda.get_device_name(0) if dev != "cpu" else f"CPU {platform.processor()} ({torch.get_num_threads()} thread)",
                     "batch": 1, "presisi": "FP32", "warmup": warmup, "n_gambar": len(arr),
                     "ms_total_median": round(statistics.median(tot), 2), "ms_total_rata2": round(statistics.mean(tot), 2), "ms_total_p90": round(q[-1], 2),
                     "ms_praproses_rata2": round(statistics.mean(pre), 2), "ms_inferensi_rata2": round(statistics.mean(inf), 2),
                     "ms_pascaproses_rata2": round(statistics.mean(post), 2), "fps_dari_median": round(1000 / statistics.median(tot), 1)})
        del m
        if dev != "cpu":
            torch.cuda.empty_cache()
    return rows


def tabel_latex(rows, judul, label):
    hdr = ("\\begin{table}[ht]\n\\centering\n\\caption{" + judul + "}\n\\label{" + label + "}\n"
           "\\begin{tabular}{llrrrrr}\n\\hline\nSubset & Kelas & P & R & F1 & mAP50 & mAP50-95 \\\\\n\\hline\n")
    body = "".join(f"{r['subset'].replace('_', ' ')} & {r['kelas'].replace('_', ' ')} & {r['precision']:.3f} & {r['recall']:.3f} & "
                   f"{r['f1']:.3f} & {r['mAP50']:.3f} & {r['mAP50_95']:.3f} \\\\\n" for r in rows)
    return hdr + body + "\\hline\n\\end{tabular}\n\\end{table}\n"


# ============================================================ DIAGNOSIS PER CITRA
# Ruang lingkup: satu penyakit per citra. Diagnosis = kelas kotak dengan confidence tertinggi; bila tidak ada kotak
# dengan confidence >= ambang -> "sehat/tidak terdeteksi". Ambang dipilih SEKALI pada VAL bersih (akurasi per citra
# maksimum), dikunci di <run>/ambang_diagnosis.json, lalu dipakai apa adanya untuk semua set uji.
TAK_TERDETEKSI = "sehat_atau_tidak_terdeteksi"
LUAR_CAKUPAN = "luar_cakupan"
GRID_AMBANG = [round(0.01 * i, 2) for i in range(1, 96)]


def prediksi_teratas(model, paths, imgsz, batch):
    """Per gambar: (confidence tertinggi, kelas) dari semua kotak conf >= 0.001; (0.0, -1) bila tanpa kotak."""
    out = []
    for i in range(0, len(paths), batch):
        for res in model.predict([str(x) for x in paths[i:i + batch]], imgsz=imgsz, conf=0.001, iou=0.7, device=0, verbose=False):
            cf = res.boxes.conf.cpu().numpy()
            if len(cf):
                k = int(cf.argmax())
                out.append((float(cf[k]), int(res.boxes.cls[k].item())))
            else:
                out.append((0.0, -1))
    return out


def diagnosa(top, ambang, names):
    return [names[c] if c >= 0 and cf >= ambang else TAK_TERDETEKSI for cf, c in top]


def label_citra(cls_list, names):
    """Label per citra dari kotak: tanpa kotak = sehat; satu kelas = kelas itu; >1 kelas = None (di luar ruang lingkup)."""
    u = set(cls_list)
    if not u:
        return TAK_TERDETEKSI
    return names[u.pop()] if len(u) == 1 else None


def metrik_citra(y_true, y_pred, kelas, grup=None):
    """Akurasi, P/R/F1 per kelas di level citra, confusion matrix (baris = benar, kolom = diagnosis)."""
    n = len(y_true)
    cm = {t: {p_: 0 for p_ in kelas} for t in kelas}
    for t, p_ in zip(y_true, y_pred):
        cm[t][p_] += 1
    benar = sum(t == p_ for t, p_ in zip(y_true, y_pred))
    per = []
    for k in kelas:
        tp = cm[k][k]
        fp = sum(cm[t][k] for t in kelas if t != k)
        fn = sum(cm[k][p_] for p_ in kelas if p_ != k)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        per.append({"kelas": k, "n_benar_kelas": tp + fn, "n_didiagnosis": tp + fp, "precision": round(pr, 4), "recall": round(rc, 4),
                    "f1": round(2 * pr * rc / (pr + rc), 4) if pr + rc else 0.0})
    ada = [r for r in per if r["n_benar_kelas"]]
    res = {"n_citra": n, "akurasi": round(benar / n, 4) if n else None,
           "f1_makro_kelas_yang_ada": round(sum(r["f1"] for r in ada) / len(ada), 4) if ada else None,
           "per_kelas": per, "confusion_matrix": cm}
    if grup:                                  # akurasi dengan bobot sama per kelompok foto kembar (group_id)
        g = defaultdict(list)
        for t, p_, gi in zip(y_true, y_pred, grup):
            g[gi].append(t == p_)
        res["n_kelompok"] = len(g)
        res["akurasi_rata2_per_kelompok"] = round(sum(sum(v) / len(v) for v in g.values()) / len(g), 4)
    return res


def pilih_ambang(y_true, top, names, kelas):
    """Ambang dengan akurasi per citra maksimum; seri -> F1 makro lebih tinggi, lalu ambang terkecil."""
    kurva = []
    for t in GRID_AMBANG:
        m = metrik_citra(y_true, diagnosa(top, t, names), kelas)
        kurva.append({"ambang": t, "akurasi": m["akurasi"], "f1_makro": m["f1_makro_kelas_yang_ada"]})
    best = max(kurva, key=lambda r: (r["akurasi"], r["f1_makro"], -r["ambang"]))
    return best, kurva


def tulis_diagnosis(out, nama, m, rinci):
    """Simpan per kelas, confusion matrix, dan rincian per citra untuk satu himpunan."""
    with open(out / f"diagnosis_{nama}_per_kelas.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(m["per_kelas"][0]))
        w.writeheader()
        w.writerows(m["per_kelas"])
    kel = list(m["confusion_matrix"])
    with open(out / f"diagnosis_{nama}_confusion.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["benar \\ diagnosis"] + kel)
        for t in kel:
            w.writerow([t] + [m["confusion_matrix"][t][p_] for p_ in kel])
    with open(out / f"diagnosis_{nama}_per_citra.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rinci[0]))
        w.writeheader()
        w.writerows(rinci)


def baca_kunci_ambang(run_dir, wsha):
    kp = run_dir / "ambang_diagnosis.json"
    if not kp.exists():
        return None, kp
    k = json.loads(kp.read_text(encoding="utf-8"))
    if k.get("weights_sha256") != wsha:
        sys.exit(f"[GAGAL] {kp} milik checkpoint lain (sha {k.get('weights_sha256', '')[:12]} != {wsha[:12]}).")
    return k, kp


def evaluasi_folder(a, model, names, weights, wsha, out):
    """Evaluasi diagnosis per citra untuk set berlabel per FOLDER (tanpa bounding box), mis. ds3. Ambang WAJIB terkunci."""
    kunci, kp = baca_kunci_ambang(weights.parent.parent, wsha)
    if kunci is None:
        sys.exit(f"[DITOLAK] ambang diagnosis belum dikunci ({kp}). Jalankan evaluasi split val dulu (memilih ambang pada val bersih).")
    root = Path(a.set_folder) if Path(a.set_folder).is_absolute() else ROOT / a.set_folder
    peta = dict(x.split("=", 1) for x in a.peta_folder)
    kelas_valid = set(names.values()) | {TAK_TERDETEKSI, LUAR_CAKUPAN}
    for f_, k in peta.items():
        if k not in kelas_valid:
            sys.exit(f"[GAGAL] peta folder {f_}={k}: kelas harus salah satu dari {sorted(kelas_valid)}")
    if a.daftar:                              # baris: <jalur relatif project atau terhadap set>[\t<group_id>]
        dp = Path(a.daftar) if Path(a.daftar).is_absolute() else ROOT / a.daftar
        baris = [ln.split("\t") for ln in dp.read_text(encoding="utf-8").splitlines() if ln.strip()]
        item = []
        for b in baris:
            pth = Path(b[0]) if Path(b[0]).is_absolute() else (ROOT / b[0] if (ROOT / b[0]).exists() else root / b[0])
            item.append((pth, b[1] if len(b) > 1 else pth.name))
    else:
        item = [(x, x.name) for x in sorted(root.rglob("*")) if x.suffix.lower() in IMG_EXT]
    item = [(x, g) for x, g in item if x.parent.name.lower() in {k.lower() for k in peta}]
    peta_l = {k.lower(): v for k, v in peta.items()}
    if not item:
        sys.exit("[GAGAL] tidak ada gambar yang cocok dengan peta folder")
    print(f"[diagnosis folder] {len(item)} gambar dari {root}; ambang terkunci {kunci['ambang']} ({kunci['dipilih_pada']})", flush=True)
    top = prediksi_teratas(model, [x for x, _ in item], a.imgsz, a.batch)
    pred = diagnosa(top, kunci["ambang"], names)
    benar = [peta_l[x.parent.name.lower()] for x, _ in item]
    rinci = [{"berkas": str(x), "folder": x.parent.name, "label": t, "group_id": g, "conf_teratas": round(cf, 4),
              "kelas_teratas": names[c] if c >= 0 else "", "diagnosis": d}
             for (x, g), t, (cf, c), d in zip(item, benar, top, pred)]
    kelas = list(names.values()) + [TAK_TERDETEKSI]
    dalam = [i for i, t in enumerate(benar) if t != LUAR_CAKUPAN]
    hasil = {"ambang": kunci["ambang"], "ambang_dipilih_pada": kunci["dipilih_pada"], "set": str(root), "daftar": a.daftar,
             "n_gambar": len(item), "per_folder": dict(Counter(x.parent.name for x, _ in item))}
    if dalam:
        m = metrik_citra([benar[i] for i in dalam], [pred[i] for i in dalam], kelas, [item[i][1] for i in dalam])
        tulis_diagnosis(out, "dalam_cakupan", m, [rinci[i] for i in dalam])
        hasil["dalam_cakupan"] = m
        print(f"  dalam cakupan: akurasi {m['akurasi']} (n={m['n_citra']}, kelompok {m.get('n_kelompok')})", flush=True)
    sf = defaultdict(Counter)
    for (x, _), d in zip(item, pred):
        sf[x.parent.name][d] += 1
    hasil["sebaran_diagnosis_per_folder"] = {k: {"n": sum(v.values()), "sebaran": dict(v),
                                               "persen_didiagnosis_penyakit": round(100 * sum(n for d, n in v.items() if d != TAK_TERDETEKSI) / sum(v.values()), 2)}
                                           for k, v in sorted(sf.items())}
    hasil["weights_sha256"] = wsha
    luar = [i for i, t in enumerate(benar) if t == LUAR_CAKUPAN]
    if luar:                                  # penyakit di luar cakupan: hanya sebaran diagnosis (tidak ada jawaban benar)
        sb = Counter(pred[i] for i in luar)
        hasil["luar_cakupan"] = {"n_citra": len(luar), "sebaran_diagnosis": dict(sb),
                                 "persen_didiagnosis_salah_satu_penyakit": round(100 * sum(v for k, v in sb.items() if k != TAK_TERDETEKSI) / len(luar), 2)}
        with open(out / "diagnosis_luar_cakupan_per_citra.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rinci[0]))
            w.writeheader()
            w.writerows([rinci[i] for i in luar])
        print(f"  luar cakupan: {hasil['luar_cakupan']}", flush=True)
    (out / "ringkasan_diagnosis.json").write_text(json.dumps(hasil, indent=2, ensure_ascii=False), encoding="utf-8")
    return hasil


def diagnosis_split(a, model, names, items, subsets, weights, wsha, out, prov_src):
    """Diagnosis per citra pada split berlabel box. Pada val: ambang dipilih di subset 'bersih' (atau seluruh val bila tidak
    ada sumber provisional) lalu dikunci; pada split lain: ambang terkunci WAJIB ada."""
    kelas = list(names.values()) + [TAK_TERDETEKSI]
    lab = [label_citra(cl, names) for _, _, cl in items]
    multi = [x[0].name for x, l_ in zip(items, lab) if l_ is None]
    top_all = prediksi_teratas(model, [x[0] for x in items], a.imgsz, a.batch)
    idx = {x[0]: i for i, x in enumerate(items)}
    kunci, kp = baca_kunci_ambang(weights.parent.parent, wsha)
    sel_nama = "bersih" if (a.ambang_pada == "bersih" and any(n == "bersih" for n, _ in subsets)) else "dengan_background"
    if a.split == "val" and (kunci is None or a.pilih_ulang_ambang):
        sub = dict(subsets)[sel_nama]
        ii = [idx[x[0]] for x in sub if lab[idx[x[0]]] is not None]
        best, kurva = pilih_ambang([lab[i] for i in ii], [top_all[i] for i in ii], names, kelas)
        kunci = {"ambang": best["ambang"], "akurasi_val_saat_dipilih": best["akurasi"], "f1_makro_val_saat_dipilih": best["f1_makro"],
                 "dipilih_pada": f"val {sel_nama} ({len(ii)} citra" + (f"; tanpa sumber {sorted(prov_src)}" if sel_nama == "bersih" else "") + ")",
                 "kriteria": "akurasi per citra maksimum; seri -> F1 makro lebih tinggi, lalu ambang terkecil",
                 "grid": f"{GRID_AMBANG[0]}..{GRID_AMBANG[-1]} langkah 0.01", "aturan_diagnosis":
                 "kelas kotak dengan confidence tertinggi; tanpa kotak >= ambang -> " + TAK_TERDETEKSI,
                 "weights": str(weights), "weights_sha256": wsha, "dikunci": datetime.now().isoformat(timespec="seconds"),
                 "catatan": "ambang ini dipakai apa adanya untuk SEMUA set uji (test, test_new_session, ds3); jangan dipilih ulang di set uji"}
        kp.write_text(json.dumps(kunci, indent=2, ensure_ascii=False), encoding="utf-8")
        with open(out / "diagnosis_kurva_ambang_val.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(kurva[0]))
            w.writeheader()
            w.writerows(kurva)
        print(f"  ambang dipilih dan DIKUNCI: {kunci['ambang']} (akurasi {best['akurasi']} pada {kunci['dipilih_pada']}) -> {kp}", flush=True)
    elif kunci is None:
        sys.exit(f"[DITOLAK] ambang diagnosis belum dikunci ({kp}); evaluasi split val dulu.")
    hasil = {"ambang": kunci["ambang"], "ambang_dipilih_pada": kunci["dipilih_pada"], "citra_multi_kelas_dikeluarkan": len(multi),
             "contoh_multi_kelas": multi[:10], "subset": {}}
    for nama, sub in subsets:
        if nama == "tanpa_background":
            continue
        ii = [idx[x[0]] for x in sub if lab[idx[x[0]]] is not None]
        pred = diagnosa([top_all[i] for i in ii], kunci["ambang"], names)
        m = metrik_citra([lab[i] for i in ii], pred, kelas)
        rinci = [{"berkas": items[i][0].name, "label": lab[i], "conf_teratas": round(top_all[i][0], 4),
                  "kelas_teratas": names[top_all[i][1]] if top_all[i][1] >= 0 else "", "diagnosis": d} for i, d in zip(ii, pred)]
        tulis_diagnosis(out, nama, m, rinci)
        hasil["subset"][nama] = m
        print(f"  {nama}: akurasi per citra {m['akurasi']} (n={m['n_citra']}), F1 makro {m['f1_makro_kelas_yang_ada']}", flush=True)
    return hasil


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="checkpoint, biasanya runs/<run>/weights/last.pt")
    ap.add_argument("--data", default="data/processed/prov_v6/data.yaml")
    ap.add_argument("--split", default="val", choices=["val", "test", "test_new_session"])
    ap.add_argument("--izinkan-test", action="store_true", help="WAJIB untuk split test / test_new_session")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=None, help="baku: imgsz latihan dari checkpoint (640 bila tidak ada)")
    ap.add_argument("--conf-fp", type=float, nargs="+", default=[0.25, 0.5])
    ap.add_argument("--bench-n", type=int, default=100, help="jumlah gambar VAL untuk benchmark kecepatan")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--devices", nargs="+", default=["0", "cpu"])
    ap.add_argument("--tanpa-benchmark", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--set-folder", default=None, help="set berlabel per FOLDER tanpa box (mis. data/raw/ds3): HANYA diagnosis per citra")
    ap.add_argument("--peta-folder", nargs="+", default=[], metavar="FOLDER=KELAS",
                    help=f"mis. blast=leaf_blast blight=bacterial_leaf_blight tungro={LUAR_CAKUPAN} healthy={TAK_TERDETEKSI}")
    ap.add_argument("--daftar", default=None, help="berkas daftar foto (jalur[TAB]group_id), mis. reports/prov_v6/ds3_eksternal/ds3_bersih_blast.txt")
    ap.add_argument("--izinkan-uji-eksternal", action="store_true", help="WAJIB untuk --set-folder (set uji eksternal)")
    ap.add_argument("--pilih-ulang-ambang", action="store_true", help="HANYA split val: pilih ulang ambang diagnosis pada val bersih")
    ap.add_argument("--ambang-pada", choices=["bersih", "semua"], default="bersih",
                    help="himpunan val untuk memilih ambang diagnosis: bersih (tanpa provisional; run final) atau semua (randsplit)")
    ap.add_argument("--ulang-dengan-alasan", default=None, metavar="TEKS",
                    help="izinkan evaluasi ULANG set uji yang sudah pernah dievaluasi (alasan dicatat di riwayat_ulang.txt)")
    a = ap.parse_args()
    if a.set_folder and not a.izinkan_uji_eksternal:
        sys.exit("[DITOLAK] --set-folder adalah set uji eksternal; jalankan hanya setelah model final dipilih, dengan --izinkan-uji-eksternal.")
    if a.pilih_ulang_ambang and a.split != "val":
        sys.exit("[DITOLAK] ambang diagnosis hanya boleh dipilih pada val.")
    if a.split in SPLIT_TERLINDUNG and not a.izinkan_test and not a.set_folder:
        sys.exit(f"[DITOLAK] split '{a.split}' terlindung. Evaluasi hanya setelah model final dipilih, dengan --izinkan-test.")

    from ultralytics import YOLO, __version__ as ul_ver
    import torch

    weights = Path(a.weights) if Path(a.weights).is_absolute() else ROOT / a.weights
    data_p = Path(a.data) if Path(a.data).is_absolute() else ROOT / a.data
    dy = yaml.safe_load(data_p.read_text(encoding="utf-8"))
    droot = Path(dy["path"])
    sha = sha256(droot / "manifest.csv")
    if dy.get("manifest_sha256") and sha != dy["manifest_sha256"]:
        sys.exit(f"[GAGAL] manifest dataset berubah: {sha} != {dy['manifest_sha256']}")
    names = {int(k): v for k, v in dy["names"].items()}
    nama_eval = f"eval_{a.split}" if not a.set_folder else "eval_folder_" + (Path(a.daftar).stem if a.daftar else Path(a.set_folder).name)
    out = jalur_keluaran(a.out) if a.out else weights.parent.parent / nama_eval
    out.mkdir(parents=True, exist_ok=True)
    uji = a.set_folder or a.split in SPLIT_TERLINDUNG
    jejak = out / ("ringkasan_diagnosis.json" if a.set_folder else "ringkasan.json")
    if uji and jejak.exists():
        if not a.ulang_dengan_alasan:
            sys.exit(f"[DITOLAK] set uji ini sudah pernah dievaluasi ({jejak}). Setiap set uji dievaluasi SEKALI; "
                     "pakai --ulang-dengan-alasan \"...\" hanya bila benar-benar perlu (dicatat).")
        with open(out / "riwayat_ulang.txt", "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat(timespec='seconds')}\t{a.ulang_dengan_alasan}\n")
    wsha = sha256(weights)

    model = YOLO(str(weights))
    targs = (getattr(model, "ckpt", None) or {}).get("train_args", {}) or {}
    peringatan = []
    if targs.get("data") and Path(targs["data"]).resolve() != data_p.resolve():
        tp = Path(targs["data"])
        induk = (yaml.safe_load(tp.read_text(encoding="utf-8")) or {}).get("induk_data") if tp.exists() else None
        if not (induk and Path(induk).resolve() == data_p.resolve()):   # data_valsel.yaml turunan dari data.yaml ini -> wajar
            peringatan.append(f"checkpoint dilatih dengan data {targs['data']}, dievaluasi dengan {data_p}")
    if a.imgsz is None:
        a.imgsz = int(targs.get("imgsz") or 640)
    if {int(k): v for k, v in model.names.items()} != names:
        sys.exit(f"[GAGAL] nama kelas model {model.names} != data.yaml {names}")
    if a.set_folder:
        evaluasi_folder(a, model, names, weights, wsha, out)
        return

    items = daftar_split(droot, a.split)
    nobg = [x for x in items if x[2]]
    subsets = [("dengan_background", items), ("tanpa_background", nobg)]
    prov_src = set(dy.get("provisional_sources") or [])
    if prov_src:
        with open(droot / "manifest.csv", encoding="utf-8") as f:
            sumber = {r["file"]: r["source"] for r in csv.DictReader(f)}
        prov_items = [x for x in items if sumber.get(x[0].name) in prov_src]
        if prov_items:
            subsets += [("bersih", [x for x in items if sumber.get(x[0].name) not in prov_src]), ("blb_provisional", prov_items)]
    print(f"[1/4] metrik: {len(items)} gambar ({len(items) - len(nobg)} background); subset: "
          + ", ".join(f"{n}={len(s_)}" for n, s_ in subsets), flush=True)
    rows, extra = [], {}
    for nama, sub in subsets:
        r, e = evaluasi_subset(model, sub, names, out, nama, a.batch, a.imgsz)
        rows += r
        extra[nama] = e
    shutil.rmtree(out / "_subset", ignore_errors=True)
    with open(out / "metrik_per_kelas.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    prov = " BLB provisional; TIDAK untuk tabel paper." if dy.get("provisional") else ""
    run = weights.parent.parent.name
    (out / "metrik_per_kelas.tex").write_text(tabel_latex(rows, f"Metrik {run} pada split {a.split}.{prov}".replace("_", "\\_"),
                                                          f"tab:{run}_{a.split}"), encoding="utf-8")

    print("[1b/4] diagnosis per citra (satu penyakit per citra)", flush=True)
    diag = diagnosis_split(a, model, names, items, subsets, weights, wsha, out, prov_src)

    print("[2/4] false positive pada gambar background", flush=True)
    fp = fp_background(model, items, names, a.conf_fp, a.imgsz, a.batch)
    if fp:
        with open(out / "fp_background.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(fp[0]))
            w.writeheader()
            w.writerows(fp)

    print("[3/4] parameter dan GFLOPs", flush=True)
    info = info_model(str(weights), a.imgsz)

    spd = []
    if not a.tanpa_benchmark:
        print("[4/4] benchmark kecepatan (gambar VAL, batch 1)", flush=True)
        bimgs = [x[0] for x in daftar_split(droot, "val")][: a.bench_n]
        spd = benchmark(str(weights), bimgs, [d if d == "cpu" else int(d) for d in a.devices], a.warmup, a.imgsz)
        with open(out / "kecepatan.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(spd[0]))
            w.writeheader()
            w.writerows(spd)

    summ = {"weights": str(weights), "weights_sha256": wsha, "run": run, "split": a.split, "dataset": dy.get("dataset_name"), "manifest_sha256": sha,
            "provisional": dy.get("provisional"), "catatan": prov.strip() or None, "peringatan": peringatan,
            "n_gambar": len(items), "n_background": len(items) - len(nobg), "metrik": rows, "lain": extra,
            "fp_background": fp, "model": info, "kecepatan": spd, "diagnosis_per_citra": diag,
            "versi": {"torch": torch.__version__, "ultralytics": ul_ver, "platform": platform.platform()},
            "parameter_eval": {"imgsz": a.imgsz, "batch_val": a.batch, "conf_val": 0.001, "iou": 0.7, "conf_benchmark": 0.25}}
    (out / "ringkasan.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: summ[k] for k in ("run", "split", "n_gambar", "n_background", "model", "peringatan")}, indent=1, ensure_ascii=False))
    for r in rows:
        print(r)
    for r in fp + spd:
        print(r)


if __name__ == "__main__":
    main()
