"""
Kontrol latar alternatif pada data VALIDASI (jawaban komentar reviewer 1 dan 3).

Model asli (dilatih dengan latar, v7) diuji pada foto validasi bersih dengan latar
yang diganti lima cara:
    original : foto asli
    gray     : latar abu-abu (114,114,114) = foto v9 apa adanya
    black    : latar hitam
    blur     : latar asli yang diburamkan kuat (daun tetap tajam)
    swap     : latar dari foto daun sehat validasi lain (acak, seed tetap)

Masker daun diturunkan dari pasangan foto v7 dan v9: piksel yang di v9 berwarna
abu-abu 114 DAN berbeda dari foto v7 dianggap latar. Tidak ada label penyakit yang
dipakai untuk membuat masker. Data UJI tidak disentuh sama sekali.

Cara pakai (dari folder rice_yolo, dengan venv proyek aktif):
    python scripts/paper/kontrol_latar_val.py --limit 20     # coba cepat dulu
    python scripts/paper/kontrol_latar_val.py                # jalankan penuh
Uji alur tanpa GPU/model (data tiruan):
    python scripts/paper/kontrol_latar_val.py --selftest
"""
import argparse
import csv
import json
import os
import re
import statistics
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import yaml

KONDISI = ["original", "gray", "black", "blur", "swap"]
SEHAT = "sehat"
# Nilai acuan dari uji latar hitam sebelumnya (reports/v9_latar/tahap1_isian_hitam_s_seed0.json)
ACUAN_HITAM_SEED0 = {"yolov8s": 0.7491, "yolo11s": 0.6617}
EKST = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


# ----------------------------------------------------------------- data
def baca_yaml(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def nama_kelas(d):
    n = d.get("names")
    if isinstance(n, dict):
        return {int(k): v for k, v in n.items()}
    return {i: v for i, v in enumerate(n)}


def daftar_val(yaml_path):
    """Kembalikan daftar path foto validasi dari data yaml (list .txt, folder, atau list)."""
    d = baca_yaml(yaml_path)
    base_cands = [Path(yaml_path).parent]
    if d.get("path"):
        base_cands.insert(0, Path(d["path"]))
    v = d.get("val")
    entri = v if isinstance(v, list) else [v]
    hasil = []
    for e in entri:
        p = cari_path(e, base_cands)
        if p is None:
            sys.exit(f"[GAGAL] tidak menemukan 'val' = {e} dari {yaml_path}")
        if p.is_dir():
            hasil += sorted(str(x) for x in p.rglob("*") if x.suffix.lower() in EKST)
        else:
            for baris in p.read_text(encoding="utf-8").splitlines():
                baris = baris.strip()
                if not baris:
                    continue
                q = cari_path(baris, [p.parent] + base_cands)
                if q is None:
                    sys.exit(f"[GAGAL] foto di daftar tidak ditemukan: {baris}")
                hasil.append(str(q))
    return d, hasil


def cari_path(s, bases):
    s = str(s).replace("\\", "/")
    p = Path(s)
    if p.is_absolute() and p.exists():
        return p
    for b in bases:
        q = Path(b) / s
        if q.exists():
            return q
    if p.exists():
        return p
    return None


def path_label(img_path):
    # Konvensi yang sama dengan Ultralytics (img2label_paths)
    s = str(Path(img_path))
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    return sb.join(s.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt"


def kelas_benar(img_path, names):
    lp = Path(path_label(img_path))
    if not lp.exists():
        return SEHAT
    ids = []
    for baris in lp.read_text(encoding="utf-8").splitlines():
        if baris.strip():
            ids.append(int(float(baris.split()[0])))
    if not ids:
        return SEHAT
    ids = sorted(set(ids))
    if len(ids) > 1:
        print(f"[PERINGATAN] lebih dari satu kelas di {lp.name}: {ids}, dipakai yang pertama")
    return names[ids[0]]


def muat_grup(manifest):
    """Peta stem -> group_id dari manifest (bila kolomnya dapat dikenali)."""
    peta = {}
    if not manifest or not Path(manifest).exists():
        return peta
    with open(manifest, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        kol_g = next((c for c in r.fieldnames if "group" in c.lower() or "kelompok" in c.lower()), None)
        kol_p = next((c for c in r.fieldnames if any(k in c.lower() for k in ("path", "file", "image", "id"))), None)
        if not kol_g or not kol_p:
            return peta
        for row in r:
            peta[Path(str(row[kol_p]).replace("\\", "/")).stem] = row[kol_g]
    return peta


# ----------------------------------------------------------------- masker dan latar
def masker_latar(a, b, tol_abu=6, tol_beda=8):
    """True = latar. a: foto v7, b: foto v9 (latar abu-abu)."""
    if b.shape != a.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST)
    a16, b16 = a.astype(np.int16), b.astype(np.int16)
    abu = (np.abs(b16 - 114) <= tol_abu).all(axis=2)
    beda = (np.abs(b16 - a16) > tol_beda).any(axis=2)
    m = (abu & beda).astype(np.uint8)
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k7)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k3)
    return m.astype(bool), b


def varian(a, b, bg, donor):
    h, w = a.shape[:2]
    m3 = bg[:, :, None]
    sigma = 0.03 * max(h, w)
    blur = cv2.GaussianBlur(a, (0, 0), sigmaX=sigma, sigmaY=sigma)
    d = cv2.resize(donor, (w, h), interpolation=cv2.INTER_AREA)
    return {
        "original": a,
        "gray": b,
        "black": np.where(m3, np.uint8(0), a),
        "blur": np.where(m3, blur, a),
        "swap": np.where(m3, d, a),
    }


# ----------------------------------------------------------------- ambang terkunci
def cari_tau(run, kunci_paths):
    pola = re.compile(r"ambang|tau|thresh", re.I)
    for kp in kunci_paths:
        kp = Path(kp)
        if not kp.exists():
            continue
        if kp.suffix == ".json":
            data = json.loads(kp.read_text(encoding="utf-8"))
            ditemukan = []

            def jalan(o, dalam_run=False):
                if isinstance(o, dict):
                    cocok = dalam_run or any(isinstance(v, str) and run in v for v in o.values())
                    for k, v in o.items():
                        if cocok and pola.search(str(k)) and isinstance(v, (int, float)) and 0 < v < 1:
                            ditemukan.append(float(v))
                        jalan(v, cocok or run in str(k))
                elif isinstance(o, list):
                    for v in o:
                        jalan(v, dalam_run)
            jalan(data)
            if ditemukan:
                return ditemukan[0], f"{kp.name}"
        elif kp.suffix == ".csv":
            with open(kp, newline="", encoding="utf-8") as f:
                r = csv.DictReader(f)
                kol = [c for c in r.fieldnames if pola.search(c)]
                for row in r:
                    if any(run in str(v) for v in row.values()) and kol:
                        try:
                            return float(str(row[kol[0]]).replace(",", ".")), f"{kp.name}:{kol[0]}"
                        except ValueError:
                            pass
    return None, None


# ----------------------------------------------------------------- model
class PelariUltralytics:
    def __init__(self, bobot, device):
        from ultralytics import YOLO
        self.m = YOLO(str(bobot))
        self.device = device

    def prediksi(self, imgs):
        # conf 0,001 dan iou 0,7 sama dengan evaluasi resmi (scripts/evaluate.py, diagnosis per citra)
        res = self.m.predict(imgs, imgsz=640, conf=0.001, iou=0.7, device=self.device, verbose=False)
        out = []
        for r in res:
            if r.boxes is None or len(r.boxes) == 0:
                out.append((None, 0.0))
                continue
            c = r.boxes.conf.cpu().numpy()
            k = r.boxes.cls.cpu().numpy().astype(int)
            i = int(c.argmax())
            out.append((int(k[i]), float(c[i])))
        return out


class PelariTiruan:
    """Untuk --selftest: kelas ditebak dari warna rata-rata daun."""
    def __init__(self, sensitif_latar):
        self.s = sensitif_latar

    def prediksi(self, imgs):
        out = []
        for im in imgs:
            mean = im.reshape(-1, 3).mean(0)
            if self.s and mean[2] < 90:          # tiruan "model peka latar"
                out.append((1, 0.9))
            elif im[..., 2].max() > 200:
                out.append((0, 0.8))
            else:
                out.append((None, 0.0))
        return out


# ----------------------------------------------------------------- utama
def jalankan(args):
    root = Path(args.root).resolve()
    v7_yaml = root / args.v7_yaml
    v9_yaml = root / args.v9_yaml
    d7, foto7 = daftar_val(v7_yaml)
    _, foto9 = daftar_val(v9_yaml)
    names = nama_kelas(d7)
    peta9 = {Path(p).stem: p for p in foto9}
    pasangan = [(p, peta9.get(Path(p).stem)) for p in foto7]
    hilang = [p for p, q in pasangan if q is None]
    pasangan = [(p, q) for p, q in pasangan if q is not None]
    if args.limit:
        # sampel berjarak rata (daftar terurut nama dimulai dengan foto sehat ds2, jadi 20 pertama hanya satu kelas)
        pasangan = pasangan[:: max(1, len(pasangan) // args.limit)][: args.limit]
    print(f"Foto validasi v7: {len(foto7)}; berpasangan dengan v9: {len(pasangan)}; tanpa pasangan v9: {len(hilang)}")
    if len(hilang) > 0.05 * len(foto7):
        sys.exit("[GAGAL] terlalu banyak foto tanpa pasangan v9; periksa --v9-yaml")

    benar = {p: kelas_benar(p, names) for p, _ in pasangan}
    print("Sebaran kelas:", {k: list(benar.values()).count(k) for k in sorted(set(benar.values()))})
    grup = muat_grup(root / args.manifest) if args.manifest else {}
    donor_pool = [p for p, _ in pasangan if benar[p] == SEHAT]
    rng = np.random.default_rng(0)

    # model
    runs = []
    for m in args.models.split(","):
        for s in args.seeds.split(","):
            runs.append((m, int(s), f"v7_final_prov_{m}_s{s}"))
    pelari, tau = {}, {}
    manual = dict(x.split("=") for x in args.tau.split(",")) if args.tau else {}
    for m, s, run in runs:
        if args.selftest:
            pelari[run] = PelariTiruan(sensitif_latar=(m == "yolo11s"))
            tau[run] = 0.5
            continue
        if run in manual:
            tau[run], sumber = float(manual[run]), "--tau"
        else:
            tau[run], sumber = cari_tau(run, [root / k for k in args.kunci.split(",")])
        if tau[run] is None:
            sys.exit(f"[GAGAL] ambang untuk {run} tidak ditemukan. Isi manual: --tau {run}=0.xx,...")
        bobot = root / args.runs_dir / run / "weights" / "best.pt"
        if not bobot.exists():
            sys.exit(f"[GAGAL] bobot tidak ditemukan: {bobot}")
        print(f"{run}: tau={tau[run]:.3f} (dari {sumber})")
        pelari[run] = PelariUltralytics(bobot, args.device)

    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)
    baris_foto, n_cadangan, contoh = [], 0, []
    selisih_abu = []
    for i, (p7, p9) in enumerate(pasangan):
        a, b = cv2.imread(p7), cv2.imread(p9)
        if a is None or b is None:
            print(f"[LEWAT] gagal membaca {p7} atau {p9}")
            continue
        bg, b = masker_latar(a, b, tol_abu=args.tol_abu, tol_beda=args.tol_beda)
        cadangan = bg.mean() < 0.01
        n_cadangan += int(cadangan)
        kandidat = [d for d in donor_pool if d != p7 and (not grup or grup.get(Path(d).stem) != grup.get(Path(p7).stem))]
        donor = cv2.imread(kandidat[int(rng.integers(len(kandidat)))]) if kandidat else a
        v = varian(a, b, bg, donor)
        if not cadangan:
            abu_dari_masker = np.where(bg[:, :, None], np.uint8(114), a)
            selisih_abu.append(float(np.abs(abu_dari_masker.astype(int) - b.astype(int)).mean()))
        if len(contoh) < 6 and not cadangan and benar[p7] != SEHAT:
            contoh.append([cv2.resize(v[k], (256, 256)) for k in KONDISI])
        imgs = [v[k] for k in KONDISI]
        for run in pelari:
            hasil = pelari[run].prediksi(imgs)
            for k, (cls, conf) in zip(KONDISI, hasil):
                diag = names[cls] if (cls is not None and conf >= tau[run]) else SEHAT
                baris_foto.append({"foto": Path(p7).name, "benar": benar[p7], "run": run, "kondisi": k,
                                   "diagnosis": diag, "conf": round(conf, 4), "cadangan": int(cadangan)})
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(pasangan)} foto")

    # simpan per foto
    with open(out / "per_foto.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(baris_foto[0].keys()))
        w.writeheader()
        w.writerows(baris_foto)

    # ringkasan per run x kondisi
    blast = next((n for n in names.values() if "blast" in n.lower()), None)
    brown = next((n for n in names.values() if "brown" in n.lower()), None)
    per_run = {}
    for run in pelari:
        for k in KONDISI:
            rr = [r for r in baris_foto if r["run"] == run and r["kondisi"] == k]
            def rec(c):
                s = [r for r in rr if r["benar"] == c]
                return sum(r["diagnosis"] == c for r in s) / len(s) if s else float("nan")
            per_run[(run, k)] = {
                "akurasi": sum(r["diagnosis"] == r["benar"] for r in rr) / len(rr),
                "blast_dR": rec(blast), "brown_dR": rec(brown), "sehat_benar": rec(SEHAT),
            }
    with open(out / "ringkasan_per_run.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run", "kondisi", "akurasi", "blast_dR", "brown_dR", "sehat_benar"])
        for (run, k), v in per_run.items():
            w.writerow([run, k] + [f"{v[x]:.4f}" for x in ("akurasi", "blast_dR", "brown_dR", "sehat_benar")])

    ringkas = {}
    for m in args.models.split(","):
        for k in KONDISI:
            rs = [per_run[(run, k)] for run in pelari if f"_{m}_" in run]
            ringkas[(m, k)] = {x: (statistics.mean(r[x] for r in rs),
                                   statistics.stdev(r[x] for r in rs) if len(rs) > 1 else 0.0)
                               for x in ("akurasi", "blast_dR", "brown_dR", "sehat_benar")}
    with open(out / "ringkasan.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "kondisi", "akurasi_mean", "akurasi_sd", "blast_dR_mean", "blast_dR_sd",
                    "brown_dR_mean", "brown_dR_sd", "sehat_benar_mean", "sehat_benar_sd"])
        for (m, k), v in ringkas.items():
            w.writerow([m, k] + [f"{v[x][j]:.4f}" for x in ("akurasi", "blast_dR", "brown_dR", "sehat_benar") for j in (0, 1)])

    if contoh:
        grid = np.vstack([np.hstack(r) for r in contoh])
        cv2.imwrite(str(out / "contoh_kondisi.jpg"), grid)

    # laporan
    teks = ["# Kontrol latar alternatif (ANALISIS VALIDASI, bukan data uji)", "",
            f"Foto: {len(pasangan)} (cadangan tanpa masker: {n_cadangan}); runs: {', '.join(pelari)}",
            f"Rata-rata selisih abu-abu dari masker vs foto v9: {statistics.mean(selisih_abu) if selisih_abu else float('nan'):.2f} (0-255; kecil = masker cocok)",
            "", "Kolom contoh_kondisi.jpg: " + ", ".join(KONDISI), "",
            "| Model | Kondisi | Akurasi | Blast dR | Brown dR | Sehat benar |", "|---|---|---|---|---|---|"]
    for (m, k), v in ringkas.items():
        teks.append(f"| {m} | {k} | " + " | ".join(f"{v[x][0]:.3f} ± {v[x][1]:.3f}" for x in ("akurasi", "blast_dR", "brown_dR", "sehat_benar")) + " |")
    teks += ["", "## Pemeriksaan terhadap hasil lama (latar hitam, seed 0)"]
    for m, acuan in ACUAN_HITAM_SEED0.items():
        run = f"v7_final_prov_{m}_s0"
        if (run, "black") in per_run:
            nilai = per_run[(run, "black")]["akurasi"]
            status = "COCOK" if abs(nilai - acuan) <= 0.01 else "BERBEDA, periksa masker atau ambang"
            teks.append(f"- {m}: sekarang {nilai:.4f}, sebelumnya {acuan:.4f} -> {status}")
    teks += ["", "## Baris tabel LaTeX (Akurasi, Blast dR; rata-rata 3 seed)"]
    label = {"original": "Original", "gray": "Gray", "black": "Black", "blur": "Blurred", "swap": "Swapped"}
    for k in KONDISI:
        sel = []
        for m in ("yolov8s", "yolo11s"):
            if (m, k) in ringkas:
                sel += [f"{ringkas[(m, k)]['akurasi'][0]:.3f}", f"{ringkas[(m, k)]['blast_dR'][0]:.3f}"]
        teks.append(f"{label[k]} & " + " & ".join(sel) + r" \\")
    (out / "RINGKASAN.md").write_text("\n".join(teks) + "\n", encoding="utf-8")
    print("\n".join(teks))
    print(f"\nHasil tersimpan di {out}")


def buat_data_tiruan(tmp):
    """Data tiruan kecil untuk --selftest."""
    rng = np.random.default_rng(1)
    for versi in ("v7", "v9"):
        (tmp / versi / "images" / "val").mkdir(parents=True)
        (tmp / versi / "labels" / "val").mkdir(parents=True)
    daftar7, daftar9 = [], []
    for i in range(12):
        kelas = [0, 1, None][i % 3]
        a = (rng.integers(20, 60, (120, 160, 3))).astype(np.uint8)       # latar gelap
        a[30:90, 60:100] = (40, 160, 60)                                  # daun
        if kelas == 0:
            a[50:60, 75:85] = (40, 40, 230)                               # bercak
        b = np.full_like(a, 114)
        b[30:90, 60:100] = a[30:90, 60:100]
        n = f"img{i:02d}.jpg"
        for versi, im, dft in (("v7", a, daftar7), ("v9", b, daftar9)):
            p = tmp / versi / "images" / "val" / n
            cv2.imwrite(str(p), im, [cv2.IMWRITE_JPEG_QUALITY, 98])
            lbl = tmp / versi / "labels" / "val" / n.replace(".jpg", ".txt")
            lbl.write_text("" if kelas is None else f"{kelas} 0.5 0.5 0.1 0.1\n")
            dft.append(str(p))
    for versi, dft in (("v7", daftar7), ("v9", daftar9)):
        (tmp / versi / "val.txt").write_text("\n".join(dft))
        (tmp / versi / "data.yaml").write_text(yaml.safe_dump(
            {"path": str(tmp / versi), "val": "val.txt", "names": {0: "leaf_blast", 1: "brown_spot", 2: "bacterial_leaf_blight"}}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--v7-yaml", default="data/derived/prov_v7_valsel/data_valsel.yaml")
    ap.add_argument("--v9-yaml", default="data/derived/prov_v9_valsel/data_valsel.yaml")
    ap.add_argument("--manifest", default="data/processed/prov_v7/manifest.csv")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--kunci", default="reports/final_eval_v7/kunci_model.json,reports/final_eval_v7/kunci_model.csv")
    ap.add_argument("--models", default="yolov8s,yolo11s")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--tau", default="", help="isi manual, mis. v7_final_prov_yolov8s_s0=0.33,...")
    ap.add_argument("--device", default="0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="reports/v9_latar/kontrol_latar_val")
    # Foto v9 disimpan JPEG kualitas 95 (bangun_prov_v9.py), jadi abu-abu 114 di dekat tepi daun bergeser. Diukur terhadap masker
    # SAM2 asli pada 120 foto val: tol 6 -> latar tertinggal 0,25%; tol 20 -> 0,07% dengan daun ikut terhapus hanya 0,004%.
    ap.add_argument("--tol-abu", type=int, default=20, help="toleransi warna abu-abu 114 pada foto v9")
    ap.add_argument("--tol-beda", type=int, default=8, help="beda minimum v9 vs v7 agar dianggap latar")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        tmp = Path(tempfile.mkdtemp())
        buat_data_tiruan(tmp)
        args.root, args.v7_yaml, args.v9_yaml = str(tmp), "v7/data.yaml", "v9/data.yaml"
        args.manifest, args.out = "", "hasil_selftest"
    jalankan(args)


if __name__ == "__main__":
    main()
