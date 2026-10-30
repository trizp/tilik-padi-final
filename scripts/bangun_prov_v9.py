#!/usr/bin/env python3
"""
bangun_prov_v9.py - prov_v9 = prov_v7 dengan LATAR DIHAPUS (KEPUTUSAN_PROYEK.md 2026-10-16, keputusan v9).

- Masker: SAM2.1-b mode otomatis (grid 16x16, batch 16) + aturan v2 + cadangan, dibuat TANPA label oleh
  scripts/v9_latar/masker_sam2_warna.py (data/derived/v9_latar/sam2_v2/masker_mentah), pasca-proses umum.pasca_proses.
- Latar diganti abu-abu (114,114,114), JPEG q95. Foto cadangan (tidak dipisah) = berkas prov_v7 apa adanya (hardlink, byte identik).
- Label: koordinat tidak diubah; kotak dengan < 50% luas di dalam masker dibuang. Foto berpenyakit yang kehilangan SEMUA kotaknya
  DIKELUARKAN dari semua split (tidak dijadikan background).
- Aturan background train v7 dipertahankan (max_background_frac_train 0,10; pangkasan acak seed 42 HANYA dari background ds4 di train)
  karena foto berpenyakit yang dikeluarkan menaikkan porsi background; jumlah pangkasan dilaporkan.
- ID foto, split, kelompok, kelas, sumber identik dengan prov_v7 untuk foto yang tersisa.

  python scripts/bangun_prov_v9.py bangun     # tulis data/processed/prov_v9 (belum beku) + pemeriksaan + audit masker
  python scripts/bangun_prov_v9.py bekukan    # setelah pemeriksaan, audit, dan verify_dataset lolos
Keluaran pemeriksaan: reports/prov_v9/{pemeriksaan_kesetaraan.json, audit_masker.json, foto_dikeluarkan.csv}
"""
import csv
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "v9_latar"))
import umum  # noqa: E402

V7 = ROOT / "data/processed/prov_v7"
V9 = ROOT / "data/processed/prov_v9"
MASKER = ROOT / "data/derived/v9_latar/sam2_v2"
REP = ROOT / "reports/prov_v9"
SPL = ("train", "val", "test", "test_new_session")
NAMES = {0: "leaf_blast", 1: "brown_spot", 2: "bacterial_leaf_blight"}
MAKS_BG, SEED_BG = 0.10, 42


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def labels_sha(D):
    h = hashlib.sha256()
    for sp in SPL:
        for lf in sorted((D / "labels" / sp).iterdir()):
            h.update(f"{sp}/{lf.name}\n".encode())
            h.update(lf.read_bytes())
    return h.hexdigest()


def kotak_dari_baris(baris, w, h):
    out = []
    for b in baris:
        c, x, y, bw, bh = map(float, b.split()[:5])
        out.append((int(c), (x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h))
    return out


def bangun():
    if V9.exists():
        sys.exit(f"[DITOLAK] {V9} sudah ada")
    dy7 = yaml.safe_load((V7 / "data.yaml").read_text(encoding="utf-8"))
    if sha(V7 / "manifest.csv") != dy7["manifest_sha256"] or not dy7.get("frozen"):
        sys.exit("[GAGAL] prov_v7 tidak beku atau manifest berubah")
    info = json.loads((MASKER / "info.json").read_text(encoding="utf-8"))
    man = list(csv.DictReader(open(V7 / "manifest.csv", encoding="utf-8")))
    hilang = [r["file"] for r in man if not (MASKER / "masker_mentah" / (Path(r["file"]).stem + ".png")).exists()]
    if hilang:
        sys.exit(f"[GAGAL] masker belum ada untuk {len(hilang)} foto, mis. {hilang[:3]}")
    for sp in SPL:
        (V9 / "images" / sp).mkdir(parents=True)
        (V9 / "labels" / sp).mkdir(parents=True)

    kol = list(man[0].keys()) + ["latar_dihapus", "luas_masker", "kotak_asli", "kotak_dibuang"]
    simpan, keluar, kotak_stat = [], [], defaultdict(lambda: {"kotak": 0, "dibuang": 0})
    luas_per = defaultdict(list)
    for i, r in enumerate(man):
        sp, f = r["split"], r["file"]
        src = V7 / "images" / sp / f
        bgr = cv2.imread(str(src))
        h, w = bgr.shape[:2]
        mm = cv2.imread(str(MASKER / "masker_mentah" / (Path(f).stem + ".png")), cv2.IMREAD_GRAYSCALE)
        cad = bool(info.get(f, {}).get("cadangan"))
        m = umum.pasca_proses(mm >= 128)
        lab = V7 / "labels" / sp / (Path(f).stem + ".txt")
        baris = [b for b in lab.read_text().splitlines() if b.strip()] if lab.exists() else []
        ret = umum.retensi_kotak(m, kotak_dari_baris(baris, w, h))
        tetap = [b for b, (_, v, _) in zip(baris, ret) if v >= 0.5]
        for (c, v, _) in ret:
            k = kotak_stat[f"{sp}|{r['source']}|{NAMES[c]}"]
            k["kotak"] += 1
            k["dibuang"] += int(v < 0.5)
        if baris and not tetap:
            keluar.append({"file": f, "split": sp, "class": r["class"], "source": r["source"], "group": r["group"],
                           "kotak_asli": len(baris), "latar_dihapus": "cadangan" if cad else "ya", "alasan": "semua kotak < 50% di dalam daun"})
            continue
        if cad:
            try:
                os.link(src, V9 / "images" / sp / f)
            except OSError:
                import shutil
                shutil.copy2(src, V9 / "images" / sp / f)
        else:
            out = bgr.copy()
            out[~m] = umum.ABU[::-1]
            cv2.imwrite(str(V9 / "images" / sp / f), out, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if lab.exists():
            (V9 / "labels" / sp / lab.name).write_text("\n".join(tetap) + ("\n" if tetap else ""))
        luas = float(m.mean()) if not cad else 1.0
        luas_per[f"{sp}|{r['source']}|{r['class']}"].append(luas)
        simpan.append(dict(r, n_boxes=str(len(tetap)), latar_dihapus="cadangan" if cad else "ya", luas_masker=f"{luas:.4f}",
                           kotak_asli=str(len(baris)), kotak_dibuang=str(len(baris) - len(tetap))))
        if i % 500 == 0:
            print(i, len(man), flush=True)

    # aturan background train v7: <= 10%, pangkas acak (seed 42) dari background ds4 di train
    tr = [r for r in simpan if r["split"] == "train"]
    bg = sorted((r for r in tr if r["class"] == "background" and r["source"] == "ds4"), key=lambda r: r["file"])
    n_bg = sum(r["class"] == "background" for r in tr)
    perlu = 0
    while (n_bg - perlu) / (len(tr) - perlu) > MAKS_BG + 1e-12:
        perlu += 1
    pangkas = set(r["file"] for r in random.Random(SEED_BG).sample(bg, perlu)) if perlu else set()
    for r in [r for r in simpan if r["file"] in pangkas]:
        keluar.append({"file": r["file"], "split": "train", "class": "background", "source": r["source"], "group": r["group"],
                       "kotak_asli": 0, "latar_dihapus": r["latar_dihapus"], "alasan": "pangkasan background train (aturan v7 maks 10%)"})
        (V9 / "images" / "train" / r["file"]).unlink()
        lp = V9 / "labels" / "train" / (Path(r["file"]).stem + ".txt")
        if lp.exists():
            lp.unlink()
    simpan = [r for r in simpan if r["file"] not in pangkas]

    with open(V9 / "manifest.csv", "w", newline="", encoding="utf-8") as fo:
        wr = csv.DictWriter(fo, fieldnames=kol)
        wr.writeheader()
        wr.writerows(simpan)
    ada = {r["file"] for r in simpan}
    (V9 / "lists").mkdir()
    for lf in sorted((V7 / "lists").iterdir()):
        isi = [b for b in lf.read_text(encoding="utf-8").splitlines() if b.strip() and Path(b).name in ada]
        (V9 / "lists" / lf.name).write_text("\n".join(isi) + ("\n" if isi else ""), encoding="utf-8")
    REP.mkdir(parents=True, exist_ok=True)
    with open(REP / "foto_dikeluarkan.csv", "w", newline="", encoding="utf-8") as fo:
        wr = csv.DictWriter(fo, fieldnames=list(keluar[0]))
        wr.writeheader()
        wr.writerows(keluar)

    dy9 = dict(dy7, path=str(V9).replace("\\", "/"), dataset_name="prov_v9", run_prefix="v9_", manifest_sha256=sha(V9 / "manifest.csv"),
               frozen=False, induk_dataset="prov_v7", induk_manifest_sha256=dy7["manifest_sha256"],
               perbaikan="latar dihapus (SAM2.1-b otomatis + aturan v2 + cadangan, latar abu-abu 114, JPEG q95); kotak < 50% di daun "
                         "dibuang; foto yang kehilangan semua kotak dikeluarkan; ID/split/kelompok sama dengan prov_v7 untuk foto yang tersisa")
    dy9.pop("perbaikan_orientasi", None)
    (V9 / "data.yaml").write_text(yaml.safe_dump(dy9, allow_unicode=True, sort_keys=False), encoding="utf-8")
    m7 = json.loads((V7 / "dataset_meta.json").read_text(encoding="utf-8"))
    m9 = dict(m7, dataset_name="prov_v9", dibuat=datetime.now().isoformat(timespec="seconds"), manifest_sha256=dy9["manifest_sha256"],
              frozen=False, freeze={"frozen": False, "alasan_tidak_dibekukan": "menunggu pemeriksaan"},
              induk={"dataset": "prov_v7", "manifest_sha256": dy7["manifest_sha256"], "grup_split_sha256": m7.get("grup_split_sha256")},
              latar_dihapus={"metode": "SAM2.1-b mode otomatis via ultralytics 8.4.171 (sam2.1_b.pt), grid 16x16, batch 16, aturan v2 + cadangan",
                             "skrip": "scripts/v9_latar/masker_sam2_warna.py; scripts/v9_latar/umum.py (pasca_proses)",
                             "masker": "data/derived/v9_latar/sam2_v2/masker_mentah", "isian": [114, 114, 114], "jpeg_quality": 95,
                             "buang_kotak": "kotak < 50% luas di dalam masker dibuang", "foto_dikeluarkan": "reports/prov_v9/foto_dikeluarkan.csv"})
    m9["catatan"] = list(m7.get("catatan", [])) + ["prov_v9: latar dihapus; BLB (ds1_provisional) paling terdampak (banyak foto cadangan dan kotak dibuang); "
                                                   "BLB dilaporkan terpisah dan tidak dipakai untuk kesimpulan tentang latar."]
    (V9 / "dataset_meta.json").write_text(json.dumps(m9, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- pemeriksaan kesetaraan
    idx7 = {r["file"]: r for r in man}
    beda_id = [r["file"] for r in simpan if any(r[k] != idx7[r["file"]][k] for k in ("split", "group", "class", "source"))]
    label_bukan_subset, cad_tidak_identik, img_hilang = 0, 0, 0
    for r in simpan:
        sp, f = r["split"], r["file"]
        p9 = V9 / "images" / sp / f
        if not p9.exists():
            img_hilang += 1
            continue
        if r["latar_dihapus"] == "cadangan" and sha(p9) != sha(V7 / "images" / sp / f):
            cad_tidak_identik += 1
        l7, l9 = V7 / "labels" / sp / (Path(f).stem + ".txt"), V9 / "labels" / sp / (Path(f).stem + ".txt")
        if l7.exists():
            b7 = Counter(b for b in l7.read_text().splitlines() if b.strip())
            b9 = Counter(b for b in l9.read_text().splitlines() if b.strip()) if l9.exists() else Counter()
            label_bukan_subset += int(bool(b9 - b7))
    disk = {sp: {p.name for p in (V9 / "images" / sp).iterdir()} for sp in SPL}
    per_split_kelas = lambda rows: {sp: dict(Counter(r["class"] for r in rows if r["split"] == sp)) for sp in SPL}  # noqa: E731
    hasil = {"id_split_kelompok_kelas_sumber_identik_untuk_foto_tersisa": not beda_id, "selisih": len(beda_id),
             "foto_v7": len(man), "foto_v9": len(simpan), "foto_dikeluarkan": len(keluar),
             "dikeluarkan_semua_kotak_hilang_per_split_kelas": {sp: dict(Counter(k["class"] for k in keluar if k["split"] == sp and k["kotak_asli"])) for sp in SPL},
             "dikeluarkan_pangkasan_background_train": len(pangkas),
             "jumlah_per_split_kelas_v7": per_split_kelas(man), "jumlah_per_split_kelas_v9": per_split_kelas(simpan),
             "gambar_disk_sama_dengan_manifest": all(disk[sp] == {r["file"] for r in simpan if r["split"] == sp} for sp in SPL),
             "gambar_hilang": img_hilang, "cadangan_tidak_byte_identik_v7": cad_tidak_identik,
             "label_bukan_subset_baris_v7": label_bukan_subset,
             "background_train_persen": round(100 * sum(r["class"] == "background" for r in simpan if r["split"] == "train")
                                              / sum(r["split"] == "train" for r in simpan), 2),
             "labels_sha256_v9": labels_sha(V9)}
    hasil["LOLOS"] = (not beda_id and hasil["gambar_disk_sama_dengan_manifest"] and img_hilang == 0 and cad_tidak_identik == 0
                      and label_bukan_subset == 0 and hasil["background_train_persen"] <= 100 * MAKS_BG + 1e-9)
    (REP / "pemeriksaan_kesetaraan.json").write_text(json.dumps(hasil, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- audit masker (statistik; penilaian manusia ada di reports/v9_latar/audit_masker/)
    for d in kotak_stat.values():
        d["persen_dibuang"] = round(100 * d["dibuang"] / d["kotak"], 1)
    cad = Counter(f"{r['split']}|{r['source']}|{r['class']}" for r in simpan if r["latar_dihapus"] == "cadangan")
    tot = Counter(f"{r['split']}|{r['source']}|{r['class']}" for r in simpan)
    import statistics as st
    audit = {"kotak_dibuang_per_split_sumber_kelas": dict(sorted(kotak_stat.items())),
             "cadangan_per_split_sumber_kelas": {k: f"{cad[k]}/{tot[k]}" for k in sorted(tot)},
             "median_luas_masker_per_split_sumber_kelas": {k: round(st.median(v), 4) for k, v in sorted(luas_per.items())},
             "lembar_audit_manusia": "reports/v9_latar/audit_masker/lembar_audit.html (50 foto VAL di luar sampel tahap 0)",
             "LOLOS_statistik": True}
    (REP / "audit_masker.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(hasil, indent=1, ensure_ascii=False))
    sys.exit(0 if hasil["LOLOS"] else 1)


def bekukan():
    pk = json.loads((REP / "pemeriksaan_kesetaraan.json").read_text(encoding="utf-8"))
    au = json.loads((REP / "audit_masker.json").read_text(encoding="utf-8"))
    vd = (REP / "verify_dataset.log").read_text(encoding="utf-8", errors="replace")
    if not (pk["LOLOS"] and au.get("LOLOS_statistik") and "HASIL: LOLOS" in vd):
        sys.exit(f"[DITOLAK] pemeriksaan belum lolos: kesetaraan={pk['LOLOS']} audit={au.get('LOLOS_statistik')} verify={'HASIL: LOLOS' in vd}")
    now = datetime.now().isoformat(timespec="seconds")
    dy = yaml.safe_load((V9 / "data.yaml").read_text(encoding="utf-8"))
    dy["frozen"] = True
    (V9 / "data.yaml").write_text(yaml.safe_dump(dy, allow_unicode=True, sort_keys=False), encoding="utf-8")
    m = json.loads((V9 / "dataset_meta.json").read_text(encoding="utf-8"))
    m["frozen"] = True
    m["freeze"] = {"frozen": True, "frozen_at": now, "alasan_tidak_dibekukan": None, "manifest_sha256": sha(V9 / "manifest.csv"),
                   "data_yaml_sha256": sha(V9 / "data.yaml"), "labels_sha256": labels_sha(V9),
                   "pemeriksaan": {"kesetaraan": "reports/prov_v9/pemeriksaan_kesetaraan.json", "audit_masker": "reports/prov_v9/audit_masker.json",
                                   "verify_dataset": "reports/prov_v9/verify_dataset.log"}}
    (V9 / "dataset_meta.json").write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    (V9 / "FROZEN.txt").write_text(f"DIBEKUKAN {now}\nmanifest_sha256={m['freeze']['manifest_sha256']}\n"
                                   "Perubahan lanjutan harus menjadi versi baru.\n", encoding="utf-8")
    print(f"[OK] prov_v9 dibekukan {now}; manifest {m['freeze']['manifest_sha256']}")


if __name__ == "__main__":
    {"bangun": bangun, "bekukan": bekukan}[sys.argv[1]]()
