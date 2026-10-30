#!/usr/bin/env python3
"""Bagian C bahan presentasi: semua tabel paper dan skripsi dalam CSV (bahan_presentasi/tabel/) serta angka_kunci.md (ID) dan
key_numbers.md (EN). Semua angka dibaca dari berkas hasil yang sudah ada; angka turunan baru ditandai [turunan].

  .venv\\Scripts\\python.exe scripts\\presentasi\\tabel_angka.py"""
import csv
import json
import re
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "bahan_presentasi"
TB = OUT / "tabel"
V7T = ROOT / "reports/final_eval_v7/tabel"
V9T = ROOT / "reports/final_eval_v9/tabel"
APP = ROOT / "app"
APV9 = ROOT / "reports/v9_latar/aplikasi_v9"


def baca(p):
    return list(csv.DictReader(open(p, encoding="utf-8-sig")))


def tulis(nama, rows, kolom=None):
    p = TB / nama
    p.parent.mkdir(parents=True, exist_ok=True)
    kolom = kolom or list(rows[0])
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=kolom, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def salin(src, nama):
    (TB / nama).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, TB / nama)


def v(s):
    return float(re.findall(r"[-+]?\d*\.\d+|\d+", s)[0])


def main():
    # ------------------------------------------------ tabel paper (I-VI)
    salin(ROOT / "paper/figures/tabel_data_prov_v7.csv", "paper/T1_dataset_foto_kelompok_prov_v7.csv")
    t10 = baca(V7T / "tabel10_ketepatan_perkiraan.csv")
    tulis("paper/T2_selisih_uji_vs_sesi_baru.csv", t10,
          ["Split latih", "Model", "Selisih mAP50-95", "Selisih akurasi", "mAP50-95 test tanpa BLB", "mAP50-95 sesi baru",
           "Akurasi test tanpa BLB", "Akurasi sesi baru"])
    t1 = baca(V7T / "tabel1_utama_per_model.csv")
    spd = {r["Model"]: r for r in baca(V7T / "tabel4_kecepatan.csv")}
    t3 = []
    for r in t1:
        if r["Set"] in ("Test bersih (tanpa BLB prov.)", "Test sesi baru"):
            t3.append({**r, **({k: spd[r["Model"]][k] for k in ("Parameter (juta)", "GFLOPs", "CPU ms (median)")}
                               if r["Set"].startswith("Test bersih") else {})})
    tulis("paper/T3_hasil_utama.csv", t3, list(t1[0]) + ["Parameter (juta)", "GFLOPs", "CPU ms (median)"])
    t8 = [r for r in baca(V7T / "tabel8_v8_vs_v11_berpasangan.csv") if r["Seed"] == "gabungan"
          and r["Set"] in ("Test bersih (tanpa BLB prov.)", "Test sesi baru") and r["Metrik"] in ("mAP50_95", "akurasi")]
    tulis("paper/T4_selisih_berpasangan_yolo11_yolov8.csv", t8)
    fb = baca(ROOT / "paper/figures/fig_background.csv")
    tulis("paper/T5_ketergantungan_latar_432_foto.csv", fb)
    tulis("paper/T6_ds3_uji_utama.csv", [r for r in baca(V9T / "tabel7_ds3.csv") if r["Set"].startswith("ds3 uji utama")])

    # ------------------------------------------------ tabel skripsi (hasil yang sudah ada, disalin apa adanya)
    for p in sorted(V7T.glob("*.csv")):
        salin(p, f"skripsi/v7_{p.name}")
    for p in sorted(V9T.glob("*.csv")):
        salin(p, f"skripsi/v9_{p.name}")
    for n in ("fase4_ambang_per_citra.csv", "fase5_alur_aplikasi_final.csv", "fase5_selisih_alur.csv"):
        salin(APV9 / n, f"skripsi/aplikasi_v9_{n}")
    m = json.loads((APV9 / "fase5_model_aplikasi_v9_vs_v7.json").read_text(encoding="utf-8"))
    tulis("skripsi/aplikasi_v9_fase5_model_aplikasi_v9_vs_v7.csv",
          [{"kondisi": k, **{a: b for a, b in d.items()}} for k, d in m.items() if isinstance(d, dict)],
          ["kondisi", "foto", "cadangan", "akurasi", "akurasi_didiagnosis", "recall_blast", "recall_blb", "mAP50", "mAP50_95",
           "disebut_BLB", "persen_BLB_dari_semua", "persen_BLB_dari_yang_didiagnosis"])
    bb = json.loads((APP / "docs/uji/blackbox.json").read_text(encoding="utf-8"))
    tulis("skripsi/aplikasi_blackbox.csv", bb["baris"], ["no", "masukan", "harapan", "aktual", "sesuai"])
    sd = json.loads((APP / "docs/uji/sumber_daya.json").read_text(encoding="utf-8"))
    tulis("skripsi/aplikasi_sumber_daya.csv", [{"ukuran": k, "nilai": x} for k, x in sd.items() if not isinstance(x, dict)])
    rb = yaml.safe_load((APP / "rbs/rules.yaml").read_text(encoding="utf-8"))
    baris = []
    for a in rb["aturan"]:
        syarat = " DAN ".join(f"{j['fakta']} {j['op']} {j['nilai']}" for j in a["jika"])
        for s in a.get("saran", []) or [None]:
            baris.append({"id": a["id"], "nama": a["nama"], "jika": syarat, "maka": json.dumps(a["maka"], ensure_ascii=False),
                          "kelompok": (s.get("kelompok") if isinstance(s, dict) else ("umum" if s else "")),
                          "saran": (s.get("teks") if isinstance(s, dict) else (s or "")), "sumber": a.get("sumber", "")})
    tulis("skripsi/rbs_aturan_lengkap.csv", baris)

    # ------------------------------------------------ angka kunci
    T1 = {(r["Set"], r["Model"]): r for r in t1}
    tb = "Test bersih (tanpa BLB prov.)"
    ns = "Test sesi baru"
    T8 = {(r["Pasangan"][5:6] if False else r["Pasangan"], r["Set"], r["Metrik"]): r for r in t8}
    t5k = {(r["Set"], r["Kontras"]): r for r in baca(V9T / "tabel5_kontras_latar.csv")}
    t2 = {(r["Set"], r["Kondisi"]): r for r in baca(V9T / "tabel2_selisih_yolo11s_yolov8s.csv")}
    t7 = {(r["Model"], r["Kondisi"]): r for r in baca(V9T / "tabel7_ds3.csv") if r["Set"].startswith("ds3 uji utama")}
    t5b = {r["Himpunan"]: r for r in baca(V7T / "tabel5b_kebocoran_split_acak.csv")}
    data = {r["Kelas"]: r for r in baca(ROOT / "paper/figures/tabel_data_prov_v7.csv")}
    tot = data["TOTAL (kelompok berbeda per split)"]
    au = {r["Kelompok"]: r for r in baca(V9T / "tabel8_audit_pemisahan.csv")}
    alur = {(r["Set"], r["Alur"]): r for r in baca(APV9 / "fase5_alur_aplikasi_final.csv")}
    pd = json.loads((APP / "docs/uji/pemisah_daun.json").read_text(encoding="utf-8"))
    cfg = yaml.safe_load((APP / "config.yaml").read_text(encoding="utf-8"))
    fbv = {(r["model"], r["train_test"], r["metrik"]): r for r in fb}

    def ci(r, k):
        return r[k]

    K = []   # (bagian, teks_id, teks_en, sumber)
    K.append(("Data", f"prov_v7: {int(tot['train foto']) + int(tot['val foto']) + int(tot['test foto']) + int(tot['test_new_session foto']):,} foto; "
              f"latih {tot['train foto']}, validasi {tot['val foto']}, uji {tot['test foto']} (434 tanpa BLB + 195 BLB), sesi baru "
              f"{tot['test_new_session foto']}; kelompok {tot['train kelompok']}/{tot['val kelompok']}/{tot['test kelompok']}/"
              f"{tot['test_new_session kelompok']}",
              f"prov_v7: {int(tot['train foto']) + int(tot['val foto']) + int(tot['test foto']) + int(tot['test_new_session foto']):,} images; "
              f"train {tot['train foto']}, val {tot['val foto']}, test {tot['test foto']} (434 without BLB + 195 BLB), new session "
              f"{tot['test_new_session foto']}; groups {tot['train kelompok']}/{tot['val kelompok']}/{tot['test kelompok']}/"
              f"{tot['test_new_session kelompok']}", "paper/figures/tabel_data_prov_v7.csv (manifest prov_v7)"))
    K.append(("Data", "RiceLeafDiseaseBD versi 3 (Mendeley, 1 Sep 2026); 931 foto ds4 diputar agar cocok dengan kotak (930 jpegtran + 1 tag EXIF)",
              "RiceLeafDiseaseBD version 3 (Mendeley, 1 Sep 2026); 931 ds4 images rotated to match the boxes (930 jpegtran + 1 EXIF tag)",
              "paper/CEK_VERSI_DS4.md; reports/prov_v7_prep/jpegtran_ringkasan.json"))
    K.append(("Kebocoran", f"Split acak: {t5b['split test']['Batas bawah (kembaran langsung)']}–{t5b['split test']['Batas atas (kelompok berantai)']} "
              "foto uji punya kembaran di data latih",
              f"Random split: {t5b['split test']['Batas bawah (kembaran langsung)']}–{t5b['split test']['Batas atas (kelompok berantai)']} "
              "of test images have a near-duplicate in training", "reports/final_eval_v7/tabel/tabel5b_kebocoran_split_acak.csv"))
    acak = [r for r in t10 if r["Split latih"].startswith("split acak")]
    bersih = [r for r in t10 if r["Split latih"].startswith("split bersih")]
    lo, hi = min(v(r["Selisih mAP50-95"]) for r in bersih), max(v(r["Selisih mAP50-95"]) for r in bersih)
    K.append(("Kebocoran", f"Selisih mAP50-95 uji − sesi baru: split acak {acak[0]['Selisih mAP50-95']} ({acak[0]['Model']}) dan "
              f"{acak[1]['Selisih mAP50-95']} ({acak[1]['Model']}); split bebas kebocoran {lo:.3f}–{hi:.3f}",
              f"Test − new-session mAP50-95 gap: random split {acak[0]['Selisih mAP50-95']} ({acak[0]['Model']}) and "
              f"{acak[1]['Selisih mAP50-95']} ({acak[1]['Model']}); leakage-free {lo:.3f}–{hi:.3f}",
              "reports/final_eval_v7/tabel/tabel10_ketepatan_perkiraan.csv"))
    for s, lab_id, lab_en in ((tb, "uji (434 foto)", "test (434 images)"), (ns, "sesi baru (299 foto)", "new session (299 images)")):
        isi = "; ".join(f"{mm} mAP50-95 {T1[(s, mm)]['mAP50-95']} {T1[(s, mm)]['IK95 mAP50-95']}, akurasi {T1[(s, mm)]['Akurasi diagnosis']} "
                        f"{T1[(s, mm)]['IK95 akurasi']}" for mm in ("YOLOv8n", "YOLO11n", "YOLOv8s", "YOLO11s"))
        K.append(("Hasil utama", f"{lab_id}: {isi}", f"{lab_en}: {isi.replace('akurasi', 'accuracy')}",
                  "reports/final_eval_v7/tabel/tabel1_utama_per_model.csv"))
    for p in ("YOLO11n − YOLOv8n", "YOLO11s − YOLOv8s"):
        isi = "; ".join(f"{'uji' if s == tb else 'sesi baru'} {'mAP50-95' if mm == 'mAP50_95' else 'akurasi'} {T8[(p, s, mm)]['Selisih']} "
                        f"{T8[(p, s, mm)]['IK95']}" for s in (tb, ns) for mm in ("mAP50_95", "akurasi"))
        K.append(("YOLO11 vs YOLOv8", f"{p}: {isi}", f"{p}: " + isi.replace("sesi baru", "new session").replace("uji", "test")
                  .replace("akurasi", "accuracy"), "reports/final_eval_v7/tabel/tabel8_v8_vs_v11_berpasangan.csv (gabungan)"))
    isi = "; ".join(f"{mm} {spd[mm]['Parameter (juta)']} M, {spd[mm]['GFLOPs']} GFLOPs, CPU {spd[mm]['CPU ms (median)']} ms"
                    for mm in ("YOLOv8n", "YOLO11n", "YOLOv8s", "YOLO11s"))
    K.append(("Kecepatan", isi, isi, "reports/final_eval_v7/tabel/tabel4_kecepatan.csv"))
    sens = t5k[("Test bersih (tanpa BLB)", "Kepekaan v7: tanpa − asli")]
    K.append(("Latar", f"Model asli, uji tanpa latar − uji asli (432 foto): perubahan recall Blast YOLOv8s {sens['YOLOv8s recall Blast']}, YOLO11s "
              f"{sens['YOLO11s recall Blast']}; selisih {sens['YOLO11s−YOLOv8s recall Blast']}; mAP50-95 {sens['YOLOv8s mAP50-95']} vs "
              f"{sens['YOLO11s mAP50-95']}",
              f"Original models, leaf-only test − original test (432 images): change in blast recall YOLOv8s {sens['YOLOv8s recall Blast']}, YOLO11s "
              f"{sens['YOLO11s recall Blast']}; difference {sens['YOLO11s−YOLOv8s recall Blast']}; mAP50-95 {sens['YOLOv8s mAP50-95']} vs "
              f"{sens['YOLO11s mAP50-95']}", "reports/final_eval_v9/tabel/tabel5_kontras_latar.csv"))
    tl = t2[("Test bersih (tanpa BLB)", "Tanpa latar (v9, foto tanpa latar)")]
    K.append(("Latar", f"Dilatih tanpa latar, YOLO11s − YOLOv8s: mAP50-95 {tl['Selisih mAP50-95']}, akurasi {tl['Selisih akurasi']} "
              f"(memuat nol), recall Blast {tl['Selisih recall Blast']}",
              f"Trained leaf-only, YOLO11s − YOLOv8s: mAP50-95 {tl['Selisih mAP50-95']}, accuracy {tl['Selisih akurasi']} "
              f"(includes 0), blast recall {tl['Selisih recall Blast']}", "reports/final_eval_v9/tabel/tabel2_selisih_yolo11s_yolov8s.csv"))
    sel = "; ".join(f"{mm} {k}: mAP50-95 {float(fbv[(mm, k, 'mAP50-95')]['rata2_3seed']):.3f}, akurasi "
                    f"{float(fbv[(mm, k, 'Diagnosis accuracy')]['rata2_3seed']):.3f}, recall Blast {float(fbv[(mm, k, 'Blast recall')]['rata2_3seed']):.3f}"
                    for mm in ("YOLOv8s", "YOLO11s") for k in ("Leaf/Orig",))
    K.append(("Latar", f"Sel Daun/Asli Tabel V (sama dengan main.tex final 27 Okt): {sel}",
              f"Leaf/Orig cells of Table V (same as final main.tex, 27 Oct): {sel.replace('akurasi', 'accuracy').replace('recall Blast', 'blast recall')}",
              "paper/figures/fig_background.csv = reports/final_eval_v9/tabel/tabel4_sel_silang.csv"))
    K.append(("Latar", f"Audit masker: ds4 {au['ds4 (Blast, Brown spot, sehat)']['Baik']}/{au['ds4 (Blast, Brown spot, sehat)']['n']} baik "
              f"({au['ds4 (Blast, Brown spot, sehat)']['% baik [IK95 Wilson]']}); BLB {au['ds1 BLB provisional']['Baik']}/{au['ds1 BLB provisional']['n']}",
              f"Mask audit: ds4 {au['ds4 (Blast, Brown spot, sehat)']['Baik']}/{au['ds4 (Blast, Brown spot, sehat)']['n']} correct "
              f"({au['ds4 (Blast, Brown spot, sehat)']['% baik [IK95 Wilson]']}); BLB {au['ds1 BLB provisional']['Baik']}/{au['ds1 BLB provisional']['n']}",
              "reports/final_eval_v9/tabel/tabel8_audit_pemisahan.csv"))
    for mm in ("YOLOv8s", "YOLO11s"):
        a, b = t7[(mm, "v7 pada ds3 asli")], t7[(mm, "v9 pada ds3 tanpa latar")]
        d = t7[(mm, "Selisih v9 − v7")]
        K.append(("ds3", f"{mm} pada ds3 (70 foto): akurasi {a['Akurasi']} → {b['Akurasi']} (selisih {d['Akurasi']} {d['IK95 akurasi']}); "
                  f"recall Blast {a['R Blast']} → {b['R Blast']}",
                  f"{mm} on ds3 (70 images): accuracy {a['Akurasi']} → {b['Akurasi']} (difference {d['Akurasi']} {d['IK95 akurasi']}); "
                  f"blast recall {a['R Blast']} → {b['R Blast']}", "reports/final_eval_v9/tabel/tabel7_ds3.csv"))
    K.append(("ds3", "Verifikasi label ds3: dua penilai non-pakar, kappa 0.431; 70 foto diterima keduanya",
              "ds3 label check: two non-expert raters, kappa 0.431; 70 images accepted by both", "reports/final_eval_v7/ds3/RINGKASAN_DS3.md"))
    K.append(("Aplikasi", f"Alur v9: pemisah daun YOLOv8n-seg (tau {cfg['pemisah_daun']['tau']}) + YOLOv8s v9; ambang terdeteksi "
              f"{cfg['ambang']['terdeteksi']}, batas kemungkinan {cfg["ambang"]["rendah"]:.2f}",
              f"App pipeline v9: YOLOv8n-seg leaf separator (tau {cfg['pemisah_daun']['tau']}) + YOLOv8s v9; detected threshold "
              f"{cfg['ambang']['terdeteksi']}, possible from {cfg["ambang"]["rendah"]:.2f}", "app/config.yaml"))
    for s in ("Test bersih (tanpa BLB)", "Test sesi baru", "Test BLB provisional (terpisah)"):
        r = alur[(s, "Alur aplikasi (YOLOv8n-seg)")]
        K.append(("Aplikasi", f"Evaluasi alur aplikasi, {s} ({r['Foto / kelompok']}): akurasi {r['Akurasi diagnosis']} {r['IK95 Akurasi diagnosis']}; "
                  f"foto ulang {r['% cadangan (foto ulang)']}; akurasi pada foto yang didiagnosis {r['Akurasi pada foto yang didiagnosis']}",
                  f"App-pipeline evaluation, {s} ({r['Foto / kelompok']}): accuracy {r['Akurasi diagnosis']} {r['IK95 Akurasi diagnosis']}; "
                  f"retake requested {r['% cadangan (foto ulang)']}; accuracy on diagnosed photos {r['Akurasi pada foto yang didiagnosis']}",
                  "reports/v9_latar/aplikasi_v9/fase5_alur_aplikasi_final.csv"))
    K.append(("Aplikasi", f"RAM diam {sd['rss_diam_setelah_model_dimuat_MB']} MB, puncak {sd['rss_puncak_saat_memproses_MB']} MB; waktu inti "
              f"{sd['waktu_inti_ms_median_20_val']} ms, ujung ke ujung {sd['waktu_ujung_ke_ujung_ms_median_val']} ms; unit test 79 lulus; "
              f"black-box {bb['foto_sesuai']}/10, {bb['khusus_sesuai']}/4, {bb['fitur_sesuai']}/{bb['fitur_n']}; pemisah daun setara "
              f"{pd['keputusan_cadangan_sama']}/{pd['foto_val']}",
              f"RAM idle {sd['rss_diam_setelah_model_dimuat_MB']} MB, peak {sd['rss_puncak_saat_memproses_MB']} MB; core time "
              f"{sd['waktu_inti_ms_median_20_val']} ms, end to end {sd['waktu_ujung_ke_ujung_ms_median_val']} ms; 79 unit tests pass; "
              f"black-box {bb['foto_sesuai']}/10, {bb['khusus_sesuai']}/4, {bb['fitur_sesuai']}/{bb['fitur_n']}; leaf separator agreement "
              f"{pd['keputusan_cadangan_sama']}/{pd['foto_val']}", "app/docs/uji/*.json"))
    K.append(("Kesalahan diagnosis", "[turunan] Daun sehat didiagnosis Brown spot (uji, 3 seed digabung): YOLOv8s 115/474 = 24,3%, "
              "YOLO11s 112/474 = 23,6%",
              "[derived] Healthy leaves diagnosed as brown spot (test, 3 seeds pooled): YOLOv8s 115/474 = 24.3%, YOLO11s 112/474 = 23.6%",
              "jumlah runs/v7_final_prov_{yolov8s,yolo11s}_s{0,1,2}/eval_test/diagnosis_dengan_background_confusion.csv"))

    for bahasa, nama, judul, kol in (("id", "angka_kunci.md", "Angka kunci paper dan skripsi", ("Bagian", "Angka", "Sumber")),
                                     ("en", "key_numbers.md", "Key numbers (paper and thesis)", ("Section", "Number", "Source"))):
        bag = {"Data": "Data", "Kebocoran": "Leakage", "Hasil utama": "Main results", "YOLO11 vs YOLOv8": "YOLO11 vs YOLOv8",
               "Kecepatan": "Speed", "Latar": "Background", "ds3": "ds3 (external)", "Aplikasi": "App", "Kesalahan diagnosis": "Diagnosis errors"}
        md = [f"# {judul}", "",
              ("Semua angka dibaca dari berkas hasil yang sudah ada (dibuat otomatis oleh `scripts/presentasi/tabel_angka.py`). "
               "IK = interval kepercayaan 95% bootstrap per kelompok foto. Angka yang dihitung baru dari keluaran tersimpan ditandai "
               "[turunan]." if bahasa == "id" else
               "All numbers are read from existing result files (generated by `scripts/presentasi/tabel_angka.py`). CI = 95% bootstrap "
               "confidence interval by image group. Numbers newly computed from stored outputs are marked [derived]."), "",
              f"| {kol[0]} | {kol[1]} | {kol[2]} |", "|---|---|---|"]
        for b_, ti, te, s in K:
            teks = ti if bahasa == "id" else te
            if bahasa == "id":                                   # ribuan titik, desimal koma
                teks = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "§", teks)
                teks = re.sub(r"(?<=\d)\.(?=\d)", ",", teks).replace("§", ".")
            md.append(f"| {b_ if bahasa == 'id' else bag[b_]} | {teks} | `{s}` |")
        (OUT / nama).write_text("\n".join(md) + "\n", encoding="utf-8")

    # nama asli foto ds4 yang dipakai (untuk atribusi)
    man = {r["file"]: r["orig_path"].split("Original images/")[-1] for r in csv.DictReader(open(ROOT / "data/processed/prov_v7/manifest.csv", encoding="utf-8"))}
    man.update({r["file"]: r["orig_path"].split("Original images/")[-1]
                for r in csv.DictReader(open(ROOT / "data/processed/prov_v7_randsplit/manifest.csv", encoding="utf-8"))})
    cat = (OUT / "_catatan_gambar_penelitian.md").read_text(encoding="utf-8")
    pakai = sorted(set(re.findall(r"`(ds4_[^`]+\.jpg)`", cat)) | {r["terpilih"] for r in baca(ROOT / "paper/figures/fig_leaf_examples.csv")})
    print("FOTO DS4:", [(f, man.get(f)) for f in pakai])
    print("tabel:", len(list(TB.rglob("*.csv"))), "| angka kunci:", len(K))


if __name__ == "__main__":
    main()
