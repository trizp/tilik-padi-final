#!/usr/bin/env python3
"""INFERENSI UNTUK ILUSTRASI (bukan evaluasi): model final v7_final_prov_yolov8s_s0 (best.pt) dijalankan HANYA pada 4 foto test yang sudah
dipilih untuk gambar 9 bahan presentasi, dengan pengaturan yang sama dengan evaluasi resmi (scripts/evaluate.py: prediksi_teratas):
imgsz 640, conf 0,001, iou 0,7, device 0, ambang diagnosis terkunci dari runs/<run>/ambang_diagnosis.json.

Hasil (kotak prediksi) hanya disimpan sebagai bahan gambar di bahan_presentasi/_ilustrasi_prediksi_gambar9.json. Tidak ada berkas di runs/
atau reports/ yang ditulis atau diubah. Sebelum dipakai, kelas teratas, keyakinan teratas, dan diagnosis dibandingkan dengan CSV evaluasi
tersimpan; bila kelas/diagnosis berbeda atau keyakinan (dibulatkan 4 desimal) berbeda lebih dari 0,0001, skrip berhenti (exit 1) tanpa menulis apa pun. Toleransi 0,0001 = keputusan pemilik skripsi 2026-10-27, pilihan (a): selisih diduga karena ukuran batch GPU (evaluasi resmi batch 8, ilustrasi batch 1).

  .venv\\Scripts\\python.exe scripts\\presentasi\\ilustrasi_prediksi.py"""
import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "false")
ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/v7_final_prov_yolov8s_s0"
W = RUN / "weights/best.pt"
CSV_RESMI = RUN / "eval_test/diagnosis_bersih_per_citra.csv"
OUT = ROOT / "bahan_presentasi/_ilustrasi_prediksi_gambar9.json"
TOL = 1e-4 + 1e-9   # keputusan 2026-10-27 (a): selisih keyakinan maks. 0,0001 setelah dibulatkan 4 desimal seperti CSV resmi


def main():
    kunci = json.loads((RUN / "ambang_diagnosis.json").read_text(encoding="utf-8"))
    sha = hashlib.sha256(W.read_bytes()).hexdigest()
    if sha != kunci["weights_sha256"]:
        sys.exit(f"[BERHENTI] sha bobot {sha[:12]} tidak sama dengan ambang terkunci {kunci['weights_sha256'][:12]}")
    cat = (ROOT / "bahan_presentasi/_catatan_gambar_penelitian.md").read_text(encoding="utf-8")
    foto = re.findall(r"Contoh kualitatif \(9\): \w+ `([^`]+)`", cat)
    assert len(foto) == 4, foto
    resmi = {r["berkas"]: r for r in csv.DictReader(open(CSV_RESMI, encoding="utf-8"))}
    from ultralytics import YOLO
    model = YOLO(str(W))
    names = model.names
    imgsz = 640
    args = json.loads(json.dumps({"imgsz": imgsz, "conf": 0.001, "iou": 0.7, "device": 0}))
    hasil, beda = [], []
    for f in foto:
        p = ROOT / "data/processed/prov_v7/images/test" / f
        res = model.predict(str(p), verbose=False, save=False, **args)[0]
        cf = res.boxes.conf.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        xyxy = res.boxes.xyxy.cpu().numpy()
        k = int(cf.argmax()) if len(cf) else -1
        top_c, top_k = (float(cf[k]), names[int(cls[k])]) if k >= 0 else (0.0, "")
        diag = top_k if k >= 0 and top_c >= kunci["ambang"] else "sehat_atau_tidak_terdeteksi"
        r = resmi[f]
        cocok = (abs(round(top_c, 4) - float(r["conf_teratas"])) <= TOL and top_k == r["kelas_teratas"] and diag == r["diagnosis"])
        if not cocok:
            beda.append({"foto": f, "baru": [round(top_c, 4), top_k, diag], "resmi": [r["conf_teratas"], r["kelas_teratas"], r["diagnosis"]]})
        hasil.append({"foto": f, "conf_teratas": round(top_c, 6), "kelas_teratas": top_k, "diagnosis": diag,
                      "resmi": {"conf_teratas": r["conf_teratas"], "kelas_teratas": r["kelas_teratas"], "diagnosis": r["diagnosis"]},
                      "kotak_ge_ambang": [{"xyxy": [round(float(v), 2) for v in b], "kelas": names[int(c)], "conf": round(float(s), 4)}
                                          for b, c, s in zip(xyxy, cls, cf) if s >= kunci["ambang"]],
                      "ukuran": list(res.orig_shape)[::-1]})
    if beda:
        for b in beda:
            print("[BEDA]", b)
        sys.exit("[BERHENTI] diagnosis/keyakinan tidak sama dengan hasil tersimpan; tidak ada yang ditulis")
    OUT.write_text(json.dumps({"jenis": "INFERENSI UNTUK ILUSTRASI, BUKAN EVALUASI", "run": RUN.name, "weights_sha256": sha,
                               "pengaturan": {**args, "ambang_diagnosis": kunci["ambang"], "kotak_digambar": "conf >= ambang diagnosis"},
                               "dibanding": str(CSV_RESMI.relative_to(ROOT)), "toleransi_conf": TOL, "foto": hasil},
                              indent=1, ensure_ascii=False), encoding="utf-8")
    for h in hasil:
        print(h["foto"], "| baru", h["conf_teratas"], h["kelas_teratas"], h["diagnosis"], "| resmi", h["resmi"], "| kotak >= ambang:",
              len(h["kotak_ge_ambang"]))
    print("COCOK 4/4; ditulis", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
