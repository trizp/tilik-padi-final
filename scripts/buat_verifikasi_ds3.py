#!/usr/bin/env python3
"""
buat_verifikasi_ds3.py - Lembar verifikasi label ds3 (blight dan blast, daftar BERSIH) untuk dinilai manusia.
TIDAK menjalankan model apa pun; foto hanya disalin dan diperkecil untuk ditanam di Excel.

Keluaran (data/annotation/verifikasi_ds3/):
  verifikasi_ds3.xlsx           - berkas utama (penilai 1): Petunjuk, Verifikasi, Kriteria, Ringkasan, Impor_Penilai_2
  verifikasi_ds3_penilai2.xlsx  - salinan BUTA untuk penilai 2: urutan acak, label asli dan nama berkas disembunyikan,
                                  penilai 2 memilih diagnosis sendiri (BLB / Blast / bukan keduanya / ragu)
  foto_penuh/                   - foto ukuran asli, nama <no>_<label>_<berkas> (untuk penilai 1)
  foto_penuh_penilai2/          - foto ukuran asli, nama <kode>.jpg (tanpa label, untuk penilai 2)
  kunci_kode_penilai2.csv       - pemetaan kode -> berkas/label (JANGAN diberikan ke penilai 2)
"""
import argparse
import csv
import io
import random
import shutil
from datetime import datetime
from pathlib import Path

from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
SEED = 20261008
THUMB_H = 520                       # piksel; tinggi baris Excel maks 409 pt (~545 px)
NILAI = ["sesuai", "tidak sesuai", "ragu"]
DIAG = ["BLB", "Blast", "bukan keduanya", "ragu"]
HARAP = {"blight": "BLB", "blast": "Blast"}   # label asli ds3 -> diagnosis yang dianggap "sesuai"

F = "Arial"
H_FILL = PatternFill("solid", fgColor="1F3864")
IN_FILL = PatternFill("solid", fgColor="FFF2CC")     # sel yang diisi penilai
FX_FILL = PatternFill("solid", fgColor="EDEDED")     # sel rumus (jangan diisi)
EX_FILL = PatternFill("solid", fgColor="E2EFDA")     # contoh
THIN = Side(style="thin", color="BFBFBF")
BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def hdr(ws, row, values, widths=None):
    for j, v in enumerate(values, 1):
        c = ws.cell(row, j, v)
        c.font = Font(name=F, bold=True, color="FFFFFF")
        c.fill = H_FILL
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = BOX
    if widths:
        for j, w in enumerate(widths, 1):
            ws.column_dimensions[get_column_letter(j)].width = w


def tulis(ws, row, col, v, bold=False, fill=None, wrap=True, size=10, color=None, align="left"):
    c = ws.cell(row, col, v)
    c.font = Font(name=F, bold=bold, size=size, color=color)
    c.alignment = Alignment(wrap_text=wrap, vertical="top", horizontal=align)
    if fill:
        c.fill = fill
    return c


def thumb(path):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    s = THUMB_H / im.height
    im = im.resize((round(im.width * s), THUMB_H), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    buf.seek(0)
    return buf, im.width, im.height


def tanam(ws, path, cell):
    buf, w, h = thumb(path)
    img = XLImage(buf)
    img.width, img.height = w, h
    ws.add_image(img, cell)
    return w, h


ATURAN_P1 = ["sesuai = ciri penyakit sesuai label (lihat sheet Kriteria) terlihat jelas pada foto.",
             "tidak sesuai = terlihat jelas ciri penyakit lain, atau tidak tampak ciri penyakit sesuai label.",
             "ragu = foto buram, gejala terlalu kecil/awal, atau ciri campuran sehingga tidak bisa diputuskan.",
             "Nilai hanya dari foto; jangan mencari informasi lain tentang foto tersebut."]
ATURAN_P2 = ["BLB = ciri BLB (lihat sheet Kriteria) terlihat jelas pada foto.",
             "Blast = ciri Blast terlihat jelas pada foto.",
             "bukan keduanya = terlihat jelas ciri penyakit lain (lihat sheet Kriteria), atau tidak tampak ciri BLB maupun Blast.",
             "ragu = foto buram, gejala terlalu kecil/awal, atau ciri campuran sehingga tidak bisa diputuskan.",
             "Nilai hanya dari foto; jangan mencari informasi lain tentang foto tersebut."]


def kriteria(ws):
    """Sheet Kriteria: HANYA ciri yang tertulis di bab 2.2 proposal (BAB2_Kajian_Pustaka.docx), bahasa sederhana.
    Isi sheet ini sama persis di berkas penilai 1 dan penilai 2."""
    ws.column_dimensions["A"].width = 24
    ws.column_dimensions["B"].width = 80
    ws.column_dimensions["C"].width = 36
    tulis(ws, 1, 1, "Kriteria visual (hanya dari bab 2.2 proposal)", bold=True, size=13, wrap=False)
    tulis(ws, 2, 1, "Sumber yang dikutip bab 2.2: Semangun (2008); Kementerian Pertanian RI (2022); Yasid dkk. (2026). "
                    "Ciri di luar bab 2.2 tidak dipakai.", wrap=True)
    ws.merge_cells("A2:C2")
    ws.row_dimensions[2].height = 30
    hdr(ws, 4, ["Kelompok", "Ciri yang dilihat pada foto", "Sumber (bab 2.2)"])
    rows = [
        ("BLB (hawar daun bakteri)", "Awalnya bercak kebasahan (tampak basah) pada tepi daun.", "Semangun (2008); Kementan (2022) — bab 2.2.4"),
        ("BLB (hawar daun bakteri)", "Bercak meluas menjadi kuning kecokelatan dari ujung hingga pangkal daun.", "Semangun (2008); Kementan (2022) — bab 2.2.4"),
        ("BLB (hawar daun bakteri)", "Pada serangan parah, seluruh helai daun dapat mengering.", "Semangun (2008); Kementan (2022) — bab 2.2.4"),
        ("Blast (blas daun)", "Bercak berbentuk belah ketupat (wajik) pada permukaan daun.", "Semangun (2008); Yasid dkk. (2026) — bab 2.2.2"),
        ("Blast (blas daun)", "Bercak berwarna abu-abu atau putih, dikelilingi tepi berwarna cokelat.", "Semangun (2008); Yasid dkk. (2026) — bab 2.2.2"),
        ("Blast (blas daun)", "Warna kuning pucat sering muncul di sekitar bercak; bercak terus membesar.", "Semangun (2008); Yasid dkk. (2026) — bab 2.2.2"),
        ("Bukan keduanya: bercak cokelat (brown spot)", "Bercak oval kecil berwarna cokelat tua dengan tepi yang lebih muda.", "Semangun (2008); Yasid dkk. (2026) — bab 2.2.3"),
        ("Bukan keduanya: tungro", "Daun berubah kuning hingga jingga dari pucuk ke arah pangkal; pertumbuhan tanaman terhambat; "
                                  "jumlah anakan dan malai berkurang.", "Semangun (2008) — bab 2.2.5"),
    ]
    r = 5
    for k, c, s_ in rows:
        tulis(ws, r, 1, k, bold=True)
        tulis(ws, r, 2, c)
        tulis(ws, r, 3, s_)
        ws.row_dimensions[r].height = 32
        r += 1


def petunjuk_utama(pt, n, n_blight, n_blast):
    pt.column_dimensions["A"].width = 3
    pt.column_dimensions["B"].width = 110
    baris = [
        ("Lembar verifikasi label ds3 (blight dan blast)", True),
        (f"Isi: {n} foto ds3 BERSIH (tanpa kembar di ds1/ds2/ds4): {n_blight} blight, {n_blast} blast. "
         "Model TIDAK pernah dijalankan pada foto ini.", False),
        ("", False),
        ("Cara mengisi (penilai 1)", True),
        ("1. Buka sheet Verifikasi. Lihat foto di kolom D (klik nama berkas di kolom B untuk membuka foto ukuran penuh).", False),
        ("2. Baca sheet Kriteria, lalu pilih nilai di kolom E (dropdown: sesuai / tidak sesuai / ragu). Tulis alasan singkat di kolom F bila perlu.", False),
        ("3. Sel KUNING = diisi penilai. Sel ABU-ABU = rumus otomatis, jangan diubah.", False),
        ("", False),
        ("Aturan memilih nilai (penilai 1)", True),
    ] + [("• " + t, False) for t in ATURAN_P1] + [
        ("", False),
        ("Penilai 2 (buta)", True),
        ("4. Berikan HANYA berkas verifikasi_ds3_penilai2.xlsx dan folder foto_penuh_penilai2 ke penilai 2 (JANGAN berkas ini dan kunci_kode_penilai2.csv).", False),
        ("5. Penilai 2 tidak melihat label, jadi ia memilih diagnosisnya sendiri: BLB / Blast / bukan keduanya / ragu.", False),
        ("6. Setelah selesai: salin kolom B:D (kode, diagnosis_penilai_2, catatan_2) dari berkas penilai 2 ke sheet Impor_Penilai_2 kolom B:D di berkas ini.", False),
        ("7. Kolom G (nilai_penilai_2) terisi otomatis: diagnosis sama dengan label asli = sesuai; ragu = ragu; selain itu (termasuk 'bukan keduanya') = tidak sesuai.", False),
        ("8. Sheet Ringkasan menghitung otomatis jumlah per nilai per kelas dan Cohen's kappa (hanya baris yang sudah dinilai kedua penilai).", False),
        ("", False),
        ("Pemakaian hasil (ditetapkan sebelum model dijalankan pada ds3; lihat KEPUTUSAN_PROYEK.md)", True),
        ("• Uji eksternal utama = foto yang dinilai 'sesuai' oleh KEDUA penilai.", False),
        ("• Analisis tambahan = semua foto dengan label asli ds3.", False),
        ("• Foto 'ragu' tidak masuk uji utama; jumlahnya dilaporkan.", False),
        ("", False),
        ("Contoh pengisian (TIDAK dihitung di Ringkasan)", True),
    ]
    for r, (t, b) in enumerate(baris, 1):
        tulis(pt, r, 2, t, bold=b, size=13 if r == 1 else 10)
    r0 = len(baris) + 1
    contoh = [["no", "nama berkas", "label asli ds3", "nilai_penilai_1", "catatan_1", "diagnosis penilai 2 (buta)", "nilai_penilai_2 (otomatis)"],
              [999, "blight/IMG_XXXX.jpg", "blight", "sesuai", "kuning kecokelatan dari ujung daun", "BLB", "sesuai"],
              [998, "blast/IMG_YYYY.jpg", "blast", "ragu", "bercak sangat kecil, foto agak buram", "bukan keduanya", "tidak sesuai"]]
    for i, rw in enumerate(contoh):
        for j, v in enumerate(rw):
            c = pt.cell(r0 + i, 2 + j, v)
            c.font = Font(name=F, bold=(i == 0), size=9)
            c.fill = EX_FILL
            c.border = BOX
            c.alignment = Alignment(wrap_text=True, vertical="top")
    for j in range(3, 9):
        pt.column_dimensions[get_column_letter(j)].width = 18
    tulis(pt, r0 + 4, 2, f"Urutan acak penilai 2 dibuat dengan seed {SEED} (skrip scripts/buat_verifikasi_ds3.py).", size=9, color="595959")


def petunjuk_p2(p2, n):
    """Petunjuk penilai 2: TANPA label, nama berkas asli, atau jumlah per kelas."""
    p2.column_dimensions["A"].width = 3
    p2.column_dimensions["B"].width = 110
    baris = [
        ("Lembar penilaian foto daun padi (penilai 2)", True),
        (f"Berisi {n} foto dalam urutan acak. Untuk tiap foto, pilih diagnosis berdasarkan apa yang terlihat di foto.", False),
        ("", False),
        ("Cara mengisi", True),
        ("1. Buka sheet Penilaian. Klik kode di kolom B untuk membuka foto ukuran penuh (folder foto_penuh_penilai2 harus di samping berkas ini).", False),
        ("2. Baca sheet Kriteria. Pilih di kolom C: BLB / Blast / bukan keduanya / ragu. Tulis alasan singkat di kolom D bila perlu.", False),
        ("3. Sel KUNING = diisi. Jangan berdiskusi dengan penilai lain sebelum selesai.", False),
        ("", False),
        ("Aturan memilih diagnosis", True),
    ] + [("• " + t, False) for t in ATURAN_P2] + [
        ("", False),
        ("Contoh (tidak dihitung): P2-999 | BLB | kuning kecokelatan menjalar dari ujung daun", False),
    ]
    for r, (t, b) in enumerate(baris, 1):
        tulis(p2, r, 2, t, bold=b, size=13 if r == 1 else 10, fill=EX_FILL if t.startswith("Contoh") else None)


def teks_saja(path, n, n_blight, n_blast):
    """Tulis HANYA sheet teks (Petunjuk utama, Petunjuk penilai 2, Kriteria) ke berkas sementara, untuk disalin
    ke berkas yang sudah ada tanpa menyentuh isian (lihat KEPUTUSAN: jangan jalankan ulang bila sudah diisi)."""
    wb = Workbook()
    petunjuk_utama(wb.active, n, n_blight, n_blast)
    wb.active.title = "Petunjuk_utama"
    petunjuk_p2(wb.create_sheet("Petunjuk_p2"), n)
    kriteria(wb.create_sheet("Kriteria"))
    wb.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--daftar-dir", default="reports/prov_v6/ds3_eksternal")
    ap.add_argument("--out", default="data/annotation/verifikasi_ds3")
    ap.add_argument("--teks-saja", default=None, metavar="XLSX", help="hanya tulis sheet teks ke berkas sementara (tidak menyentuh berkas penilaian)")
    ap.add_argument("--timpa", action="store_true", help="izinkan menimpa berkas penilaian yang sudah ada (ISIAN HILANG)")
    a = ap.parse_args()
    out = ROOT / a.out
    if not a.teks_saja and not a.timpa and ((out / "verifikasi_ds3.xlsx").exists() or (out / "verifikasi_ds3_penilai2.xlsx").exists()):
        raise SystemExit("[DITOLAK] berkas penilaian sudah ada; menjalankan ulang akan MENGHAPUS isian. "
                         "Pakai --teks-saja untuk memperbarui teks, atau --timpa bila memang ingin membuat ulang.")
    out.mkdir(parents=True, exist_ok=True)

    item = []
    for kelas in ("blight", "blast"):
        for ln in (ROOT / a.daftar_dir / f"ds3_bersih_{kelas}.txt").read_text(encoding="utf-8").splitlines():
            if ln.strip():
                p, g = ln.split("\t")
                item.append({"path": ROOT / p, "rel": p, "kelas": kelas, "group": g.strip()})
    item.sort(key=lambda x: (x["kelas"] != "blight", x["path"].name))
    for i, x in enumerate(item, 1):
        x["no"] = i
    acak = item[:]
    random.Random(SEED).shuffle(acak)
    for i, x in enumerate(acak, 1):
        x["kode"] = f"P2-{i:03d}"
        x["urut2"] = i
    n = len(item)
    last = n + 1                                   # baris data terakhir (data mulai baris 2)
    n_blight, n_blast = sum(x["kelas"] == "blight" for x in item), sum(x["kelas"] == "blast" for x in item)
    if a.teks_saja:
        teks_saja(a.teks_saja, n, n_blight, n_blast)
        print(f"[OK] sheet teks -> {a.teks_saja}")
        return

    # ---- folder foto ukuran penuh
    d1, d2 = out / "foto_penuh", out / "foto_penuh_penilai2"
    for d in (d1, d2):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir()
    for x in item:
        x["f1"] = f"{x['no']:03d}_{x['kelas']}_{x['path'].name}"
        shutil.copy2(x["path"], d1 / x["f1"])
        shutil.copyfile(x["path"], d2 / f"{x['kode']}{x['path'].suffix.lower()}")   # copyfile: tanpa cap waktu asli
    with open(out / "kunci_kode_penilai2.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["kode", "no", "berkas", "label_asli_ds3", "group_id"])
        for x in sorted(item, key=lambda x: x["kode"]):
            w.writerow([x["kode"], x["no"], x["rel"], x["kelas"], x["group"]])

    # ================= berkas utama =================
    wb = Workbook()
    pt = wb.active
    pt.title = "Petunjuk"
    petunjuk_utama(pt, n, n_blight, n_blast)

    ws = wb.create_sheet("Verifikasi")
    kol = ["no", "nama berkas", "label asli ds3", "foto", "nilai_penilai_1", "catatan_1", "nilai_penilai_2", "catatan_2",
           "group_id", "kode_penilai2", "diagnosis_buta_penilai_2"]
    hdr(ws, 1, kol, [6, 24, 12, 58, 16, 34, 16, 34, 10, 13, 18])
    ws.freeze_panes = "E2"
    dv = DataValidation(type="list", formula1='"' + ",".join(NILAI) + '"', allow_blank=True,
                        error="Pilih: sesuai, tidak sesuai, atau ragu", errorTitle="Nilai tidak valid", showErrorMessage=True)
    ws.add_data_validation(dv)
    dv.add(f"E2:E{last}")
    for x in item:
        r = x["no"] + 1
        tulis(ws, r, 1, x["no"], align="center")
        c = tulis(ws, r, 2, f"{x['kelas']}/{x['path'].name}")
        c.hyperlink = f"foto_penuh/{x['f1']}"
        c.font = Font(name=F, size=10, color="0563C1", underline="single")
        tulis(ws, r, 3, x["kelas"], align="center")
        w, h = tanam(ws, x["path"], f"D{r}")
        ws.row_dimensions[r].height = h * 0.75 + 6
        for col in (5, 6):
            ws.cell(r, col).fill = IN_FILL
            ws.cell(r, col).alignment = Alignment(wrap_text=True, vertical="top")
            ws.cell(r, col).font = Font(name=F, size=10)
        tulis(ws, r, 9, x["group"], align="center")
        tulis(ws, r, 10, x["kode"], align="center")
        # diagnosis buta penilai 2 diambil dari sheet Impor_Penilai_2 berdasarkan kode
        ws.cell(r, 11, f'=IFERROR(INDEX(Impor_Penilai_2!$C$2:$C${last},MATCH($J{r},Impor_Penilai_2!$B$2:$B${last},0))&"","")')
        harap = HARAP[x["kelas"]]
        ws.cell(r, 7, f'=IF($K{r}="","",IF($K{r}="ragu","ragu",IF($K{r}="{harap}","sesuai","tidak sesuai")))')
        ws.cell(r, 8, f'=IFERROR(INDEX(Impor_Penilai_2!$D$2:$D${last},MATCH($J{r},Impor_Penilai_2!$B$2:$B${last},0))&"","")')
        for col in (7, 8, 11):
            ws.cell(r, col).fill = FX_FILL
            ws.cell(r, col).font = Font(name=F, size=10)
            ws.cell(r, col).alignment = Alignment(wrap_text=True, vertical="top")
        for col in range(1, 12):
            ws.cell(r, col).border = BOX

    kriteria(wb.create_sheet("Kriteria"))

    # ---- Ringkasan
    rs = wb.create_sheet("Ringkasan")
    rs.column_dimensions["A"].width = 26
    for j in range(2, 9):
        rs.column_dimensions[get_column_letter(j)].width = 15
    V = f"Verifikasi!$C$2:$C${last}"
    P1, P2, DB = f"Verifikasi!$E$2:$E${last}", f"Verifikasi!$G$2:$G${last}", f"Verifikasi!$K$2:$K${last}"
    tulis(rs, 1, 1, "Ringkasan verifikasi label ds3 (terhitung otomatis)", bold=True, size=13, wrap=False)
    r = 3
    for judul, rng in (("Penilai 1", P1), ("Penilai 2 (dari diagnosis buta)", P2)):
        tulis(rs, r, 1, judul, bold=True, wrap=False)
        r += 1
        hdr(rs, r, ["kelas"] + NILAI + ["belum diisi", "jumlah", "% sesuai (dari yang dinilai)"])
        r += 1
        top = r
        for kelas in ("blight", "blast"):
            tulis(rs, r, 1, kelas)
            for j, v in enumerate(NILAI, 2):
                rs.cell(r, j, f'=COUNTIFS({V},$A{r},{rng},"{v}")')
            rs.cell(r, 5, f"=F{r}-SUM(B{r}:D{r})")
            rs.cell(r, 6, f'=COUNTIFS({V},$A{r})')
            rs.cell(r, 7, f'=IF(SUM(B{r}:D{r})=0,"",B{r}/SUM(B{r}:D{r}))')
            r += 1
        tulis(rs, r, 1, "total", bold=True)
        for j in range(2, 7):
            col = get_column_letter(j)
            rs.cell(r, j, f"=SUM({col}{top}:{col}{r - 1})").font = Font(name=F, bold=True)
        rs.cell(r, 7, f'=IF(SUM(B{r}:D{r})=0,"",B{r}/SUM(B{r}:D{r}))')
        for rr in range(top, r + 1):
            rs.cell(rr, 7).number_format = "0.0%"
        r += 2

    tulis(rs, r, 1, "Diagnosis buta penilai 2 per label asli", bold=True, wrap=False)
    r += 1
    hdr(rs, r, ["label asli"] + DIAG + ["belum diisi"])
    r += 1
    for kelas in ("blight", "blast"):
        tulis(rs, r, 1, kelas)
        for j, v in enumerate(DIAG, 2):
            rs.cell(r, j, f'=COUNTIFS({V},$A{r},{DB},"{v}")')
        rs.cell(r, 6, f'=COUNTIFS({V},$A{r})-SUM(B{r}:E{r})')
        r += 1
    r += 1

    tulis(rs, r, 1, "Kesepakatan antar penilai (Cohen's kappa)", bold=True, wrap=False)
    r += 1
    tulis(rs, r, 1, "Hanya baris yang sudah dinilai KEDUA penilai. Baris = penilai 1, kolom = penilai 2. "
                    "kappa = (po - pe) / (1 - pe); po = proporsi sama; pe = kesepakatan karena kebetulan.", size=9, color="595959")
    rs.merge_cells(start_row=r, start_column=1, end_row=r, end_column=8)
    rs.row_dimensions[r].height = 26
    r += 2
    hasil_kappa = []
    for judul, kelas in (("Semua", None), ("blight", "blight"), ("blast", "blast")):
        tulis(rs, r, 1, f"Matriks: {judul}", bold=True, wrap=False)
        r += 1
        hdr(rs, r, ["P1 \\ P2"] + NILAI + ["jumlah"])
        r += 1
        m0 = r
        filt = f",{V},\"{kelas}\"" if kelas else ""
        for i, v1 in enumerate(NILAI):
            tulis(rs, r, 1, v1)
            for j, v2 in enumerate(NILAI, 2):
                rs.cell(r, j, f'=COUNTIFS({P1},"{v1}",{P2},"{v2}"{filt})')
            rs.cell(r, 5, f"=SUM(B{r}:D{r})")
            r += 1
        tulis(rs, r, 1, "jumlah", bold=True)
        for j in range(2, 6):
            col = get_column_letter(j)
            rs.cell(r, j, f"=SUM({col}{m0}:{col}{r - 1})")
        sr = r
        r += 1
        lab = [("n (dinilai keduanya)", f"=E{sr}"),
               ("po (kesepakatan teramati)", f'=IF(E{sr}=0,"",(B{m0}+C{m0 + 1}+D{m0 + 2})/E{sr})'),
               ("pe (kesepakatan kebetulan)", f'=IF(E{sr}=0,"",(E{m0}*B{sr}+E{m0 + 1}*C{sr}+E{m0 + 2}*D{sr})/E{sr}^2)'),
               ("Cohen's kappa", f'=IF(OR(E{sr}=0,B{r + 2}=1),"",(B{r + 1}-B{r + 2})/(1-B{r + 2}))')]
        for k, (t, fml) in enumerate(lab):
            tulis(rs, r, 1, t, bold=(k == 3))
            c = rs.cell(r, 2, fml)
            c.number_format = "0" if k == 0 else "0.000"
            if k == 3:
                c.font = Font(name=F, bold=True)
                hasil_kappa.append(f"B{r}")
            r += 1
        r += 1
    tulis(rs, r, 1, "Tafsiran kappa (Landis & Koch, 1977, Biometrics 33(1):159-174)", bold=True, wrap=False)
    r += 1
    for t in ["< 0,00 buruk", "0,00-0,20 sangat lemah", "0,21-0,40 lemah", "0,41-0,60 sedang", "0,61-0,80 kuat", "0,81-1,00 hampir sempurna"]:
        tulis(rs, r, 1, t, size=9)
        r += 1
    tulis(rs, r + 1, 1, "Catatan: bila semua penilaian berada di satu kategori, pe = 1 dan kappa tidak terdefinisi (sel kosong).", size=9, color="595959")
    rs.merge_cells(start_row=r + 1, start_column=1, end_row=r + 1, end_column=8)

    # ---- Impor penilai 2
    im = wb.create_sheet("Impor_Penilai_2")
    hdr(im, 1, ["urut_penilai2", "kode", "diagnosis_penilai_2", "catatan_2"], [14, 12, 22, 50])
    for x in acak:
        rr = x["urut2"] + 1
        tulis(im, rr, 1, x["urut2"], align="center")
        tulis(im, rr, 2, x["kode"], align="center")
        for col in (3, 4):
            im.cell(rr, col).fill = IN_FILL
    tulis(im, last + 2, 1, "Tempel kolom B:D dari berkas penilai 2 ke B:D di sini (kode ikut ditempel agar pencocokan aman walau urutan berubah).",
          size=9, color="595959", wrap=False)

    wb.save(out / "verifikasi_ds3.xlsx")

    # ================= salinan buta penilai 2 =================
    wb2 = Workbook()
    p2 = wb2.active
    p2.title = "Petunjuk"
    petunjuk_p2(p2, n)
    w2 = wb2.create_sheet("Penilaian")
    hdr(w2, 1, ["urut", "kode", "diagnosis_penilai_2", "catatan_2", "foto"], [7, 12, 20, 40, 58])
    w2.freeze_panes = "C2"
    dv2 = DataValidation(type="list", formula1='"' + ",".join(DIAG) + '"', allow_blank=True,
                         error="Pilih: BLB, Blast, bukan keduanya, atau ragu", errorTitle="Nilai tidak valid", showErrorMessage=True)
    w2.add_data_validation(dv2)
    dv2.add(f"C2:C{last}")
    for x in acak:
        r = x["urut2"] + 1
        tulis(w2, r, 1, x["urut2"], align="center")
        c = tulis(w2, r, 2, x["kode"], align="center")
        c.hyperlink = f"foto_penuh_penilai2/{x['kode']}{x['path'].suffix.lower()}"
        c.font = Font(name=F, size=10, color="0563C1", underline="single")
        for col in (3, 4):
            w2.cell(r, col).fill = IN_FILL
            w2.cell(r, col).font = Font(name=F, size=10)
            w2.cell(r, col).alignment = Alignment(wrap_text=True, vertical="top")
        _, h = tanam(w2, x["path"], f"E{r}")
        w2.row_dimensions[r].height = h * 0.75 + 6
        for col in range(1, 6):
            w2.cell(r, col).border = BOX
    kriteria(wb2.create_sheet("Kriteria"))
    wb2.save(out / "verifikasi_ds3_penilai2.xlsx")
    print(f"[OK] {n} foto -> {out}")


if __name__ == "__main__":
    main()
