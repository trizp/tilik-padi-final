#!/usr/bin/env python3
"""
siapkan_unggah.py - Susun folder siap-unggah untuk repositori GitHub publik khusus aplikasi (TIDAK mengunggah apa pun).

- Menyalin hanya berkas dalam daftar izin IZIN (WAJIB, DOKUMENTASI INTI, TEST) + LICENSE AGPL-3.0 (teks resmi dari paket Ultralytics
  di venv proyek, hanya dibaca). Berkas kerja (catatan kritik, tangkapan layar lama, alat pengembangan, arsip aturan) tidak ikut.
- Tidak menyalin: .venv, __pycache__, .pytest_cache, data/, runs/, foto uji, docs/uji/pindai_val.csv.
- Memindai semua berkas (teks dan biner) untuk jalur pribadi dan kunci rahasia.
- Mengukur ukuran dan menguji .gitignore dengan repositori git sementara (di folder sementara sistem, lalu dihapus).

  app/.venv/Scripts/python.exe app/tools/siapkan_unggah.py [--tujuan <folder>] [--timpa]
Keluaran: folder tujuan (bawaan: D:/TELKOM/SKRIPSI/tilik-padi, yaitu <folder induk proyek>/tilik-padi); laporan hanya di app/docs/siap_unggah.md (tidak ikut repositori).
"""
import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

APP = Path(__file__).resolve().parent.parent
PROYEK = APP.parent
LISENSI = PROYEK / ".venv/Lib/site-packages/ultralytics-8.4.171.dist-info/licenses/LICENSE"
# Daftar izin eksplisit (disetujui pemilik skripsi): WAJIB, DOKUMENTASI INTI, TEST. LICENSE ditambahkan terpisah.
IZIN = [
    # WAJIB (aplikasi berjalan di Streamlit Cloud)
    "streamlit_app.py", "config.yaml", "requirements.txt", ".streamlit/config.toml",
    "core/__init__.py", "core/alur.py", "core/diagnosis.py", "core/gambar.py", "core/inferensi.py", "core/konfigurasi.py",
    "core/pemisah_daun.py", "core/visual.py",
    "rbs/__init__.py", "rbs/mesin.py", "rbs/rules.yaml",
    "ui/__init__.py", "ui/gaya.css", "ui/ilustrasi.py", "ui/komponen.py",
    # alur v9 (KEPUTUSAN_PROYEK.md 2026-10-20): detektor YOLOv8s v9 + pemisah daun YOLOv8n-seg; model v7 hanya diarsipkan di app/
    "models/v9_deploy_yolov8s_s2.onnx", "models/v9_pemisah_daun_yolov8n_seg.onnx",
    # DOKUMENTASI INTI (+ LICENSE)
    "README.md", ".gitignore", "docs/arsitektur.md", "docs/menjalankan_lokal.md", "docs/deploy.md", "docs/desain.md",
    "docs/tangkapan_layar/awal_390.png", "docs/tangkapan_layar/terdeteksi_390.png", "docs/tangkapan_layar/tidak_terdeteksi_390.png",
    "docs/tangkapan_layar/kemungkinan_lain_390.png", "docs/tangkapan_layar/kemungkinan_390.png", "docs/tangkapan_layar/foto_ulang_390.png",
    "docs/tangkapan_layar/terdeteksi_1280.png", "docs/tangkapan_layar/tidak_terdeteksi_1280.png",
    "docs/tangkapan_layar/kemungkinan_lain_1280.png",
    # TEST
    "tests/conftest.py", "tests/test_inti.py", "tests/test_rbs.py", "tests/test_blackbox.py", "requirements-dev.txt",
    "tools/uji_blackbox.py", "docs/uji/blackbox.md", "docs/uji/sumber_daya.md", "docs/uji_kesetaraan.md", "docs/uji/pemisah_daun.md",
    "docs/uji/sumber_daya_v7_ulang_20261020.md",   # pembanding v7 pada kondisi mesin yang sama (dirujuk README)
]
POLA = {  # pola berkas sensitif (teks); untuk biner dicari versi bytes
    "jalur Windows pribadi": re.compile(r"[A-Za-z]:\\\\?(Users|TELKOM)\\|[A-Za-z]:/(Users|TELKOM)/", re.I),
    "folder proyek skripsi": re.compile(r"TELKOM[\\/]+SKRIPSI|rice_yolo[\\/]+(data|runs)", re.I),
    "nama pengguna lokal": re.compile(r"\bTRISTAN\b", re.I),
    "kunci API / token": re.compile(r"(api[_-]?key|secret|password|token)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{12,}", re.I),
    "variabel kredensial": re.compile(r"ROBOFLOW_API_KEY|KAGGLE_KEY|KAGGLE_USERNAME|\.kaggle[\\/]", re.I),
    "alamat surel": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
}
SUREL_BOLEH = {"noreply@anthropic.com"}
TEKS = {".py", ".md", ".yaml", ".yml", ".toml", ".txt", ".json", ".css", ".gitignore", ""}


def salin(tujuan: Path):
    for f in IZIN:
        (tujuan / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(APP / f, tujuan / f)
    shutil.copyfile(LISENSI, tujuan / "LICENSE")


def pindai(tujuan: Path):
    temuan = []
    for p in sorted(tujuan.rglob("*")):
        if p.is_dir() or ".git" in p.parts:
            continue
        r = p.relative_to(tujuan).as_posix()
        data = p.read_bytes()
        if p.suffix.lower() in TEKS or p.name in (".gitignore", "LICENSE"):
            teks = data.decode("utf-8", errors="replace")
            for nama, pola in POLA.items():
                for m in pola.finditer(teks):
                    if nama == "alamat surel" and (m.group(0) in SUREL_BOLEH or r == "LICENSE"):
                        continue
                    baris = teks.count("\n", 0, m.start()) + 1
                    temuan.append((r, baris, nama, m.group(0)[:60]))
        else:
            for kata in (b"TELKOM\\SKRIPSI", b"TELKOM/SKRIPSI", b"Users\\TRISTAN", b"Users/TRISTAN", b"TRISTAN", b"ROBOFLOW_API_KEY"):
                if kata in data:
                    temuan.append((r, 0, "biner memuat teks pribadi", kata.decode()))
    return temuan


UMPAN = [".venv/umpan.txt", "data/umpan.jpg", "runs/umpan.txt", "foto_uji.jpg", "foto_uji.png", "model_latih.pt",
         "docs/uji/pindai_val.csv", ".streamlit/secrets.toml", "core/__pycache__/x.pyc"]


def _cocok(jalur: str, pola: str) -> bool:
    """Pencocok pola .gitignore sederhana (cukup untuk pola di .gitignore aplikasi): 'dir/', '*.ext', '!pengecualian', '**'."""
    import fnmatch
    if pola.endswith("/"):
        d = pola.rstrip("/")
        return jalur.startswith(d + "/") or ("/" + d + "/") in ("/" + jalur)
    if "/" in pola.strip("/"):
        return fnmatch.fnmatch(jalur, pola.lstrip("/").replace("**/", "*")) or fnmatch.fnmatch(jalur, pola.lstrip("/"))
    return fnmatch.fnmatch(jalur.rsplit("/", 1)[-1], pola)


def diabaikan(jalur: str, pola_list) -> bool:
    hasil = False
    for p in pola_list:
        neg = p.startswith("!")
        if _cocok(jalur, p[1:] if neg else p):
            hasil = not neg
    return hasil


def uji_gitignore_python(tujuan: Path):
    """Uji .gitignore tanpa menjalankan git: berkas umpan harus diabaikan, semua berkas folder unggah harus terlacak."""
    pola = [l.strip() for l in (tujuan / ".gitignore").read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
    berkas = [p.relative_to(tujuan).as_posix() for p in tujuan.rglob("*") if p.is_file()]
    return {"git": "tidak dijalankan (pencocok pola Python)", "berkas_terlacak": sum(not diabaikan(b, pola) for b in berkas),
            "berkas_folder_yang_terabaikan": [b for b in berkas if diabaikan(b, pola)],
            "umpan_ikut_terlacak": [u for u in UMPAN if not diabaikan(u, pola)], "count_objects": ""}


def uji_git(tujuan: Path):
    git = shutil.which("git")
    if not git:
        return {"git": "tidak ditemukan; uji .gitignore dilewati"}
    tmp = Path(tempfile.mkdtemp(prefix="uji_git_"))
    try:
        salinan = tmp / "repo"
        shutil.copytree(tujuan, salinan)
        # berkas umpan yang HARUS diabaikan .gitignore
        umpan = UMPAN
        for u in umpan:
            (salinan / u).parent.mkdir(parents=True, exist_ok=True)
            (salinan / u).write_text("umpan", encoding="utf-8")
        run = lambda *a: subprocess.run([git, *a], cwd=salinan, capture_output=True, text=True)  # noqa: E731
        run("init", "-q")
        run("add", "-A")
        terlacak = run("ls-files").stdout.split("\n")
        bocor = [u for u in umpan if u in terlacak]
        ukuran = run("count-objects", "-vH").stdout
        return {"git": run("--version").stdout.strip(), "berkas_terlacak": len([t for t in terlacak if t]),
                "umpan_ikut_terlacak": bocor, "count_objects": ukuran.strip()}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def mb(n):
    return f"{n / 2 ** 20:.1f} MB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tujuan", default=str(PROYEK.parent / "tilik-padi"))
    ap.add_argument("--timpa", action="store_true")
    ap.add_argument("--tanpa-git", action="store_true", help="uji .gitignore dengan pencocok pola Python, tanpa menjalankan git")
    a = ap.parse_args()
    tujuan = Path(a.tujuan).resolve()
    if tujuan == APP or APP in tujuan.parents or tujuan in APP.parents:
        sys.exit("[DITOLAK] folder tujuan tidak boleh di dalam app/ atau induknya")
    if tujuan.exists() and any(tujuan.iterdir()):
        if not a.timpa:
            sys.exit(f"[DITOLAK] {tujuan} sudah berisi (pakai --timpa)")
        if (tujuan / ".git").exists():
            sys.exit(f"[DITOLAK] {tujuan} sudah menjadi repositori git; perbarui isinya lewat git, bukan skrip ini")
        for x in tujuan.iterdir():                      # kosongkan isi (folder tetap ada; aman bila sedang dibuka)
            shutil.rmtree(x) if x.is_dir() else x.unlink()
    tujuan.mkdir(parents=True, exist_ok=True)
    salin(tujuan)
    temuan = pindai(tujuan)
    git = uji_gitignore_python(tujuan) if a.tanpa_git else uji_git(tujuan)
    berkas = [p for p in tujuan.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in berkas)
    besar = sorted(berkas, key=lambda p: -p.stat().st_size)[:8]
    lolos = (not temuan and not git.get("umpan_ikut_terlacak") and not git.get("berkas_folder_yang_terabaikan")
             and max(p.stat().st_size for p in berkas) < 100 * 2 ** 20)
    md = ["# Pemeriksaan folder siap-unggah", "",
          f"**Hasil: {'LOLOS' if lolos else 'PERLU DIPERIKSA'}**", "",
          f"- Berkas: {len(berkas)}; ukuran total {mb(total)}; berkas terbesar {mb(besar[0].stat().st_size)} (batas GitHub 100 MB per berkas).",
          f"- Pemindaian jalur pribadi dan kunci rahasia (teks dan biner): {len(temuan)} temuan.",
          f"- Uji .gitignore ({git.get('git')}): {git.get('berkas_terlacak', '-')} berkas akan terlacak; "
          f"berkas umpan (.venv, data/, runs/, foto .jpg/.png di luar docs, .pt, pindai_val.csv, secrets.toml, cache) yang ikut terlacak: "
          f"{git.get('umpan_ikut_terlacak', '-') or 'tidak ada'}.", "",
          "Berkas terbesar:", ""] + [f"- `{p.relative_to(tujuan).as_posix()}`: {mb(p.stat().st_size)}" for p in besar]
    if temuan:
        md += ["", "Temuan:", ""] + [f"- `{r}` baris {b}: {n} ({t})" for r, b, n, t in temuan]
    md += ["", "```", git.get("count_objects", ""), "```"]
    laporan = "\n".join(md) + "\n"
    (APP / "docs/siap_unggah.md").write_text(laporan, encoding="utf-8")
    print(laporan)
    print(f"Folder: {tujuan}")
    return lolos


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
