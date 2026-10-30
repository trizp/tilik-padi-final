"""Memuat app/config.yaml (atau berkas yang ditunjuk variabel lingkungan TILIK_CONFIG, untuk mencoba konfigurasi alternatif
tanpa mengaktifkannya). Jalur relatif di config dihitung dari folder app/."""
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

APP_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Konfigurasi:
    model_path: Path
    nama_model: str
    model_sha256: str
    imgsz: int
    iou: float
    conf_min: float
    max_det: int
    ambang_terdeteksi: float
    ambang_rendah: float
    ukuran_maks_mb: float
    sisi_maks_proses: int
    jenis_berkas: tuple
    kelas: tuple
    teks: dict = field(default_factory=dict)
    sumber_data: tuple = ()
    batas_tampil_rendah: float | None = 0.10   # batas TAMPILAN bercak berkeyakinan rendah (bukan ambang diagnosis); None = nonaktif
    pemisah_path: Path | None = None          # pemisah daun YOLOv8n-seg (alur v9); None = tanpa pemisahan (config arsip v7)
    pemisah_sha256: str = ""
    pemisah_tau: float = 0.35
    hasil_evaluasi: dict = field(default_factory=dict)   # teks hasil evaluasi alur aplikasi untuk halaman Tentang


def muat(path=None) -> Konfigurasi:
    if path is None and os.environ.get("TILIK_CONFIG"):
        path = APP_DIR / os.environ["TILIK_CONFIG"]
    path = Path(path) if path else APP_DIR / "config.yaml"
    c = yaml.safe_load(path.read_text(encoding="utf-8"))
    m, a, u = c["model"], c["ambang"], c["unggahan"]
    if not (0 < a["rendah"] <= a["terdeteksi"] < 1):        # rendah = terdeteksi: tingkat "kemungkinan" tidak dipakai
        raise ValueError("config.yaml: syarat 0 < ambang.rendah <= ambang.terdeteksi < 1 tidak terpenuhi")
    tp = c.get("tampilan", {})
    tampil = float(tp.get("batas_bercak_rendah", 0.10)) if tp.get("bercak_rendah_aktif", True) else None
    if tampil is not None and not (0 < tampil <= a["rendah"]):
        raise ValueError("config.yaml: syarat 0 < tampilan.batas_bercak_rendah <= ambang.rendah tidak terpenuhi")
    pd = c.get("pemisah_daun")
    if pd is not None and not (0 < float(pd["tau"]) < 1):
        raise ValueError("config.yaml: syarat 0 < pemisah_daun.tau < 1 tidak terpenuhi")
    return Konfigurasi(
        model_path=(APP_DIR / m["path"]).resolve(),
        nama_model=str(m.get("nama", "YOLO11s")),
        model_sha256=m.get("sha256", ""),
        imgsz=int(m["imgsz"]),
        iou=float(m["iou"]),
        conf_min=float(m["conf_min"]),
        max_det=int(m["max_det"]),
        ambang_terdeteksi=float(a["terdeteksi"]),
        ambang_rendah=float(a["rendah"]),
        ukuran_maks_mb=float(u["ukuran_maks_mb"]),
        sisi_maks_proses=int(u.get("sisi_maks_proses", 2048)),
        jenis_berkas=tuple(u["jenis"]),
        kelas=tuple(m["kelas"]),
        teks=c.get("teks", {}),
        sumber_data=tuple(c.get("sumber_data", [])),
        batas_tampil_rendah=tampil,
        pemisah_path=(APP_DIR / pd["path"]).resolve() if pd else None,
        pemisah_sha256=pd.get("sha256", "") if pd else "",
        pemisah_tau=float(pd["tau"]) if pd else 0.35,
        hasil_evaluasi=c.get("hasil_evaluasi") or {},
    )
