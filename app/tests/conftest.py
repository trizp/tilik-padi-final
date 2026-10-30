import sys
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP))
PROYEK = APP.parent
VAL = PROYEK / "data/processed/prov_v7/images/val"          # foto ASLI (alur v9 memisahkan daun sendiri)
VAL_V9 = PROYEK / "data/processed/prov_v9/images/val"       # foto VAL v9 yang latarnya sudah dihapus SAM2 (acuan uji detektor)
MASKER_VAL = PROYEK / "data/derived/v9_latar/seg_n_val"     # masker pemisah daun fase 1 (acuan uji pemisah daun)

butuh_data = pytest.mark.skipif(not VAL.exists(), reason="foto VAL proyek tidak tersedia (repositori aplikasi tidak memuat dataset)")


@pytest.fixture(scope="session")
def cfg():
    from core import konfigurasi
    return konfigurasi.muat()


@pytest.fixture(scope="session")
def aturan():
    from rbs import mesin
    return mesin.muat_aturan()


@pytest.fixture(scope="session")
def detektor(cfg):
    from core import inferensi
    return inferensi.Detektor(cfg.model_path, cfg.imgsz, cfg.conf_min, cfg.iou, cfg.max_det)


@pytest.fixture(scope="session")
def pemisah(cfg):
    from core import pemisah_daun
    return pemisah_daun.PemisahDaun(cfg.pemisah_path, cfg.pemisah_tau)
