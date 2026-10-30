"""Unit test modul inti: pembacaan foto, aturan tingkat, pra-/pasca-proses, dan inferensi ONNX."""
import csv
import hashlib
import io

import numpy as np
import pytest
from PIL import Image

from conftest import MASKER_VAL, PROYEK, VAL, VAL_V9, butuh_data
from core import diagnosis as dg
from core import gambar, inferensi


def jpeg(w=200, h=100, warna=(40, 120, 60), exif_orientasi=None):
    im = Image.new("RGB", (w, h), warna)
    buf = io.BytesIO()
    if exif_orientasi:
        ex = Image.Exif()
        ex[0x0112] = exif_orientasi
        im.save(buf, "JPEG", exif=ex.tobytes())
    else:
        im.save(buf, "JPEG")
    return buf.getvalue()


# ---------- pembacaan foto
def test_tolak_bukan_gambar():
    with pytest.raises(gambar.FotoDitolak, match="bukan foto"):
        gambar.baca_foto(b"ini teks, bukan foto")


def test_tolak_kosong_dan_terlalu_besar():
    with pytest.raises(gambar.FotoDitolak, match="kosong"):
        gambar.baca_foto(b"")
    with pytest.raises(gambar.FotoDitolak, match="lebih dari batas"):
        gambar.baca_foto(b"\xff" * (10 * 1024 * 1024 + 1), 10)


def test_tolak_terlalu_kecil_dan_format_lain():
    with pytest.raises(gambar.FotoDitolak, match="terlalu kecil"):
        gambar.baca_foto(jpeg(20, 20))
    buf = io.BytesIO()
    Image.new("RGB", (100, 100)).save(buf, "GIF")
    with pytest.raises(gambar.FotoDitolak, match="tidak didukung"):
        gambar.baca_foto(buf.getvalue())


def test_png_rgba_menjadi_rgb():
    buf = io.BytesIO()
    Image.new("RGBA", (64, 48), (10, 20, 30, 128)).save(buf, "PNG")
    im = gambar.baca_foto(buf.getvalue())
    assert im.mode == "RGB" and im.size == (64, 48)


def test_koreksi_orientasi_exif():
    im = gambar.baca_foto(jpeg(200, 100, exif_orientasi=6))      # 6 = putar 90 derajat
    assert im.size == (100, 200)


# ---------- aturan tingkat
@pytest.mark.parametrize("k,t", [(0.35, "terdeteksi"), (0.3499, "kemungkinan"), (0.29, "kemungkinan"), (0.2899, "tidak_terdeteksi"),
                                 (0.0, "tidak_terdeteksi"), (1.0, "terdeteksi")])
def test_tingkat_batas(k, t):
    assert dg.tentukan_tingkat(k, 0.35, 0.29) == t


def test_diagnosis_kotak_teratas_dan_tanpa_kotak():
    K = inferensi.Kotak
    kelas = ("leaf_blast", "brown_spot", "bacterial_leaf_blight")
    h = dg.diagnosis([K(0, 0, 1, 1, 0.30, 1), K(0, 0, 1, 1, 0.62, 0), K(2, 2, 3, 3, 0.31, 0)], kelas, 0.35, 0.29)
    assert (h.penyakit, h.tingkat, h.keyakinan) == ("leaf_blast", "terdeteksi", 0.62)
    assert len(dg.kotak_untuk_digambar(h, kelas, 0.29)) == 2       # hanya kelas diagnosis dan >= 0,29
    h0 = dg.diagnosis([], kelas, 0.35, 0.29)
    assert (h0.penyakit, h0.tingkat, h0.keyakinan) == (None, "tidak_terdeteksi", 0.0)
    assert dg.kotak_untuk_digambar(h0, kelas, 0.29) == []


# ---------- pra-proses dan pasca-proses
def test_letterbox_persegi_panjang():
    img = np.zeros((480, 640, 3), np.uint8)
    lb, r, (kiri, atas) = inferensi.letterbox(img, 640)
    assert lb.shape == (640, 640, 3) and r == 1.0 and (kiri, atas) == (0, 80)
    assert (lb[:80] == 114).all() and (lb[-80:] == 114).all()


def test_letterbox_potret_besar():
    img = np.zeros((4000, 3000, 3), np.uint8)
    lb, r, (kiri, atas) = inferensi.letterbox(img, 640)
    assert lb.shape == (640, 640, 3) and r == pytest.approx(0.16) and (kiri, atas) == (80, 0)


def test_bilinear_identitas_dan_rata():
    img = np.random.default_rng(0).integers(0, 255, (50, 70, 3), dtype=np.uint8)
    assert np.array_equal(inferensi.ubah_ukuran_bilinear(img, 70, 50), img)
    rata = np.full((40, 40, 3), 77, np.uint8)
    assert (inferensi.ubah_ukuran_bilinear(rata, 17, 23) == 77).all()


def test_nms_per_kelas():
    kotak = np.array([[0, 0, 10, 10], [0.5, 0.5, 10.5, 10.5], [100, 100, 110, 110]], np.float32)   # IoU kotak 0-1 = 0,82
    skor = np.array([0.9, 0.8, 0.7], np.float32)
    assert list(inferensi.nms(kotak, skor, 0.7)) == [0, 2]
    # kelas berbeda tidak saling menekan karena offset 7680
    k2 = kotak.copy()
    k2[1] += 7680
    assert list(inferensi.nms(k2, skor, 0.7)) == [0, 1, 2]


def test_pasca_proses_koordinat():
    out = np.zeros((1, 7, 4), np.float32)
    out[0, :4, 0] = [320, 320, 64, 64]          # kotak di tengah input 640
    out[0, 4 + 2, 0] = 0.9                       # kelas 2
    out[0, 4, 1] = 0.0005                        # di bawah conf_min -> dibuang
    k = inferensi.pasca_proses(out, 0.5, (0, 80), (960, 1280), 0.001, 0.7, 300)
    assert len(k) == 1 and k[0].kelas == 2 and k[0].keyakinan == pytest.approx(0.9)
    # x: (288 - 0) / 0,5 = 576 ; y: (288 - 80) / 0,5 = 416
    assert (k[0].x1, k[0].y1, k[0].x2, k[0].y2) == pytest.approx((576, 416, 704, 544))


# ---------- model ONNX
def test_model_sesuai_sha256(cfg):
    assert hashlib.sha256(cfg.model_path.read_bytes()).hexdigest() == cfg.model_sha256


def test_inferensi_gambar_sintetis(detektor):
    k = detektor.prediksi(np.full((480, 640, 3), 120, np.uint8))
    assert isinstance(k, list)
    for b in k:
        assert 0 <= b.x1 <= b.x2 <= 640 and 0 <= b.y1 <= b.y2 <= 480 and 0 < b.keyakinan <= 1 and b.kelas in (0, 1, 2)


@butuh_data
def test_inferensi_setara_hasil_tersimpan(detektor, cfg):
    """Kelas teratas dan keyakinan sama dengan evaluasi tersimpan Ultralytics (selisih < 0,01) pada 20 foto val."""
    v7 = PROYEK / "reports/final_eval_v7/aplikasi"
    acuan_per_model = {"v7_deploy_yolo11s_s2.onnx": (v7 / "onnx_vs_pt_50_val.csv", VAL),
                       "v7_deploy_yolov8s_s2.onnx": (v7 / "onnx_vs_pt_20_val_yolov8s.csv", VAL),
                       # v9: acuan fase 4 pada foto VAL v9 (latar sudah dihapus), jadi yang diuji hanya pra-/pasca-proses detektor
                       "v9_deploy_yolov8s_s2.onnx": (PROYEK / "reports/v9_latar/aplikasi_v9/fase4_onnx_vs_pt_20_val.csv", VAL_V9)}
    berkas, folder = acuan_per_model[cfg.model_path.name]
    acuan = list(csv.DictReader(open(berkas, encoding="utf-8")))[:20]
    assert len(acuan) == 20
    for r in acuan:
        im = gambar.baca_foto((folder / r["gambar"]).read_bytes(), cfg.ukuran_maks_mb)
        k = detektor.prediksi(gambar.ke_array(im))
        top = max(k, key=lambda b: b.keyakinan) if k else None
        kelas, conf = (top.kelas, top.keyakinan) if top else (-1, 0.0)      # acuan -1 = tanpa kotak
        # Kelas dibandingkan bila keyakinan acuan >= 0,01. Di bawahnya (daun sehat, 2 dari 20 foto v9 dengan keyakinan ±0,003) dua kotak
        # teratas hampir seri sehingga urutannya bisa tertukar karena selisih resize; diagnosisnya tetap sama (tidak terdeteksi).
        if float(r["conf_onnx"]) >= 0.01:
            assert kelas == int(r["kelas_onnx"]), r["gambar"]
        assert abs(conf - float(r["conf_onnx"])) < 0.01, r["gambar"]
        assert dg.tentukan_tingkat(conf, cfg.ambang_terdeteksi, cfg.ambang_rendah) == \
            dg.tentukan_tingkat(float(r["conf_onnx"]), cfg.ambang_terdeteksi, cfg.ambang_rendah), r["gambar"]


def test_foto_besar_diperkecil_dan_orientasi_tetap():
    im = gambar.baca_foto(jpeg(4000, 3000), 10, 2048)
    assert max(im.size) == 2048 and im.size == (2048, 1536)
    im = gambar.baca_foto(jpeg(4000, 3000, exif_orientasi=6), 10, 2048)   # potret setelah koreksi EXIF
    assert im.size == (1536, 2048)
    assert gambar.baca_foto(jpeg(1024, 1024), 10, 2048).size == (1024, 1024)  # foto kecil tidak diubah


def test_saran_tindakan_sekarang_lebih_dulu(aturan):
    from rbs import mesin
    from ui import komponen as ui
    h = mesin.jalankan(mesin.fakta_dari_diagnosis("leaf_blast", "terdeteksi", 0.7), aturan)
    out = ui.daftar_saran(h.saran, mesin.muat_meta())
    assert out.index("Tindakan sekarang") < out.index("Pencegahan musim berikutnya")
    assert out.index("Bakar jerami") < out.index("Pemupukan berimbang")
    assert "Gunakan pestisida sesuai dosis pada label dan anjuran penyuluh setempat." in out
    assert "Teknik Mengendalikan Hama dan Penyakit Padi" in out


def test_pita_tanpa_angka_batas():
    from ui import komponen as ui
    s = ui.pita_keyakinan(0.66, "terdeteksi")
    assert "66%" in s and "29%" not in s and "35%" not in s


def test_potongan_diperbesar():
    from core import visual
    im = Image.new("RGB", (1024, 1024), (60, 120, 60))
    jpg, f = visual.potongan_diperbesar(im, inferensi.Kotak(500, 500, 520, 512, 0.8, 1), "terdeteksi")
    assert Image.open(io.BytesIO(jpg)).size == (480, 480) and f == pytest.approx(3.0)
    _, f2 = visual.potongan_diperbesar(im, inferensi.Kotak(100, 100, 900, 600, 0.8, 2), "terdeteksi")
    assert f2 < 1.0                                        # kotak besar: potongan memuat seluruh kotak, tidak diperbesar


def test_css_tanpa_has_dan_skrip_label():
    from ui import komponen as ui
    assert ":has(" not in ui.css()                         # tampilan tidak bergantung pada CSS :has()
    s = ui.skrip_label()
    assert '"Take Photo": "Ambil foto"' in s and "Unggah foto lain" in s and "MutationObserver" in s


def test_kepala_hasil_menandai_ilustrasi_bila_ada_potongan():
    from ui import komponen as ui
    assert "ada-potongan" in ui.kepala_hasil("leaf_blast", "terdeteksi", ada_potongan=True)
    assert "ada-potongan" not in ui.kepala_hasil("leaf_blast", "terdeteksi", ada_potongan=False)


def test_ambang_sama_tanpa_tingkat_kemungkinan():
    """rendah = terdeteksi (hasil prosedur tiga tingkat untuk YOLOv8s): tidak ada tingkat kemungkinan, pita dua zona."""
    from ui import komponen as ui
    assert dg.tentukan_tingkat(0.37, 0.37, 0.37) == "terdeteksi"
    assert dg.tentukan_tingkat(0.369, 0.37, 0.37) == "tidak_terdeteksi"
    lama = dict(ui.BATAS)
    try:
        ui.atur_batas(0.37, 0.37)
        assert ui.posisi_pita(0.37) == pytest.approx(50.0) and ui.posisi_pita(0.0) == 0.0 and ui.posisi_pita(1.0) == pytest.approx(100.0)
        s = ui.pita_keyakinan(0.5, "terdeteksi")
        assert "kemungkinan" not in s and "width:50%" in s
    finally:
        ui.atur_batas(lama["rendah"], lama["terdeteksi"])


def test_config_aktif_alur_v9_tiga_tingkat(cfg):
    """Keputusan 2026-10-20 (pilihan a): alur v9 = pemisah daun YOLOv8n-seg (tau 0,35) + YOLOv8s v9; tiga tingkat (0,30 / 0,36)
    dikunci di VAL; bercak berkeyakinan rendah nonaktif."""
    assert cfg.nama_model == "YOLOv8s" and cfg.model_path.name == "v9_deploy_yolov8s_s2.onnx"
    assert cfg.pemisah_path.name == "v9_pemisah_daun_yolov8n_seg.onnx" and cfg.pemisah_tau == 0.35
    assert (cfg.ambang_rendah, cfg.ambang_terdeteksi) == (0.30, 0.36)
    assert cfg.batas_tampil_rendah is None
    assert cfg.hasil_evaluasi.get("butir") and "Foto ulang" in cfg.teks["pesan_foto_ulang"]


@pytest.mark.parametrize("berkas_cfg", ["config.yaml", "config_yolov8.yaml", "config_yolo11s_arsip.yaml"])
def test_config_memuat_semua_teks_yang_dipakai_tampilan(berkas_cfg):
    """Kunci teks yang dibaca streamlit_app.py dan ui/ ada di config (bukan tersisip di blok lain)."""
    from core import konfigurasi
    p = konfigurasi.APP_DIR / berkas_cfg
    if not p.exists():
        pytest.skip(f"{berkas_cfg} hanya ada di folder proyek (tidak ikut repositori)")
    c = konfigurasi.muat(p)
    wajib = {"judul", "subjudul", "catatan_unggah", "petunjuk_benar", "petunjuk_salah", "petunjuk_jarak_cahaya", "peringatan_singkat"}
    assert wajib <= set(c.teks), wajib - set(c.teks)
    assert set(c.hasil_evaluasi) <= {"pengantar", "butir"}


def test_pemisah_daun_sesuai_sha256(cfg):
    assert hashlib.sha256(cfg.pemisah_path.read_bytes()).hexdigest() == cfg.pemisah_sha256


def test_config_arsip_v7_tanpa_pemisah_daun():
    from core import konfigurasi
    p = konfigurasi.APP_DIR / "config_yolov8.yaml"
    if not p.exists():
        pytest.skip("arsip v7 hanya ada di folder proyek (tidak ikut repositori)")
    c = konfigurasi.muat(p)
    assert c.model_path.name == "v7_deploy_yolov8s_s2.onnx" and c.pemisah_path is None and not c.hasil_evaluasi
    assert (c.ambang_rendah, c.ambang_terdeteksi) == (0.37, 0.37)


def test_tau_pemisah_divalidasi(tmp_path):
    from core import konfigurasi
    teks = (konfigurasi.APP_DIR / "config.yaml").read_text(encoding="utf-8")
    p = tmp_path / "c.yaml"
    p.write_text(teks.replace("tau: 0.35", "tau: 1.5"), encoding="utf-8")
    with pytest.raises(ValueError):
        konfigurasi.muat(p)


def test_bercak_rendah_nonaktif_lewat_config(cfg, aturan):
    """Config aktif: tidak ada kotak putus-putus walau keyakinan teratas tepat di bawah ambang rendah (batas tidak terdeteksi)."""
    from core import alur
    K = inferensi.Kotak
    assert dg.bercak_rendah(_dx([K(0, 0, 9, 9, 0.2, 1)]), None) is None
    h = alur.periksa_foto(jpeg(400, 300), cfg, _DetektorPalsu([K(10, 10, 90, 60, cfg.ambang_rendah - 0.005, 1)]), aturan,
                          _PemisahPalsu())
    assert h.diagnosis.tingkat == "tidak_terdeteksi" and h.bercak_rendah is None and "bercak_keyakinan_rendah" not in h.fakta_awal
    assert [L.aturan.id for L in h.inferensi.jejak] == ["N01", "N02"]


def test_config_yolo11s_arsip_dan_model_tetap_ada():
    from core import konfigurasi
    p = konfigurasi.APP_DIR / "config_yolo11s_arsip.yaml"
    if not p.exists():
        pytest.skip("arsip YOLO11s hanya ada di folder proyek (tidak ikut repositori)")
    c = konfigurasi.muat(p)
    assert c.nama_model == "YOLO11s" and c.model_path.exists()
    assert hashlib.sha256(c.model_path.read_bytes()).hexdigest() == c.model_sha256
    assert (c.ambang_rendah, c.ambang_terdeteksi) == (0.29, 0.35)



# ---------- penyampaian tambahan (KEPUTUSAN_PROYEK.md 2026-10-15): kemungkinan lain, cara membedakan, bercak berkeyakinan rendah
KELAS = ("leaf_blast", "brown_spot", "bacterial_leaf_blight")


def _dx(kotak, t=0.35, r=0.29):
    return dg.diagnosis(kotak, KELAS, t, r)


def test_kemungkinan_lain_satu_tertinggi_dari_penyakit_lain():
    K = inferensi.Kotak
    h = _dx([K(0, 0, 9, 9, 0.62, 0), K(0, 0, 9, 9, 0.50, 0), K(0, 0, 9, 9, 0.30, 1), K(1, 1, 9, 9, 0.33, 1), K(0, 0, 5, 5, 0.31, 2)])
    lain = dg.kemungkinan_lain(h, KELAS, 0.29)
    assert (lain.penyakit, lain.keyakinan) == ("brown_spot", 0.33)          # bukan kelas diagnosis; satu, tertinggi
    assert dg.kemungkinan_lain(_dx([K(0, 0, 9, 9, 0.62, 0), K(0, 0, 9, 9, 0.2899, 1)]), KELAS, 0.29) is None   # < t_rendah
    assert dg.kemungkinan_lain(_dx([K(0, 0, 9, 9, 0.62, 0), K(0, 0, 9, 9, 0.29, 1)]), KELAS, 0.29).keyakinan == 0.29  # tepat batas


def test_kemungkinan_lain_pada_tingkat_kemungkinan_dan_tidak_pada_tidak_terdeteksi():
    K = inferensi.Kotak
    h = _dx([K(0, 0, 9, 9, 0.33, 0), K(0, 0, 9, 9, 0.30, 1)])
    assert h.tingkat == "kemungkinan" and dg.kemungkinan_lain(h, KELAS, 0.29).penyakit == "brown_spot"
    h0 = _dx([K(0, 0, 9, 9, 0.28, 0), K(0, 0, 9, 9, 0.27, 1)])
    assert h0.tingkat == "tidak_terdeteksi" and dg.kemungkinan_lain(h0, KELAS, 0.29) is None
    assert dg.kemungkinan_lain(_dx([]), KELAS, 0.29) is None


def test_kemungkinan_lain_dua_tingkat_yolov8():
    """rendah = terdeteksi (config YOLOv8s): kemungkinan lain butuh keyakinan >= 0,37."""
    K = inferensi.Kotak
    assert dg.kemungkinan_lain(dg.diagnosis([K(0, 0, 9, 9, 0.6, 0), K(0, 0, 9, 9, 0.369, 1)], KELAS, 0.37, 0.37), KELAS, 0.37) is None
    assert dg.kemungkinan_lain(dg.diagnosis([K(0, 0, 9, 9, 0.6, 0), K(0, 0, 9, 9, 0.37, 1)], KELAS, 0.37, 0.37), KELAS, 0.37).penyakit == "brown_spot"


def test_bercak_rendah_hanya_tidak_terdeteksi_dan_minimal_batas_tampil():
    K = inferensi.Kotak
    b = dg.bercak_rendah(_dx([K(0, 0, 9, 9, 0.12, 1), K(0, 0, 9, 9, 0.05, 0)]), 0.10)
    assert b is not None and (b.keyakinan, b.kelas) == (0.12, 1)              # kotak teratas
    assert dg.bercak_rendah(_dx([K(0, 0, 9, 9, 0.10, 1)]), 0.10) is not None  # tepat batas tampilan
    assert dg.bercak_rendah(_dx([K(0, 0, 9, 9, 0.0999, 1)]), 0.10) is None
    assert dg.bercak_rendah(_dx([K(0, 0, 9, 9, 0.30, 1)]), 0.10) is None      # kemungkinan, bukan tidak terdeteksi
    assert dg.bercak_rendah(_dx([K(0, 0, 9, 9, 0.60, 1)]), 0.10) is None
    assert dg.bercak_rendah(_dx([]), 0.10) is None


def test_perlu_cara_membedakan():
    K = inferensi.Kotak
    L = dg.KemungkinanLain
    assert dg.perlu_cara_membedakan(_dx([K(0, 0, 9, 9, 0.6, 0)]), None)                         # Blast utama
    assert dg.perlu_cara_membedakan(_dx([K(0, 0, 9, 9, 0.31, 1)]), None)                        # Brown spot, kemungkinan
    assert not dg.perlu_cara_membedakan(_dx([K(0, 0, 9, 9, 0.6, 2)]), None)                     # BLB tanpa kemungkinan lain
    assert dg.perlu_cara_membedakan(_dx([K(0, 0, 9, 9, 0.6, 2)]), L("brown_spot", 0.3))         # BLB + kemungkinan lain Brown spot
    assert not dg.perlu_cara_membedakan(_dx([K(0, 0, 9, 9, 0.2, 0)]), None)                     # tidak terdeteksi
    assert not dg.perlu_cara_membedakan(_dx([]), None)


def test_mode_cara_membedakan_terbuka_hanya_pasangan_blast_brown_spot():
    K = inferensi.Kotak
    L = dg.KemungkinanLain
    assert dg.mode_cara_membedakan(_dx([K(0, 0, 9, 9, 0.6, 0)]), L("brown_spot", 0.3)) == "terbuka"     # Blast + Brown spot
    assert dg.mode_cara_membedakan(_dx([K(0, 0, 9, 9, 0.6, 1)]), L("leaf_blast", 0.3)) == "terbuka"     # Brown spot + Blast
    assert dg.mode_cara_membedakan(_dx([K(0, 0, 9, 9, 0.6, 0)]), None) == "terlipat"                   # Blast saja
    assert dg.mode_cara_membedakan(_dx([K(0, 0, 9, 9, 0.6, 0)]), L("bacterial_leaf_blight", 0.3)) == "terlipat"
    assert dg.mode_cara_membedakan(_dx([K(0, 0, 9, 9, 0.6, 2)]), L("brown_spot", 0.3)) == "terlipat"   # BLB + Brown spot
    assert dg.mode_cara_membedakan(_dx([K(0, 0, 9, 9, 0.6, 2)]), None) is None                         # BLB saja
    assert dg.mode_cara_membedakan(_dx([K(0, 0, 9, 9, 0.2, 0)]), None) is None                         # tidak terdeteksi


def test_kepala_hasil_kemungkinan_lain():
    from ui import komponen as ui
    s = ui.kepala_hasil("leaf_blast", "terdeteksi", lain=dg.KemungkinanLain("brown_spot", 0.334))
    assert "Kemungkinan lain: <span class=\"utuh\"><strong>Bercak cokelat</strong>" in s and "(33%)" in s
    assert s.index("Blas daun") < s.index("Kemungkinan lain")                                   # di bawah nama penyakit
    assert "Kemungkinan lain" not in ui.kepala_hasil("leaf_blast", "terdeteksi")
    assert "Kemungkinan lain" not in ui.kepala_hasil(None, "tidak_terdeteksi", lain=dg.KemungkinanLain("brown_spot", 0.3))


def test_kepala_hasil_bercak_rendah_teks_persis():
    from ui import komponen as ui
    s = ui.kepala_hasil("brown_spot", "tidak_terdeteksi", keyakinan_rendah=0.184)
    assert ("Sistem melihat kemungkinan bercak cokelat, tetapi keyakinannya terlalu rendah untuk didiagnosis (18%). "
            "Foto ulang dari dekat sesuai petunjuk, atau konfirmasi ke penyuluh.") in s
    assert "Tidak ada penyakit yang dikenali" in s and "tidak menemukan bercak" not in s
    assert "kemungkinan bercak blas daun" in ui.kepala_hasil("leaf_blast", "tidak_terdeteksi", keyakinan_rendah=0.2)
    assert "kemungkinan bercak hawar daun bakteri" in ui.kepala_hasil("bacterial_leaf_blight", "tidak_terdeteksi", keyakinan_rendah=0.2)
    assert "tidak menemukan bercak" in ui.kepala_hasil("brown_spot", "tidak_terdeteksi")         # di bawah batas tampilan


def test_cara_membedakan_dua_ilustrasi_ciri_dan_penyuluh():
    from ui import komponen as ui
    s = ui.cara_membedakan()
    assert s.count("<svg") == 2 and "potong-beda-leaf_blast" in s and "potong-beda-brown_spot" in s   # id clipPath unik
    assert "belah ketupat; pusat abu-abu, tepi cokelat" in s and "oval kecil berwarna cokelat tua; tepinya lebih muda" in s
    assert "penyuluh" in s and 'viewBox="40 6 200 52"' in s
    s2 = ui.cara_membedakan(judul=False)                                 # di dalam lipatan berjudul sama: tanpa judul ganda
    assert "<h3" not in s2 and s2.count("<svg") == 2 and ui.JUDUL_PEMBEDA == "Cara membedakan blas daun dan bercak cokelat"


def test_jejak_mencatat_kemungkinan_lain_tanpa_mengubah_saran(aturan):
    from rbs import mesin
    from ui import komponen as ui
    biasa = mesin.jalankan(mesin.fakta_dari_diagnosis("leaf_blast", "terdeteksi", 0.6), aturan)
    fakta = mesin.fakta_dari_diagnosis("leaf_blast", "terdeteksi", 0.6, dg.KemungkinanLain("brown_spot", 0.334))
    h = mesin.jalankan(fakta, aturan)
    assert h.saran == biasa.saran and [L.aturan.id for L in h.jejak] == ["D01", "R001"]
    s = ui.jejak(h, fakta)
    assert "Kemungkinan lain:" in s and "bercak cokelat" in s and "33%" in s and "tidak mengubah diagnosis dan saran" in s


def test_jejak_bercak_rendah_tanpa_saran_penyakit(aturan):
    from rbs import mesin
    from ui import komponen as ui
    K = inferensi.Kotak
    fakta = mesin.fakta_dari_diagnosis("brown_spot", "tidak_terdeteksi", 0.184, None, K(0, 0, 9, 9, 0.184, 1))
    h = mesin.jalankan(fakta, aturan)
    assert [L.aturan.id for L in h.jejak] == ["N01", "N02"] and all(not s.kelompok for s in h.saran)   # tetap saran umum
    lama = dict(ui.BATAS)
    try:
        ui.atur_batas(0.29, 0.35, 0.10)
        s = ui.jejak(h, fakta)
    finally:
        ui.atur_batas(lama["rendah"], lama["terdeteksi"], lama["tampil"])
    assert "Bercak berkeyakinan rendah:" in s and "18%" in s and "10%" in s and "batas tampilan, bukan ambang diagnosis" in s
    assert "Kemungkinan lain:" not in s


def test_kotak_rendah_putus_putus_netral():
    from core import visual
    im = Image.new("RGB", (800, 600), (60, 120, 60))
    out = visual.gambar_kotak(im, [], "tidak_terdeteksi", kotak_rendah=inferensi.Kotak(200, 200, 600, 400, 0.18, 1))
    a = np.asarray(out)
    tepi = a[400, 210:590]                                               # tepi bawah kotak
    terang = (tepi > 220).all(axis=1)
    asli = (tepi == (60, 120, 60)).all(axis=1)
    assert terang.any() and asli.any()                                  # ada garis dan ada jeda: putus-putus
    warna = {tuple(int(visual.WARNA_TINGKAT[t][i:i + 2], 16) for i in (1, 3, 5)) for t in visual.WARNA_TINGKAT}
    assert not any(tuple(p) in warna for p in a.reshape(-1, 3))         # bukan warna tingkat
    assert (a[150:200, 200:420] == (31, 45, 36)).all(axis=2).any()      # label gelap di atas kotak


class _DetektorPalsu:
    def __init__(self, kotak):
        self.kotak = kotak
        self.masukan = []

    def prediksi(self, arr):
        self.masukan.append(arr.copy())
        return list(self.kotak)


class _PemisahPalsu:
    """Daun = seluruh foto kecuali pita kiri selebar `kiri` piksel; cadangan=True meniru foto tanpa daun yang jelas."""

    def __init__(self, cadangan=False, kiri=0):
        self.cadangan, self.kiri = cadangan, kiri

    def pisahkan(self, rgb):
        m = np.ones(rgb.shape[:2], bool)
        m[:, :self.kiri] = False
        return m, self.cadangan


@pytest.mark.parametrize("berkas_cfg", ["config.yaml", "config_yolov8.yaml", "config_yolo11s_arsip.yaml"])
def test_alur_kemungkinan_lain_dan_bercak_rendah(berkas_cfg, aturan):
    from core import alur, konfigurasi
    if not (konfigurasi.APP_DIR / berkas_cfg).exists():
        pytest.skip(f"{berkas_cfg} hanya ada di folder proyek (tidak ikut repositori)")
    cfg = konfigurasi.muat(konfigurasi.APP_DIR / berkas_cfg)
    K = inferensi.Kotak
    r = cfg.ambang_rendah
    h = alur.periksa_foto(jpeg(400, 300), cfg, _DetektorPalsu([K(10, 10, 90, 60, 0.6, 2), K(100, 100, 150, 140, r + 0.01, 1)]), aturan, _PemisahPalsu())
    assert h.kemungkinan_lain.penyakit == "brown_spot" and h.cara_membedakan == "terlipat" and h.bercak_rendah is None
    assert h.fakta_awal["kemungkinan_lain"] == "brown_spot" and [L.aturan.id for L in h.inferensi.jejak] == ["D01", "R002"]
    hp = alur.periksa_foto(jpeg(400, 300), cfg, _DetektorPalsu([K(10, 10, 90, 60, 0.6, 0), K(100, 100, 150, 140, r + 0.01, 1)]), aturan, _PemisahPalsu())
    assert hp.cara_membedakan == "terbuka" and [L.aturan.id for L in hp.inferensi.jejak] == ["D01", "R001"]
    b = cfg.batas_tampil_rendah
    if b is None:                                                # fitur nonaktif di config ini (lihat test_bercak_rendah_nonaktif_lewat_config)
        return
    h2 = alur.periksa_foto(jpeg(400, 300), cfg, _DetektorPalsu([K(10, 10, 90, 60, b + 0.005, 0)]), aturan, _PemisahPalsu())
    assert h2.diagnosis.tingkat == "tidak_terdeteksi" and h2.bercak_rendah is not None and h2.kemungkinan_lain is None
    assert h2.kotak_digambar == 0 and not h2.cara_membedakan and h2.potongan is None
    assert [L.aturan.id for L in h2.inferensi.jejak] == ["N01", "N02"] and h2.fakta_awal["bercak_keyakinan_rendah"] == "ya"
    h3 = alur.periksa_foto(jpeg(400, 300), cfg, _DetektorPalsu([K(10, 10, 90, 60, b - 0.005, 0)]), aturan, _PemisahPalsu())
    assert h3.bercak_rendah is None and "bercak_keyakinan_rendah" not in h3.fakta_awal


def test_batas_tampil_dari_config_dan_validasi(cfg, tmp_path):
    from core import konfigurasi
    teks = (konfigurasi.APP_DIR / "config.yaml").read_text(encoding="utf-8")
    p = tmp_path / "c.yaml"
    aktif = "bercak_rendah_aktif: true" + chr(10) + "  batas_bercak_rendah: "
    p.write_text(teks.replace("bercak_rendah_aktif: false", aktif + "0.25"), encoding="utf-8")
    assert konfigurasi.muat(p).batas_tampil_rendah == 0.25        # bila diaktifkan, nilai dari config yang dipakai
    p.write_text(teks.replace("bercak_rendah_aktif: false", aktif + "0.33"), encoding="utf-8")
    with pytest.raises(ValueError):
        konfigurasi.muat(p)                                              # batas tampilan tidak boleh melebihi ambang rendah (0,30)


# ---------- alur v9: pemisah daun (KEPUTUSAN_PROYEK.md 2026-10-20)
def test_alur_foto_cadangan_diminta_foto_ulang_tanpa_deteksi(cfg, aturan):
    from core import alur
    det = _DetektorPalsu([inferensi.Kotak(10, 10, 90, 60, 0.9, 0)])
    with pytest.raises(gambar.FotoPerluDiulang, match="Foto ulang"):
        alur.periksa_foto(jpeg(400, 300), cfg, det, aturan, _PemisahPalsu(cadangan=True))
    assert det.masukan == []                                             # tidak didiagnosis sama sekali
    assert issubclass(gambar.FotoPerluDiulang, gambar.FotoDitolak)       # tampil lewat pesan galat yang sama


def test_alur_latar_diganti_abu_tetapi_foto_tampil_asli(cfg, aturan):
    from core import alur
    det = _DetektorPalsu([inferensi.Kotak(150, 100, 300, 200, 0.9, 0)])
    h = alur.periksa_foto(jpeg(400, 300, warna=(40, 120, 60)), cfg, det, aturan, _PemisahPalsu(kiri=100))
    x = det.masukan[0]
    assert (x[:, :100] == 114).all()                                     # latar -> abu 114 (sama dengan data latih v9)
    assert np.abs(x[:, 100:].astype(int) - (40, 120, 60)).max() <= 3     # daun tidak diubah (toleransi JPEG)
    tampil = np.asarray(Image.open(io.BytesIO(h.foto_bertanda)).convert("RGB"))
    assert np.abs(tampil[150, 20].astype(int) - (40, 120, 60)).max() <= 6   # pengguna tetap melihat foto aslinya
    assert h.diagnosis.penyakit == "leaf_blast" and h.diagnosis.tingkat == "terdeteksi"


def test_alur_tanpa_pemisah_untuk_config_arsip(cfg, aturan):
    from core import alur
    det = _DetektorPalsu([])
    alur.periksa_foto(jpeg(400, 300), cfg, det, aturan, None)
    assert (det.masukan[0] != 114).any()


def test_pemisah_daun_asli_foto_tanpa_daun_jadi_cadangan(pemisah):
    m, cadangan = pemisah.pisahkan(np.full((480, 640, 3), 128, np.uint8))   # abu polos: tidak ada vegetasi
    assert cadangan and m.shape == (480, 640) and m.all()


@butuh_data
def test_pemisah_daun_setara_masker_fase1(pemisah, cfg):
    """Jalur aplikasi (baca_foto + pisahkan) menghasilkan cadangan yang sama dan masker yang praktis sama (IoU >= 0,98) dengan masker
    fase 1 (scripts/v9_latar/fase1_pemisah_daun.py, cv2.imread) pada 20 foto VAL acak (seed 0). Uji penuh: tools/uji_pemisah_daun.py."""
    import json
    import random
    info = json.loads((MASKER_VAL / "info.json").read_text(encoding="utf-8"))
    assert info["tau"] == cfg.pemisah_tau
    for f in random.Random(0).sample(sorted(info["cadangan"]), 20):
        arr = gambar.ke_array(gambar.baca_foto((VAL / f).read_bytes(), cfg.ukuran_maks_mb, cfg.sisi_maks_proses))
        m, cad = pemisah.pisahkan(arr)
        assert cad == info["cadangan"][f], f
        if not cad:
            acuan = np.asarray(Image.open(MASKER_VAL / "masker" / (f.rsplit(".", 1)[0] + ".png"))) > 0
            assert acuan.shape == m.shape, f
            assert (m & acuan).sum() / max((m | acuan).sum(), 1) >= 0.98, f


def test_persen_di_bawah_ambang_dibulatkan_ke_bawah():
    """36,7% (tidak terdeteksi, ambang 37%) tidak boleh tertulis 37%."""
    from ui import komponen as ui
    assert "(36%)" in ui.kepala_hasil("brown_spot", "tidak_terdeteksi", keyakinan_rendah=0.3666)
    assert ">36%<" in ui.pita_keyakinan(0.3666, "tidak_terdeteksi") and "36 persen" in ui.pita_keyakinan(0.3666, "tidak_terdeteksi")
    assert ">37%<" in ui.pita_keyakinan(0.3666, "terdeteksi")              # terdeteksi tetap dibulatkan biasa
    assert ui.persen(0.29, turun=True) == "29%" and ui.persen(0.2899, turun=True) == "28%"

