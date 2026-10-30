"""Alur lengkap satu foto: baca -> pisahkan daun (latar diganti abu) -> deteksi -> diagnosis tiga tingkat (+ kemungkinan lain / bercak berkeyakinan rendah)
-> forward chaining -> foto bertanda kotak.
Dipakai oleh tampilan Streamlit dan oleh uji black-box (jalur yang sama persis)."""
from dataclasses import dataclass

from core import diagnosis as dg
from core import gambar, pemisah_daun, visual
from rbs import mesin


@dataclass
class HasilPemeriksaan:
    foto_bertanda: bytes            # JPEG siap tampil (lebih kecil daripada PNG bawaan st.image)
    diagnosis: dg.HasilDiagnosis
    inferensi: mesin.HasilInferensi
    fakta_awal: dict
    kotak_digambar: int
    potongan: bytes | None = None   # JPEG potongan diperbesar dari kotak berkeyakinan tertinggi (None bila tidak terdeteksi)
    perbesaran: float = 0.0
    kemungkinan_lain: dg.KemungkinanLain | None = None   # penyakit lain >= ambang_rendah (bukan pada tidak terdeteksi)
    bercak_rendah: object = None    # Kotak teratas berkeyakinan rendah (tidak terdeteksi, >= batas tampilan); digambar putus-putus
    cara_membedakan: str | None = None   # perbandingan blas daun vs bercak cokelat: "terbuka", "terlipat", atau None


PESAN_FOTO_ULANG = "Daun tidak terlihat jelas pada foto ini. Foto ulang satu helai daun dari dekat sesuai petunjuk memotret."


def periksa_foto(data: bytes, cfg, detektor, aturan, pemisah) -> HasilPemeriksaan:
    """Melempar gambar.FotoDitolak bila berkas bukan foto yang valid, dan gambar.FotoPerluDiulang bila pemisah daun tidak menemukan
    satu helai daun yang jelas. pemisah = None: tanpa pemisahan daun (config arsip v7). Foto tidak ditulis ke disk.
    Detektor melihat foto dengan latar abu; yang ditampilkan ke pengguna tetap foto aslinya."""
    im = gambar.baca_foto(data, cfg.ukuran_maks_mb, cfg.sisi_maks_proses)
    arr = gambar.ke_array(im)
    if pemisah is not None:
        masker, cadangan = pemisah.pisahkan(arr)
        if cadangan:
            raise gambar.FotoPerluDiulang(cfg.teks.get("pesan_foto_ulang", PESAN_FOTO_ULANG))
        arr = pemisah_daun.ganti_latar(arr, masker)
        del masker
    kotak = detektor.prediksi(arr)
    del arr
    hasil = dg.diagnosis(kotak, cfg.kelas, cfg.ambang_terdeteksi, cfg.ambang_rendah)
    gambar_kotak = dg.kotak_untuk_digambar(hasil, cfg.kelas, cfg.ambang_rendah)
    lain = dg.kemungkinan_lain(hasil, cfg.kelas, cfg.ambang_rendah)
    rendah = dg.bercak_rendah(hasil, cfg.batas_tampil_rendah)
    tampil = visual.ke_jpeg(visual.gambar_kotak(im, gambar_kotak, hasil.tingkat, kotak_rendah=rendah))
    fakta = mesin.fakta_dari_diagnosis(hasil.penyakit, hasil.tingkat, hasil.keyakinan, lain, rendah)
    inf = mesin.jalankan(fakta, aturan)
    potongan, perbesaran = None, 0.0
    if gambar_kotak:
        teratas = max(gambar_kotak, key=lambda b: b.keyakinan)
        potongan, perbesaran = visual.potongan_diperbesar(im, teratas, hasil.tingkat)
    del im
    return HasilPemeriksaan(tampil, hasil, inf, fakta, len(gambar_kotak), potongan, perbesaran, lain, rendah,
                            dg.mode_cara_membedakan(hasil, lain))
