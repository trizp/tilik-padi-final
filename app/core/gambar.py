"""Membaca dan memeriksa foto unggahan. Foto hanya diproses di memori; tidak ada yang ditulis ke disk."""
import io

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

FORMAT_DIIZINKAN = {"JPEG", "PNG", "MPO"}       # MPO = JPEG multi-gambar dari sebagian kamera HP


class FotoDitolak(ValueError):
    """Foto tidak bisa diproses. Pesannya ditulis untuk pengguna: apa yang salah dan cara memperbaikinya."""


class FotoPerluDiulang(FotoDitolak):
    """Foto terbaca, tetapi pemisah daun tidak menemukan satu helai daun yang jelas (foto cadangan). Tidak didiagnosis."""


def baca_foto(data: bytes, ukuran_maks_mb: float = 10.0, sisi_maks: int = 2048) -> Image.Image:
    """bytes -> PIL RGB yang orientasinya sudah dikoreksi sesuai EXIF (ImageOps.exif_transpose).
    Foto yang sisi terpanjangnya > sisi_maks diperkecil (JPEG langsung didekode kecil lewat draft) agar hemat memori; model
    hanya melihat 640 px, jadi foto <= sisi_maks (termasuk semua foto uji kesetaraan) tidak berubah."""
    if not data:
        raise FotoDitolak("Berkas kosong. Pilih atau ambil foto daun lagi.")
    if len(data) > ukuran_maks_mb * 1024 * 1024:
        mb = f"{len(data) / 1024 / 1024:.1f}".replace(".", ",")          # koma desimal
        raise FotoDitolak(f"Ukuran foto {mb} MB, lebih dari batas {ukuran_maks_mb:.0f} MB. "
                          "Kecilkan foto atau ambil ulang dengan kamera HP.")
    try:
        with Image.open(io.BytesIO(data)) as uji:
            fmt = uji.format
            uji.verify()                          # memeriksa berkas rusak tanpa memuat seluruh piksel
    except (UnidentifiedImageError, OSError, SyntaxError):
        raise FotoDitolak("Berkas ini bukan foto yang bisa dibaca. Unggah foto berformat JPG atau PNG.") from None
    except Image.DecompressionBombError:
        raise FotoDitolak("Resolusi foto terlalu besar. Kecilkan foto lalu unggah lagi.") from None
    if fmt not in FORMAT_DIIZINKAN:
        raise FotoDitolak(f"Format {fmt} tidak didukung. Unggah foto berformat JPG atau PNG.")
    im = Image.open(io.BytesIO(data))
    if im.format in ("JPEG", "MPO") and max(im.size) > sisi_maks:
        im.draft("RGB", (sisi_maks, sisi_maks))   # dekode berskala 1/2, 1/4, 1/8 (tetap >= sisi_maks)
    im = ImageOps.exif_transpose(im)
    im = im.convert("RGB")
    if max(im.size) > sisi_maks:
        im.thumbnail((sisi_maks, sisi_maks), Image.BILINEAR)
    if min(im.size) < 32:
        raise FotoDitolak("Foto terlalu kecil untuk diperiksa. Ambil foto daun dari jarak 15–30 cm.")
    return im


def ke_array(im: Image.Image) -> np.ndarray:
    return np.asarray(im, dtype=np.uint8)
