"""Menggambar kotak bercak pada salinan foto yang diperkecil untuk tampilan."""
import io

from PIL import Image, ImageDraw, ImageFont

WARNA_TINGKAT = {"terdeteksi": "#A4442A", "kemungkinan": "#C9A227", "tidak_terdeteksi": "#4F6D8A"}
NETRAL, TINTA = "#F4F4F0", "#1F2D24"     # kotak bercak berkeyakinan rendah: garis putus-putus terang di atas garis gelap


def _putus(d, xy, warna, tebal, panjang, jeda):
    """Persegi panjang bergaris putus-putus (PIL tidak punya gaya garis)."""
    x1, y1, x2, y2 = xy
    for (a, b), (c, e) in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
        n = max(1.0, ((c - a) ** 2 + (e - b) ** 2) ** 0.5)
        t = 0.0
        while t < n:
            u = min(t + panjang, n)
            d.line([(a + (c - a) * t / n, b + (e - b) * t / n), (a + (c - a) * u / n, b + (e - b) * u / n)], fill=warna, width=tebal)
            t += panjang + jeda


def gambar_kotak_rendah(d, xy, ukuran_foto):
    """Kotak netral putus-putus berlabel "keyakinan rendah" (bukan warna tingkat: ini bukan diagnosis)."""
    sisi = max(ukuran_foto)
    tebal = max(4, round(sisi / 120))
    # bercak sangat kecil: kotak diperlebar ke sisi minimum agar terlihat di layar HP (pusat tetap)
    minimum = sisi / 10
    cx, cy = (xy[0] + xy[2]) / 2, (xy[1] + xy[3]) / 2
    lw, lh = max(xy[2] - xy[0], minimum) / 2, max(xy[3] - xy[1], minimum) / 2
    xy = [max(cx - lw, 0), max(cy - lh, 0), min(cx + lw, ukuran_foto[0] - 1), min(cy + lh, ukuran_foto[1] - 1)]
    panjang, jeda = 3 * tebal, 2 * tebal
    _putus(d, [xy[0] - 1, xy[1] - 1, xy[2] + 1, xy[3] + 1], TINTA, tebal + 4, panjang, jeda)
    _putus(d, xy, NETRAL, tebal, panjang, jeda)
    huruf = ImageFont.load_default(size=max(14, round(sisi / 24)))
    teks = "keyakinan rendah"
    l, t, r, b = d.textbbox((0, 0), teks, font=huruf)
    pad = max(4, (b - t) // 3)
    w, h = r - l + 2 * pad, b - t + 2 * pad
    x = min(max(xy[0], 0), ukuran_foto[0] - w)
    y = xy[1] - h - tebal if xy[1] - h - tebal >= 0 else min(xy[3] + tebal, ukuran_foto[1] - h)
    d.rectangle([x, y, x + w, y + h], fill=TINTA)
    d.text((x + pad - l, y + pad - t), teks, fill=NETRAL, font=huruf)


def gambar_kotak(im: Image.Image, kotak, tingkat: str, sisi_maks: int = 1280, kotak_rendah=None) -> Image.Image:
    s = min(1.0, sisi_maks / max(im.size))
    tampil = im.resize((round(im.width * s), round(im.height * s)), Image.BILINEAR) if s < 1 else im.copy()
    d = ImageDraw.Draw(tampil)
    tebal = max(6, round(max(tampil.size) / 90))      # tetap tebal setelah foto diperkecil di layar HP
    warna = WARNA_TINGKAT[tingkat]
    for b in kotak:
        xy = [b.x1 * s, b.y1 * s, b.x2 * s, b.y2 * s]
        luar = max(2, tebal // 3)                      # garis luar Tinta: kotak tetap terlihat di daun terang/kuning
        d.rectangle([xy[0] - luar, xy[1] - luar, xy[2] + luar, xy[3] + luar], outline="#1F2D24", width=tebal + 2 * luar)
        d.rectangle(xy, outline=warna, width=tebal)
    if kotak_rendah is not None:
        b = kotak_rendah
        gambar_kotak_rendah(d, [b.x1 * s, b.y1 * s, b.x2 * s, b.y2 * s], tampil.size)
    return tampil


def ke_jpeg(im: Image.Image, kualitas: int = 85) -> bytes:
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=kualitas)
    return buf.getvalue()


def potongan_diperbesar(im: Image.Image, kotak, tingkat: str, keluaran: int = 480):
    """Potongan persegi di sekitar kotak (keyakinan tertinggi), untuk ditampilkan selebar setengah foto utama.
    Sisi potongan = maks(1,6 x sisi kotak terpanjang, sisi foto terpendek / 6), dibatasi sisi foto terpendek. Dengan tampilan
    setengah lebar foto, perbesaran relatif terhadap foto utama = lebar_foto / (2 x sisi potongan): 3x untuk bercak kecil,
    makin kecil untuk bercak besar. Kembalikan (JPEG bytes, perbesaran)."""
    W, H = im.size
    pendek = min(W, H)
    bw, bh = kotak.x2 - kotak.x1, kotak.y2 - kotak.y1
    sisi = min(max(1.6 * max(bw, bh), pendek / 6), pendek)
    cx, cy = (kotak.x1 + kotak.x2) / 2, (kotak.y1 + kotak.y2) / 2
    x0 = min(max(cx - sisi / 2, 0), W - sisi)
    y0 = min(max(cy - sisi / 2, 0), H - sisi)
    pot = im.crop((round(x0), round(y0), round(x0 + sisi), round(y0 + sisi))).resize((keluaran, keluaran), Image.LANCZOS)
    s = keluaran / sisi
    d = ImageDraw.Draw(pot)
    xy = [(kotak.x1 - x0) * s, (kotak.y1 - y0) * s, (kotak.x2 - x0) * s, (kotak.y2 - y0) * s]
    d.rectangle([xy[0] - 3, xy[1] - 3, xy[2] + 3, xy[3] + 3], outline="#1F2D24", width=9)
    d.rectangle(xy, outline=WARNA_TINGKAT[tingkat], width=5)
    return ke_jpeg(pot), W / (2 * sisi)
