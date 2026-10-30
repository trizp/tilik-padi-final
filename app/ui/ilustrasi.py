"""Ilustrasi SVG buatan sendiri (tanpa foto dataset). Ciri bercak mengikuti bab 2.2 proposal:
- blas daun: bercak belah ketupat, pusat abu-abu/putih, tepi cokelat, sering dikelilingi kuning pucat;
- bercak cokelat: bercak oval kecil cokelat tua dengan tepi lebih muda;
- hawar daun bakteri: menguning kecokelatan dari ujung dan tepi daun, batas bergelombang.
Warna ilustrasi (alami, redup) hanya dipakai di dalam gambar; teks tetap memakai palet aplikasi."""
from html import escape

DAUN, URAT, URAT_TENGAH = "#6F9A57", "#5C8747", "#93B57C"
TINTA, PADI = "#1F2D24", "#2F6B3B"

# bentuk helai daun memanjang: pangkal lebar di kiri, ujung runcing di kanan (viewBox 0 0 320 64)
HELAI = "M4 18 C 90 13, 190 12, 246 17 C 278 21, 300 27, 316 32 C 300 37, 278 43, 246 47 C 190 52, 90 51, 4 46 Z"


def ikon_padi(ukuran=28) -> str:
    """Ikon padi satu garis (stroke): batang, malai merunduk dengan bulir, dan satu helai daun. Warna Daun padi."""
    return (f'<svg class="ikon-padi" width="{ukuran}" height="{ukuran}" viewBox="0 0 24 24" fill="none" stroke="{PADI}" '
            'stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false">'
            '<path d="M7 22.5C8.4 16.5 9.4 11 11 6"/>'
            '<path d="M11 6c2.6-1.1 5.2.4 6.6 3.8.9 2.4.9 4.8 0 7"/>'
            '<path d="M14.3 6.2l1.3-2.1M16.6 8.1l2.2-1.2M17.9 11.2l2.4-.5M18.1 14.5l2.4.3"/>'
            '<path d="M9.7 13C7.2 12 5.2 10 4 7"/></svg>')


def _dasar(isi: str, label: str, id_: str) -> str:
    urat = "".join(f'<path d="M6 {y} C 100 {y - 3}, 200 {y - 3}, 300 {31 + (y - 32) * 0.15:.1f}" stroke="{URAT}" stroke-width="0.8" fill="none" opacity="0.7"/>'
                   for y in (23, 27, 37, 41))
    return (f'<svg class="helai" viewBox="0 0 320 64" role="img" aria-label="{escape(label)}" preserveAspectRatio="xMidYMid meet">'
            f'<defs><clipPath id="potong-{id_}"><path d="{HELAI}"/></clipPath></defs>'
            f'<path d="{HELAI}" fill="{DAUN}"/>'
            f'<g clip-path="url(#potong-{id_})">{urat}'
            f'<path d="M6 32 C 100 30, 200 30, 314 32" stroke="{URAT_TENGAH}" stroke-width="1.6" fill="none"/>{isi}</g>'
            f'<path d="{HELAI}" fill="none" stroke="#4E7A3C" stroke-width="1"/></svg>')


def _belah_ketupat(cx, cy, p, l):
    return f"M{cx - p} {cy} L{cx} {cy - l} L{cx + p} {cy} L{cx} {cy + l} Z"


def blas(id_="blas") -> str:
    isi = ""
    for cx, cy, p, l in ((70, 30, 20, 7), (150, 36, 16, 6), (215, 28, 13, 5), (110, 40, 9, 3.5)):
        isi += (f'<ellipse cx="{cx}" cy="{cy}" rx="{p + 7}" ry="{l + 4}" fill="#E2D88C" opacity="0.85"/>'
                f'<path d="{_belah_ketupat(cx, cy, p, l)}" fill="#7B4A26"/>'
                f'<path d="{_belah_ketupat(cx, cy, p * 0.62, l * 0.55)}" fill="#D3D5CE"/>')
    return _dasar(isi, "Ilustrasi blas daun: bercak belah ketupat, pusat abu-abu, tepi cokelat", id_)


def bercak_cokelat(id_="bercak") -> str:
    titik = ((40, 27), (62, 38), (88, 30), (104, 41), (131, 26), (150, 35), (176, 40), (194, 28), (222, 36), (246, 31), (120, 33), (75, 22))
    isi = "".join(f'<ellipse cx="{x}" cy="{y}" rx="6.2" ry="3.6" fill="#B58149"/><ellipse cx="{x}" cy="{y}" rx="4" ry="2.3" fill="#55301A"/>'
                  for x, y in titik)
    return _dasar(isi, "Ilustrasi bercak cokelat: bercak oval kecil cokelat tua dengan tepi lebih muda", id_)


def hawar(id_="hawar") -> str:
    # daerah menguning dari ujung (kanan) dan menjalar di tepi atas, batas bergelombang
    zona = ("M320 0 L320 64 L250 64 C 244 58, 236 60, 232 53 C 228 46, 220 50, 214 43 C 209 37, 202 41, 196 34 "
            "C 190 28, 182 32, 174 25 C 167 19, 158 24, 150 18 C 142 13, 133 19, 124 14 C 116 10, 106 15, 98 11 "
            "C 90 8, 82 12, 76 9 L 76 0 Z")
    isi = (f'<path d="{zona}" fill="#D8BF6B"/>'
           f'<path d="{zona}" fill="none" stroke="#A88A3C" stroke-width="1.4" stroke-dasharray="0" />'
           '<path d="M320 0 L320 64 L270 64 C 262 54, 258 46, 256 40 C 252 34, 246 30, 240 26 C 230 20, 222 18, 214 14 L 214 0 Z" fill="#BE9A4E"/>')
    return _dasar(isi, "Ilustrasi hawar daun bakteri: daun menguning kecokelatan dari ujung dan tepi, batas bergelombang", id_)


ILUSTRASI = {"leaf_blast": blas, "brown_spot": bercak_cokelat, "bacterial_leaf_blight": hawar}


def penyakit(kode: str, id_: str | None = None, dekat: bool = False) -> str:
    """dekat=True: bagian tengah helai diperbesar (untuk perbandingan blas daun vs bercak cokelat)."""
    s = ILUSTRASI[kode](id_ or kode)
    return s.replace('viewBox="0 0 320 64"', 'viewBox="40 6 200 52"', 1).replace('class="helai"', 'class="helai dekat"', 1) if dekat else s


def foto_benar() -> str:
    """Daun di tanamannya, satu helai memenuhi bingkai (dengan sudut jendela bidik)."""
    return ('<svg class="petunjuk-gambar" viewBox="0 0 160 120" role="img" aria-label="Contoh benar: satu helai daun di tanamannya memenuhi bingkai foto">'
            '<rect x="0" y="0" width="160" height="120" fill="#E4ECDF"/>'
            '<path d="M18 120 C 30 80, 40 50, 46 0" stroke="#9DBA8A" stroke-width="6" fill="none"/>'
            '<path d="M138 120 C 130 86, 126 50, 132 0" stroke="#9DBA8A" stroke-width="6" fill="none"/>'
            '<path d="M60 120 C 66 90, 74 60, 70 20" stroke="#A9C497" stroke-width="5" fill="none"/>'
            f'<path d="M20 112 C 50 92, 100 46, 150 10 L 156 18 C 108 54, 58 100, 30 120 Z" fill="{DAUN}"/>'
            '<path d="M26 114 C 60 90, 104 50, 152 14" stroke="#93B57C" stroke-width="1.4" fill="none"/>'
            '<ellipse cx="88" cy="66" rx="7" ry="3.2" fill="#B58149" transform="rotate(-38 88 66)"/>'
            '<ellipse cx="88" cy="66" rx="4.4" ry="2" fill="#55301A" transform="rotate(-38 88 66)"/>'
            f'<g stroke="{TINTA}" stroke-width="3" fill="none" stroke-linecap="square">'
            '<path d="M8 22 V8 H22"/><path d="M138 8 H152 V22"/><path d="M152 98 V112 H138"/><path d="M22 112 H8 V98"/></g>'
            '</svg>')


def foto_salah() -> str:
    """Daun dipetik dan diletakkan di atas kertas putih."""
    return ('<svg class="petunjuk-gambar" viewBox="0 0 160 120" role="img" aria-label="Contoh salah: daun dipetik diletakkan di atas kertas putih">'
            '<rect x="0" y="0" width="160" height="120" fill="#CFC9BD"/>'
            '<rect x="22" y="14" width="116" height="92" fill="#FFFFFF" stroke="#B9B3A6" stroke-width="1"/>'
            f'<path d="M40 74 C 70 66, 104 52, 124 40 L 126 44 C 106 56, 72 72, 42 78 Z" fill="#A9B86A"/>'
            '<path d="M42 76 C 72 68, 104 54, 125 42" stroke="#C7C98A" stroke-width="1" fill="none"/>'
            f'<g stroke="{TINTA}" stroke-width="3" stroke-linecap="round"><path d="M128 92 L144 108"/><path d="M144 92 L128 108"/></g></svg>')


def tanda_benar() -> str:
    return (f'<svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true" focusable="false"><path d="M3 9.5 L7.5 14 L15 4.5" '
            f'stroke="{PADI}" stroke-width="2.6" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>')


def tanda_salah() -> str:
    return (f'<svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true" focusable="false"><path d="M4 4 L14 14 M14 4 L4 14" '
            f'stroke="{TINTA}" stroke-width="2.6" fill="none" stroke-linecap="round"/></svg>')
