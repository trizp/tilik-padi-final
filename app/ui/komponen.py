"""Potongan HTML untuk tampilan. Semua teks dari luar di-escape."""
import base64
import math
from html import escape
from pathlib import Path

from ui import ilustrasi as il

PENYAKIT = {
    "leaf_blast": {"nama": "Blas daun", "istilah": "Blast", "patogen": "Magnaporthe oryzae", "jenis": "jamur",
                   "ciri": "Bercak belah ketupat berwarna abu-abu atau putih dengan tepi cokelat."},
    "brown_spot": {"nama": "Bercak cokelat", "istilah": "Brown spot", "patogen": "Bipolaris oryzae", "jenis": "jamur",
                   "ciri": "Bercak oval kecil berwarna cokelat tua dengan tepi yang lebih muda."},
    "bacterial_leaf_blight": {"nama": "Hawar daun bakteri", "istilah": "BLB", "patogen": "Xanthomonas oryzae pv. oryzae", "jenis": "bakteri",
                              "ciri": "Daun menguning kecokelatan dari ujung atau tepi daun, dengan batas bergelombang."},
}
LABEL_TINGKAT = {"terdeteksi": "Terdeteksi", "kemungkinan": "Kemungkinan", "tidak_terdeteksi": "Tidak terdeteksi"}
NAMA_FAKTA = {"penyakit": "penyakit hasil deteksi", "tingkat": "tingkat", "keyakinan": "tingkat keyakinan", "diagnosis": "diagnosis",
              "dugaan": "dugaan", "perlu_konfirmasi": "perlu konfirmasi", "perlu_foto_ulang": "perlu foto ulang", "rekomendasi": "jenis saran",
              "kemungkinan_lain": "kemungkinan lain", "keyakinan_kemungkinan_lain": "tingkat keyakinan kemungkinan lain",
              "bercak_keyakinan_rendah": "bercak berkeyakinan rendah ditampilkan"}
NILAI_FAKTA = {"leaf_blast": "blas daun", "brown_spot": "bercak cokelat", "bacterial_leaf_blight": "hawar daun bakteri", "tidak_ada": "tidak ada",
               "terdeteksi": "terdeteksi", "kemungkinan": "kemungkinan", "tidak_terdeteksi": "tidak terdeteksi", "ya": "ya",
               "khusus": "khusus penyakit", "umum": "umum"}
OP = {"==": "adalah", "!=": "bukan", ">=": "minimal", "<": "kurang dari", "ada": "tersedia"}


def css() -> str:
    return (Path(__file__).with_name("gaya.css")).read_text(encoding="utf-8")


def persen(k: float, turun: bool = False) -> str:
    """turun=True untuk keyakinan di bawah ambang terdeteksi: dibulatkan ke bawah, supaya misalnya 36,7% tidak tertulis "37%"
    (= ambang terdeteksi YOLOv8s) pada hasil yang tidak terdeteksi."""
    return f"{math.floor(100 * k + 1e-9):d}%" if turun else f"{round(100 * k):d}%"


# ---------------------------------------------------------------- kepala dan halaman awal
def navigasi(aktif: str) -> str:
    a = ' aria-current="page"' if aktif == "beranda" else ""
    b = ' aria-current="page"' if aktif == "tentang" else ""
    return (f'<nav class="nav-atas" aria-label="Navigasi utama"><a class="merek" href="./" target="_self"{a}>{il.ikon_padi()}'
            f'<span>Tilik Padi</span></a><a href="tentang" target="_self"{b}>Tentang</a></nav>')


def judul_awal(t: dict) -> str:
    return f'<h1 class="judul">{escape(t["judul"])}</h1><p class="subjudul">{escape(t["subjudul"])}</p>'


def catatan_unggah(t: dict) -> str:
    return f'<p class="catatan-unggah">{escape(t["catatan_unggah"])}</p>'


def tiga_penyakit(judul: bool = True) -> str:
    butir = ""
    for kode, p in PENYAKIT.items():
        butir += (f'<li class="penyakit">{il.penyakit(kode, "awal-" + kode)}'
                  f'<p class="penyakit-nama">{escape(p["nama"])} <span class="istilah">({escape(p["istilah"])})</span></p>'
                  f'<p class="penyakit-patogen"><em>{escape(p["patogen"])}</em></p>'
                  f'<p class="penyakit-ciri">{escape(p["ciri"])}</p></li>')
    kepala = '<h2 class="judul-bagian">Tiga penyakit yang dikenali</h2>' if judul else ""
    return f'<section class="tiga-penyakit" aria-label="Tiga penyakit yang dikenali">{kepala}<ul>{butir}</ul></section>'


def petunjuk(t: dict, judul: bool = True) -> str:
    kepala = '<h2 class="judul-bagian">Cara memotret</h2>' if judul else ""
    return (f'<section class="petunjuk" aria-label="Cara memotret">{kepala}<div class="petunjuk-dua">'
            f'<figure class="petunjuk-benar">{il.foto_benar()}<figcaption><strong>{il.tanda_benar()} Seperti ini.</strong> '
            f'{escape(t["petunjuk_benar"])}</figcaption></figure>'
            f'<figure class="petunjuk-salah">{il.foto_salah()}<figcaption><strong>{il.tanda_salah()} Bukan seperti ini.</strong> '
            f'{escape(t["petunjuk_salah"])}</figcaption></figure></div>'
            f'<p class="petunjuk-jarak">{escape(t["petunjuk_jarak_cahaya"])}</p></section>')


# ---------------------------------------------------------------- hasil
def _bercak_nama(penyakit: str) -> str:
    """'bercak blas daun', 'bercak cokelat' (tanpa 'bercak bercak'), 'bercak hawar daun bakteri'."""
    n = PENYAKIT[penyakit]["nama"].lower()
    return n if n.startswith("bercak") else "bercak " + n


def teks_bercak_rendah(penyakit: str, keyakinan: float) -> str:
    return (f"Sistem melihat kemungkinan {_bercak_nama(penyakit)}, tetapi keyakinannya terlalu rendah untuk didiagnosis "
            f"({persen(keyakinan, turun=True)}). Foto ulang dari dekat sesuai petunjuk, atau konfirmasi ke penyuluh.")


def kepala_hasil(penyakit, tingkat: str, ada_potongan: bool = False, lain=None, keyakinan_rendah: float | None = None) -> str:
    """lain: kemungkinan lain (objek dengan .penyakit dan .keyakinan), hanya untuk terdeteksi/kemungkinan.
    keyakinan_rendah: keyakinan bercak berkeyakinan rendah (tingkat tidak terdeteksi, >= batas tampilan)."""
    lab = f'<span class="tingkat tingkat-{tingkat}">{LABEL_TINGKAT[tingkat]}</span>'
    if tingkat == "tidak_terdeteksi" or penyakit is None:
        if keyakinan_rendah is not None and penyakit in PENYAKIT:
            isi = f'<p class="catatan-rendah">{escape(teks_bercak_rendah(penyakit, keyakinan_rendah))}</p>'
        else:
            isi = '<p class="patogen-polos">Sistem tidak menemukan bercak Blast, Brown spot, atau BLB pada foto ini.</p>'
        return f'<div class="hasil-kepala">{lab}<h2 class="nama-penyakit">Tidak ada penyakit yang dikenali</h2>{isi}</div>'
    p = PENYAKIT[penyakit]
    awal = "Dugaan: " if tingkat == "kemungkinan" else ""
    kelas = "hasil-ilustrasi ada-potongan" if ada_potongan else "hasil-ilustrasi"
    baris_lain = ""
    if lain is not None and lain.penyakit in PENYAKIT and lain.penyakit != penyakit:
        baris_lain = (f'<p class="kemungkinan-lain">Kemungkinan lain: <span class="utuh"><strong>{escape(PENYAKIT[lain.penyakit]["nama"])}</strong> '
                      f'<span class="angka">({persen(lain.keyakinan)})</span></span></p>')
    return (f'<div class="hasil-kepala">{lab}<div class="hasil-baris"><div class="hasil-nama">'
            f'<h2 class="nama-penyakit">{awal}{escape(p["nama"])}</h2>'
            f'<p class="patogen">{escape(p["istilah"])}, disebabkan {p["jenis"]} <em>{escape(p["patogen"])}</em></p>{baris_lain}</div>'
            f'<div class="{kelas}">{il.penyakit(penyakit, "hasil-" + penyakit)}</div></div>'
            f'<p class="penyakit-ciri ciri-hasil">Ciri: {escape(p["ciri"][0].lower() + p["ciri"][1:])}</p></div>')


def potongan(jpeg: bytes, penyakit: str) -> str:
    b64 = base64.b64encode(jpeg).decode()
    p = PENYAKIT[penyakit]
    return (f'<div class="potongan"><figure class="potongan-foto"><img src="data:image/jpeg;base64,{b64}" '
            f'alt="Potongan foto yang diperbesar di sekitar bercak dengan tingkat keyakinan tertinggi">'
            f'<figcaption>Bercak yang paling diyakini sistem</figcaption></figure>'
            f'<figure class="potongan-ilustrasi">{il.penyakit(penyakit, "potong-" + penyakit)}<figcaption>Gambaran ciri {escape(p["nama"].lower())}</figcaption></figure></div>')


CIRI_PEMBEDA = {   # bab 2.2 proposal
    "leaf_blast": "Bercak belah ketupat; pusat abu-abu, tepi cokelat.",
    "brown_spot": "Bercak oval kecil berwarna cokelat tua; tepinya lebih muda.",
}


JUDUL_PEMBEDA = "Cara membedakan blas daun dan bercak cokelat"


def cara_membedakan(judul: bool = True) -> str:
    """Blas daun dan bercak cokelat berdampingan (ilustrasi diperbesar) dengan ciri bab 2.2 dan saran konfirmasi ke penyuluh.
    judul=False: untuk di dalam lipatan yang sudah berjudul sama."""
    fig = "".join(
        f'<figure>{il.penyakit(k, "beda-" + k, dekat=True)}<figcaption><strong>{escape(PENYAKIT[k]["nama"])}</strong> '
        f'({escape(PENYAKIT[k]["istilah"])})<br>{escape(CIRI_PEMBEDA[k])}</figcaption></figure>' for k in ("leaf_blast", "brown_spot"))
    return (f'<section class="beda{"" if judul else " dalam-lipatan"}" aria-label="{JUDUL_PEMBEDA}">'
            + (f'<h3 class="judul-bagian">{JUDUL_PEMBEDA}</h3>' if judul else "") +
            f'<div class="beda-dua">{fig}</div>'
            '<p class="beda-saran">Kedua penyakit ini mudah tertukar, juga oleh sistem. Bandingkan bentuk dan warna bercak pada daun Anda '
            'dengan gambar di atas. Bila ragu, tunjukkan daunnya ke penyuluh pertanian atau petugas POPT sebelum menyemprot fungisida.</p>'
            '</section>')


def pita_keyakinan(k: float, tingkat: str) -> str:
    """Label zona dan angka keyakinan saja; angka batas ada di panel jejak. Skala bertingkat 36/28/36 (lihat docs/desain.md)."""
    x = posisi_pita(k)
    zona = [("tidak_terdeteksi", "tidak terdeteksi", 36), ("kemungkinan", "kemungkinan", 28), ("terdeteksi", "terdeteksi", 36)]
    if BATAS["terdeteksi"] <= BATAS["rendah"]:
        zona = [("tidak_terdeteksi", "tidak terdeteksi", 50), ("terdeteksi", "terdeteksi", 50)]
    label = "".join(f'<span class="pk-zona-label{" aktif" if z == tingkat else ""}" style="width:{w}%">{t}</span>' for z, t, w in zona)
    bar = "".join(f'<span class="pk-zona pk-{z}{" aktif" if z == tingkat else ""}" style="width:{w}%"></span>' for z, _, w in zona)
    angka = persen(k, turun=tingkat != "terdeteksi")
    aria = f"Tingkat keyakinan {angka[:-1]} persen, zona {LABEL_TINGKAT[tingkat].lower()}"
    return (f'<div class="pita" role="img" aria-label="{escape(aria)}">'
            f'<p class="pk-judul" aria-hidden="true">Tingkat keyakinan <strong class="angka">{angka}</strong></p>'
            f'<div class="pk-label" aria-hidden="true">{label}</div>'
            f'<div class="pk-bar" aria-hidden="true">{bar}<span class="pk-penanda" style="left:{x:.2f}%"></span></div></div>')


BATAS = {"rendah": 0.29, "terdeteksi": 0.35, "tampil": 0.10}       # diisi dari config oleh atur_batas(); tampil None = nonaktif


def atur_batas(rendah: float, terdeteksi: float, tampil: float | None = None):
    BATAS.update(rendah=rendah, terdeteksi=terdeteksi, tampil=tampil)


def posisi_pita(k: float, rendah: float | None = None, terdeteksi: float | None = None) -> float:
    """Keyakinan (0-1) -> posisi % pada pita bertingkat 36/28/36."""
    rendah = BATAS["rendah"] if rendah is None else rendah
    terdeteksi = BATAS["terdeteksi"] if terdeteksi is None else terdeteksi
    k = min(max(k, 0.0), 1.0)
    if terdeteksi <= rendah:                          # tanpa tingkat "kemungkinan": pita dua zona 50/50
        return 50.0 * k / terdeteksi if k < terdeteksi else 50.0 + 50.0 * (k - terdeteksi) / (1.0 - terdeteksi)
    if k < rendah:
        return 36.0 * k / rendah
    if k < terdeteksi:
        return 36.0 + 28.0 * (k - rendah) / (terdeteksi - rendah)
    return 64.0 + 36.0 * (k - terdeteksi) / (1.0 - terdeteksi)


def daftar_saran(saran, meta: dict) -> str:
    """Butir aturan penyakit: 'Tindakan sekarang' (S) lebih dulu, lalu 'Pencegahan musim berikutnya' (P); kata-kata tidak diubah.
    Butir umum (tanpa kelompok) di bawah judul 'Yang perlu dilakukan'."""
    if not saran:
        return ""
    nama = meta.get("kelompok", {})
    bagian = []
    for kel in ("S", "P"):
        b = [s for s in saran if s.kelompok == kel]
        if b:
            bagian.append(f'<h4 class="kelompok-saran">{escape(nama.get(kel, kel))}</h4><ul class="saran">'
                          + "".join(f"<li>{escape(s.teks)}</li>" for s in b) + "</ul>")
    umum = [s for s in saran if not s.kelompok]
    if umum:
        bagian.append('<ol class="saran">' + "".join(f"<li>{escape(s.teks)}</li>" for s in umum) + "</ol>")
    khusus = any(s.kelompok for s in saran)
    sumber = (f'<p class="sumber-aturan">Sumber aturan: {escape(meta.get("sumber_aturan", ""))}</p>' if khusus else "")
    catatan = f'<p class="catatan-tetap">{escape(meta.get("catatan_tetap", ""))}</p>' if meta.get("catatan_tetap") else ""
    return f'<h3 class="judul-bagian">Yang perlu dilakukan</h3>{"".join(bagian)}{catatan}{sumber}'


def _nilai(v, turun: bool = False):
    if isinstance(v, float):
        return persen(v, turun)
    return NILAI_FAKTA.get(str(v), str(v))


def jejak(hasil_inferensi, fakta_awal: dict) -> str:
    turun = fakta_awal.get("tingkat") != "terdeteksi"
    awal = "".join(f"<li>{escape(NAMA_FAKTA.get(k, k))}: <strong>{escape(_nilai(v, turun and k == 'keyakinan'))}</strong></li>"
                   for k, v in fakta_awal.items())
    if BATAS["terdeteksi"] <= BATAS["rendah"]:
        batas = (f"<p>Batas tingkat: <strong>tidak terdeteksi</strong> di bawah {persen(BATAS['terdeteksi'])}; <strong>terdeteksi</strong> "
                 f"{persen(BATAS['terdeteksi'])} atau lebih. Tingkat keyakinan diambil dari bercak yang paling diyakini sistem.</p>")
    else:
        batas = (f"<p>Batas tingkat: <strong>tidak terdeteksi</strong> di bawah {persen(BATAS['rendah'])}; <strong>kemungkinan</strong> "
                 f"{persen(BATAS['rendah'])} sampai di bawah {persen(BATAS['terdeteksi'])}; <strong>terdeteksi</strong> "
                 f"{persen(BATAS['terdeteksi'])} atau lebih. Tingkat keyakinan diambil dari bercak yang paling diyakini sistem.</p>")
    catatan = []
    if "kemungkinan_lain" in fakta_awal:
        catatan.append(f"<li><strong>Kemungkinan lain:</strong> selain bercak yang paling diyakini, sistem juga menemukan bercak "
                       f"{escape(_nilai(fakta_awal['kemungkinan_lain']))} dengan tingkat keyakinan "
                       f"{escape(_nilai(fakta_awal.get('keyakinan_kemungkinan_lain', 0.0)))} (minimal {persen(BATAS['rendah'])}). "
                       "Hanya ditampilkan sebagai informasi; diagnosis dan saran tetap mengikuti bercak yang paling diyakini.</li>")
    if fakta_awal.get("bercak_keyakinan_rendah") == "ya":
        catatan.append(f"<li><strong>Bercak berkeyakinan rendah:</strong> bercak {escape(_nilai(fakta_awal.get('penyakit')))} dengan tingkat "
                       f"keyakinan {escape(_nilai(fakta_awal.get('keyakinan', 0.0), True))}, di bawah {persen(BATAS['rendah'])}, sehingga tidak "
                       f"didiagnosis. Kotaknya digambar dengan garis putus-putus karena keyakinannya minimal {persen(BATAS['tampil'] or 0.0)} "
                       "(batas tampilan, bukan ambang diagnosis). Saran penanganan penyakit tidak diberikan.</li>")
    catatan_html = (f"<p>Catatan tampilan (tidak mengubah diagnosis dan saran):</p><ul>{''.join(catatan)}</ul>" if catatan else "")
    langkah = []
    for L in hasil_inferensi.jejak:
        syarat = "; ".join(f"{escape(NAMA_FAKTA.get(s['fakta'], s['fakta']))} {OP[s['op']]} {escape(_nilai(s.get('nilai')))}" for s in L.aturan.jika)
        baru = ", ".join(f"{escape(NAMA_FAKTA.get(k, k))} = {escape(_nilai(v))}" for k, v in L.fakta_baru.items())
        sumber = f'<br><span class="kecil">Sumber: {escape(L.aturan.sumber)}</span>' if L.aturan.sumber else ""
        langkah.append(f"<li><strong>Putaran {L.putaran}: aturan {escape(L.aturan.id)}, {escape(L.aturan.nama[0].lower() + L.aturan.nama[1:])}.</strong><br>"
                       f"Syarat terpenuhi: {syarat}.<br>Fakta baru: {baru}.<br>{escape(L.aturan.alasan)}{sumber}</li>")
    if not langkah:
        langkah.append("<li>Tidak ada aturan yang terpenuhi.</li>")
    akhir = f"<p>Putaran {hasil_inferensi.putaran + 1}: tidak ada aturan baru yang terpenuhi, jadi penalaran berhenti.</p>"
    return (f'<div class="jejak">{batas}<p>Fakta awal dari hasil pemeriksaan foto:</p><ul>{awal}</ul>{catatan_html}'
            f"<p>Aturan yang aktif secara berurutan (forward chaining):</p><ol>{''.join(langkah)}</ol>{akhir}</div>")


def peringatan(teks: str) -> str:
    return f'<div class="peringatan"><p>{escape(teks)}</p></div>'


def pesan_salah(teks: str) -> str:
    return f'<div class="pesan-salah" role="alert"><p><strong>Foto belum bisa diperiksa.</strong> {escape(teks)}</p></div>'


def skrip_gulir(target: str) -> str:
    """Gulir instan (tanpa animasi) ke elemen #target di halaman induk; dipanggil sekali per hasil baru."""
    if target == "atas":
        aksi = "(d.querySelector('[data-testid=\"stMain\"]')||d.scrollingElement).scrollTo(0,0);"
    else:
        aksi = f"var e=d.getElementById('{target}'); if(e) e.scrollIntoView({{block:'start'}});"
    return f"<script>var d=window.parent.document; setTimeout(function(){{ {aksi} }}, 60);</script>"


LABEL_INDONESIA = {"Take Photo": "Ambil foto", "Take photo": "Ambil foto", "Clear photo": "Hapus foto dan ambil ulang",
                   "Clear Photo": "Hapus foto dan ambil ulang"}


def skrip_label() -> str:
    """Mengganti teks tombol kamera bawaan Streamlit dan nama aksesibel tombol unggah ke bahasa Indonesia, tanpa CSS :has().
    Hanya nilai simpul teks dan atribut aria-label yang diubah (tidak menambah/menghapus elemen yang dikelola React).
    Bila skrip gagal, teks bawaan tetap terlihat (tidak pernah tombol kosong)."""
    import json
    peta = json.dumps(LABEL_INDONESIA)
    return ("<script>(function(){var d=window.parent.document; var peta=" + peta + ";"
            "function ubah(){"
            " d.querySelectorAll('[data-testid=stCameraInputButton]').forEach(function(b){"
            "  var w=d.createTreeWalker(b,4), n; while((n=w.nextNode())){ var t=n.nodeValue.trim(); if(peta[t]) n.nodeValue=' '+peta[t]; } });"
            " d.querySelectorAll('[data-testid=stFileUploaderDropzone] button[data-testid=stBaseButton-secondary]').forEach(function(b){"
            "  if(b.getAttribute('aria-label')!=='Unggah foto') b.setAttribute('aria-label','Unggah foto'); });"
            " d.querySelectorAll('[data-testid=stFileUploaderDropzone] button[data-testid=stBaseButton-borderlessIcon]').forEach(function(b){"
            "  if(b.getAttribute('aria-label')!=='Unggah foto lain') b.setAttribute('aria-label','Unggah foto lain'); });"
            "}"
            "ubah(); if(!window.__pengamatLabel){ window.__pengamatLabel=new MutationObserver(ubah);"
            " window.__pengamatLabel.observe(d.body,{subtree:true,childList:true,characterData:true}); }"
            "})();</script>")
