"""Tampilan aplikasi "Tilik Padi" (Streamlit). Logika ada di core/ dan rbs/; berkas ini hanya menampilkan.

Jalankan dari folder app/:  streamlit run streamlit_app.py
"""
import gc

import streamlit as st
import streamlit.components.v1 as components

from core import alur, gambar, inferensi, konfigurasi, pemisah_daun
from rbs import mesin
from ui import komponen as ui

st.set_page_config(page_title="Tilik Padi", page_icon=None, layout="wide", initial_sidebar_state="collapsed")


@st.cache_resource(show_spinner=False)
def sumber_daya():
    cfg = konfigurasi.muat()
    det = inferensi.Detektor(cfg.model_path, cfg.imgsz, cfg.conf_min, cfg.iou, cfg.max_det)
    pemisah = pemisah_daun.PemisahDaun(cfg.pemisah_path, cfg.pemisah_tau) if cfg.pemisah_path else None
    ui.atur_batas(cfg.ambang_rendah, cfg.ambang_terdeteksi, cfg.batas_tampil_rendah)
    return cfg, det, pemisah, mesin.muat_aturan(), mesin.muat_meta()


def html(s: str):
    st.markdown(s, unsafe_allow_html=True)


# ---------------------------------------------------------------- status sesi
def _awal():
    for k, v in {"n": 0, "kamera_buka": False, "sumber": None, "digulir": None, "gulir_atas": False}.items():
        st.session_state.setdefault(k, v)


def _alih_kamera():
    st.session_state.kamera_buka = not st.session_state.kamera_buka


def _dari(sumber):
    def f():
        st.session_state.sumber = sumber
    return f


def _ulang():
    st.session_state.n += 1                     # kunci widget baru = unggahan dan foto kamera lama terhapus
    st.session_state.kamera_buka = False
    st.session_state.sumber = None
    st.session_state.gulir_atas = True


def berkas_aktif():
    n = st.session_state.n
    u = st.session_state.get(f"unggah_{n}")
    k = st.session_state.get(f"kamera_{n}") if st.session_state.kamera_buka else None
    if st.session_state.sumber == "kamera" and k is not None:
        return k
    if st.session_state.sumber == "unggah" and u is not None:
        return u
    return u or k


def aksi(cfg):
    """Dua aksi: kamera (buka/tutup) dan unggah (file uploader bergaya tombol). Di HP bertumpuk penuh lebar."""
    n = st.session_state.n
    k1, k2 = st.columns(2, gap="small")
    with k1:
        st.button("Tutup kamera" if st.session_state.kamera_buka else "Ambil foto daun", type="primary", width="stretch",
                  on_click=_alih_kamera, key=f"tombol_kamera_{n}")
    with k2:
        st.file_uploader("Unggah foto", type=list(cfg.jenis_berkas), accept_multiple_files=False, label_visibility="collapsed",
                         key=f"unggah_{n}", on_change=_dari("unggah"))      # filter jenis: di HP pemilih langsung membuka galeri
        html(ui.catatan_unggah(cfg.teks))
    if st.session_state.kamera_buka:
        st.camera_input("Arahkan kamera ke satu helai daun, lalu tekan Ambil foto.", key=f"kamera_{n}", on_change=_dari("kamera"))


# ---------------------------------------------------------------- halaman
def halaman_diagnosis():
    _awal()
    cfg, det, pemisah, aturan, meta = sumber_daya()
    t = cfg.teks
    html(ui.navigasi("beranda"))

    with st.container(key="label_id"):
        components.html(ui.skrip_label(), height=0)               # label Indonesia untuk tombol bawaan (tanpa CSS :has)

    hasil, galat = None, None
    berkas = berkas_aktif()
    if berkas is not None:
        try:
            data = berkas.getvalue()
            hasil = alur.periksa_foto(data, cfg, det, aturan, pemisah)
        except gambar.FotoDitolak as e:
            galat = str(e)
        finally:
            data = None
            gc.collect()

    if st.session_state.gulir_atas:
        st.session_state.gulir_atas = False
        with st.container(key="gulir_atas"):
            components.html(ui.skrip_gulir("atas"), height=0)

    if hasil is None:
        # ---- halaman awal: kiri = judul, aksi, petunjuk; kanan = tiga penyakit (>= 1024 px)
        with st.container(key="tata_awal"):
            kiri, kanan = st.columns([1.1, 1], gap="large")
            with kiri:
                html(ui.judul_awal(t))
                aksi(cfg)
                if galat:
                    html(ui.pesan_salah(galat))
                html(ui.petunjuk(t))
            with kanan:
                html(ui.tiga_penyakit())
        return

    # ---- halaman hasil: aksi di atas, hasil langsung di bawahnya
    with st.container(key="aksi_hasil"):
        aksi(cfg)
    html('<div id="hasil" class="jangkar-hasil"></div>')
    d = hasil.diagnosis
    with st.container(key="tata_hasil"):
        kiri, kanan = st.columns([1, 1.05], gap="large")
        with kiri:
            keterangan = ("Kotak menandai bercak yang dikenali sistem." if hasil.kotak_digambar else
                          "Garis putus-putus menandai bercak berkeyakinan rendah. Ini bukan diagnosis." if hasil.bercak_rendah else None)
            st.image(hasil.foto_bertanda, width="stretch", caption=keterangan)
            if hasil.potongan:
                html(ui.potongan(hasil.potongan, d.penyakit))
        with kanan:
            html(ui.kepala_hasil(d.penyakit, d.tingkat, ada_potongan=bool(hasil.potongan), lain=hasil.kemungkinan_lain,
                                 keyakinan_rendah=hasil.bercak_rendah.keyakinan if hasil.bercak_rendah else None))
            html(ui.pita_keyakinan(d.keyakinan, d.tingkat))
            if hasil.cara_membedakan == "terbuka":            # pasangan Blast dan Brown spot
                html(ui.cara_membedakan())
            elif hasil.cara_membedakan == "terlipat":
                with st.expander(ui.JUDUL_PEMBEDA):
                    html(ui.cara_membedakan(judul=False))
            html(ui.daftar_saran(hasil.inferensi.saran, meta))
            with st.expander("Mengapa sistem menyimpulkan ini"):
                html(ui.jejak(hasil.inferensi, hasil.fakta_awal))
            html(ui.peringatan(t["peringatan_singkat"]))
            st.page_link(HALAMAN_TENTANG, label="Baca selengkapnya di halaman Tentang")
    with st.container(key="akhir_hasil"):
        st.button("Periksa foto lain", type="secondary", on_click=_ulang, key=f"lagi_{st.session_state.n}")
        with st.expander("Cara memotret"):
            html(f'<p class="judul-lipat">{t["judul"]}. {t["subjudul"]}</p>')
            html(ui.petunjuk(t, judul=False))
            html(ui.tiga_penyakit())

    penanda = (st.session_state.n, getattr(berkas, "file_id", None))
    if st.session_state.digulir != penanda:
        st.session_state.digulir = penanda
        with st.container(key="gulir_hasil"):
            components.html(ui.skrip_gulir("hasil"), height=0)


def teks_tingkat(cfg) -> str:
    t, r = round(100 * cfg.ambang_terdeteksi), round(100 * cfg.ambang_rendah)
    b = round(100 * cfg.batas_tampil_rendah) if cfg.batas_tampil_rendah is not None else None
    rendah = (f" Bila bercak teratas berkeyakinan {b}% sampai di bawah {r}%, bercak itu tetap ditandai dengan garis putus-putus berlabel "
              "<strong>keyakinan rendah</strong> supaya Anda bisa memeriksanya sendiri. Ini bukan diagnosis, sehingga saran obat tidak "
              f"diberikan. Batas tampilan {b}% dipilih dari data validasi agar tanda ini jarang muncul pada daun sehat."
              if b is not None and cfg.batas_tampil_rendah < cfg.ambang_rendah else "")
    lain = (f" Bila ada bercak penyakit lain dengan keyakinan {r}% atau lebih, namanya ditampilkan sebagai <strong>kemungkinan "
            "lain</strong>; diagnosis dan saran tetap mengikuti bercak yang paling diyakini.")
    if cfg.ambang_rendah >= cfg.ambang_terdeteksi:
        return (f"Ada dua tingkat jawaban. Tingkat keyakinan {t}% atau lebih berarti <strong>terdeteksi</strong>. Di bawah {t}% berarti "
                "<strong>tidak terdeteksi</strong>: sistem tidak menyimpulkan penyakit. Tingkat \"kemungkinan\" tidak dipakai, karena "
                f"menurut prosedur pemilihan pada data validasi batas bawahnya jatuh tepat pada {t}%." + rendah + lain)
    return (f"Tingkat keyakinan {t}% atau lebih berarti <strong>terdeteksi</strong>. Antara {r}% dan {t}% berarti "
            f"<strong>kemungkinan</strong> (belum pasti). Di bawah {r}% berarti <strong>tidak terdeteksi</strong>." + rendah + lain)


def halaman_tentang():
    cfg, _, pemisah, _, meta = sumber_daya()
    html(ui.navigasi("tentang"))
    sumber = "".join(
        f"<li><strong>{s['nama']}</strong>. {s.get('keterangan', '')}"
        + (f"<br>{s['sitasi']}" if s.get("sitasi") else "")
        + (f"<br>Lisensi: {s['lisensi']}." if s.get("lisensi") else "") + "</li>"
        for s in cfg.sumber_data)
    langkah_pemisah = (
        "<li>Model segmentasi YOLOv8n-seg mencari satu helai daun pada foto, lalu latar di sekitarnya ditutup dengan warna abu supaya "
        "penilaian hanya didasarkan pada daun. Bila tidak ada daun yang terlihat jelas, sistem tidak memberi diagnosis dan meminta Anda "
        "memotret ulang.</li>\n" if pemisah else "")
    peringatan_foto_ulang = (
        "<li>Foto BLB dan foto daun yang dipegang tangan lebih sering diminta foto ulang, karena pada foto seperti itu sistem sering tidak "
        "menemukan satu helai daun yang jelas. Dalam pengujian, 29% foto BLB diminta foto ulang. Untuk hasil terbaik, potret daun yang "
        "masih di tanamannya.</li>\n" if pemisah else "")
    ev = cfg.hasil_evaluasi
    evaluasi = (f'<h2>Hasil pengujian</h2>\n<p>{ev.get("pengantar", "")}</p>\n<ul>'
                + "".join(f"<li>{b}</li>" for b in ev.get("butir", [])) + "</ul>\n\n" if ev else "")
    html(f"""
<div class="tentang">
<h1 class="judul">Tentang Tilik Padi</h1>
<p>Tilik Padi membantu mengenali tiga penyakit daun padi dari foto: blas daun (Blast), bercak cokelat (Brown spot), dan hawar
daun bakteri (BLB). Aplikasi ini dibuat untuk penelitian skripsi di Universitas Telkom.</p>

<h2>Cara kerja</h2>
<ol>
{langkah_pemisah}<li>Model deteksi objek {cfg.nama_model} mencari bercak penyakit pada {"daun itu" if pemisah else "foto"} dan memberi tingkat keyakinan untuk setiap bercak.</li>
<li>Bercak dengan tingkat keyakinan tertinggi menentukan diagnosis. {teks_tingkat(cfg)}</li>
<li>Sistem berbasis aturan (forward chaining) memilih saran tindakan dari basis aturan. Sumber basis aturan: {meta['sumber_aturan']}
Langkah penalarannya bisa dibuka di bagian "Mengapa sistem menyimpulkan ini".</li>
</ol>

<h2>Peringatan</h2>
<ul>
<li>Model dilatih terutama dengan foto lapangan dari Bangladesh. Pada foto dari Indonesia, ketepatannya masih terbatas, terutama untuk
Blast.</li>
{peringatan_foto_ulang}<li>Diagnosis BLB belum dapat diandalkan untuk daun yang dipetik, atau daun yang menguning karena sebab lain (misalnya Tungro). Dalam
pengujian dengan daun petik dari Indonesia, sistem sering menyebut daun Tungro sebagai BLB.</li>
<li>Sistem hanya mengenal Blast, Brown spot, dan BLB. Penyakit lain (misalnya Tungro, Leaf smut, Sheath blight) bisa salah dikenali
sebagai salah satu dari ketiganya.</li>
<li>Hasil aplikasi ini adalah alat bantu. Selalu konfirmasi ke penyuluh pertanian atau petugas POPT sebelum menyemprot obat.
{meta['catatan_tetap']}</li>
</ul>

{evaluasi}<h2>Privasi foto</h2>
<p>Foto Anda tidak disimpan. Foto hanya diproses di memori server untuk pemeriksaan ini dan tidak ditulis ke disk. Foto terhapus dari
memori ketika Anda memeriksa foto lain atau menutup halaman.</p>

<h2>Sumber data latih</h2>
<ul class="sumber">{sumber}</ul>

<h2>Gambar di aplikasi ini</h2>
<p>Gambar daun dan petunjuk memotret adalah ilustrasi buatan sendiri, bukan foto dari dataset.</p>

<h2>Model dan perangkat lunak</h2>
<p>{"Model YOLOv8n-seg (pemisah daun) dan " if pemisah else "Model "}{cfg.nama_model} dari Ultralytics (lisensi AGPL-3.0), dilatih ulang
untuk {"memisahkan daun dan " if pemisah else ""}tiga penyakit lalu dijalankan dengan ONNX Runtime di CPU. Antarmuka dibuat dengan Streamlit.</p>
</div>""")


HALAMAN_DIAGNOSIS = st.Page(halaman_diagnosis, title="Tilik Padi", url_path="beranda", default=True)
HALAMAN_TENTANG = st.Page(halaman_tentang, title="Tentang Tilik Padi", url_path="tentang")
html(f"<style>{ui.css()}</style>")
st.navigation([HALAMAN_DIAGNOSIS, HALAMAN_TENTANG], position="hidden").run()
