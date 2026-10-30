"""Unit test basis aturan (setiap aturan) dan mesin forward chaining."""
from pathlib import Path

import pytest

from rbs import mesin


def jalankan(aturan, penyakit, tingkat, keyakinan=0.5):
    return mesin.jalankan(mesin.fakta_dari_diagnosis(penyakit, tingkat, keyakinan), aturan)


def ids(hasil):
    return [L.aturan.id for L in hasil.jejak]


# ---------- setiap aturan
# Tabel 3.1 proposal (Kementerian Pertanian RI, 2023): kata-kata butir harus PERSIS sama, beserta kelompoknya.
TABEL_3_1 = {
    "R001": [("P", "Pemupukan berimbang, hindari pupuk N lebih dari 90 kg/Ha."),
             ("P", "Tanam varietas tahan dan lakukan pergiliran varietas secara berkala."),
             ("P", "Hindari penggunaan benih dari daerah endemis blas; lakukan perlakuan benih dengan fungisida."),
             ("P", "Atur waktu tanam agar fase keluar malai tidak bertepatan dengan musim embun tinggi."),
             ("S", "Bakar jerami dari pertanaman yang terinfeksi untuk mengurangi sumber inokulum."),
             ("S", "Aplikasikan fungisida berbahan aktif: trisiklazol, isoprotiolan, heksakonazol, atau azoksistrobin + difenokonazol.")],
    "R002": [("P", "Tanam varietas yang tahan penyakit hawar daun bakteri."),
             ("S", "Bersihkan tunggul dan jerami yang terinfeksi dari areal pertanaman."),
             ("P", "Lakukan seleksi dan perlakuan benih menggunakan agens hayati sesuai anjuran."),
             ("P", "Jika menggunakan kompos jerami, pastikan jerami sakit telah terdekomposisi sempurna sebelum tanam."),
             ("S", "Aplikasikan agens hayati pada umur 14, 28, dan 42 HST di sore hari."),
             ("P", "Gunakan pupuk N sesuai takaran anjuran.")],
    "R003": [("P", "Gunakan benih bermutu, perbaiki tata air, dan lakukan pemupukan berimbang."),
             ("P", "Sanitasi lahan dan pergiliran tanaman dengan tanaman bukan inang untuk mematikan patogen di dalam tanah."),
             ("P", "Lakukan perlakuan benih dengan perendaman air panas yang dicampur fungisida."),
             ("P", "Tanam varietas yang tahan atau toleran terhadap Brown Spot."),
             ("S", "Aplikasikan fungisida berbahan aktif: difenokonazol, tebukonazol, heksakonazol, mankozeb, atau azoksistrobin + difenokonazol.")],
}


@pytest.mark.parametrize("penyakit,kode", [("leaf_blast", "R001"), ("bacterial_leaf_blight", "R002"), ("brown_spot", "R003")])
def test_aturan_penyakit_terdeteksi(aturan, penyakit, kode):
    h = jalankan(aturan, penyakit, "terdeteksi", 0.8)
    assert ids(h) == ["D01", kode]                       # dua tahap: diagnosis diterima -> tindakan khusus
    assert h.fakta["diagnosis"] == penyakit
    assert h.fakta["rekomendasi"] == "khusus"
    assert {s.aturan for s in h.saran} == {kode}
    assert [(s.kelompok, s.teks) for s in h.saran] == TABEL_3_1[kode]     # kata-kata dan kelompok persis Tabel 3.1


def test_meta_sumber_dan_catatan_tetap():
    m = mesin.muat_meta()
    assert "Teknik Mengendalikan Hama dan Penyakit Padi" in m["sumber_aturan"] and "2023" in m["sumber_aturan"]
    assert m["catatan_tetap"] == "Gunakan pestisida sesuai dosis pada label dan anjuran penyuluh setempat."
    assert m["kelompok"] == {"S": "Tindakan sekarang", "P": "Pencegahan musim berikutnya"}


ARSIP = Path(mesin.__file__).with_name("rules_slide14_arsip.yaml")


@pytest.mark.skipif(not ARSIP.exists(), reason="arsip aturan versi slide hanya ada di folder proyek (tidak ikut repositori)")
def test_arsip_slide14_tetap_bisa_dibaca():
    a = mesin.muat_aturan(ARSIP)
    assert {x.id for x in a} >= {"R001", "R002", "R003"}


def test_D01_tidak_aktif_tanpa_penyakit(aturan):
    h = jalankan(aturan, None, "terdeteksi")
    assert "D01" not in ids(h)


@pytest.mark.parametrize("penyakit", ["leaf_blast", "brown_spot", "bacterial_leaf_blight"])
def test_K01_K02_kemungkinan(aturan, penyakit):
    h = jalankan(aturan, penyakit, "kemungkinan", 0.31)
    assert ids(h) == ["K01", "K02"]
    assert h.fakta["dugaan"] == penyakit                 # templat {penyakit} disalin
    assert h.fakta["perlu_konfirmasi"] == "ya"
    assert "diagnosis" not in h.fakta                    # tidak ada obat khusus untuk hasil belum pasti
    assert all(s.aturan == "K02" and s.kelompok == "" for s in h.saran)
    assert any("penyuluh" in s.teks for s in h.saran)


def test_N01_N02_tidak_terdeteksi(aturan):
    h = jalankan(aturan, None, "tidak_terdeteksi", 0.05)
    assert ids(h) == ["N01", "N02"]
    assert h.fakta["perlu_foto_ulang"] == "ya"
    assert all(s.aturan == "N02" for s in h.saran)


def test_tidak_terdeteksi_dengan_penyakit_berkeyakinan_rendah(aturan):
    h = jalankan(aturan, "brown_spot", "tidak_terdeteksi", 0.2)
    assert ids(h) == ["N01", "N02"]
    assert "diagnosis" not in h.fakta


def test_tidak_ada_R004_sampai_R006(aturan):
    assert not {a.id for a in aturan} & {"R004", "R005", "R006"}


def test_setiap_aturan_punya_alasan_dan_saran_tertulis(aturan):
    for a in aturan:
        assert a.alasan, a.id
        if a.id.startswith(("R", "K02", "N02")):
            assert a.saran and a.sumber, a.id


# ---------- mesin forward chaining
def test_tanpa_aturan_aktif(aturan):
    h = mesin.jalankan({"penyakit": "tidak_ada", "tingkat": "lainnya", "keyakinan": 0.0}, aturan)
    assert h.jejak == [] and h.saran == [] and h.putaran == 0
    assert h.fakta == {"penyakit": "tidak_ada", "tingkat": "lainnya", "keyakinan": 0.0}


def test_basis_aturan_kosong():
    h = mesin.jalankan({"tingkat": "terdeteksi"}, [])
    assert h.jejak == [] and h.putaran == 0


def test_rantai_berurutan_dan_refraksi():
    A = mesin.Aturan
    a = [A("X2", "dua", ({"fakta": "b", "op": "==", "nilai": 1},), {"c": 1}, ("s2",)),
         A("X1", "satu", ({"fakta": "a", "op": "==", "nilai": 1},), {"b": 1}, ("s1",)),
         A("X3", "selalu", ({"fakta": "a", "op": "ada"},), {}, ())]
    h = mesin.jalankan({"a": 1}, a)
    assert ids(h) == ["X1", "X2", "X3"]                   # X2 baru aktif setelah X1 menambah fakta b
    assert h.fakta["c"] == 1 and h.putaran == 3
    assert [L.putaran for L in h.jejak] == [1, 2, 3]
    assert [(s.aturan, s.teks) for s in h.saran] == [("X1", "s1"), ("X2", "s2")]


def test_operator_numerik():
    A = mesin.Aturan
    a = [A("T", "tinggi", ({"fakta": "keyakinan", "op": ">=", "nilai": 0.35},), {"tinggi": "ya"}),
         A("L", "rendah", ({"fakta": "keyakinan", "op": "<", "nilai": 0.29},), {"rendah": "ya"})]
    assert ids(mesin.jalankan({"keyakinan": 0.35}, a)) == ["T"]
    assert ids(mesin.jalankan({"keyakinan": 0.2}, a)) == ["L"]
    assert ids(mesin.jalankan({}, a)) == []               # fakta tidak ada -> syarat numerik salah


def test_batas_putaran():
    A = mesin.Aturan
    a = [A("A", "a", ({"fakta": "x", "op": "ada"},), {"y": 1}), A("B", "b", ({"fakta": "y", "op": "ada"},), {"z": 1})]
    with pytest.raises(RuntimeError):
        mesin.jalankan({"x": 1}, a, maks_putaran=1)


def test_jejak_mencatat_syarat_dan_fakta_baru(aturan):
    h = jalankan(aturan, "leaf_blast", "terdeteksi", 0.66)
    L = h.jejak[0]
    assert L.fakta_baru == {"diagnosis": "leaf_blast"}
    assert {s["fakta"] for s in L.syarat_terpenuhi} == {"tingkat", "penyakit"}


def test_muat_aturan_menolak_id_ganda_dan_operator_salah(tmp_path):
    p = tmp_path / "r.yaml"
    p.write_text("aturan:\n- {id: A, nama: a, jika: [{fakta: x, op: '==', nilai: 1}]}\n"
                 "- {id: A, nama: b, jika: [{fakta: x, op: '==', nilai: 1}]}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        mesin.muat_aturan(p)
    p.write_text("aturan:\n- {id: A, nama: a, jika: [{fakta: x, op: '~', nilai: 1}]}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        mesin.muat_aturan(p)
