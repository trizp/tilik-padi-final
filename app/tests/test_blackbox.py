"""Uji black-box (10 foto VAL + 4 skenario masukan + skenario penyampaian). Yang dipastikan: perilaku sistem konsisten untuk setiap
masukan (aturan aktif sesuai tingkat, foto ulang saat daun tidak jelas, masukan buruk ditolak). Kebenaran diagnosis tidak diuji di sini;
rujukannya hasil evaluasi model (README)."""
import sys

from conftest import APP, butuh_data

sys.path.insert(0, str(APP / "tools"))

POLA = {"terdeteksi": "D01+R", "kemungkinan": "K01+K02", "tidak_terdeteksi": "N01+N02"}


@butuh_data
def test_blackbox():
    import uji_blackbox
    baris = uji_blackbox.main()
    assert len(baris) == 14 + len(uji_blackbox.FITUR)
    for b in baris[:10] + baris[14:]:
        if b.get("foto_ulang"):                                  # alur v9: tidak didiagnosis, tanpa aturan dan tanpa kotak
            assert b["aturan"] == "" and b["kotak_digambar"] == 0 and b["tingkat"] is None, b
            continue
        assert b["aturan"].startswith(POLA[b["tingkat"]]), b
        assert (b["kotak_digambar"] > 0) == (b["tingkat"] != "tidak_terdeteksi"), b
        # penyampaian tambahan konsisten dengan tingkat; aturan (saran) tidak berubah karenanya
        assert b["kemungkinan_lain"] is None or (b["tingkat"] != "tidak_terdeteksi" and b["kemungkinan_lain"] != b["diagnosis"]), b
        assert b["bercak_rendah"] is None or (b["tingkat"] == "tidak_terdeteksi" and b["bercak_rendah"] >= 0.10), b
        relevan = b["tingkat"] != "tidak_terdeteksi" and bool({b["diagnosis"], b["kemungkinan_lain"]} & {"leaf_blast", "brown_spot"})
        pasangan = {b["diagnosis"], b["kemungkinan_lain"]} == {"leaf_blast", "brown_spot"}
        assert b["cara_membedakan"] == (("terbuka" if pasangan else "terlipat") if relevan else None), b
    assert all(b["sesuai"] for b in baris), [b for b in baris if not b["sesuai"]]      # perilaku sistem, bukan kebenaran diagnosis
