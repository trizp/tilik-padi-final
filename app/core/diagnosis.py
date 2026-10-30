"""Aturan diagnosis per foto (sama dengan evaluasi skripsi dan keputusan tiga tingkat 2026-10-13):

- diagnosis = kelas kotak dengan keyakinan tertinggi;
- terdeteksi       : keyakinan >= ambang_terdeteksi (0,35);
- kemungkinan      : ambang_rendah (0,29) <= keyakinan < ambang_terdeteksi;
- tidak terdeteksi : keyakinan < ambang_rendah, atau tidak ada kotak.

Penyampaian tambahan (tidak mengubah diagnosis, tingkat, atau saran; KEPUTUSAN_PROYEK.md 2026-10-15):
- kemungkinan lain : kotak berkelas penyakit LAIN dari diagnosis utama dengan keyakinan >= ambang_rendah (satu, tertinggi);
                     tidak berlaku pada tingkat tidak terdeteksi;
- bercak berkeyakinan rendah : pada tingkat tidak terdeteksi, kotak teratas bila keyakinannya >= batas tampilan.
                     Batas tampilan, BUKAN ambang diagnosis. Bisa dinonaktifkan di config (tampilan.bercak_rendah_aktif: false).
"""
from dataclasses import dataclass, field

TERDETEKSI, KEMUNGKINAN, TIDAK_TERDETEKSI = "terdeteksi", "kemungkinan", "tidak_terdeteksi"


@dataclass(frozen=True)
class HasilDiagnosis:
    penyakit: str | None            # kode kelas (leaf_blast, brown_spot, bacterial_leaf_blight) atau None
    tingkat: str
    keyakinan: float                # keyakinan kotak teratas (0 bila tanpa kotak)
    kotak: tuple = field(default_factory=tuple)   # semua kotak hasil NMS (untuk digambar sesuai aturan tampilan)


def tentukan_tingkat(keyakinan: float, ambang_terdeteksi: float, ambang_rendah: float) -> str:
    if keyakinan >= ambang_terdeteksi:
        return TERDETEKSI
    if keyakinan >= ambang_rendah:
        return KEMUNGKINAN
    return TIDAK_TERDETEKSI


def diagnosis(kotak, kelas, ambang_terdeteksi: float, ambang_rendah: float) -> HasilDiagnosis:
    if not kotak:
        return HasilDiagnosis(None, TIDAK_TERDETEKSI, 0.0, ())
    top = max(kotak, key=lambda b: b.keyakinan)
    return HasilDiagnosis(kelas[top.kelas], tentukan_tingkat(top.keyakinan, ambang_terdeteksi, ambang_rendah), top.keyakinan, tuple(kotak))


@dataclass(frozen=True)
class KemungkinanLain:
    penyakit: str                   # kode kelas
    keyakinan: float


def kemungkinan_lain(hasil: HasilDiagnosis, kelas, ambang_rendah: float) -> KemungkinanLain | None:
    """Penyakit lain (bukan diagnosis utama) dengan kotak berkeyakinan >= ambang_rendah; satu saja, keyakinan tertinggi."""
    if hasil.tingkat == TIDAK_TERDETEKSI or hasil.penyakit is None:
        return None
    calon = [b for b in hasil.kotak if kelas[b.kelas] != hasil.penyakit and b.keyakinan >= ambang_rendah]
    if not calon:
        return None
    b = max(calon, key=lambda x: x.keyakinan)
    return KemungkinanLain(kelas[b.kelas], b.keyakinan)


def bercak_rendah(hasil: HasilDiagnosis, batas_tampil: float | None):
    """Kotak teratas pada tingkat tidak terdeteksi bila keyakinannya >= batas_tampil (batas tampilan, bukan ambang diagnosis).
    batas_tampil None = fitur dinonaktifkan di config."""
    if batas_tampil is None or hasil.tingkat != TIDAK_TERDETEKSI or not hasil.kotak:
        return None
    top = max(hasil.kotak, key=lambda b: b.keyakinan)
    return top if top.keyakinan >= batas_tampil else None


PERLU_PEMBEDA = {"leaf_blast", "brown_spot"}


def perlu_cara_membedakan(hasil: HasilDiagnosis, lain: KemungkinanLain | None) -> bool:
    """Blast atau Brown spot muncul sebagai diagnosis utama atau kemungkinan lain (bukan pada tingkat tidak terdeteksi)."""
    if hasil.tingkat == TIDAK_TERDETEKSI or hasil.penyakit is None:
        return False
    return hasil.penyakit in PERLU_PEMBEDA or (lain is not None and lain.penyakit in PERLU_PEMBEDA)


TERBUKA, TERLIPAT = "terbuka", "terlipat"


def mode_cara_membedakan(hasil: HasilDiagnosis, lain: KemungkinanLain | None) -> str | None:
    """'terbuka' bila diagnosis utama dan kemungkinan lain adalah pasangan Blast dan Brown spot (KEPUTUSAN_PROYEK.md 2026-10-16);
    'terlipat' bila Blast atau Brown spot muncul tanpa pasangan itu; None bila tidak relevan (BLB saja, atau tidak terdeteksi)."""
    if not perlu_cara_membedakan(hasil, lain):
        return None
    if lain is not None and {hasil.penyakit, lain.penyakit} == PERLU_PEMBEDA:
        return TERBUKA
    return TERLIPAT


def kotak_untuk_digambar(hasil: HasilDiagnosis, kelas, ambang_rendah: float):
    """Kotak berkelas sama dengan diagnosis dan berkeyakinan >= ambang_rendah. Tidak ada kotak bila tidak terdeteksi."""
    if hasil.tingkat == TIDAK_TERDETEKSI or hasil.penyakit is None:
        return []
    return [b for b in hasil.kotak if kelas[b.kelas] == hasil.penyakit and b.keyakinan >= ambang_rendah]
