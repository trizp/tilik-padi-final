"""Mesin forward chaining.

Siklus: cari aturan yang belum pernah aktif dan semua syaratnya benar terhadap memori kerja -> aktifkan aturan pertama (urutan di
rules.yaml) -> tambahkan fakta baru dan saran -> ulangi sampai tidak ada aturan baru yang aktif. Setiap aturan aktif paling banyak
sekali (refraksi). Jejak menyimpan urutan aturan aktif, syarat yang terpenuhi, fakta baru, dan alasannya.
"""
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

OPERATOR = {
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    ">=": lambda a, b: a is not None and float(a) >= float(b),
    "<": lambda a, b: a is not None and float(a) < float(b),
    "ada": lambda a, b: a is not None,
}


@dataclass(frozen=True)
class Aturan:
    id: str
    nama: str
    jika: tuple
    maka: dict
    saran: tuple = ()        # (teks, kelompok)
    sumber: str = ""
    alasan: str = ""


@dataclass
class LangkahJejak:
    putaran: int
    aturan: Aturan
    syarat_terpenuhi: list
    fakta_baru: dict


@dataclass(frozen=True)
class Saran:
    aturan: str          # id aturan asal
    teks: str
    kelompok: str = ""   # "S" tindakan sekarang, "P" pencegahan musim berikutnya, "" umum


@dataclass
class HasilInferensi:
    fakta: dict
    saran: list = field(default_factory=list)          # Saran
    jejak: list = field(default_factory=list)          # LangkahJejak
    putaran: int = 0


def _baca(path):
    path = Path(path) if path else Path(__file__).with_name("rules.yaml")
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def muat_meta(path=None) -> dict:
    """Sumber aturan, catatan tetap, dan nama kelompok saran."""
    d = _baca(path)
    return {"sumber_aturan": d.get("sumber_aturan", ""), "catatan_tetap": d.get("catatan_tetap", ""),
            "kelompok": d.get("kelompok", {})}


def _saran(a):
    out = []
    for s in a.get("saran", []):
        if isinstance(s, dict):
            if s.get("kelompok", "") not in ("S", "P", ""):
                raise ValueError(f"{a['id']}: kelompok saran tidak dikenal {s.get('kelompok')}")
            out.append((s["teks"], s.get("kelompok", "")))
        else:
            out.append((str(s), ""))
    return tuple(out)


def muat_aturan(path=None):
    data = _baca(path)
    out, ids = [], set()
    for a in data["aturan"]:
        if a["id"] in ids:
            raise ValueError(f"id aturan ganda: {a['id']}")
        ids.add(a["id"])
        for s in a["jika"]:
            if s["op"] not in OPERATOR:
                raise ValueError(f"{a['id']}: operator tidak dikenal {s['op']}")
        out.append(Aturan(a["id"], a["nama"], tuple(a["jika"]), dict(a.get("maka", {})), _saran(a),
                          a.get("sumber", ""), a.get("alasan", "")))
    return out


def _isi(nilai, fakta):
    if isinstance(nilai, str):
        return re.sub(r"\{(\w+)\}", lambda m: str(fakta.get(m.group(1), m.group(0))), nilai)
    return nilai


def jalankan(fakta_awal: dict, aturan, maks_putaran: int = 100) -> HasilInferensi:
    mem = dict(fakta_awal)
    hasil = HasilInferensi(fakta=mem)
    aktif = set()
    for putaran in range(1, maks_putaran + 1):
        kandidat = None
        for a in aturan:
            if a.id in aktif:
                continue
            if all(OPERATOR[s["op"]](mem.get(s["fakta"]), s.get("nilai")) for s in a.jika):
                kandidat = a
                break
        if kandidat is None:
            hasil.putaran = putaran - 1
            return hasil
        aktif.add(kandidat.id)
        baru = {k: _isi(v, mem) for k, v in kandidat.maka.items()}
        mem.update(baru)
        hasil.saran += [Saran(kandidat.id, *(s if isinstance(s, tuple) else (s, ""))) for s in kandidat.saran]
        hasil.jejak.append(LangkahJejak(putaran, kandidat, [dict(s, nilai_fakta=mem.get(s["fakta"])) for s in kandidat.jika], baru))
    raise RuntimeError(f"forward chaining tidak berhenti dalam {maks_putaran} putaran")


def fakta_dari_diagnosis(penyakit, tingkat, keyakinan, kemungkinan_lain=None, bercak_rendah=None) -> dict:
    """Fakta awal. Fakta tambahan (kemungkinan lain, bercak berkeyakinan rendah) hanya dicatat untuk jejak; tidak ada aturan yang
    memakainya, jadi diagnosis dan saran tidak berubah."""
    f = {"penyakit": penyakit or "tidak_ada", "tingkat": tingkat, "keyakinan": round(float(keyakinan), 4)}
    if kemungkinan_lain is not None:
        f["kemungkinan_lain"] = kemungkinan_lain.penyakit
        f["keyakinan_kemungkinan_lain"] = round(float(kemungkinan_lain.keyakinan), 4)
    if bercak_rendah is not None:
        f["bercak_keyakinan_rendah"] = "ya"
    return f
