"""
jalur.py - Pengaman jalur keluaran untuk semua skrip.

Keluaran hanya boleh ke folder project (rice_yolo/...) atau ke folder sementara Windows (%TEMP%, absolut).
Jalur seperti "/Users/..." atau "../../../Users/..." DITOLAK: di Windows jalur itu terbaca sebagai D:\\Users (akar drive
tempat project berada), bukan C:\\Users.
"""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def jalur_keluaran(p, root=ROOT):
    """Kembalikan Path absolut yang aman untuk keluaran; hentikan skrip bila di luar project/TEMP."""
    raw = str(p)
    q = Path(raw)
    if not q.is_absolute() or (raw.startswith(("/", "\\")) and not q.drive):
        if raw.startswith(("/", "\\")):
            sys.exit(f"[DITOLAK] jalur keluaran '{raw}' tanpa huruf drive (di Windows menjadi {Path(raw).resolve()}). "
                     f"Pakai jalur relatif terhadap project atau jalur absolut lengkap (C:\\...).")
        q = root / q
    q = q.resolve()
    izin = [root.resolve(), Path(tempfile.gettempdir()).resolve()]
    if not any(q == b or b in q.parents for b in izin):
        sys.exit(f"[DITOLAK] jalur keluaran {q} berada di luar folder project ({izin[0]}) dan folder sementara ({izin[1]}).")
    return q
