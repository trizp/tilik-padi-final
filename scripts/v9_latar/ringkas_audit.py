#!/usr/bin/env python3
"""
ringkas_audit.py - Ringkas penilaian manusia atas lembar audit masker v9 (50 foto VAL di luar sampel tahap 0).
Masukan: reports/v9_latar/audit_masker/penilaian.csv (dari lembar_audit.html) + daftar_audit.csv (sumber, kelas, tanda cadangan sistem).
Keluaran: reports/v9_latar/audit_masker/ringkasan_audit.{json,md}. Hanya membaca; tidak mengubah dataset, masker, atau model.
"""
import csv
import json
import math
from collections import Counter
from pathlib import Path

D = Path(__file__).resolve().parents[2] / "reports/v9_latar/audit_masker"
KAT = ["baik", "salah_objek", "terpotong", "cadangan"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    a = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    return (round(100 * (c - a), 1), round(100 * (c + a), 1))


def main():
    sis = {r["berkas"]: r for r in csv.DictReader(open(D / "daftar_audit.csv", encoding="utf-8"))}
    nil = list(csv.DictReader(open(D / "penilaian.csv", encoding="utf-8")))
    assert len(nil) == 50 and all(r["berkas"] in sis for r in nil) and all(r["penilaian"] in KAT for r in nil), "penilaian tidak lengkap"
    for r in nil:
        r.update(sumber=sis[r["berkas"]]["sumber"], kelas=sis[r["berkas"]]["kelas"], sistem_cadangan=sis[r["berkas"]]["sistem_cadangan"])
    grup = {"semua": nil, "ds4 (Blast, Brown spot, sehat)": [r for r in nil if r["sumber"] == "ds4"],
            "ds1 BLB provisional": [r for r in nil if r["sumber"] == "ds1_provisional"]}
    for k in ("leaf_blast", "brown_spot", "background"):
        grup[f"ds4 {k}"] = [r for r in nil if r["sumber"] == "ds4" and r["kelas"] == k]
    ring = {}
    for g, rs in grup.items():
        c = Counter(r["penilaian"] for r in rs)
        ring[g] = {"n": len(rs), **{k: c[k] for k in KAT}, "persen_baik": round(100 * c["baik"] / len(rs), 1) if rs else None,
                   "IK95_baik_wilson": wilson(c["baik"], len(rs)),
                   "persen_dipisah_baik_di_antara_yang_dipisah": round(100 * c["baik"] / (len(rs) - c["cadangan"]), 1) if len(rs) - c["cadangan"] else None}
    cocok = Counter((r["sistem_cadangan"], "ya" if r["penilaian"] == "cadangan" else "tidak") for r in nil)
    beda = [{"no": r["no"], "berkas": r["berkas"], "sistem_cadangan": r["sistem_cadangan"], "penilaian": r["penilaian"]}
            for r in nil if (r["sistem_cadangan"] == "ya") != (r["penilaian"] == "cadangan")]
    catatan = [{"no": r["no"], "berkas": r["berkas"], "penilaian": r["penilaian"], "catatan": r["catatan"]} for r in nil if r["catatan"].strip()]
    out = {"LABEL": "audit manusia (pemilik skripsi) atas 50 foto VAL acak di luar sampel tahap 0; SAM2 aturan v2",
           "per_kelompok": ring, "tanda_cadangan_sistem_vs_penilai": {f"sistem {a} / penilai {b}": n for (a, b), n in sorted(cocok.items())},
           "ketidakcocokan_cadangan": beda, "catatan_penilai": catatan, "salah_objek": [r["berkas"] for r in nil if r["penilaian"] == "salah_objek"],
           "terpotong": [r["berkas"] for r in nil if r["penilaian"] == "terpotong"]}
    (D / "ringkasan_audit.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    md = ["# Ringkasan audit masker v9 (penilaian pemilik skripsi)", "",
          "50 foto VAL acak (seed tetap) di luar 100 foto tahap 0; SAM2 aturan v2 + cadangan. IK 95% Wilson.", "",
          "| Kelompok | n | Baik | Salah objek | Terpotong | Cadangan | % baik [IK 95%] | % baik di antara yang dipisah |", "|---|---|---|---|---|---|---|---|"]
    for g, v in ring.items():
        ik = v["IK95_baik_wilson"]
        md.append(f"| {g} | {v['n']} | {v['baik']} | {v['salah_objek']} | {v['terpotong']} | {v['cadangan']} | {v['persen_baik']}% [{ik[0]}; {ik[1]}] | "
                  f"{v['persen_dipisah_baik_di_antara_yang_dipisah']}% |")
    md += ["", "Tanda cadangan sistem vs penilai: " + "; ".join(f"{k}: {n}" for k, n in out["tanda_cadangan_sistem_vs_penilai"].items()), ""]
    if beda:
        md += ["Ketidakcocokan cadangan:"] + [f"- no. {b['no']} {b['berkas']}: sistem {b['sistem_cadangan']}, penilai {b['penilaian']}" for b in beda] + [""]
    md += ["Salah objek: " + (", ".join(out["salah_objek"]) or "-"), "", "Catatan penilai:"]
    md += [f"- no. {c['no']} ({c['penilaian']}): {c['catatan']}" for c in catatan]
    (D / "ringkasan_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
