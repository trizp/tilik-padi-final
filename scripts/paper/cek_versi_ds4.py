#!/usr/bin/env python3
"""cek_versi_ds4.py - Cocokkan isi data/raw/ds4 dengan rilis RiceLeafDiseaseBD di Mendeley Data TANPA mengunduh zip 5,6 GB.

1. Metadata versi dan daftar berkas (ukuran, sha256) dari API publik Mendeley untuk V1, V2, V3.
2. Central directory zip V3 diambil lewat HTTP Range (hanya beberapa MB dari akhir berkas): nama, ukuran, CRC32 setiap berkas.
3. CRC32 dan ukuran setiap berkas lokal di data/raw/ds4/images_labels dibandingkan dengan entri zip V3 berdasarkan jalur relatif.
Hanya membaca; tidak mengubah data, model, atau hasil. Keluaran: reports/ds4_versi/{ringkasan.json, beda_berkas.csv, zip_v3_isi.csv}"""
import csv
import json
import struct
import urllib.request
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/ds4"
OUT = ROOT / "reports/ds4_versi"
API = "https://data.mendeley.com/public-api/datasets/86s4jzj2m4"
UA = {"User-Agent": "curl/8.9.1 (cek-versi-ds4; skripsi)"}   # tanpa User-Agent, API menolak urllib (403)


def get_json(url):
    return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60))


def ambil(url, a, b):
    req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={a}-{b}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        assert r.status == 206, r.status
        return r.read()


def central_directory(url, size):
    ekor = ambil(url, size - 65536, size - 1)
    i = ekor.rfind(b"PK\x05\x06")
    assert i >= 0, "EOCD tidak ditemukan"
    n_ent, cd_size, cd_off = struct.unpack("<HII", ekor[i + 10:i + 20])
    j = ekor.rfind(b"PK\x06\x07")                      # ZIP64 EOCD locator
    if j >= 0:
        z64_off = struct.unpack("<Q", ekor[j + 8:j + 16])[0]
        z64 = ambil(url, z64_off, z64_off + 55)
        assert z64[:4] == b"PK\x06\x06"
        n_ent, cd_size, cd_off = struct.unpack("<QQQ", z64[32:56])
    cd = ambil(url, cd_off, cd_off + cd_size - 1)
    out, p = [], 0
    while p < len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        (crc, csz, usz, fnl, exl, cml) = struct.unpack("<IIIHHH", cd[p + 16:p + 34])
        nama = cd[p + 46:p + 46 + fnl].decode("utf-8", errors="replace")
        extra = cd[p + 46 + fnl:p + 46 + fnl + exl]
        q = 0
        while q + 4 <= len(extra):                   # ZIP64 extra: ukuran asli bila 0xFFFFFFFF
            hid, hsz = struct.unpack("<HH", extra[q:q + 4])
            if hid == 1 and usz == 0xFFFFFFFF:
                usz = struct.unpack("<Q", extra[q + 4:q + 12])[0]
            q += 4 + hsz
        out.append({"nama": nama, "crc32": crc, "ukuran": usz})
        p += 46 + fnl + exl + cml
    assert len(out) == n_ent, (len(out), n_ent)
    return out


def crc_berkas(p):
    c = 0
    with open(p, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                return c & 0xFFFFFFFF
            c = zlib.crc32(b, c)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ring = {"versi": {}}
    for v in (1, 2, 3):
        d = get_json(f"{API}?version={v}")
        f = get_json(f"{API}/files?folder_id=root&version={v}")
        ring["versi"][f"V{v}"] = {"publish_date": d.get("publish_date"),
                                  "deskripsi_kotak": next((s for s in d.get("description", "").split(". ") if "bounding box" in s), ""),
                                  "berkas": {x["filename"]: {"size": x["content_details"]["size"], "sha256": x["content_details"]["sha256_hash"]}
                                             for x in f}}
    # dokumen lokal vs tiap versi (sha256)
    import hashlib
    lokal = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (RAW / "docs").iterdir() if p.is_file()}
    ring["dokumen_lokal_cocok"] = {nama: [v for v, info in ring["versi"].items()
                                          if any(b["sha256"] == h for b in info["berkas"].values())] for nama, h in lokal.items()}
    # central directory zip V3
    f3 = get_json(f"{API}/files?folder_id=root&version=3")
    z = next(x for x in f3 if x["filename"].endswith(".zip"))
    isi = central_directory(z["content_details"]["download_url"], z["content_details"]["size"])
    with open(OUT / "zip_v3_isi.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["nama", "crc32", "ukuran"])
        w.writeheader()
        w.writerows(isi)
    # pencocokan jalur: entri zip "RiceLeafDiseaseBD/..." vs lokal images_labels/RiceLeafDiseaseBD/...
    berkas_zip = {e["nama"]: e for e in isi if not e["nama"].endswith("/")}
    akar_lokal = RAW / "images_labels"
    lok = {p.relative_to(akar_lokal).as_posix(): p for p in akar_lokal.rglob("*") if p.is_file()}
    sama, beda, hanya_lokal = 0, [], []
    for rel, p in sorted(lok.items()):
        e = berkas_zip.get(rel)
        if e is None:
            hanya_lokal.append(rel)
            continue
        st = p.stat().st_size
        c = crc_berkas(p)
        if st == e["ukuran"] and c == e["crc32"]:
            sama += 1
        else:
            beda.append({"berkas": rel, "ukuran_lokal": st, "ukuran_zip": e["ukuran"], "crc_lokal": c, "crc_zip": e["crc32"]})
    hanya_zip = sorted(set(berkas_zip) - set(lok))
    with open(OUT / "beda_berkas.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["jenis", "berkas"])
        for b in beda:
            w.writerow(["isi_beda", b["berkas"]])
        for b in hanya_lokal:
            w.writerow(["hanya_lokal", b])
        for b in hanya_zip:
            w.writerow(["hanya_zip_v3", b])
    ring["zip_v3"] = {"entri": len(isi), "berkas": len(berkas_zip), "sha256_zip": z["content_details"]["sha256_hash"]}
    ring["pencocokan_lokal_vs_zip_v3"] = {"berkas_lokal": len(lok), "identik_crc32_dan_ukuran": sama, "isi_beda": len(beda),
                                          "hanya_lokal": len(hanya_lokal), "hanya_di_zip_v3": len(hanya_zip),
                                          "contoh_isi_beda": beda[:5], "contoh_hanya_lokal": hanya_lokal[:5], "contoh_hanya_zip": hanya_zip[:5]}
    (OUT / "ringkasan.json").write_text(json.dumps(ring, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in ring.items() if k != "versi"}, indent=1, ensure_ascii=False))
    for v, info in ring["versi"].items():
        print(v, info["publish_date"], "|", info["deskripsi_kotak"][:110])


if __name__ == "__main__":
    main()
