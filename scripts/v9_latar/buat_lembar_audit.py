#!/usr/bin/env python3
"""
buat_lembar_audit.py - Lembar audit masker v9 (KEPUTUSAN_PROYEK.md 2026-10-16, butir 1): 50 foto VAL acak (seed tetap) yang BUKAN bagian
dari 100 foto tahap 0. Foto asli dan hasil (latar abu-abu) berdampingan; pilihan: baik / salah objek / terpotong / cadangan.
Kotak label TIDAK digambar (penilaian murni pemisahan daun). Berkas lokal (tidak dipublikasikan: berisi foto dataset).

  .venv/Scripts/python scripts/v9_latar/buat_lembar_audit.py
Keluaran: reports/v9_latar/audit_masker/{lembar_audit.html, daftar_audit.csv (templat isian)}
"""
import base64
import csv
import json
import random
import sys
from html import escape
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent))
import umum  # noqa: E402

N, SEED = 50, umum.SEED + 7
OUT = umum.LAPORAN / "audit_masker"


def jpg64(img, sisi=420):
    h, w = img.shape[:2]
    s = sisi / max(h, w)
    kecil = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return base64.b64encode(cv2.imencode(".jpg", kecil, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()).decode()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tahap0 = {x[0] for x in json.loads((umum.LAPORAN / "sampel_tahap0.json").read_text(encoding="utf-8"))}
    rows = sorted((r for r in umum.manifest_val() if r["file"] not in tahap0), key=lambda r: r["file"])
    pilih = random.Random(SEED).sample(rows, N)
    info = json.loads((umum.KERJA / "sam2_v2/info.json").read_text(encoding="utf-8"))
    kartu, daftar = [], []
    for i, r in enumerate(pilih, 1):
        f = r["file"]
        bgr = cv2.imread(str(umum.VAL_IMG / f))
        m = umum.pasca_proses(cv2.imread(str(umum.KERJA / "sam2_v2/masker_mentah" / (Path(f).stem + ".png")), cv2.IMREAD_GRAYSCALE) >= 128)
        hasil = bgr.copy()
        hasil[~m] = umum.ABU[::-1]
        cad = bool(info.get(f, {}).get("cadangan"))
        daftar.append({"no": i, "berkas": f, "sumber": r["source"], "kelas": r["class"], "sistem_cadangan": "ya" if cad else "tidak",
                       "penilaian": "", "catatan": ""})
        opsi = "".join(f'<label><input type="radio" name="p{i}" value="{v}"> {t}</label>'
                       for v, t in (("baik", "baik"), ("salah_objek", "salah objek"), ("terpotong", "terpotong"), ("cadangan", "cadangan")))
        kartu.append(f'''<section class="kartu" data-no="{i}" data-berkas="{escape(f)}">
<h2>{i}. <span class="kecil">{escape(f)}</span></h2>
<p class="meta">{escape(r["source"])} | {escape(r["class"])} | sistem: {"CADANGAN (tidak dipisah)" if cad else "dipisah"}</p>
<div class="dua"><figure><img src="data:image/jpeg;base64,{jpg64(bgr)}" alt="asli"><figcaption>Asli</figcaption></figure>
<figure><img src="data:image/jpeg;base64,{jpg64(hasil)}" alt="hasil"><figcaption>Hasil (latar abu-abu)</figcaption></figure></div>
<div class="opsi">{opsi}</div><input class="catatan" type="text" placeholder="catatan (opsional)" data-no="{i}"></section>''')
    with open(OUT / "daftar_audit.csv", "w", newline="", encoding="utf-8") as fo:
        w = csv.DictWriter(fo, fieldnames=list(daftar[0]))
        w.writeheader()
        w.writerows(daftar)
    html = f'''<!doctype html><html lang="id"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Audit masker v9</title><style>
:root{{--latar:#f4f5f2;--teks:#1f2d24;--garis:#cfd6cc;--aksen:#2f6b3b}}
body{{margin:0;background:var(--latar);color:var(--teks);font:16px/1.5 system-ui,sans-serif}}
main{{max-width:980px;margin:0 auto;padding:16px}} h1{{font-size:24px;margin:8px 0}}
.kartu{{background:#fff;border:1px solid var(--garis);border-radius:6px;padding:12px;margin:14px 0}}
.kartu h2{{font-size:17px;margin:0}} .kecil{{font-weight:400;font-size:13px;word-break:break-all}} .meta{{margin:2px 0 8px;font-size:14px;color:#4a5a50}}
.dua{{display:flex;gap:10px;flex-wrap:wrap}} figure{{margin:0;flex:1 1 300px}} img{{width:100%;height:auto;border:1px solid var(--garis)}}
figcaption{{font-size:13px}} .opsi{{display:flex;gap:16px;flex-wrap:wrap;margin:8px 0}} .opsi label{{min-height:32px;display:flex;align-items:center;gap:4px}}
.catatan{{width:100%;box-sizing:border-box;padding:6px;border:1px solid var(--garis);border-radius:4px}}
.bar{{position:sticky;top:0;background:var(--latar);padding:8px 0;border-bottom:1px solid var(--garis);display:flex;gap:12px;align-items:center;flex-wrap:wrap}}
button{{background:var(--aksen);color:#fff;border:0;border-radius:4px;padding:10px 14px;font-size:15px;cursor:pointer}}
</style></head><body><main>
<h1>Audit masker v9 (50 foto VAL di luar sampel tahap 0)</h1>
<p>Nilai pemisahan daun pada tiap foto: <b>baik</b> (daun utama utuh, latar terbuang), <b>salah objek</b> (yang tersisa bukan daun utama),
<b>terpotong</b> (daun utama terpotong sehingga bagian penting hilang), <b>cadangan</b> (foto tidak dipisah). Pilihan tersimpan di browser ini;
tekan "Unduh CSV" bila selesai lalu simpan sebagai <code>reports/v9_latar/audit_masker/penilaian.csv</code>.</p>
<div class="bar"><button id="unduh">Unduh CSV</button><span id="hitung"></span></div>
{"".join(kartu)}
</main><script>
const K='audit_v9_'; const kartu=[...document.querySelectorAll('.kartu')];
function muat(){{try{{const d=JSON.parse(localStorage.getItem(K+'data')||'{{}}');kartu.forEach(k=>{{const n=k.dataset.no,v=d[n]||{{}};
 if(v.p){{const r=k.querySelector(`input[value="${{v.p}}"]`);if(r)r.checked=true}} if(v.c)k.querySelector('.catatan').value=v.c}})}}catch(e){{}}}}
function kumpul(){{const d={{}};kartu.forEach(k=>{{const n=k.dataset.no,r=k.querySelector('input[type=radio]:checked');
 d[n]={{p:r?r.value:'',c:k.querySelector('.catatan').value}}}});return d}}
function simpan(){{const d=kumpul();try{{localStorage.setItem(K+'data',JSON.stringify(d))}}catch(e){{}}
 document.getElementById('hitung').textContent=Object.values(d).filter(v=>v.p).length+' dari {N} dinilai'}}
document.addEventListener('change',simpan);document.addEventListener('input',simpan);
document.getElementById('unduh').onclick=()=>{{const d=kumpul();const q=s=>'"'+String(s).replace(/"/g,'""')+'"';
 const baris=['no,berkas,penilaian,catatan'].concat(kartu.map(k=>[k.dataset.no,q(k.dataset.berkas),d[k.dataset.no].p,q(d[k.dataset.no].c)].join(',')));
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([baris.join('\\n')+'\\n'],{{type:'text/csv'}}));a.download='penilaian.csv';a.click()}};
muat();simpan();
</script></body></html>'''
    (OUT / "lembar_audit.html").write_text(html, encoding="utf-8")
    print("ok", OUT / "lembar_audit.html", "cadangan:", sum(d["sistem_cadangan"] == "ya" for d in daftar),
          {k: sum(1 for d in daftar if d["sumber"] + "|" + d["kelas"] == k) for k in sorted({d["sumber"] + "|" + d["kelas"] for d in daftar})})


if __name__ == "__main__":
    main()
