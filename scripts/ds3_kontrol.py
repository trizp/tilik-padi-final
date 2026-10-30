#!/usr/bin/env python3
"""
ds3_kontrol.py - Kontrol untuk ds3_kembar.py (hasil 0 kembar luar perlu dibuktikan bukan karena pipa rusak).

1. Kontrol positif: 12 foto ds3 (4 per kelas) diberi perubahan (A: rotasi 90 + potong tengah 80% + perkecil 0.6 + JPEG q70;
   B: flip horizontal + kecerahan 1.25 + potong 90% + JPEG q80). Salinan disimpan di folder sementara (bukan di project);
   tiap salinan harus terdeteksi sebagai kembar foto aslinya oleh minimal satu kriteria (phash<=14 / cos>=0.97 / SIFT>=21).
2. Jarak tetangga terdekat ds3 -> tiap sumber (ds1, ds2, ds4): phash minimum (8 varian), cosine maksimum, dan inlier SIFT
   maksimum pada 10 tetangga embedding terdekat per sumber. Menunjukkan seberapa jauh dari ambang.
Keluaran: reports/prov_v6/ds3_eksternal/kontrol.json
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

sys.path.insert(0, str(Path(__file__).parent))
import aug_detect as ad  # noqa: E402
import dedupe_rules as dr  # noqa: E402
import ds3_kembar as dk  # noqa: E402
import prepare_dataset as pd_  # noqa: E402

ROOT = dr.ROOT


def ubah(p, mode, dst):
    im = Image.open(p).convert("RGB")
    w, h = im.size
    if mode == "A":
        im = im.rotate(90, expand=True)
        w, h = im.size
        c = 0.8
        im = im.crop((int(w * (1 - c) / 2), int(h * (1 - c) / 2), int(w * (1 + c) / 2), int(h * (1 + c) / 2)))
        im = im.resize((int(im.width * 0.6), int(im.height * 0.6)))
        q = 70
    else:
        im = ImageEnhance.Brightness(im.transpose(Image.FLIP_LEFT_RIGHT)).enhance(1.25)
        im = im.crop((int(w * 0.05), int(h * 0.05), int(w * 0.95), int(h * 0.95)))
        q = 80
    out = dst / f"{p.parent.name}_{p.stem}_{mode}.jpg"
    im.save(out, quality=q)
    return out


def main():
    q = dk.imgs("data/raw/ds3")
    pilih = [p for kf in ("blast", "blight", "tungro") for p in [x for x in q if x.parent.name.lower() == kf][::20][:4]]
    tmp = Path(tempfile.mkdtemp(prefix="ds3_kontrol_"))
    salinan = [(p, ubah(p, m, tmp)) for p in pilih for m in ("A", "B")]
    asli, kop = [p for p, _ in salinan], [c for _, c in salinan]
    va, _ = dk.hashes(asli)
    vc, _ = dk.hashes(kop)
    Ea, Ec = ad.embeddings(asli), ad.embeddings(kop)
    dr.ensure_descriptors(asli + kop)
    sift = dr.verify_pairs(list(zip(kop, asli)))
    pos = []
    for i, (p, c) in enumerate(salinan):
        ph = int(min(pd_.min_dist(vc[i:i + 1, v], va[i:i + 1, 0])[0] for v in range(8)))
        cs = float(ad.sim_matrix(Ec[i:i + 1], Ea[i:i + 1, 0])[0, 0])
        si = int(sift[i][0])
        pos.append({"asli": dr.ROOT.joinpath(p).relative_to(ROOT).as_posix() if p.is_absolute() else str(p), "perubahan": c.stem[-1],
                    "phash": ph, "cos": round(cs, 4), "sift_inlier": si,
                    "terdeteksi": ph <= dk.PHASH_THR or cs >= dk.EMB_THR or si >= dr.MIN_INLIERS})

    # tetangga terdekat ds3 -> sumber
    refs = dk.ref_sets()
    per_sumber = {}
    Eq = ad.embeddings(q)
    var_q, _ = dk.hashes(q)
    for src in ("ds1", "ds2", "ds4"):
        ps = [p for k, v in refs.items() if k.startswith(src) for p in v]
        vr, _ = dk.hashes(ps)
        Er = ad.embeddings(ps)
        S = ad.sim_matrix(Eq, Er[:, 0])
        ph = np.min([pd_.min_dist(var_q[:, v], vr[:, 0]) for v in range(8)], axis=0)
        P = dr.knn_pairs_cross(Eq, Er, 10)
        res = dr.verify_pairs([(q[i], ps[j]) for i, j in P])
        mx = np.zeros(len(q), int)
        for (i, _), r in zip(P, res):
            mx[i] = max(mx[i], int(r[0]))
        per_sumber[src] = {"n_pembanding": len(ps),
                           "cos_maks": {"maks": round(float(S.max()), 4), "p95": round(float(np.percentile(S.max(1), 95)), 4), "ambang": dk.EMB_THR},
                           "phash_min": {"min": int(ph.min()), "p5": float(np.percentile(ph, 5)), "ambang": dk.PHASH_THR},
                           "sift_inlier_maks_10_tetangga": {"maks": int(mx.max()), "p95": float(np.percentile(mx, 95)), "ambang": dr.MIN_INLIERS}}
    hasil = {"kontrol_positif": {"n": len(pos), "terdeteksi": sum(r["terdeteksi"] for r in pos), "rincian": pos},
             "tetangga_terdekat_ds3_ke_sumber": per_sumber}
    (ROOT / "reports/prov_v6/ds3_eksternal/kontrol.json").write_text(json.dumps(hasil, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"kontrol_positif": {k: v for k, v in hasil["kontrol_positif"].items() if k != "rincian"},
                      "tetangga": per_sumber}, indent=1, ensure_ascii=False))
    for r in pos:
        print(r)


if __name__ == "__main__":
    main()
