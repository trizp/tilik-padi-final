# Cek versi RiceLeafDiseaseBD (ds4)

Ditulis 2026-10-26. Hanya membaca: tidak ada data, model, atau hasil yang diubah, dan tidak ada pelatihan ulang. Dari Mendeley hanya
diambil metadata API publik dan *central directory* zip (daftar nama, ukuran, CRC32 berkas) lewat HTTP Range, bukan isi foto.

Skrip: `scripts/paper/{cek_versi_ds4, banding_zip_v1_v3, orientasi_visual_ds4}.py`. Hasil: `reports/ds4_versi/`.

## 1. Versi yang dipakai: **V3** (1 Sep 2026, doi:10.17632/86s4jzj2m4.3)

| Bukti | Hasil |
|---|---|
| Isi foto dan label lokal (`data/raw/ds4/images_labels`) vs central directory zip V3 | **26.158 dari 26.158 berkas identik** (CRC32 dan ukuran), tidak ada berkas yang hilang atau berlebih |
| Dokumen lokal (`data/raw/ds4/docs`) vs berkas Mendeley (sha256) | README.md, Annotation_Protocol.pdf, dan Dataset folder Structure.png hanya cocok dengan V3; dua berkas lain sama di beberapa versi |
| Jumlah kotak di label lokal | 21.460, sama dengan deskripsi Mendeley V3. Deskripsi V1/V2 menyebut 74.836, dan README V3 menyebut berkas V1 berisi 70.067 |
| README lokal | Mengutip "Mendeley Data, V3, doi: 10.17632/86s4jzj2m4.3" |
| Tanggal | Dokumen bertanggal 2026-09-01 (tanggal terbit V3); data diekstrak 2026-10-02. Catatan unduhan tersendiri tidak ditemukan |
| Label prov_v7 vs label mentah | 3.384 dari 3.384 foto ds4 berpenyakit di prov_v7 memakai label mentah V3 apa adanya (ID kelas dan koordinat identik) |

Riwayat versi menurut API Mendeley:
- V1: terbit 19 Feb 2026.
- V2: terbit 16 Mar 2026. Zipnya **sama persis** dengan V1 (sha256 `ce1362d6…`); V2 hanya menambah `Dataset metadata.xlsx`.
- V3: terbit 1 Sep 2026, zip baru (sha256 `00999f0a…`).

## 2. Perbandingan dengan V3: tidak diperlukan

Karena yang dipakai sudah V3, V3 tidak diunduh ke `data/raw/ds4_v3_cek/`. Untuk butir (a)–(c), data proyek **sama dengan V3**: jumlah
foto, jumlah kotak, ID kelas, dan koordinat identik (bukti CRC32 di atas).

**(d) Orientasi.** Ini tetap diperiksa, karena pertanyaannya apakah V3 sudah memperbaiki 931 foto itu.
- **V3 tidak mengubah foto.** Seluruh 9.769 foto asli dan 8.194 foto bertanda identik antara V1/V2 dan V3. Yang berubah hanya seluruh
  8.194 berkas label (`reports/ds4_versi/banding_v1_v3.json`). Ke-931 foto yang kita putar juga identik di semua versi, sedangkan
  labelnya berubah di V3.
- **Label V3 mengikuti orientasi foto bertanda.** Pada semua 3.384 foto ds4 berlabel di prov_v7, orientasi foto bertanda penulis
  (`Annotated images/.../visuals`, sumber koordinat V3 menurut README) sama dengan hasil perbaikan kita: 2.453 tanpa putaran,
  823 diputar 90°, 106 diputar 270°, dan 2 diputar 180°. Tidak ada yang tidak sesuai (`reports/ds4_versi/orientasi_visual.json`).
- Kesimpulan: pada 931 foto itu, piksel foto asli V3 berorientasi berbeda dari foto bertanda yang menjadi dasar koordinat V3. Perbaikan
  kita (memutar foto asli tanpa mengubah label) membuat foto cocok dengan label resmi V3. Perbaikan itu masih diperlukan di V3; tidak
  ada perbaikan dari penulis yang terlewat atau dobel.

## 3. Dampak ke prov_v7

**Nol foto** di train, val, test, maupun test sesi baru yang kotaknya berbeda kelas dari V3, karena label prov_v7 identik dengan V3.
Tidak ada angka hasil yang perlu diubah.

Catatan dokumentasi V3 yang perlu diketahui (bukan masalah data proyek):
- **Pemetaan ID kelas di README V3 tidak sesuai dengan berkas labelnya.** README menulis 0 = Brown Spot dan 1 = Blast, tetapi berkas di
  folder Blast memakai ID 0 dan berkas di folder Brown spot memakai ID 1. Leaf smut = 2, Tungro = 3, dan Sheath blight = 4; Healthy
  tanpa berkas label. Proyek ini benar karena kelas foto di prov_v7 diturunkan dari nama folder, dan ID di berkas kebetulan sama dengan
  urutan kelas proyek (`leaf_blast` = 0, `brown_spot` = 1). Pengguna yang mengikuti tabel README akan menukar Blast dan Brown spot.
- README V3 tidak konsisten soal jumlah kotak. Ringkasannya menulis 21.460 (sama dengan berkas dan deskripsi Mendeley), tetapi
  pemberitahuan versi dan total tabelnya menulis 22.123. Tabel per kelas menulis Sheath blight 2.605, sedangkan berkasnya berisi 2.607.
  Untuk kelas yang dipakai proyek, angkanya cocok (Blast 4.844, Brown spot 9.352).
- Satu berkas label Blast kosong (1 dari 1.326).

## 4. Rekomendasi kalimat untuk paper

III-A, sumber data (menggantikan kalimat pembuka sumber utama):

> The main source was RiceLeafDiseaseBD, version 3 (Mendeley Data, released 1 September 2026) \cite{ds4}, a collection of field
> photographs from Bangladesh with bounding boxes for individual blast and brown spot lesions. We used the version-3 boxes, which
> replace the over-detected boxes of earlier versions. Class labels were taken from the class folders, because the class identifiers
> in the version-3 label files (0 = blast, 1 = brown spot) do not follow the mapping given in the dataset README.

III-A, orientasi (menggantikan kalimat "We found that 930 images ..."; sekaligus memperbaiki 930 → 931 dari audit sebelumnya):

> All images were inspected before use. In 931 images of the main source, the stored pixels were rotated relative to the box
> coordinates, which the dataset authors derived from their annotated overlay images; in every case the overlay matched the rotated
> orientation. We rotated these images losslessly (930 with \texttt{jpegtran} and one by applying its EXIF orientation tag) and left
> the boxes unchanged.

Klaim "raised mAP50-95 by roughly 65% in a preliminary comparison" tetap belum bersumber (lihat `LAPORAN_PAPER.md`). Usul pengganti
yang bersumber:

> ... which raised mAP50-95 on the test set without BLB from 0.15--0.18 to 0.25--0.28 across the four models.

Sumbernya `reports/final_eval_v7/tabel/lampiran_v6_vs_v7.csv`: v6 0.147–0.180, v7 0.249–0.276.

Entri `references.bib` untuk `ds4` (saat ini masih `\todo`; berkas tidak diubah). Data dari README V3 dan API Mendeley:

```bibtex
@misc{ds4,
  author       = {Munna, Mohammad Mehedi Hasan and Islam, Toriqul and Saha, Smita and Safy, Iqbal Hossain and Bari, Saiful
                  and Akter, Jasmin and Rahman, Md Momtajur and Akter, Riya and Antu, Nahid Hossain and Kabir, Acramul Haque
                  and Tulona, Reefka Fabliha and Islam, Ashraful and Amin, M Ashraful},
  title        = {{RiceLeafDiseaseBD}: A Field-Based Annotated Smartphone Image Dataset of Healthy and Diseased Rice Leaves from Bangladesh},
  howpublished = {Mendeley Data, version 3, CC BY 4.0},
  year         = {2026},
  doi          = {10.17632/86s4jzj2m4.3}
}
```
