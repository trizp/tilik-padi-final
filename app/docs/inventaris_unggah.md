# Inventaris folder unggah tilik-padi (usulan pembersihan, BELUM dijalankan)

| Kelompok | Jalur | Ukuran | Fungsi | Usulan |
|---|---|---|---|---|
| WAJIB | `,streamlit/config,toml` | 0,7 KB | tema. batas unggah. menu tersembunyi | pertahankan |
| WAJIB | `config,yaml` | 3,0 KB | jalur model. ambang. teks. atribusi | pertahankan |
| WAJIB | `core/__init__,py` | 0,1 KB | paket inti | pertahankan |
| WAJIB | `core/alur,py` | 1,7 KB | alur satu foto | pertahankan |
| WAJIB | `core/diagnosis,py` | 1,8 KB | aturan tiga tingkat | pertahankan |
| WAJIB | `core/gambar,py` | 2,5 KB | baca foto. E.IF. validasi | pertahankan |
| WAJIB | `core/inferensi,py` | 5,4 KB | inferensi ONN.. letterbox. NMS | pertahankan |
| WAJIB | `core/konfigurasi,py` | 1,6 KB | muat config,yaml | pertahankan |
| WAJIB | `core/visual,py` | 2,4 KB | kotak bercak. potongan diperbesar | pertahankan |
| WAJIB | `models/v7_deploy_yolo11s_s2,onnx` | 37.042,4 KB | model YOLO11s | pertahankan |
| WAJIB | `rbs/__init__,py` | 0,1 KB | paket RBS | pertahankan |
| WAJIB | `rbs/mesin,py` | 4,2 KB | mesin forward chaining | pertahankan |
| WAJIB | `rbs/rules,yaml` | 6,4 KB | basis aturan (Tabel 3,1) | pertahankan |
| WAJIB | `requirements,txt` | 0,2 KB | paket yang dipasang Streamlit Cloud | pertahankan |
| WAJIB | `streamlit_app,py` | 8,9 KB | titik masuk aplikasi (tampilan) | pertahankan |
| WAJIB | `ui/__init__,py` | 0,1 KB | paket tampilan | pertahankan |
| WAJIB | `ui/gaya,css` | 12,6 KB | gaya tampilan | pertahankan |
| WAJIB | `ui/ilustrasi,py` | 6,9 KB | ilustrasi SVG | pertahankan |
| WAJIB | `ui/komponen,py` | 13,0 KB | potongan HTML | pertahankan |
| DOKUMENTASI INTI | `,gitignore` | 0,2 KB | pengecualian git (repo) | pertahankan |
| DOKUMENTASI INTI | `LICENSE` | 33,7 KB | AGPL-3,0 | pertahankan |
| DOKUMENTASI INTI | `README,md` | 5,7 KB | ringkasan. cara menjalankan. atribusi. lisensi | pertahankan |
| DOKUMENTASI INTI | `docs/arsitektur,md` | 2,3 KB | diagram arsitektur & forward chaining | pertahankan |
| DOKUMENTASI INTI | `docs/deploy,md` | 7,1 KB | langkah pemasangan | pertahankan |
| DOKUMENTASI INTI | `docs/desain,md` | 12,9 KB | desain akhir (palet. huruf. keputusan) | pertahankan |
| DOKUMENTASI INTI | `docs/menjalankan_lokal,md` | 2,2 KB | cara menjalankan | pertahankan |
| DOKUMENTASI INTI | `docs/tangkapan_layar/awal_390,png` | 111,9 KB | tangkapan layar final untuk README (foto ds4) | pertahankan |
| DOKUMENTASI INTI | `docs/tangkapan_layar/kemungkinan_390,png` | 453,9 KB | tangkapan layar final untuk README (foto ds4) | pertahankan |
| DOKUMENTASI INTI | `docs/tangkapan_layar/terdeteksi_1280,png` | 734,9 KB | tangkapan layar final untuk README (foto ds4) | pertahankan |
| DOKUMENTASI INTI | `docs/tangkapan_layar/terdeteksi_390,png` | 416,9 KB | tangkapan layar final untuk README (foto ds4) | pertahankan |
| TEST | `docs/uji/blackbox,md` | 2,9 KB | hasil black-box (dirujuk README) | pertahankan |
| TEST | `docs/uji/sumber_daya,md` | 0,8 KB | hasil ukur RAM/waktu (dirujuk README) | pertahankan |
| TEST | `docs/uji_kesetaraan,md` | 2,0 KB | hasil kesetaraan ONN. (dirujuk README) | pertahankan |
| TEST | `requirements-dev,txt` | 0,2 KB | paket test | pertahankan |
| TEST | `tests/conftest,py` | 0,7 KB | fixture | pertahankan |
| TEST | `tests/test_blackbox,py` | 0,8 KB | uji black-box (butuh dataset; dilewati di repo) | pertahankan |
| TEST | `tests/test_inti,py` | 8,1 KB | unit test inti/UI | pertahankan |
| TEST | `tests/test_rbs,py` | 7,5 KB | unit test aturan & mesin | pertahankan |
| TEST | `tools/uji_blackbox,py` | 6,3 KB | penggerak test_blackbox | pertahankan |
| BERKAS KERJA | `docs/siap_unggah,md` | 1,0 KB | laporan pemeriksaan unggah | HAPUS dari folder unggah |
| BERKAS KERJA | `docs/tangkapan_layar,md` | 7,5 KB | catatan kritik desain | HAPUS dari folder unggah |
| BERKAS KERJA | `docs/tangkapan_layar/awal_1280,png` | 125,5 KB | tangkapan layar final lain (tidak dipakai README) | HAPUS dari folder unggah |
| BERKAS KERJA | `docs/tangkapan_layar/banding/awal_1280,png` | 135,2 KB | gambar banding sebelum/sesudah | HAPUS dari folder unggah |
| BERKAS KERJA | `docs/tangkapan_layar/banding/awal_390,png` | 168,5 KB | gambar banding sebelum/sesudah | HAPUS dari folder unggah |
| BERKAS KERJA | `docs/tangkapan_layar/banding/kemungkinan_1280,png` | 618,7 KB | gambar banding sebelum/sesudah | HAPUS dari folder unggah |
| BERKAS KERJA | `docs/tangkapan_layar/banding/kemungkinan_390,png` | 901,9 KB | gambar banding sebelum/sesudah | HAPUS dari folder unggah |
| BERKAS KERJA | `docs/tangkapan_layar/banding/terdeteksi_1280,png` | 556,3 KB | gambar banding sebelum/sesudah | HAPUS dari folder unggah |
| BERKAS KERJA | `docs/tangkapan_layar/banding/terdeteksi_390,png` | 786,5 KB | gambar banding sebelum/sesudah | HAPUS dari folder unggah |
| BERKAS KERJA | `docs/tangkapan_layar/banding/tidak_terdeteksi_1280,png` | 474,2 KB | gambar banding sebelum/sesudah | HAPUS dari folder unggah |
| BERKAS KERJA | `docs/tangkapan_layar/banding/tidak_terdeteksi_390,png` | 706,9 KB | gambar banding sebelum/sesudah | HAPUS dari folder unggah |
| BERKAS KERJA | `docs/tangkapan_layar/bercak_kecil_1280,png` | 784,8 KB | tangkapan layar final lain (tidak dipakai README) | HAPUS dari folder unggah |
| BERKAS KERJA | `docs/tangkapan_layar/bercak_kecil_390,png` | 442,2 KB | tangkapan layar final lain (tidak dipakai README) | HAPUS dari folder unggah |
| BERKAS KERJA | `docs/tangkapan_layar/kamera_1280,png` | 138,7 KB | tangkapan layar final lain (tidak dipakai README) | HAPUS dari folder unggah |
| BERKAS KERJA | `docs/tangkapan_layar/kamera_390,png` | 122,6 KB | tangkapan layar final lain (tidak dipakai README) | HAPUS dari folder unggah |
| BERKAS KERJA | `docs/tangkapan_layar/kamera_tanpa_has_390,png` | 122,1 KB | tangkapan layar final lain (tidak dipakai README) | HAPUS dari folder unggah |
| BERKAS KERJA | `docs/tangkapan_layar/kemungkinan_1280,png` | 833,3 KB | tangkapan layar final lain (tidak dipakai README) | HAPUS dari folder unggah |
| BERKAS KERJA | `docs/tangkapan_layar/putaran_awal/awal_1280,png` | 46,3 KB | tangkapan layar versi awal (v1) | HAPUS dari folder unggah |
| BERKAS KERJA | `docs/tangkapan_layar/putaran_awal/awal_390,png` | 39,9 KB | tangkapan layar versi awal (v1) | HAPUS dari folder unggah |
| BERKAS KERJA | `docs/tangkapan_layar/putaran_awal/kemungkinan_1280,png` | 732,1 KB | tangkapan layar versi awal (v1) | HAPUS dari folder unggah |
| BERKAS KERJA | `docs/tangkapan_layar/putaran_awal/kemungkinan_390,png` | 419,4 KB | tangkapan layar versi awal (v1) | HAPUS dari folder unggah |
| BERKAS KERJA | `docs/tangkapan_layar/putaran_awal/tentang_1280,png` | 263,2 KB | tangkapan layar versi awal (v1) | HAPUS dari folder unggah |
| BERKAS KERJA | `docs/tangkapan_layar/putaran_awal/tentang_390,png` | 237,7 KB | tangkapan layar versi awal (v1) | HAPUS dari folder unggah |
| BERKAS KERJA | `docs/tangkapan_layar/putaran_awal/terdeteksi_1280,png` | 592,9 KB | tangkapan layar versi awal (v1) | HAPUS dari folder unggah |
| BERKAS KERJA | `docs/tangkapan_layar/putaran_awal/terdeteksi_390,png` | 349,5 KB | tangkapan layar versi awal (v1) | HAPUS dari folder unggah |
| BERKAS KERJA | `docs/tangkapan_layar/putaran_awal/tidak_terdeteksi_1280,png` | 583,8 KB | tangkapan layar versi awal (v1) | HAPUS dari folder unggah |
| BERKAS KERJA | `docs/tangkapan_layar/putaran_awal/tidak_terdeteksi_390,png` | 350,0 KB | tangkapan layar versi awal (v1) | HAPUS dari folder unggah |
| BERKAS KERJA | `docs/tangkapan_layar/sebelum_revisi/awal_1280,png` | 46,7 KB | tangkapan layar sebelum revisi | HAPUS dari folder unggah |
| BERKAS KERJA | `docs/tangkapan_layar/sebelum_revisi/awal_390,png` | 40,2 KB | tangkapan layar sebelum revisi | HAPUS dari folder unggah |
| BERKAS KERJA | `docs/tangkapan_layar/sebelum_revisi/jejak_kemungkinan_390,png` | 494,4 KB | tangkapan layar sebelum revisi | HAPUS dari folder unggah |
| BERKAS KERJA | `docs/tangkapan_layar/sebelum_revisi/kemungkinan_1280,png` | 877,5 KB | tangkapan layar sebelum revisi | HAPUS dari folder unggah |
| BERKAS KERJA | `docs/tangkapan_layar/sebelum_revisi/kemungkinan_390,png` | 417,2 KB | tangkapan layar sebelum revisi | HAPUS dari folder unggah |
| BERKAS KERJA | `docs/tangkapan_layar/sebelum_revisi/tentang_1280,png` | 263,2 KB | tangkapan layar sebelum revisi | HAPUS dari folder unggah |
| BERKAS KERJA | `docs/tangkapan_layar/sebelum_revisi/tentang_390,png` | 237,7 KB | tangkapan layar sebelum revisi | HAPUS dari folder unggah |
| BERKAS KERJA | `docs/tangkapan_layar/sebelum_revisi/terdeteksi_1280,png` | 708,7 KB | tangkapan layar sebelum revisi | HAPUS dari folder unggah |
| BERKAS KERJA | `docs/tangkapan_layar/sebelum_revisi/terdeteksi_390,png` | 345,1 KB | tangkapan layar sebelum revisi | HAPUS dari folder unggah |
| BERKAS KERJA | `docs/tangkapan_layar/sebelum_revisi/tidak_terdeteksi_1280,png` | 578,1 KB | tangkapan layar sebelum revisi | HAPUS dari folder unggah |
| BERKAS KERJA | `docs/tangkapan_layar/sebelum_revisi/tidak_terdeteksi_390,png` | 343,0 KB | tangkapan layar sebelum revisi | HAPUS dari folder unggah |
| BERKAS KERJA | `docs/tangkapan_layar/tentang_1280,png` | 302,0 KB | tangkapan layar final lain (tidak dipakai README) | HAPUS dari folder unggah |
| BERKAS KERJA | `docs/tangkapan_layar/tentang_390,png` | 277,3 KB | tangkapan layar final lain (tidak dipakai README) | HAPUS dari folder unggah |
| BERKAS KERJA | `docs/tangkapan_layar/tidak_terdeteksi_1280,png` | 556,4 KB | tangkapan layar final lain (tidak dipakai README) | HAPUS dari folder unggah |
| BERKAS KERJA | `docs/tangkapan_layar/tidak_terdeteksi_390,png` | 320,5 KB | tangkapan layar final lain (tidak dipakai README) | HAPUS dari folder unggah |
| BERKAS KERJA | `docs/uji/blackbox,json` | 5,7 KB | duplikat mesin dari ,md | HAPUS dari folder unggah |
| BERKAS KERJA | `docs/uji/kamera_tanpa_has,md` | 0,5 KB | hasil uji kamera tanpa :has (pengembangan) | HAPUS dari folder unggah |
| BERKAS KERJA | `docs/uji/sumber_daya,json` | 0,7 KB | duplikat mesin dari ,md | HAPUS dari folder unggah |
| BERKAS KERJA | `docs/uji_kesetaraan,json` | 7,1 KB | duplikat mesin dari ,md | HAPUS dari folder unggah |
| BERKAS KERJA | `rbs/rules_slide14_arsip,yaml` | 5,0 KB | arsip aturan versi slide (tidak dipakai) | HAPUS dari folder unggah |
| BERKAS KERJA | `tools/pindai_val,py` | 1,7 KB | pindai VAL (butuh dataset) | HAPUS dari folder unggah |
| BERKAS KERJA | `tools/tangkap_layar,py` | 6,6 KB | tangkapan layar otomatis | HAPUS dari folder unggah |
| BERKAS KERJA | `tools/uji_kamera_tanpa_has,py` | 4,7 KB | uji kamera (Chrome) | HAPUS dari folder unggah |
| BERKAS KERJA | `tools/uji_kesetaraan,py` | 3,7 KB | uji kesetaraan (butuh folder proyek) | HAPUS dari folder unggah |
| BERKAS KERJA | `tools/ukur_sumber_daya,py` | 7,9 KB | ukur RAM/waktu | HAPUS dari folder unggah |

| Kelompok | Berkas | Ukuran |
|---|---|---|
| WAJIB | 19 | 36,2 MB |
| DOKUMENTASI INTI | 11 | 1,7 MB |
| TEST | 9 | 0,0 MB |
| BERKAS KERJA | 52 | 16,0 MB |
