"""Inferensi YOLO (ONNX Runtime) dengan pra- dan pasca-proses setara Ultralytics 8.4.171 untuk model ONNX statis 640:

- LetterBox(640, auto=False, center=True): skala r = min(640/h, 640/w), ukuran baru round(w*r) x round(h*r), interpolasi bilinear
  (setara cv2.INTER_LINEAR), padding nilai 114, kiri/atas = round(d/2 - 0.1), kanan/bawah = round(d/2 + 0.1).
- RGB, float32 / 255, NCHW.
- non_max_suppression: kandidat = skor kelas maksimum > conf; satu kelas per kotak; maks 30000 kandidat; NMS per kelas (offset
  koordinat kelas x 7680); IoU 0,7; maks 300 kotak.
- scale_boxes: kurangi padding, bagi skala, potong ke batas gambar.
"""
from dataclasses import dataclass

import numpy as np

MAX_WH, MAX_NMS = 7680, 30000


@dataclass(frozen=True)
class Kotak:
    x1: float
    y1: float
    x2: float
    y2: float
    keyakinan: float
    kelas: int


def ubah_ukuran_bilinear(img: np.ndarray, lebar: int, tinggi: int) -> np.ndarray:
    """Bilinear seperti cv2.INTER_LINEAR (pusat piksel di +0,5; koordinat di luar batas dijepit)."""
    h, w = img.shape[:2]
    if (w, h) == (lebar, tinggi):
        return img

    def koordinat(n_baru, n_lama):
        s = n_lama / n_baru
        f = (np.arange(n_baru, dtype=np.float64) + 0.5) * s - 0.5
        i0 = np.floor(f).astype(np.int64)
        t = f - i0
        t[i0 < 0] = 0.0
        i0 = np.clip(i0, 0, n_lama - 1)
        i1 = np.clip(i0 + 1, 0, n_lama - 1)
        return i0, i1, t.astype(np.float32)

    y0, y1, ty = koordinat(tinggi, h)
    x0, x1, tx = koordinat(lebar, w)
    atas, bawah = img[y0].astype(np.float32), img[y1].astype(np.float32)
    baris = atas + (bawah - atas) * ty[:, None, None]
    kiri, kanan = baris[:, x0], baris[:, x1]
    hasil = kiri + (kanan - kiri) * tx[None, :, None]
    return np.clip(np.rint(hasil), 0, 255).astype(np.uint8)


def letterbox(img: np.ndarray, ukuran: int = 640, isi: int = 114):
    """Kembalikan (gambar ukuran x ukuran, skala, (pad_kiri, pad_atas))."""
    h, w = img.shape[:2]
    r = min(ukuran / h, ukuran / w)
    lb, tb = round(w * r), round(h * r)
    dw, dh = (ukuran - lb) / 2, (ukuran - tb) / 2
    atas, bawah = round(dh - 0.1), round(dh + 0.1)
    kiri, kanan = round(dw - 0.1), round(dw + 0.1)
    kecil = ubah_ukuran_bilinear(img, lb, tb)
    out = np.full((tb + atas + bawah, lb + kiri + kanan, 3), isi, dtype=np.uint8)
    out[atas:atas + tb, kiri:kiri + lb] = kecil
    return out, r, (kiri, atas)


def nms(kotak: np.ndarray, skor: np.ndarray, iou: float) -> np.ndarray:
    """NMS serakah (urut skor turun). kotak xyxy (N, 4). Kembalikan indeks yang dipertahankan."""
    urut = np.argsort(-skor, kind="stable")
    x1, y1, x2, y2 = kotak.T
    luas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    simpan = []
    while urut.size:
        i = urut[0]
        simpan.append(i)
        if urut.size == 1:
            break
        sisa = urut[1:]
        iw = (np.minimum(x2[i], x2[sisa]) - np.maximum(x1[i], x1[sisa])).clip(0)
        ih = (np.minimum(y2[i], y2[sisa]) - np.maximum(y1[i], y1[sisa])).clip(0)
        inter = iw * ih
        iou_ = inter / (luas[i] + luas[sisa] - inter + 1e-9)
        urut = sisa[iou_ <= iou]
    return np.array(simpan, dtype=np.int64)


def pasca_proses(keluaran: np.ndarray, skala: float, pad, ukuran_asli, conf: float, iou: float, max_det: int):
    """keluaran (1, 4+nc, N) -> list Kotak dalam koordinat gambar asli, urut keyakinan turun."""
    p = keluaran[0].T                                   # (N, 4+nc)
    skor_kelas = p[:, 4:]
    k = skor_kelas.argmax(1)
    s = skor_kelas[np.arange(len(p)), k]
    m = s > conf
    if not m.any():
        return []
    p, k, s = p[m], k[m], s[m]
    if len(s) > MAX_NMS:
        top = np.argsort(-s, kind="stable")[:MAX_NMS]
        p, k, s = p[top], k[top], s[top]
    cx, cy, w, h = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    simpan = nms(xyxy + (k[:, None] * MAX_WH), s, iou)[:max_det]
    xyxy, k, s = xyxy[simpan], k[simpan], s[simpan]
    H, W = ukuran_asli
    xyxy[:, [0, 2]] = ((xyxy[:, [0, 2]] - pad[0]) / skala).clip(0, W)
    xyxy[:, [1, 3]] = ((xyxy[:, [1, 3]] - pad[1]) / skala).clip(0, H)
    return [Kotak(*map(float, b), float(c), int(kk)) for b, c, kk in zip(xyxy, s, k)]


class Detektor:
    """Model ONNX dimuat sekali; prediksi() aman dipanggil berulang (satu thread per panggilan)."""

    def __init__(self, path, imgsz=640, conf=0.001, iou=0.7, max_det=300, threads=None, arena_memori=False):
        import onnxruntime as ort
        so = ort.SessionOptions()
        # arena memori CPU dimatikan: RSS tetap ~110 MB setelah banyak foto (dengan arena ~260 MB), waktu ~9% lebih lambat
        so.enable_cpu_mem_arena = arena_memori
        # dua sesi ONNX (pemisah daun + detektor) dipanggil bergantian; thread yang "berputar" menunggu membuat sesi lain melambat
        # (prototipe v9: waktu inti ±400 ms dengan spinning, 177 ms tanpa; reports/v9_latar/aplikasi_v9/RINGKASAN_FASE4_5.md)
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        if threads:
            so.intra_op_num_threads = threads
        self.sesi = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
        self.masukan = self.sesi.get_inputs()[0].name
        self.imgsz, self.conf, self.iou, self.max_det = imgsz, conf, iou, max_det

    def prediksi(self, img_rgb: np.ndarray):
        lb, r, pad = letterbox(img_rgb, self.imgsz)
        x = (lb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        out = self.sesi.run(None, {self.masukan: np.ascontiguousarray(x)})[0]
        return pasca_proses(out, r, pad, img_rgb.shape[:2], self.conf, self.iou, self.max_det)
