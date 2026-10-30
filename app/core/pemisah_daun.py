"""Pemisah daun: YOLOv8n-seg (ONNX Runtime CPU) + aturan pemilih satu helai daun, lalu latar diganti abu 114 sebelum deteksi bercak.
Alur aplikasi v9 (KEPUTUSAN_PROYEK.md 2026-10-20). Logikanya sama dengan yang divalidasi di skripsi (scripts/v9_latar/pemisah_daun_seg.py
dan umum.pasca_proses): instance dengan keyakinan >= tau, luas 3-95% foto, dan >= 50% pikselnya berwarna vegetasi; dipilih skor
sqrt(luas) x (0,2 + cakupan kotak tengah 40%). Tidak ada daun yang memenuhi (atau luasnya < 5%) -> foto cadangan: pengguna diminta
memotret ulang. Tanpa Streamlit."""
import cv2
import numpy as np

LATAR = 114   # nilai abu pengganti latar, sama dengan data latih model v9


def vegetasi(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (h >= 8) & (h <= 100) & (s >= 40) & (v >= 40)


def _isi_lubang(m):
    m = m.astype(np.uint8)
    h, w = m.shape
    pad = np.zeros((h + 2, w + 2), np.uint8)
    pad[1:-1, 1:-1] = m
    luar = pad.copy()
    cv2.floodFill(luar, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 1)
    return (m > 0) | (luar == 0)[1:-1, 1:-1]


def pasca_proses(m, min_relatif=0.10):
    """Buang potongan kecil (< 10% potongan terbesar), tutup celah, isi lubang."""
    m = (m > 0).astype(np.uint8)
    if m.sum() == 0:
        return m.astype(bool)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 2:
        luas = stats[1:, cv2.CC_STAT_AREA]
        m = np.isin(lab, 1 + np.where(luas >= min_relatif * luas.max())[0]).astype(np.uint8)
    k = max(3, round(min(m.shape) / 150)) | 1
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return _isi_lubang(m)


class PemisahDaun:
    """Model ONNX dimuat sekali; pisahkan() aman dipanggil berulang."""

    def __init__(self, path, tau, imgsz=640, conf_min=0.05, iou=0.7, max_det=30, threads=None):
        import onnxruntime as ort
        so = ort.SessionOptions()
        # dua sesi ONNX (pemisah daun + detektor) dipanggil bergantian; thread yang "berputar" menunggu membuat sesi lain melambat
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        so.enable_cpu_mem_arena = False
        if threads:
            so.intra_op_num_threads = threads
        self.sesi = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
        self.masukan = self.sesi.get_inputs()[0].name
        self.tau, self.imgsz, self.conf_min, self.iou, self.max_det = tau, imgsz, conf_min, iou, max_det

    def _nms(self, b, s):
        x1, y1, x2, y2 = b.T
        ar = (x2 - x1) * (y2 - y1)
        o = s.argsort()[::-1]
        keep = []
        while o.size:
            i = o[0]
            keep.append(i)
            xx1, yy1 = np.maximum(x1[i], x1[o[1:]]), np.maximum(y1[i], y1[o[1:]])
            xx2, yy2 = np.minimum(x2[i], x2[o[1:]]), np.minimum(y2[i], y2[o[1:]])
            inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
            o = o[1:][inter / (ar[i] + ar[o[1:]] - inter + 1e-9) <= self.iou]
        return np.array(keep, int)

    def _instance(self, rgb):
        h0, w0 = rgb.shape[:2]
        r = min(self.imgsz / h0, self.imgsz / w0)
        nh, nw = round(h0 * r), round(w0 * r)
        im = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nh, nw) != (h0, w0) else rgb
        top, left = (self.imgsz - nh) // 2, (self.imgsz - nw) // 2
        lb = np.full((self.imgsz, self.imgsz, 3), 114, np.uint8)
        lb[top:top + nh, left:left + nw] = im
        out0, protos = self.sesi.run(None, {self.masukan: lb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0})
        p = out0[0].T
        p = p[p[:, 4] >= self.conf_min]
        if not len(p):
            return [], []
        conf, xywh, coef = p[:, 4], p[:, :4], p[:, 5:]
        b = np.c_[xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2, xywh[:, 0] + xywh[:, 2] / 2, xywh[:, 1] + xywh[:, 3] / 2]
        keep = self._nms(b, conf)[: self.max_det]
        b, conf, coef = b[keep], conf[keep], coef[keep]
        pr = protos[0]
        c, mh, mw = pr.shape
        m = 1 / (1 + np.exp(-(coef @ pr.reshape(c, -1)))).reshape(-1, mh, mw)
        s = mw / self.imgsz
        masks = []
        for mi, bi in zip(m, b):
            bx = (bi * s).round().astype(int)
            crop = np.zeros_like(mi)
            crop[max(bx[1], 0):max(bx[3], 0), max(bx[0], 0):max(bx[2], 0)] = 1
            up = cv2.resize(mi * crop, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)[top:top + nh, left:left + nw]
            masks.append(cv2.resize(up, (w0, h0), interpolation=cv2.INTER_LINEAR) > 0.5)
        return masks, conf.tolist()

    def pisahkan(self, rgb):
        """rgb (H, W, 3) uint8 -> (masker daun bool (H, W), cadangan bool). Cadangan = tidak ada satu helai daun yang jelas."""
        masks, confs = self._instance(rgb)
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        h, w = rgb.shape[:2]
        veg = vegetasi(bgr)
        y0, y1, x0, x1 = int(0.3 * h), int(0.7 * h), int(0.3 * w), int(0.7 * w)
        best, skor_best, luas_best = None, -1.0, 0.0
        for m, c in zip(masks, confs):
            luas = float(m.mean())
            if c < self.tau or not (0.03 <= luas <= 0.95) or not m.any() or veg[m].mean() < 0.5:
                continue
            skor = np.sqrt(luas) * (0.2 + float(m[y0:y1, x0:x1].mean()))
            if skor > skor_best:
                best, skor_best, luas_best = m, skor, luas
        if best is None or luas_best < 0.05:
            return np.ones((h, w), bool), True
        return pasca_proses(best), False


def ganti_latar(rgb, masker):
    """Salinan foto dengan piksel di luar daun diganti abu 114 (foto asli tidak diubah)."""
    out = rgb.copy()
    out[~masker] = LATAR
    return out
