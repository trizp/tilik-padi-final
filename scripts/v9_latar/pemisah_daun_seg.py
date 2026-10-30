"""
pemisah_daun_seg.py - Pemisah daun ringan (YOLOv8n-seg hasil distilasi SAM2) dan aturan pemilih daun (tiruan aturan v2).
Dipakai fase 1-3 rencana aplikasi v9 (KEPUTUSAN_PROYEK.md 2026-10-20). Tidak memakai label.

- prediksi_pt(): ultralytics .pt (GPU), masker resolusi asli (retina_masks) -> daftar instance (keyakinan, masker bool).
- OnnxSeg: ONNX Runtime CPU + pasca-proses numpy murni (letterbox 640, NMS, masker dari prototipe), untuk mengukur waktu/RAM seperti aplikasi.
- pilih_daun(): kandidat keyakinan >= tau, luas 3-95% foto, >= 50% piksel vegetasi; skor sqrt(luas) x (0,2 + cakupan kotak tengah 40%);
  pasca-proses umum.pasca_proses; tidak ada kandidat atau terpilih < 5% foto -> cadangan (masker penuh).
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import umum  # noqa: E402


def vegetasi(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (h >= 8) & (h <= 100) & (s >= 40) & (v >= 40)


def ciri_kandidat(bgr, masks, confs):
    """Untuk tiap instance: (keyakinan, luas, lolos_vegetasi, skor). Skor = sqrt(luas) x (0,2 + cakupan tengah)."""
    h, w = bgr.shape[:2]
    veg = vegetasi(bgr)
    y0, y1, x0, x1 = int(0.3 * h), int(0.7 * h), int(0.3 * w), int(0.7 * w)
    out = []
    for m, c in zip(masks, confs):
        luas = float(m.mean())
        vg = float(veg[m].mean()) if m.any() else 0.0
        tengah = float(m[y0:y1, x0:x1].mean())
        out.append((float(c), luas, vg >= 0.5, float(np.sqrt(luas) * (0.2 + tengah))))
    return out


def pilih_indeks(ciri, tau):
    """Indeks instance terpilih atau None (cadangan) untuk ambang tau."""
    best = None
    for i, (c, luas, veg_ok, skor) in enumerate(ciri):
        if c < tau or not (0.03 <= luas <= 0.95) or not veg_ok:
            continue
        if best is None or skor > ciri[best][3]:
            best = i
    if best is None or ciri[best][1] < 0.05:
        return None
    return best


def pilih_daun(bgr, masks, confs, tau):
    """Kembalikan (masker akhir bool, cadangan bool)."""
    i = pilih_indeks(ciri_kandidat(bgr, masks, confs), tau)
    if i is None:
        return np.ones(bgr.shape[:2], bool), True
    return umum.pasca_proses(masks[i]), False


def prediksi_pt(model, bgr, conf_min=0.05):
    r = model.predict(bgr, imgsz=640, conf=conf_min, iou=0.7, retina_masks=True, device=0, verbose=False)[0]
    if r.masks is None or len(r.boxes) == 0:
        return [], []
    m = r.masks.data.cpu().numpy() > 0.5
    if m.shape[1:] != bgr.shape[:2]:
        m = np.stack([cv2.resize(x.astype(np.uint8), (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST) > 0 for x in m])
    return list(m), r.boxes.conf.cpu().numpy().tolist()


# ------------------------------------------------------------------ ONNX Runtime + numpy (seperti aplikasi)
class OnnxSeg:
    def __init__(self, path, imgsz=640, conf_min=0.05, iou=0.7, max_det=30, threads=None):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.enable_cpu_mem_arena = False
        if threads:
            so.intra_op_num_threads = threads
        self.s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.inp = self.s.get_inputs()[0].name
        self.imgsz, self.conf_min, self.iou, self.max_det = imgsz, conf_min, iou, max_det

    def _letterbox(self, rgb):
        h, w = rgb.shape[:2]
        r = min(self.imgsz / h, self.imgsz / w)
        nh, nw = round(h * r), round(w * r)
        im = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nh, nw) != (h, w) else rgb
        top, left = (self.imgsz - nh) // 2, (self.imgsz - nw) // 2
        out = np.full((self.imgsz, self.imgsz, 3), 114, np.uint8)
        out[top:top + nh, left:left + nw] = im
        return out, r, top, left, nh, nw

    @staticmethod
    def _nms(b, s, thr):
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
            o = o[1:][inter / (ar[i] + ar[o[1:]] - inter + 1e-9) <= thr]
        return np.array(keep, int)

    def prediksi(self, bgr):
        rgb = bgr[..., ::-1]
        lb, r, top, left, nh, nw = self._letterbox(np.ascontiguousarray(rgb))
        x = (lb.transpose(2, 0, 1)[None].astype(np.float32)) / 255.0
        out0, protos = self.s.run(None, {self.inp: x})
        p = out0[0].T                                     # (8400, 4 + 1 + 32)
        conf = p[:, 4]
        k = conf >= self.conf_min
        p, conf = p[k], conf[k]
        if not len(p):
            return [], []
        xywh, coef = p[:, :4], p[:, 5:]
        b = np.c_[xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2, xywh[:, 0] + xywh[:, 2] / 2, xywh[:, 1] + xywh[:, 3] / 2]
        keep = self._nms(b, conf, self.iou)[: self.max_det]
        b, conf, coef = b[keep], conf[keep], coef[keep]
        pr = protos[0]                                    # (32, 160, 160)
        c, mh, mw = pr.shape
        m = 1 / (1 + np.exp(-(coef @ pr.reshape(c, -1)))).reshape(-1, mh, mw)
        s = mw / self.imgsz
        h0, w0 = bgr.shape[:2]
        masks = []
        for mi, bi in zip(m, b):
            bx = (bi * s).round().astype(int)
            crop = np.zeros_like(mi)
            crop[max(bx[1], 0):max(bx[3], 0), max(bx[0], 0):max(bx[2], 0)] = 1
            mi = mi * crop
            up = cv2.resize(mi, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)[top:top + nh, left:left + nw]
            masks.append(cv2.resize(up, (w0, h0), interpolation=cv2.INTER_LINEAR) > 0.5)
        return masks, conf.tolist()
