import math
import os
from ultralytics import YOLO
import cv2
import cvzone
import torch
from sort import *
import numpy as np
import arabic_reshaper
from bidi.algorithm import get_display
from PIL import ImageFont, ImageDraw, Image

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

# ── Arabic font ────────────────────────────────────────────────────────────────
_FONT_PATHS = [
    "C:/Windows/Fonts/tahoma.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
]
ARABIC_FONT = next((p for p in _FONT_PATHS if os.path.exists(p)), None)
if not ARABIC_FONT:
    raise FileNotFoundError("Arabic font not found. Tahoma / Arial / Segoe UI required.")

_font_cache = {}

def get_font(size):
    if size not in _font_cache:
        _font_cache[size] = ImageFont.truetype(ARABIC_FONT, size)
    return _font_cache[size]

def ar(text):
    """Reshape + BiDi an Arabic string so PIL renders it correctly."""
    return get_display(arabic_reshaper.reshape(str(text)))


# ── Model ──────────────────────────────────────────────────────────────────────
model = YOLO("D:/yolo/yolo_weights/yolov8x-worldv2.pt")
model.to("cuda")

TRACK_CLASSES = ["knife"]
model.set_classes(TRACK_CLASSES)
classNames = model.names

# ── Arabic labels & colours ────────────────────────────────────────────────────
AR = {
    "knife":   "سكين",
    "unknown": "غير محدد",
}
KNIFE_RGB = (255,  59,  48)   # red — PIL (R,G,B)
KNIFE_BGR = ( 48,  59, 255)   # red — OpenCV (B,G,R)


# ── Video & tracker ────────────────────────────────────────────────────────────
cap     = cv2.VideoCapture("D:/yolo/videos/4.mp4")
tracker = Sort(max_age=20, min_hits=2, iou_threshold=0.3)
id_best = {}   # {id: {"label": str, "conf": float}}

# ── Safe-zone geometry (TikTok / Instagram Reels = 720×1280 display) ──────────
orig_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
_sx, _sy = orig_w / 720, orig_h / 1280

HUD_X    = max(16, int(28 * _sx))    # left margin in original-frame px
HUD_Y    = max(10, int(220 * _sy))   # below top danger zone (~17 %)
FS_WARN  = max(36, int(56 * _sy))    # large threat-warning font
FS_LABEL = max(18, int(26 * _sy))    # detection / track label font
BORDER_T = max(6,  int(12 * _sx))    # threat-border thickness


# ── IoU helper ─────────────────────────────────────────────────────────────────
def box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if not inter:
        return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter + 1e-9)


# ── Frame-level Arabic text batcher ───────────────────────────────────────────
class FrameAR:
    """
    Collect all Arabic draw calls, then composite them in one PIL round-trip
    per frame.
    """

    def __init__(self):
        self._q = []

    def text(self, txt, pos, size=28, fg=(255, 255, 255),
             bg=(15, 15, 15, 210), pad=8, radius=8):
        self._q.append(("text", txt, pos, size, fg, bg, pad, radius))

    def flush(self, img):
        if not self._q:
            return img
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGBA")
        ov  = Image.new("RGBA", pil.size, (0, 0, 0, 0))
        d   = ImageDraw.Draw(ov)
        for item in self._q:
            if item[0] == "text":
                _, txt, (x, y), size, fg, bg, pad, r = item
                shaped = ar(txt)
                font   = get_font(size)
                bb     = d.textbbox((0, 0), shaped, font=font)
                tw, th = bb[2] - bb[0], bb[3] - bb[1]
                d.rounded_rectangle(
                    [x - pad, y - pad, x + tw + pad, y + th + pad],
                    radius=r, fill=bg,
                )
                d.text((x, y), shaped, font=font, fill=fg)
        self._q.clear()
        return cv2.cvtColor(
            np.array(Image.alpha_composite(pil, ov).convert("RGB")),
            cv2.COLOR_RGB2BGR,
        )


# ── Threat-border overlay ──────────────────────────────────────────────────────
def draw_threat_border(img):
    """Draw a layered red border around the frame to signal danger."""
    h, w = img.shape[:2]
    # Outer glow: semi-transparent thick border
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w - 1, h - 1), KNIFE_BGR, BORDER_T * 3)
    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
    # Inner solid border
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), KNIFE_BGR, BORDER_T)


# ── Main loop ──────────────────────────────────────────────────────────────────
while True:
    success, img = cap.read()
    if not success:
        break

    fa         = FrameAR()
    results    = model(img, stream=True, imgsz=736, device=0)
    detections = np.empty((0, 5))
    det_meta   = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            conf  = math.ceil(box.conf[0] * 100) / 100
            cls   = int(box.cls[0])
            label = TRACK_CLASSES[cls] if 0 <= cls < len(TRACK_CLASSES) else "unknown"

            cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1),
                              colorR=KNIFE_BGR, colorC=KNIFE_BGR)

            if conf >= 0.35:
                detections = np.vstack((detections, [x1, y1, x2, y2, conf]))
                det_meta.append((x1, y1, x2, y2, float(conf), label))

    resultTracker = tracker.update(detections)

    # Threat alert
    knife_detected = detections.shape[0] > 0
    if knife_detected:
        draw_threat_border(img)
        fa.text(
            "تحذير: سكين مرصود",
            (HUD_X, HUD_Y),
            size=FS_WARN,
            fg=(255, 255, 255),
            bg=(160, 10, 10, 240),
            pad=18,
            radius=14,
        )
    else:
        fa.text(
            "آمن",
            (HUD_X, HUD_Y),
            size=FS_WARN,
            fg=(255, 255, 255),
            bg=(30, 130, 30, 200),
            pad=18,
            radius=14,
        )

    # Track labels
    for result in resultTracker:
        x1, y1, x2, y2, Id = result
        x1, y1, x2, y2, Id = int(x1), int(y1), int(x2), int(y2), int(Id)

        best_label, best_conf, best_v = "unknown", 0.0, 0.0
        for dx1, dy1, dx2, dy2, dconf, dlabel in det_meta:
            v = box_iou((x1, y1, x2, y2), (dx1, dy1, dx2, dy2))
            if v > best_v:
                best_v, best_label, best_conf = v, dlabel, dconf

        prev = id_best.get(Id)
        if prev is None or best_conf > prev["conf"]:
            id_best[Id] = {"label": best_label, "conf": best_conf}

        stable = id_best[Id]["label"]
        cv2.rectangle(img, (x1, y1), (x2, y2), KNIFE_BGR, 2)

        fa.text(
            f"{AR.get(stable, stable)}  #{Id}",
            (x1, max(44, y1 - 44)),
            size=FS_LABEL,
            fg=(255, 255, 255),
            bg=(*KNIFE_RGB, 215),
            pad=7,
            radius=7,
        )

    img     = fa.flush(img)
    display = cv2.resize(img, (720, 1280))
    cv2.imshow("Image", display)
    cv2.waitKey(1)
