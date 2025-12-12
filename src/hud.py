from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from .emotion import EmotionResult
from .metrics import CalibrationState, FaceMetrics


# ---------- small UI helpers ----------

def _clamp01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _rounded_rect(img, x: int, y: int, w: int, h: int, r: int, color, thickness: int = -1) -> None:
    """Draw a rounded rectangle. thickness=-1 => filled."""
    r = max(0, min(r, min(w, h) // 2))
    if thickness < 0:
        # filled: draw center + side rects + 4 circles
        cv2.rectangle(img, (x + r, y), (x + w - r, y + h), color, -1)
        cv2.rectangle(img, (x, y + r), (x + w, y + h - r), color, -1)
        cv2.circle(img, (x + r, y + r), r, color, -1)
        cv2.circle(img, (x + w - r, y + r), r, color, -1)
        cv2.circle(img, (x + r, y + h - r), r, color, -1)
        cv2.circle(img, (x + w - r, y + h - r), r, color, -1)
    else:
        # outline: use a mask trick by drawing filled then eroding? keep it simple with polylines approximation
        overlay = img.copy()
        _rounded_rect(overlay, x, y, w, h, r, color, -1)
        _rounded_rect(overlay, x + thickness, y + thickness, w - 2 * thickness, h - 2 * thickness, max(0, r - thickness), (0, 0, 0), -1)
        # composite: keep only ring
        ring = cv2.subtract(overlay, img)  # crude but works ok
        img[:] = cv2.add(img, ring)


def _drop_shadow(img, x: int, y: int, w: int, h: int, r: int, blur: int = 17, alpha: float = 0.35) -> None:
    """Soft shadow behind a rounded rect."""
    H, W = img.shape[:2]
    shadow = np.zeros((H, W), dtype=np.uint8)
    _rounded_rect(shadow, x, y, w, h, r, 255, -1)
    shadow = cv2.GaussianBlur(shadow, (blur | 1, blur | 1), 0)

    # Apply as darkening
    shadow_f = (shadow.astype(np.float32) / 255.0) * alpha
    for c in range(3):
        img[:, :, c] = np.clip(img[:, :, c].astype(np.float32) * (1.0 - shadow_f), 0, 255).astype(np.uint8)


def _glass_panel(img, x: int, y: int, w: int, h: int, r: int = 18, tint=(20, 20, 20), alpha: float = 0.55) -> None:
    """Blurred glass card with a subtle tint."""
    H, W = img.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return

    roi = img[y0:y1, x0:x1].copy()
    roi_blur = cv2.GaussianBlur(roi, (21, 21), 0)
    tinted = np.full_like(roi_blur, tint, dtype=np.uint8)
    glass = cv2.addWeighted(roi_blur, 1.0 - alpha, tinted, alpha, 0)

    # mask rounded corners
    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    _rounded_rect(mask, 0, 0, x1 - x0, y1 - y0, r, 255, -1)

    out = roi.copy()
    out[mask == 255] = glass[mask == 255]
    img[y0:y1, x0:x1] = out

    # subtle border
    border = img.copy()
    _rounded_rect(border, x, y, w, h, r, (255, 255, 255), -1)
    cv2.addWeighted(border, 0.10, img, 0.90, 0, img)


def _text(img, s: str, x: int, y: int, scale: float, color, thickness: int = 1, shadow: bool = True) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    if shadow:
        cv2.putText(img, s, (x + 1, y + 1), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def _pill(img, x: int, y: int, text: str, bg, fg=(255, 255, 255), pad_x: int = 10, pad_y: int = 7, r: int = 999) -> Tuple[int, int]:
    """Draw a rounded 'pill' label. Returns (w, h)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
    w = tw + 2 * pad_x
    h = th + 2 * pad_y
    _rounded_rect(img, x, y, w, h, min(r, h // 2), bg, -1)
    _text(img, text, x + pad_x, y + pad_y + th, scale, fg, 1, shadow=False)
    return w, h


def _progress(img, x: int, y: int, w: int, label: str, pct: float, color_fill, show_value: bool = True) -> None:
    """Modern progress bar with track + fill + right-side value."""
    pct = _clamp01(pct)
    track_h = 10
    r = 6

    # label
    _text(img, label, x, y, 0.52, (240, 240, 240), 1, shadow=True)

    # bar geometry
    bar_y = y + 10
    # track
    _rounded_rect(img, x, bar_y, w, track_h, r, (255, 255, 255), -1)
    cv2.addWeighted(img, 0.0, img, 1.0, 0, img)  # no-op; keeps pattern consistent

    # dim track (so fill pops)
    overlay = img.copy()
    _rounded_rect(overlay, x, bar_y, w, track_h, r, (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.10, img, 0.90, 0, img)

    # fill
    fw = max(0, int(w * pct))
    if fw > 0:
        _rounded_rect(img, x, bar_y, fw, track_h, r, color_fill, -1)

    if show_value:
        val = f"{int(round(pct * 100)):d}%"
        (tw, th), _ = cv2.getTextSize(val, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        _text(img, val, x + w - tw, bar_y + track_h + th + 6, 0.48, (220, 220, 220), 1, shadow=True)


# ---------- main HUD ----------

def draw_hud(frame, m: FaceMetrics, emo: EmotionResult, calib: CalibrationState, fps: float) -> None:
    pad = 22
    card_w = 420
    card_h = 210
    x, y = pad, pad
    r = 18

    # shadow + glass
    _drop_shadow(frame, x + 6, y + 8, card_w, card_h, r, blur=21, alpha=0.38)
    _glass_panel(frame, x, y, card_w, card_h, r=r, tint=(18, 18, 18), alpha=0.58)

    # Title row
    _text(frame, "EmotionMirror", x + 18, y + 34, 0.92, (255, 255, 255), 2, shadow=True)
    _text(frame, f"{fps:0.0f} FPS", x + card_w - 95, y + 34, 0.55, (210, 210, 210), 1, shadow=True)

    # Status / emotion pill
    if not calib.ready:
        _pill(frame, x + 18, y + 48, "Calibrating... hold still", bg=(80, 80, 80), fg=(255, 255, 255))
    else:
        # color by emotion label
        label = emo.label.lower()
        if "happy" in label:
            bg = (70, 170, 90)
        elif "surpris" in label:
            bg = (70, 140, 200)
        elif "tired" in label or "concern" in label:
            bg = (60, 90, 170)
        else:
            bg = (90, 90, 90)

        _pill(frame, x + 18, y + 48, f"{emo.label} - {emo.confidence:.2f}", bg=bg, fg=(255, 255, 255))

    # Bars (tighter + cleaner)
    bx = x + 18
    bw = card_w - 36
    y0 = y + 92

    _progress(frame, bx, y0,       bw, "Smile",       m.smile,      (60, 190, 230))
    _progress(frame, bx, y0 + 44,  bw, "Engagement",  m.engagement, (90, 220, 140))
    _progress(frame, bx, y0 + 88,  bw, "Focus",       m.focus,      (255, 200, 90))

    # Footer stats
    g = f"Gaze: {m.gaze:.2f}"
    b = f"Blinks: {m.blink_rate:.1f}/min"
    _text(frame, g, x + 18, y + card_h - 18, 0.52, (210, 210, 210), 1, shadow=True)
    (tw, _), _ = cv2.getTextSize(b, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    _text(frame, b, x + card_w - 18 - tw, y + card_h - 18, 0.52, (210, 210, 210), 1, shadow=True)

    # Mini head widget (top-right)
    _draw_head_widget(frame, m)


def _draw_head_widget(frame, m: FaceMetrics) -> None:
    H, W = frame.shape[:2]
    w, h = 150, 54
    x = W - w - 22
    y = 18
    r = 16

    _drop_shadow(frame, x + 5, y + 7, w, h, r, blur=19, alpha=0.35)
    _glass_panel(frame, x, y, w, h, r=r, tint=(18, 18, 18), alpha=0.55)

    _text(frame, "Head", x + 14, y + 34, 0.60, (240, 240, 240), 1, shadow=True)

    cx, cy = x + w - 34, y + h // 2 + 2
    cv2.circle(frame, (cx, cy), 13, (255, 255, 255), 1, cv2.LINE_AA)

    dx = int(np.clip(m.yaw, -1, 1) * 9)
    dy = int(np.clip(m.pitch, -1, 1) * 9)
    cv2.circle(frame, (cx + dx, cy + dy), 4, (90, 240, 230), -1, cv2.LINE_AA)


def draw_landmarks_debug(frame, face_lm) -> None:
    h, w = frame.shape[:2]
    for p in face_lm.landmark[::2]:
        x, y = int(p.x * w), int(p.y * h)
        cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)
