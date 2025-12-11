from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from .emotion import EmotionResult
from .metrics import CalibrationState, FaceMetrics


def _panel(img, x: int, y: int, w: int, h: int, alpha: float = 0.35) -> None:
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 1)


def _bar(img, x: int, y: int, w: int, h: int, pct: float, label: str, color: Tuple[int, int, int]) -> None:
    pct = float(np.clip(pct, 0.0, 1.0))
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 1)
    cv2.rectangle(img, (x, y), (x + int(w * pct), y + h), color, -1)
    cv2.putText(img, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2, cv2.LINE_AA)


def draw_hud(frame, m: FaceMetrics, emo: EmotionResult, calib: CalibrationState, fps: float) -> None:
    pad = 20
    panel_w = 440
    panel_h = 265
    x, y = pad, pad

    _panel(frame, x, y, panel_w, panel_h)

    cv2.putText(frame, "EmotionMirror", (x + 15, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    if not calib.ready:
        cv2.putText(frame, "Calibrating... hold still", (x + 15, y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, f"Emotion: {emo.label} ({emo.confidence:.2f})", (x + 15, y + 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 2, cv2.LINE_AA)

    _bar(frame, x + 15, y + 98, 400, 18, m.smile, "Smile meter", (0, 200, 255))
    _bar(frame, x + 15, y + 134, 400, 18, m.engagement, "Engagement", (0, 255, 120))
    _bar(frame, x + 15, y + 170, 400, 18, m.focus, "Focus score", (255, 200, 0))

    cv2.putText(frame, f"Gaze centered: {m.gaze:.2f}", (x + 15, y + 210), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 210, 210), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Blink rate (EMA): {m.blink_rate:.1f}/min", (x + 15, y + 235), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 210, 210), 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.0f}", (x + 335, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (210, 210, 210), 2, cv2.LINE_AA)

    # mini head pose indicator
    H, W = frame.shape[:2]
    cx, cy = W - 160, 40
    cv2.circle(frame, (cx, cy), 12, (255, 255, 255), 1)
    dx = int(np.clip(m.yaw, -1, 1) * 10)
    dy = int(np.clip(m.pitch, -1, 1) * 10)
    cv2.circle(frame, (cx + dx, cy + dy), 4, (0, 255, 255), -1)
    cv2.putText(frame, "Head", (cx + 20, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 2, cv2.LINE_AA)


def draw_landmarks_debug(frame, face_lm) -> None:
    h, w = frame.shape[:2]
    for p in face_lm.landmark[::2]:
        x, y = int(p.x * w), int(p.y * h)
        cv2.circle(frame, (x, y), 1, (255, 255, 255), -1)
