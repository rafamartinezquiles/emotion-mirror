from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

# MediaPipe FaceMesh landmark indices (subset).
RIGHT_EYE = [33, 160, 158, 133, 153, 144]   # outer, upper1, upper2, inner, lower1, lower2
LEFT_EYE  = [362, 385, 387, 263, 373, 380]

MOUTH = {
    "left": 61,
    "right": 291,
    "upper": 13,
    "lower": 14,
}


def _lm_to_xy(face_lm, idx: int, w: int, h: int) -> np.ndarray:
    p = face_lm.landmark[idx]
    return np.array([p.x * w, p.y * h], dtype=np.float32)


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def eye_aspect_ratio(face_lm, eye_idx, w: int, h: int) -> float:
    # EAR = (||p2-p6|| + ||p3-p5||) / (2*||p1-p4||)
    p1 = _lm_to_xy(face_lm, eye_idx[0], w, h)
    p2 = _lm_to_xy(face_lm, eye_idx[1], w, h)
    p3 = _lm_to_xy(face_lm, eye_idx[2], w, h)
    p4 = _lm_to_xy(face_lm, eye_idx[3], w, h)
    p5 = _lm_to_xy(face_lm, eye_idx[4], w, h)
    p6 = _lm_to_xy(face_lm, eye_idx[5], w, h)
    return (_dist(p2, p6) + _dist(p3, p5)) / (2.0 * _dist(p1, p4) + 1e-6)


def mouth_aspect_ratio(face_lm, w: int, h: int) -> float:
    left = _lm_to_xy(face_lm, MOUTH["left"], w, h)
    right = _lm_to_xy(face_lm, MOUTH["right"], w, h)
    upper = _lm_to_xy(face_lm, MOUTH["upper"], w, h)
    lower = _lm_to_xy(face_lm, MOUTH["lower"], w, h)
    return _dist(upper, lower) / (_dist(left, right) + 1e-6)


def head_yaw_pitch(face_lm, w: int, h: int) -> Tuple[float, float]:
    '''
    A cheap 2D yaw/pitch proxy from landmark geometry.
    Not true 3D pose, but stable enough for a "focus" signal.
    Returns (yaw, pitch) in roughly [-1..1].
    '''
    nose = _lm_to_xy(face_lm, 1, w, h)
    left = _lm_to_xy(face_lm, 234, w, h)
    right = _lm_to_xy(face_lm, 454, w, h)
    chin = _lm_to_xy(face_lm, 152, w, h)

    face_w = _dist(left, right) + 1e-6
    face_h = _dist(nose, chin) + 1e-6

    center = (left + right) / 2.0
    yaw = float((nose[0] - center[0]) / face_w) * 2.2
    pitch = float((nose[1] - center[1]) / face_h) * 2.0
    return float(np.clip(yaw, -1.0, 1.0)), float(np.clip(pitch, -1.0, 1.0))


def gaze_centered(face_lm, w: int, h: int) -> float:
    '''
    Uses iris landmarks (refine_landmarks=True) to estimate if gaze is centered.
    Returns 0..1 (1 = centered).
    '''
    def iris_center(start: int) -> np.ndarray:
        pts = np.stack([_lm_to_xy(face_lm, i, w, h) for i in range(start, start + 5)], axis=0)
        return pts.mean(axis=0)

    re_outer = _lm_to_xy(face_lm, 33, w, h)
    re_inner = _lm_to_xy(face_lm, 133, w, h)
    le_outer = _lm_to_xy(face_lm, 362, w, h)
    le_inner = _lm_to_xy(face_lm, 263, w, h)

    re_c = iris_center(468)
    le_c = iris_center(473)

    def norm_pos(c, outer, inner):
        d = _dist(outer, inner) + 1e-6
        return float(_dist(c, outer) / d)

    re_pos = norm_pos(re_c, re_outer, re_inner)
    le_pos = norm_pos(le_c, le_outer, le_inner)

    err = abs(re_pos - 0.5) + abs(le_pos - 0.5)
    return float(np.clip(1.0 - err * 1.8, 0.0, 1.0))


@dataclass
class FaceMetrics:
    ear: float
    mar: float
    yaw: float
    pitch: float
    gaze: float
    smile: float
    blink_rate: float = 0.0
    focus: float = 0.0
    engagement: float = 0.0


@dataclass
class CalibrationState:
    started_at: float = 0.0
    ready: bool = False
    samples: int = 0

    ear_baseline: float = 0.0
    mar_baseline: float = 0.0
    yaw_baseline: float = 0.0
    pitch_baseline: float = 0.0

    blink_ema: float = 0.0
    _blink_counter: int = 0
    _eye_closed_frames: int = 0


def compute_face_metrics(face_lm, w: int, h: int) -> FaceMetrics:
    ear_r = eye_aspect_ratio(face_lm, RIGHT_EYE, w, h)
    ear_l = eye_aspect_ratio(face_lm, LEFT_EYE, w, h)
    ear = float((ear_r + ear_l) / 2.0)

    mar = float(mouth_aspect_ratio(face_lm, w, h))
    yaw, pitch = head_yaw_pitch(face_lm, w, h)
    gaze = float(gaze_centered(face_lm, w, h))

    # Smile meter: mouth width normalized by face width (cheek-to-cheek)
    left = _lm_to_xy(face_lm, MOUTH["left"], w, h)
    right = _lm_to_xy(face_lm, MOUTH["right"], w, h)
    cheek_l = _lm_to_xy(face_lm, 234, w, h)
    cheek_r = _lm_to_xy(face_lm, 454, w, h)
    face_w = _dist(cheek_l, cheek_r) + 1e-6
    mouth_w = _dist(left, right) / face_w
    smile = float(np.clip((mouth_w - 0.32) / 0.12, 0.0, 1.0))

    return FaceMetrics(ear=ear, mar=mar, yaw=yaw, pitch=pitch, gaze=gaze, smile=smile)


def update_calibration(calib: CalibrationState, m: FaceMetrics, dt: float, window_s: float = 3.0) -> None:
    import time
    now = time.time()
    if calib.started_at == 0.0:
        calib.started_at = now

    # Blink detection (simple)
    eye_closed = m.ear < 0.19
    if eye_closed:
        calib._eye_closed_frames += 1
    else:
        if calib._eye_closed_frames >= 2:
            calib._blink_counter += 1
        calib._eye_closed_frames = 0

    elapsed = max(1e-3, now - calib.started_at)
    blinks_per_min = (calib._blink_counter / elapsed) * 60.0
    calib.blink_ema = 0.9 * calib.blink_ema + 0.1 * blinks_per_min

    if not calib.ready:
        calib.samples += 1
        a = 1.0 / calib.samples
        calib.ear_baseline = (1 - a) * calib.ear_baseline + a * m.ear
        calib.mar_baseline = (1 - a) * calib.mar_baseline + a * m.mar
        calib.yaw_baseline = (1 - a) * calib.yaw_baseline + a * m.yaw
        calib.pitch_baseline = (1 - a) * calib.pitch_baseline + a * m.pitch
        if elapsed >= window_s:
            calib.ready = True
        return

    # Scores (0..1)
    head_forward = 1.0 - min(1.0, abs(m.yaw - calib.yaw_baseline) * 1.8 + abs(m.pitch - calib.pitch_baseline) * 1.2)
    eyes_open = float(np.clip((m.ear - 0.16) / 0.10, 0.0, 1.0))
    focus = float(np.clip(0.55 * m.gaze + 0.30 * head_forward + 0.15 * eyes_open, 0.0, 1.0))

    blink_penalty = float(np.clip((calib.blink_ema - 25.0) / 35.0, 0.0, 1.0))
    alertness = float(np.clip(0.7 * eyes_open + 0.3 * (1.0 - blink_penalty), 0.0, 1.0))
    engagement = float(np.clip(0.60 * focus + 0.25 * m.smile + 0.15 * alertness, 0.0, 1.0))

    m.blink_rate = float(calib.blink_ema)
    m.focus = focus
    m.engagement = engagement
