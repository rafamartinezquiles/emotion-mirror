from __future__ import annotations

from dataclasses import dataclass

from .metrics import CalibrationState, FaceMetrics


@dataclass
class EmotionResult:
    label: str
    confidence: float
    valence: float   # -1..1
    arousal: float   # 0..1


def infer_emotion(m: FaceMetrics, calib: CalibrationState) -> EmotionResult:
    '''
    Lightweight heuristic emotion inference.
    It’s for fun demos; it is NOT a scientifically validated emotion detector.
    '''
    smile = m.smile
    eyes_open = max(0.0, min(1.0, (m.ear - 0.16) / 0.10))
    focus = m.focus if calib.ready else 0.5

    mar_base = calib.mar_baseline if calib.mar_baseline > 0 else m.mar
    mouth_open = max(0.0, (m.mar - mar_base) * 4.0)

    if smile > 0.65 and eyes_open > 0.35:
        return EmotionResult("Happy", confidence=min(1.0, 0.6 + 0.5 * smile), valence=0.8, arousal=0.6)
    if mouth_open > 0.55 and eyes_open > 0.55:
        return EmotionResult("Surprised", confidence=min(1.0, 0.55 + 0.45 * mouth_open), valence=0.2, arousal=0.9)
    if eyes_open < 0.25 or (calib.ready and (m.pitch - calib.pitch_baseline) > 0.35):
        return EmotionResult("Tired/Concerned", confidence=0.55 + 0.4 * (1.0 - eyes_open), valence=-0.4, arousal=0.25)

    return EmotionResult("Neutral", confidence=min(1.0, 0.55 + 0.35 * focus), valence=0.0, arousal=0.35)
