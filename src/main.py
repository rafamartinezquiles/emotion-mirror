from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import cv2
import mediapipe as mp

from .metrics import CalibrationState, compute_face_metrics, update_calibration
from .emotion import infer_emotion
from .hud import draw_hud, draw_landmarks_debug


@dataclass
class AppConfig:
    camera_index: int = 0
    width: int = 1280
    height: int = 720
    mirror: bool = True
    hud: bool = True
    debug_landmarks: bool = False


def parse_args() -> AppConfig:
    p = argparse.ArgumentParser(description="EmotionMirror — real-time emotion & attention camera")
    p.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    p.add_argument("--width", type=int, default=1280, help="Capture width")
    p.add_argument("--height", type=int, default=720, help="Capture height")
    p.add_argument("--mirror", type=int, default=1, help="Mirror the preview (1=yes, 0=no)")
    p.add_argument("--no-hud", action="store_true", help="Start with HUD disabled")
    p.add_argument("--debug-landmarks", action="store_true", help="Draw face mesh landmarks (costly)")
    a = p.parse_args()
    return AppConfig(
        camera_index=a.camera,
        width=a.width,
        height=a.height,
        mirror=bool(a.mirror),
        hud=not a.no_hud,
        debug_landmarks=a.debug_landmarks,
    )


def main() -> None:
    cfg = parse_args()

    cap = cv2.VideoCapture(cfg.camera_index)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {cfg.camera_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,  # iris landmarks included
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    calib = CalibrationState()
    last_ts = time.time()

    win = "EmotionMirror — q quit | h HUD | r reset | d debug"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if cfg.mirror:
            frame = cv2.flip(frame, 1)

        now = time.time()
        dt = max(1e-3, now - last_ts)
        last_ts = now

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        metrics = None
        if res.multi_face_landmarks:
            face_lm = res.multi_face_landmarks[0]
            metrics = compute_face_metrics(face_lm, frame.shape[1], frame.shape[0])
            update_calibration(calib, metrics, dt=dt)

        if metrics is not None:
            emo = infer_emotion(metrics, calib)
            if cfg.hud:
                draw_hud(frame, metrics, emo, calib, fps=1.0 / dt)
            if cfg.debug_landmarks:
                draw_landmarks_debug(frame, res.multi_face_landmarks[0])
        else:
            if cfg.hud:
                cv2.putText(frame, "No face detected", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 255), 2)

        cv2.imshow(win, frame)
        k = cv2.waitKey(1) & 0xFF

        if k == ord("q"):
            break
        if k == ord("h"):
            cfg.hud = not cfg.hud
        if k == ord("r"):
            calib = CalibrationState()
        if k == ord("d"):
            cfg.debug_landmarks = not cfg.debug_landmarks

    face_mesh.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
