"""
main.py — Entry point for the Autonomous Driving Safety Demo.

Usage:
    python3 main.py
    python3 main.py --source path/to/video.mp4   # use a video file instead of webcam
    python3 main.py --lat 44.519 --lng -88.019    # supply GPS coords for Google context

Press  Q  to quit.

⚠️  DEMO ONLY — This software does not control any vehicle.
"""

import argparse
import sys
import time

import cv2
import numpy as np

import config
from tm_model import TeachableMachineModel
from safety_logic import decide
from google_context import build_context_line
from logger import DetectionLogger


# ── HUD drawing helpers ───────────────────────────────────────────────────────

def _put(img, text: str, pos: tuple, scale: float, color, thickness: int = 1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def draw_hud(frame: np.ndarray, decision: dict, context_line: str, fps: float) -> np.ndarray:
    """Overlay the safety HUD onto `frame` (in-place)."""
    h, w = frame.shape[:2]
    action     = decision["action"]
    confidence = decision["confidence"]
    class_name = decision["class_name"]
    reason     = decision["reason"]

    color = config.ACTION_COLORS.get(action, (200, 200, 200))

    # ── Semi-transparent panel ────────────────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 150), (20, 20, 20), -1)
    cv2.addWeighted(overlay, config.OVERLAY_ALPHA, frame, 1 - config.OVERLAY_ALPHA, 0, frame)

    # ── Action banner ─────────────────────────────────────────────────────────
    banner_h = 56
    cv2.rectangle(frame, (0, 0), (w, banner_h), color, -1)
    _put(frame, f"  {action}", (10, 40), 1.2, (255, 255, 255), thickness=2)

    # ── Stats line ────────────────────────────────────────────────────────────
    _put(frame, f"Class: {class_name}   Confidence: {confidence:.0%}   FPS: {fps:.1f}",
         (12, 78), 0.6, (220, 220, 220))

    # ── Reason ────────────────────────────────────────────────────────────────
    _put(frame, f"Reason: {reason}", (12, 104), 0.55, (200, 200, 200))

    # ── Google context ────────────────────────────────────────────────────────
    _put(frame, context_line, (12, 130), 0.50, (180, 180, 180))

    # ── Alert border ─────────────────────────────────────────────────────────
    level = decision.get("alert_level", "LOW")
    if level == "HIGH":
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 6)

    return frame


# ── Main loop ─────────────────────────────────────────────────────────────────

def run(source, lat, lng):
    print("\n⚠️  DEMO ONLY — not for vehicle control.\n")

    model  = TeachableMachineModel()
    logger = DetectionLogger()
    print(f"[main] Logging detections to: {logger.log_path}")

    context_line = build_context_line(lat, lng)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[main] ERROR: Cannot open source '{source}'. Check WEBCAM_INDEX in config.py.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    prev_time = time.time()
    print("[main] Running — press Q to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[main] End of stream or read error.")
            break

        # ── Inference ─────────────────────────────────────────────────────────
        class_name, confidence, _ = model.predict(frame)
        decision = decide(class_name, confidence)

        # ── FPS ───────────────────────────────────────────────────────────────
        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        # ── HUD ───────────────────────────────────────────────────────────────
        draw_hud(frame, decision, context_line, fps)

        # ── Log ───────────────────────────────────────────────────────────────
        logger.log(decision)

        cv2.imshow("Autonomous Driving Safety Demo  [Q = quit]", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[main] Done.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Autonomous Driving Safety Demo")
    p.add_argument("--source", default=None,
                   help="Video source: integer webcam index or path to video file "
                        f"(default: {config.WEBCAM_INDEX})")
    p.add_argument("--lat",  type=float, default=None, help="GPS latitude  (for Google context)")
    p.add_argument("--lng",  type=float, default=None, help="GPS longitude (for Google context)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    source = int(args.source) if (args.source and args.source.isdigit()) else (args.source or config.WEBCAM_INDEX)
    run(source, args.lat, args.lng)
