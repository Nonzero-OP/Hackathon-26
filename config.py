"""
config.py — Central configuration for the autonomous driving safety demo.
Adjust paths, thresholds, and API keys here.
"""

import os

# ── Model ────────────────────────────────────────────────────────────────────
MODEL_PATH  = os.path.join("models", "keras_model.h5")
LABELS_PATH = os.path.join("models", "labels.txt")
IMAGE_SIZE  = (224, 224)          # Teachable Machine default input size

# ── Confidence ────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.70       # Below this → CAUTION regardless of class
LOW_CONF_THRESHOLD   = 0.40       # Below this → treat as UNKNOWN

# ── Webcam ────────────────────────────────────────────────────────────────────
WEBCAM_INDEX = 0                  # 0 = built-in camera; change for external
FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR      = "logs"
LOG_FILENAME = "detections.csv"
LOG_INTERVAL = 1.0                # Minimum seconds between CSV writes

# ── Google API (optional) ─────────────────────────────────────────────────────
# Set GOOGLE_API_KEY as an environment variable, or paste it here for dev use.
# Never commit a real key to version control.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# ── Display ───────────────────────────────────────────────────────────────────
FONT_SCALE    = 0.7
FONT_THICKNESS = 2
OVERLAY_ALPHA  = 0.55             # Translucency of the HUD overlay panel

# ── Action colours (BGR for OpenCV) ──────────────────────────────────────────
ACTION_COLORS = {
    "STOP":       (0,   0,   220),   # Red
    "SLOW DOWN":  (0,  165,  255),   # Orange
    "CAUTION":    (0,  220,  220),   # Yellow
    "PROCEED":    (0,  200,   50),   # Green
}
