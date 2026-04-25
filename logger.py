"""
logger.py — Thread-safe CSV logger for detection events.

Each row records:
  timestamp, class_name, confidence, action, reason, alert_level
"""

import csv
import os
import time
import threading
from config import LOG_DIR, LOG_FILENAME, LOG_INTERVAL

FIELDNAMES = ["timestamp", "class_name", "confidence", "action", "reason", "alert_level"]


class DetectionLogger:
    def __init__(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        self._path = os.path.join(LOG_DIR, LOG_FILENAME)
        self._lock = threading.Lock()
        self._last_write = 0.0
        self._ensure_header()

    def _ensure_header(self):
        """Write CSV header if the file doesn't exist yet."""
        if not os.path.exists(self._path):
            with open(self._path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    def log(self, decision: dict) -> bool:
        """
        Write one detection row, rate-limited by LOG_INTERVAL.

        Args:
            decision: dict returned by safety_logic.decide()

        Returns:
            True if a row was written, False if rate-limited.
        """
        now = time.time()
        if now - self._last_write < LOG_INTERVAL:
            return False

        row = {
            "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%S"),
            "class_name":  decision.get("class_name", ""),
            "confidence":  f"{decision.get('confidence', 0):.4f}",
            "action":      decision.get("action", ""),
            "reason":      decision.get("reason", ""),
            "alert_level": decision.get("alert_level", ""),
        }

        with self._lock:
            with open(self._path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)

        self._last_write = now
        return True

    @property
    def log_path(self) -> str:
        return self._path
