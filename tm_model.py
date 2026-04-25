"""
tm_model.py — Loads a Google Teachable Machine Keras model and runs inference.

Teachable Machine exports:
  • keras_model.h5   – the saved Keras model
  • labels.txt       – one label per line, e.g. "0 red_light"
"""

import numpy as np
import cv2
from config import MODEL_PATH, LABELS_PATH, IMAGE_SIZE


def load_labels(path: str) -> list[str]:
    """Parse labels.txt → list of class names (strips leading index)."""
    labels = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Format can be "0 red_light" or just "red_light"
            parts = line.split(maxsplit=1)
            labels.append(parts[-1])
    return labels


class TeachableMachineModel:
    """Wraps a Teachable Machine Keras model for single-frame inference."""

    def __init__(self, model_path: str = MODEL_PATH, labels_path: str = LABELS_PATH):
        # Import TF/Keras here so startup errors are local to this class
        try:
            import tensorflow as tf  # noqa: F401
            import keras
        except ImportError:
            raise ImportError(
                "TensorFlow is not installed.\n"
                "Run:  pip3 install tensorflow"
            )

        print(f"[tm_model] Loading model from '{model_path}' …")
        self.model = keras.models.load_model(model_path, compile=False)
        self.labels = load_labels(labels_path)
        self.input_size = IMAGE_SIZE
        print(f"[tm_model] Loaded {len(self.labels)} classes: {self.labels}")

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Resize & normalise a BGR OpenCV frame for Teachable Machine.
        TM models expect float32 in [-1, 1] by default.
        """
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, self.input_size)
        arr   = resized.astype(np.float32)
        arr   = (arr / 127.5) - 1.0          # TM default normalisation
        return np.expand_dims(arr, axis=0)   # shape: (1, H, W, 3)

    def predict(self, frame: np.ndarray) -> tuple[str, float, np.ndarray]:
        """
        Run inference on one frame.

        Returns:
            class_name  – predicted label string
            confidence  – float in [0, 1]
            scores      – full probability vector
        """
        tensor = self.preprocess(frame)
        scores = self.model.predict(tensor, verbose=0)[0]
        idx    = int(np.argmax(scores))
        return self.labels[idx], float(scores[idx]), scores
