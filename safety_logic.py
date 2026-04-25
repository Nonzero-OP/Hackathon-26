"""
safety_logic.py — Maps a predicted class + confidence → a safe driving action.

This module is pure logic (no CV, no I/O) so it is easy to unit-test.
"""

from config import CONFIDENCE_THRESHOLD, LOW_CONF_THRESHOLD

# ── Rule table ────────────────────────────────────────────────────────────────
# Map normalised label → (action, reason)
# Labels are lowercased + stripped before lookup.

RULES: dict[str, tuple[str, str]] = {
    "red light":    ("STOP",       "Red traffic signal detected — must stop."),
    "yellow light": ("SLOW DOWN",  "Yellow signal — prepare to stop."),
    "green light":  ("PROCEED",    "Green signal — road clear to proceed."),
    "obstacle":     ("STOP",       "Obstacle detected ahead — stop or yield."),
    "other":        ("CAUTION",    "Unrecognised scene — proceed with caution."),
    "unknown":      ("CAUTION",    "Scene not recognised — proceed with caution."),
}

# ── Decision function ─────────────────────────────────────────────────────────

def decide(class_name: str, confidence: float) -> dict:
    """
    Determine the recommended action given a prediction.

    Args:
        class_name:  Raw predicted label from the model.
        confidence:  Probability score in [0, 1].

    Returns:x
        dict with keys:
            action      – str, e.g. "STOP"
            reason      – str, human-readable explanation
            class_name  – normalised label used for lookup
            confidence  – original float
            alert_level – "HIGH" | "MEDIUM" | "LOW"
    """
    normalised = class_name.strip().lower()

    # Very low confidence → treat as unknown regardless of predicted class
    if confidence < LOW_CONF_THRESHOLD:
        normalised = "unknown"

    action, reason = RULES.get(normalised, ("CAUTION", f"Label '{class_name}' not in ruleset."))

    # Downgrade PROCEED/SLOW_DOWN to CAUTION when confidence is borderline
    if confidence < CONFIDENCE_THRESHOLD and action == "PROCEED":
        action = "CAUTION"
        reason = f"Confidence too low ({confidence:.0%}) to confirm clear road."

    alert_level = _alert_level(action)

    return {
        "action":     action,
        "reason":     reason,
        "class_name": normalised,
        "confidence": confidence,
        "alert_level": alert_level,
    }


def _alert_level(action: str) -> str:
    return {"STOP": "HIGH", "SLOW DOWN": "MEDIUM", "CAUTION": "MEDIUM", "PROCEED": "LOW"}.get(action, "MEDIUM")
