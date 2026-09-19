from typing import Any

from ml.inference.predictor import Predictor


_predictor: Predictor | None = None


def get_predictor() -> Predictor:
    """Load the model once, then reuse it for later requests."""
    global _predictor
    if _predictor is None:
        _predictor = Predictor()
    return _predictor


def predict_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Send an already validated profile to the existing ML predictor."""
    return get_predictor().predict(profile)