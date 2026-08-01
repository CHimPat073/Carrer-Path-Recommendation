"""
CareerPilot-AI – Confidence Engine
==================================

A standalone, UI-agnostic module that interprets the probability
output produced by the **Predictor** and converts it into a
structured confidence report.

This module is **read-only with respect to ML artefacts**:

* It does NOT load any model.
* It does NOT preprocess data.
* It does NOT call the Predictor.
* It does NOT use SHAP.
* It does NOT access the Knowledge Base.
* It does NOT generate recommendations, explanations or
  learning-roadmap content.

It accepts the dictionary returned by ``Predictor.predict()`` and
returns a dictionary that downstream layers (Streamlit UI, REST
API, recommendation engine, etc.) can render or post-process.
"""

from __future__ import annotations

import logging
import math
from enum import Enum
from typing import Any, Final

from ml.config import setup_logging


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Number of decimal places for percentages shown to the user.
PERCENT_DECIMALS: Final[int] = 2

#: Number of decimal places for raw probability carries in the report.
PROBABILITY_DECIMALS: Final[int] = 4

#: Tolerance for "probabilities sum to 1.0" checks.
PROBABILITY_SUM_TOLERANCE: Final[float] = 1e-6


# ---------------------------------------------------------------------------
# Confidence levels (public enum)
# ---------------------------------------------------------------------------

class ConfidenceLevel(str, Enum):
    """Discrete confidence bands derived from ``confidence_score`` (%)."""

    VERY_HIGH = "Very High"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


#: Lower-bound (inclusive) percentage thresholds for each level. Levels are
#: evaluated from highest to lowest and the first match wins.
_CONFIDENCE_BANDS: Final[tuple[tuple[float, ConfidenceLevel], ...]] = (
    (95.0, ConfidenceLevel.VERY_HIGH),
    (90.0, ConfidenceLevel.HIGH),
    (75.0, ConfidenceLevel.MEDIUM),
    (0.0,  ConfidenceLevel.LOW),
)


# ---------------------------------------------------------------------------
# Reliability
# ---------------------------------------------------------------------------

class Reliability(str, Enum):
    """Whether the prediction can be trusted or is uncertain."""

    RELIABLE = "Reliable"
    UNCERTAIN = "Uncertain"


#: A prediction is RELIABLE when either:
#:   * the top prediction is very strong and clearly separated, or
#:   * it is at least high-confidence, with a healthy margin and low entropy.
MIN_MARGIN_PERCENT: Final[float] = 5.0     # pp
VERY_STRONG_MARGIN_PERCENT: Final[float] = 10.0
MAX_ENTROPY: Final[float] = 1.0            # bits (top-3 normalised)
RELIABLE_MIN_CONFIDENCE_PERCENT: Final[float] = 90.0
VERY_HIGH_CONFIDENCE_PERCENT: Final[float] = 95.0


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

logger: Final[logging.Logger] = setup_logging("ml.inference.confidence_engine")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ConfidenceEngineError(ValueError):
    """Base class for all confidence-engine errors."""


class InvalidProbabilityError(ConfidenceEngineError):
    """Raised when a probability value is non-numeric or outside [0, 1]."""


class ProbabilitiesSumError(ConfidenceEngineError):
    """Raised when the probability vector does not sum to ~1.0."""

    def __init__(self, actual_sum: float) -> None:
        self.actual_sum: float = actual_sum
        super().__init__(
            f"Probabilities must sum to 1.0 (got {actual_sum:.6f})."
        )


class EmptyPredictionError(ConfidenceEngineError):
    """Raised when the prediction input is empty or missing the top list."""


class InvalidPredictionStructureError(ConfidenceEngineError):
    """Raised when the input dictionary does not match the expected shape."""


# ---------------------------------------------------------------------------
# ConfidenceEngine
# ---------------------------------------------------------------------------

class ConfidenceEngine:
    """Interpret the Predictor output and produce a confidence report.

    The class is purely functional: it carries no model, no
    preprocessor, no KB.  It is therefore safe to instantiate at
    module-load time and to reuse across requests in any web
    framework (Streamlit, Flask, FastAPI).

    Example
    -------
    >>> engine = ConfidenceEngine()
    >>> report = engine.evaluate({
    ...     "prediction": "ML Engineer",
    ...     "top3": [
    ...         {"career": "ML Engineer",     "probability": 0.9923},
    ...         {"career": "Data Scientist",  "probability": 0.8734},
    ...         {"career": "Data Engineer",   "probability": 0.7121},
    ...     ],
    ... })
    >>> report["confidence_level"]
    'Very High'
    """

    # ----- public API ------------------------------------------------------

    def evaluate(self, prediction_result: dict[str, Any]) -> dict[str, Any]:
        """Build a structured confidence report for a single prediction.

        Parameters
        ----------
        prediction_result:
            The dictionary returned by ``Predictor.predict()``. Required
            keys: ``"prediction"`` (``str``) and ``"top3"``
            (``list[dict]``) where each entry has ``"career"`` (``str``)
            and ``"probability"`` (``float`` in ``[0, 1]``).

        Returns
        -------
        dict
            ``{
                "predicted_career":      str,
                "confidence_score":      float,   # 0 – 100, 2 dp
                "confidence_level":      str,     # Very High / High / Medium / Low
                "top3":                  list[dict[str, Any]],
                "margin_from_second":    float,   # percentage points, 2 dp
                "entropy":               float,   # bits,           4 dp
                "prediction_reliability":str,     # Reliable / Uncertain
            }``
        """
        self._validate_structure(prediction_result)

        # Use a defensive copy so callers' data is never mutated.
        top3: list[dict[str, Any]] = prediction_result["top3"]
        probabilities: list[float] = [self._coerce_probability(item) for item in top3]

        # ----- scoring ------------------------------------------------------
        confidence_score_pct = probabilities[0] * 100.0
        confidence_level = self._band(confidence_score_pct)

        if len(probabilities) >= 2:
            margin_pct = (probabilities[0] - probabilities[1]) * 100.0
        else:
            # If only one candidate exists, treat the gap as the full
            # probability (no competition) – still finite and useful.
            margin_pct = probabilities[0] * 100.0

        entropy_bits = self._entropy(probabilities)

        reliability = self._reliability(
            confidence_score_pct=confidence_score_pct,
            margin_pct=margin_pct,
            entropy_bits=entropy_bits,
        )

        # ----- assembly -----------------------------------------------------
        formatted_top3 = [
            {
                "career": str(item["career"]),
                "probability": round(float(item["probability"]),
                                    PROBABILITY_DECIMALS),
            }
            for item in top3
        ]

        report: dict[str, Any] = {
            "predicted_career": str(prediction_result["prediction"]),
            "confidence_score": round(confidence_score_pct, PERCENT_DECIMALS),
            "confidence_level": confidence_level.value,
            "top3": formatted_top3,
            "margin_from_second": round(margin_pct, PERCENT_DECIMALS),
            "entropy": round(entropy_bits, PROBABILITY_DECIMALS),
            "prediction_reliability": reliability.value,
        }

        logger.info(
            "Confidence report: career=%s score=%.2f%% level=%s "
            "margin=%.2fpp entropy=%.4fbits reliability=%s",
            report["predicted_career"],
            report["confidence_score"],
            report["confidence_level"],
            report["margin_from_second"],
            report["entropy"],
            report["prediction_reliability"],
        )
        return report

    # ----- internals: validation ------------------------------------------

    @staticmethod
    def _validate_structure(prediction_result: Any) -> None:
        """Ensure the input has the expected dict / list shape."""
        if not isinstance(prediction_result, dict):
            raise InvalidPredictionStructureError(
                f"Expected dict, got {type(prediction_result).__name__}."
            )

        if "prediction" not in prediction_result:
            raise InvalidPredictionStructureError(
                "Missing 'prediction' key."
            )
        if "top3" not in prediction_result:
            raise InvalidPredictionStructureError(
                "Missing 'top3' key."
            )

        top3 = prediction_result["top3"]
        if not isinstance(top3, list) or len(top3) == 0:
            raise EmptyPredictionError(
                "'top3' must be a non-empty list."
            )
        if len(top3) > 3:
            # The engine is lenient with longer lists but only the
            # top-3 are used. This is a soft warning via the logger.
            logger.debug("top3 list contains %d entries; using all of them.",
                         len(top3))

        for idx, item in enumerate(top3):
            if not isinstance(item, dict):
                raise InvalidPredictionStructureError(
                    f"top3[{idx}] must be a dict, got {type(item).__name__}."
                )
            if "career" not in item or "probability" not in item:
                raise InvalidPredictionStructureError(
                    f"top3[{idx}] must contain 'career' and 'probability' keys."
                )

    @staticmethod
    def _coerce_probability(item: dict[str, Any]) -> float:
        """Convert ``probability`` to a finite float."""
        raw = item.get("probability")
        if raw is None:
            raise InvalidProbabilityError(
                f"Probability missing in entry {item!r}."
            )
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise InvalidProbabilityError(
                f"Probability must be numeric, got {raw!r}."
            ) from exc

        if math.isnan(value) or math.isinf(value):
            raise InvalidProbabilityError(
                f"Probability must be finite, got {raw!r}."
            )
        if not (0.0 <= value <= 1.0):
            raise InvalidProbabilityError(
                f"Probability must lie in [0, 1], got {value!r}."
            )
        return value

    @staticmethod
    def _validate_probabilities(probabilities: list[float]) -> None:
        """Reject empty or degenerate probability vectors.

        The engine tolerates partial top-k slices that do not sum to 1.0,
        because Predictor returns only the top ranked careers. Validation is
        limited to guarding against empty or all-zero vectors after the
        individual values have already been checked by ``_coerce_probability``.
        """
        if not probabilities:
            raise EmptyPredictionError("'top3' must contain at least one probability.")
        if sum(probabilities) <= 0.0:
            raise InvalidProbabilityError(
                "At least one probability must be greater than zero."
            )

    # ----- internals: scoring ---------------------------------------------

    @staticmethod
    def _band(percent: float) -> ConfidenceLevel:
        """Return the discrete confidence band for a percentage in [0, 100]."""
        for lower_bound, level in _CONFIDENCE_BANDS:
            if percent >= lower_bound:
                return level
        # Unreachable – the LOW band has a lower bound of 0 – but
        # defensive default keeps the type checker happy.
        return ConfidenceLevel.LOW

    @staticmethod
    def _entropy(probabilities: list[float]) -> float:
        """Shannon entropy (bits) of the probability vector.

        Uniform distributions maximise entropy; confident predictions
        minimise it.  The vector is renormalised across the supplied
        entries so that a 2-entry ``[0.6, 0.4]`` is compared on equal
        footing with a 20-entry uniform.
        """
        total = sum(probabilities)
        if total <= 0.0:
            return 0.0

        normalised = [p / total for p in probabilities]
        entropy = 0.0
        for p in normalised:
            if p > 0.0:
                entropy -= p * math.log2(p)
        return entropy

    @staticmethod
    def _reliability(
        confidence_score_pct: float,
        margin_pct: float,
        entropy_bits: float,
    ) -> Reliability:
        """Apply the reliability rule.

        Reliable ⇔
            (confidence_score_pct >= 95 % AND margin_pct >= 10 pp)
            OR
            (confidence_score_pct >= 90 % AND margin_pct >= 5 pp AND entropy_bits <= 1.0)
        """
        is_reliable = (
            confidence_score_pct >= VERY_HIGH_CONFIDENCE_PERCENT
            and margin_pct >= VERY_STRONG_MARGIN_PERCENT
        ) or (
            confidence_score_pct >= RELIABLE_MIN_CONFIDENCE_PERCENT
            and margin_pct >= MIN_MARGIN_PERCENT
            and entropy_bits <= MAX_ENTROPY
        )
        return Reliability.RELIABLE if is_reliable else Reliability.UNCERTAIN


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

__all__: Final[tuple[str, ...]] = (
    "ConfidenceEngine",
    "ConfidenceLevel",
    "Reliability",
    "ConfidenceEngineError",
    "InvalidProbabilityError",
    "ProbabilitiesSumError",
    "EmptyPredictionError",
    "InvalidPredictionStructureError",
)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(
        "This module is meant to be imported.\n"
        "Example:\n"
        "    from ml.inference.confidence_engine import ConfidenceEngine\n"
        "    print(ConfidenceEngine().evaluate(prediction_dict))"
    )
