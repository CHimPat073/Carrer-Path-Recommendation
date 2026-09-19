"""
CareerPilot-AI – Prediction Engine
==================================

A standalone, UI-agnostic inference module that:

    1. Loads the persisted best-performing model.
    2. Loads the persisted preprocessing pipeline.
    3. Loads the persisted target (label) encoder.
    4. Accepts a raw user input dictionary that follows the
       original *pre-preprocessing* dataset schema.
    5. Validates the input.
    6. Runs preprocessing.
    7. Predicts the encoded career label and decodes it.
    8. Computes class probabilities, sorts them and returns
       the Top-3 careers with their probabilities rounded to
       four decimal places.

This module **only** predicts.  It contains no UI, no SHAP
code, no recommendation logic, no knowledge-base lookup and no
career explanation.  It is reusable from Streamlit, Flask,
FastAPI, plain scripts or any REST/RPC layer without code
changes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Final

import joblib
import pandas as pd

from ml.config import setup_logging


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Project root resolved relative to this file.
#: ``predictor.py`` lives at ``ml/inference/predictor.py``.
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

#: Directory holding the trained model artefacts.
MODELS_DIR: Final[Path] = PROJECT_ROOT / "models"

#: Path of the production-ready trained model.
BEST_MODEL_PATH: Final[Path] = MODELS_DIR / "best_model.pkl"

#: Path of the fitted preprocessing pipeline.
PREPROCESSOR_PATH: Final[Path] = MODELS_DIR / "preprocessor.joblib"

#: Path of the fitted target encoder.
TARGET_ENCODER_PATH: Final[Path] = MODELS_DIR / "target_encoder.joblib"

#: Number of candidates returned to the caller.
TOP_K: Final[int] = 3

#: Probability rounding precision.
PROB_DECIMALS: Final[int] = 4


#: Exact column order expected by the fitted preprocessor.
#: This order matches ``feature_names_in_`` of the persisted
#: ``ColumnTransformer`` and matches the original dataset
#: schema before preprocessing.
PREPROCESSOR_COLUMNS: Final[tuple[str, ...]] = (
    # --- base numeric -------------------------------------------------------
    "years_experience",
    "projects_completed",
    "certifications",
    # --- skill scores -------------------------------------------------------
    "python_score",
    "java_score",
    "javascript_score",
    "sql_score",
    "machine_learning_score",
    "deep_learning_score",
    "cloud_score",
    "devops_score",
    "cybersecurity_score",
    "data_analysis_score",
    "database_score",
    "networking_score",
    "mobile_score",
    "game_dev_score",
    "testing_score",
    "business_analysis_score",
    "product_management_score",
    "ui_design_score",
    "ux_research_score",
    "communication_score",
    "leadership_score",
    "problem_solving_score",
    "teamwork_score",
    "agile_score",
    "research_score",
    "salary_band",
    "career_growth_score",
    "job_satisfaction",
    "work_hours_per_week",
    # --- categorical --------------------------------------------------------
    "education_level",        # ordinal
    "remote_preference",     # nominal
    "country",               # nominal
    "industry",              # nominal
    "employment_type",       # nominal
)


#: Valid scalar ranges for numeric columns.
#: ``None`` means *no upper bound* or *no lower bound*.
NUMERIC_RANGES: Final[dict[str, tuple[float | None, float | None]]] = {
    "years_experience":    (0.0, 60.0),
    "projects_completed":  (0.0, 500.0),
    "certifications":      (0.0, 50.0),
    "python_score":        (0.0, 10.0),
    "java_score":          (0.0, 10.0),
    "javascript_score":    (0.0, 10.0),
    "sql_score":           (0.0, 10.0),
    "machine_learning_score":     (0.0, 10.0),
    "deep_learning_score":        (0.0, 10.0),
    "cloud_score":                (0.0, 10.0),
    "devops_score":               (0.0, 10.0),
    "cybersecurity_score":        (0.0, 10.0),
    "data_analysis_score":        (0.0, 10.0),
    "database_score":             (0.0, 10.0),
    "networking_score":           (0.0, 10.0),
    "mobile_score":               (0.0, 10.0),
    "game_dev_score":             (0.0, 10.0),
    "testing_score":              (0.0, 10.0),
    "business_analysis_score":    (0.0, 10.0),
    "product_management_score":   (0.0, 10.0),
    "ui_design_score":            (0.0, 10.0),
    "ux_research_score":          (0.0, 10.0),
    "communication_score":        (0.0, 10.0),
    "leadership_score":           (0.0, 10.0),
    "problem_solving_score":      (0.0, 10.0),
    "teamwork_score":             (0.0, 10.0),
    "agile_score":                (0.0, 10.0),
    "research_score":             (0.0, 10.0),
    "career_growth_score":        (0.0, 10.0),
    "job_satisfaction":           (0.0, 10.0),
    "salary_band":                (0.0, None),
    "work_hours_per_week":        (0.0, 80.0),
}

#: Valid categorical levels for each categorical column.
CATEGORICAL_VALUES: Final[dict[str, tuple[str, ...]]] = {
    "education_level": (
        "High School", "Associate", "Bachelor", "Master", "PhD",
    ),
    "remote_preference": ("Hybrid", "On-site", "Remote"),
    "country": (
        "Australia", "Canada", "Germany", "India", "UK", "USA",
    ),
    "industry": (
        "Consulting", "Education", "Finance", "Gaming",
        "Healthcare", "Retail", "Technology",
    ),
    "employment_type": ("Contract", "Freelance", "Full-time", "Part-time"),
}


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

logger: Final[logging.Logger] = setup_logging("ml.inference.predictor")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class PredictorError(ValueError):
    """Base class for all predictor-level errors."""


class MissingFieldError(PredictorError):
    """Raised when one or more required fields are absent."""

    def __init__(self, missing: list[str]) -> None:
        self.missing: list[str] = missing
        super().__init__(
            "Missing required input field(s): " + ", ".join(missing)
        )


class InvalidValueError(PredictorError):
    """Raised when a field has an unsupported value or out-of-range value."""

    def __init__(self, field: str, message: str) -> None:
        self.field: str = field
        super().__init__(f"Invalid value for '{field}': {message}")


class UnknownCategoryError(InvalidValueError):
    """Raised when a categorical field has an unsupported label."""


class OutOfRangeError(InvalidValueError):
    """Raised when a numeric field is outside its permitted range."""


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class Predictor:
    """Standalone, framework-agnostic career prediction engine.

    The class is responsible **only** for:

        * loading artefacts,
        * validating & preprocessing raw user input,
        * returning the predicted career and the Top-3 ranked
          candidates together with their probabilities.

    It does not handle UI concerns, explanations, recommendations
    or any post-prediction analytics.

    Example
    -------
    >>> predictor = Predictor()
    >>> result = predictor.predict({
    ...     "years_experience": 3,
    ...     "education_level": "Bachelor",
    ...     "python_score": 9,
    ...     "java_score": 4,
    ...     # ...remaining fields...
    ... })
    >>> result["prediction"]
    'ML Engineer'
    """

    # ----- ctor ------------------------------------------------------------

    def __init__(
        self,
        model_path: Path | str | None = None,
        preprocessor_path: Path | str | None = None,
        target_encoder_path: Path | str | None = None,
    ) -> None:
        # Allow explicit overrides (mainly useful for testing),
        # otherwise fall back to the project defaults resolved
        # via pathlib so no absolute paths are hard-coded here.
        self.model_path: Path = (
            Path(model_path) if model_path is not None else BEST_MODEL_PATH
        )
        self.preprocessor_path: Path = (
            Path(preprocessor_path) if preprocessor_path is not None
            else PREPROCESSOR_PATH
        )
        self.target_encoder_path: Path = (
            Path(target_encoder_path) if target_encoder_path is not None
            else TARGET_ENCODER_PATH
        )

        self._model: Any | None = None
        self._preprocessor: Any | None = None
        self._target_encoder: Any | None = None

        self._load_artifacts()

    # ----- artefact loading ----------------------------------------------

    def _load_artifacts(self) -> None:
        """Load the model, preprocessor and target encoder from disk."""
        logger.info(
            "Loading predictor artefacts (model=%s, preprocessor=%s, "
            "target_encoder=%s)",
            self.model_path.name,
            self.preprocessor_path.name,
            self.target_encoder_path.name,
        )

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Trained model not found at: {self.model_path}"
            )
        if not self.preprocessor_path.is_file():
            raise FileNotFoundError(
                f"Preprocessor not found at: {self.preprocessor_path}"
            )
        if not self.target_encoder_path.is_file():
            raise FileNotFoundError(
                f"Target encoder not found at: {self.target_encoder_path}"
            )

        self._model = joblib.load(self.model_path)
        self._preprocessor = joblib.load(self.preprocessor_path)
        self._target_encoder = joblib.load(self.target_encoder_path)

        logger.info("Predictor artefacts loaded successfully.")

    # ----- public API ------------------------------------------------------

    def predict(self, user_input: dict[str, Any]) -> dict[str, Any]:
        """Run the full inference pipeline for a single user profile.

        Parameters
        ----------
        user_input:
            Raw user input matching the *original* dataset schema
            (i.e. the schema used **before** preprocessing).

        Returns
        -------
        dict
            ``{"prediction": str, "top3": list[dict[str, Any]]}``
            where each entry of ``top3`` has ``career`` and
            ``probability`` keys (probabilities rounded to four
            decimal places).

        Raises
        ------
        MissingFieldError
            If one or more required fields are absent.
        UnknownCategoryError
            If a categorical field contains an unsupported label.
        OutOfRangeError
            If a numeric field is outside its permitted range.
        InvalidValueError
            If a field cannot be coerced to its expected dtype.
        """
        logger.debug("Validating user input…")
        self._validate_input(user_input)

        logger.debug("Converting dict → DataFrame…")
        df = self._to_dataframe(user_input)

        logger.debug("Applying preprocessor…")
        features = self._preprocessor.transform(df)

        # scikit-learn estimators always expect 2-D input.
        if hasattr(features, "toarray"):
            features = features.toarray()

        logger.debug("Running model inference…")
        encoded_label = int(self._model.predict(features)[0])
        probabilities = self._model.predict_proba(features)[0]

        classes = list(self._target_encoder.classes_)
        if len(classes) != len(probabilities):
            raise PredictorError(
                "Class/probability length mismatch: "
                f"{len(classes)} classes vs {len(probabilities)} probabilities."
            )

        # Pair each class label with its probability and sort
        # in descending order of probability.
        sorted_pairs = sorted(
            zip(classes, probabilities),
            key=lambda pair: pair[1],
            reverse=True,
        )

        top_k = sorted_pairs[:TOP_K]
        predicted_career = classes[encoded_label]

        result: dict[str, Any] = {
            "prediction": predicted_career,
            "top3": [
                {
                    "career": career,
                    "probability": round(float(prob), PROB_DECIMALS),
                }
                for career, prob in top_k
            ],
        }

        logger.info(
            "Prediction: %s (confidence=%.4f)",
            result["prediction"],
            result["top3"][0]["probability"],
        )
        return result

    # ----- helpers ---------------------------------------------------------

    def _validate_input(self, user_input: dict[str, Any]) -> None:
        """Validate required presence, ranges and categorical values."""
        if not isinstance(user_input, dict):
            raise InvalidValueError(
                "user_input",
                f"Expected dict, got {type(user_input).__name__}.",
            )

        missing = [c for c in PREPROCESSOR_COLUMNS if c not in user_input]
        if missing:
            raise MissingFieldError(missing)

        for column in PREPROCESSOR_COLUMNS:
            value = user_input[column]

            if column in NUMERIC_RANGES:
                self._validate_numeric(column, value)
            elif column in CATEGORICAL_VALUES:
                self._validate_category(column, value)
            else:  # pragma: no cover - schema drift safeguard
                raise InvalidValueError(
                    column,
                    "Column is neither numeric nor categorical in the schema.",
                )

    @staticmethod
    def _validate_numeric(column: str, value: Any) -> None:
        """Coerce and range-check a numeric column."""
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise InvalidValueError(
                column, f"cannot be converted to float: {value!r}",
            ) from exc

        if numeric_value != numeric_value:  # NaN guard
            raise OutOfRangeError(column, f"value is NaN.")

        lower, upper = NUMERIC_RANGES[column]
        if lower is not None and numeric_value < lower:
            raise OutOfRangeError(
                column,
                f"{numeric_value} is below the minimum allowed value "
                f"of {lower}.",
            )
        if upper is not None and numeric_value > upper:
            raise OutOfRangeError(
                column,
                f"{numeric_value} is above the maximum allowed value "
                f"of {upper}.",
            )

    @staticmethod
    def _validate_category(column: str, value: Any) -> None:
        """Check that a categorical value is in the allowed set."""
        if not isinstance(value, str):
            raise UnknownCategoryError(
                column,
                f"expected string, got {type(value).__name__} "
                f"with value {value!r}.",
            )

        allowed = CATEGORICAL_VALUES[column]
        if value not in allowed:
            raise UnknownCategoryError(
                column,
                f"{value!r} not in allowed values "
                f"{list(allowed)}.",
            )

    def _to_dataframe(self, user_input: dict[str, Any]) -> pd.DataFrame:
        """Build a single-row DataFrame with the expected column order."""
        ordered: dict[str, Any] = {
            col: [user_input[col]] for col in PREPROCESSOR_COLUMNS
        }
        return pd.DataFrame(ordered)


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

__all__: Final[tuple[str, ...]] = (
    "Predictor",
    "PredictorError",
    "MissingFieldError",
    "InvalidValueError",
    "UnknownCategoryError",
    "OutOfRangeError",
)


if __name__ == "__main__:":  # pragma: no cover
    raise SystemExit(
        "This module is meant to be imported. "
        "Example:\n"
        "    from ml.inference.predictor import Predictor\n"
        "    predictor = Predictor()\n"
        "    print(predictor.predict({...}))"
    )
