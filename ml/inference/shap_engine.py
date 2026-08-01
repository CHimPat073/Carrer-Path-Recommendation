"""CareerPilot-AI SHAP Explainability Engine.

This module provides reusable SHAP-based explanations for the trained
production model. It is intentionally independent from Streamlit,
Flask, FastAPI, recommendation logic, and confidence scoring.

Responsibilities:

* load the persisted model and preprocessor once
* create and reuse a SHAP explainer once
* explain a single prediction from raw user input or a preprocessed vector
* compute global feature importance summaries
* save SHAP visualizations to disk without displaying them

The engine only explains model output. It does not preprocess business
logic beyond applying the persisted transformer, does not retrain, and
does not access any knowledge base.
"""

from __future__ import annotations

import logging
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Iterable

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap  # type: ignore[import-not-found]

from ml.config import setup_logging
from ml.inference.predictor import (
    BEST_MODEL_PATH,
    PREPROCESSOR_COLUMNS,
    PREPROCESSOR_PATH,
    TARGET_ENCODER_PATH,
)


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
PLOTS_DIR: Final[Path] = PROJECT_ROOT / "plots"
DISPLAY_NAMES_PATH: Final[Path] = PROJECT_ROOT / "knowledge_base" / "display_names.json"
SUMMARY_PLOT_PATH: Final[Path] = PLOTS_DIR / "shap_summary.png"
BAR_PLOT_PATH: Final[Path] = PLOTS_DIR / "shap_bar.png"
WATERFALL_PLOT_PATH: Final[Path] = PLOTS_DIR / "shap_waterfall.png"
BEESWARM_PLOT_PATH: Final[Path] = PLOTS_DIR / "shap_beeswarm.png"
TOP_FEATURES: Final[int] = 10
TOP_LOCAL_FEATURES: Final[int] = 5

logger: Final[logging.Logger] = setup_logging("ml.inference.shap_engine")


class SHAPEngineError(ValueError):
    """Base class for SHAP engine failures."""


class MissingModelError(SHAPEngineError):
    """Raised when the persisted model cannot be found."""


class MissingPreprocessorError(SHAPEngineError):
    """Raised when the persisted preprocessor cannot be found."""


class UnsupportedModelError(SHAPEngineError):
    """Raised when the model type is not supported by the engine."""


class InvalidInputSchemaError(SHAPEngineError):
    """Raised when the explain input does not match the supported schema."""


@dataclass(frozen=True)
class _PreparedInput:
    frame: pd.DataFrame
    raw_values: dict[str, Any]
    prediction: str | None = None


class SHAPEngine:
    """Reusable SHAP explanation service for the production model."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        preprocessor_path: Path | str | None = None,
        plots_dir: Path | str | None = None,
    ) -> None:
        self.model_path = Path(model_path) if model_path is not None else BEST_MODEL_PATH
        self.preprocessor_path = (
            Path(preprocessor_path) if preprocessor_path is not None else PREPROCESSOR_PATH
        )
        self.plots_dir = Path(plots_dir) if plots_dir is not None else PLOTS_DIR

        self._model: Any | None = None
        self._preprocessor: Any | None = None
        self._target_encoder: Any | None = None
        self._explainer: Any | None = None
        self._feature_names: list[str] | None = None
        self._raw_feature_names: list[str] | None = None
        self._display_names_by_original: dict[str, str] = {}
        self._background_matrix: np.ndarray | None = None

        self._load_artifacts()
        self._build_explainer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(
        self,
        user_input: dict[str, Any] | None = None,
        *,
        processed_features: Any | None = None,
        prediction_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Explain one prediction using raw input or a preprocessed vector."""
        prepared = self._prepare_input(
            user_input=user_input,
            processed_features=processed_features,
            prediction_result=prediction_result,
        )

        transformed = self._transform_frame(prepared.frame) if processed_features is None else self._coerce_processed(processed_features)
        shap_values = self._get_shap_values(transformed)
        predicted_class_index = self._resolve_prediction_index(prepared.prediction, transformed)
        class_shap = self._select_class_values(shap_values, predicted_class_index)
        raw_feature_shap = self._aggregate_to_raw_features(class_shap, prepared.frame.iloc[0].to_dict())

        top_positive = self._top_contributors(raw_feature_shap, prepared.frame.iloc[0].to_dict(), positive=True)
        top_negative = self._top_contributors(raw_feature_shap, prepared.frame.iloc[0].to_dict(), positive=False)

        explanation = {
            "prediction": prepared.prediction or self._predict_label(transformed),
            "top_positive_features": top_positive,
            "top_negative_features": top_negative,
        }

        self._save_plots(raw_feature_shap, prepared.frame.iloc[0].to_dict(), predicted_class_index)
        return explanation

    def global_feature_importance(self, data: Any | None = None, *, max_samples: int = 200) -> list[dict[str, Any]]:
        """Return mean absolute SHAP values aggregated by raw feature."""
        matrix = self._background_matrix if data is None else self._coerce_processed(data)
        if data is not None and getattr(matrix, "shape", (0,))[0] > max_samples:
            matrix = matrix[:max_samples]

        shap_values = self._get_shap_values(matrix)
        class_index = self._default_class_index()
        class_shap = self._select_class_values(shap_values, class_index)
        return self._mean_absolute_shap_values(class_shap)

    def shap_summary_values(self, data: Any | None = None) -> list[dict[str, Any]]:
        """Return per-feature SHAP summary values for the default class."""
        matrix = self._background_matrix if data is None else self._coerce_processed(data)
        shap_values = self._get_shap_values(matrix)
        class_shap = self._select_class_values(shap_values, self._default_class_index())
        grouped: dict[str, dict[str, Any]] = {}
        for feature_name, value in zip(self._raw_feature_names or [], class_shap, strict=False):
            if feature_name not in grouped:
                grouped[feature_name] = {
                    "feature": self._display_feature_name(feature_name),
                    "display_feature_name": self._display_feature_name(feature_name),
                    "original_feature_name": feature_name,
                    "mean_shap_value": 0.0,
                    "mean_abs_shap_values": [],
                }
            grouped[feature_name]["mean_shap_value"] += float(value)
            grouped[feature_name]["mean_abs_shap_values"].append(abs(float(value)))

        summary: list[dict[str, Any]] = []
        for item in grouped.values():
            summary.append(
                {
                    "feature": item["feature"],
                    "display_feature_name": item["display_feature_name"],
                    "original_feature_name": item["original_feature_name"],
                    "mean_shap_value": round(float(item["mean_shap_value"]), 6),
                    "mean_abs_shap_value": round(float(np.mean(item["mean_abs_shap_values"])), 6),
                }
            )
        summary.sort(key=lambda item: item["mean_abs_shap_value"], reverse=True)
        return summary

    def mean_absolute_shap_values(self, data: Any | None = None) -> list[dict[str, Any]]:
        """Alias for global importance output."""
        return self.global_feature_importance(data=data)

    def explain_global(self, data: Any | None = None) -> dict[str, Any]:
        """Generate the saved global SHAP plots and return summary values."""
        matrix = self._background_matrix if data is None else self._coerce_processed(data)
        shap_values = self._get_shap_values(matrix)
        class_index = self._default_class_index()
        class_shap = self._select_class_values(shap_values, class_index)
        aggregated = self._aggregate_to_raw_features(class_shap, self._background_feature_row())
        self._save_global_plots(aggregated)
        return {
            "mean_absolute_shap_values": self._mean_absolute_shap_values(class_shap),
            "shap_summary_values": self.shap_summary_values(matrix),
        }

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    def _prepare_input(
        self,
        *,
        user_input: dict[str, Any] | None,
        processed_features: Any | None,
        prediction_result: dict[str, Any] | None,
    ) -> _PreparedInput:
        if user_input is not None and processed_features is not None:
            raise InvalidInputSchemaError("Provide either user_input or processed_features, not both.")

        if user_input is None and processed_features is None:
            raise InvalidInputSchemaError("Either user_input or processed_features must be provided.")

        if user_input is not None:
            frame = self._validate_and_build_frame(user_input)
            return _PreparedInput(
                frame=frame,
                raw_values=frame.iloc[0].to_dict(),
                prediction=self._extract_prediction(prediction_result),
            )

        raw_values = self._prediction_result_to_raw_values(prediction_result)
        frame = self._validate_and_build_frame(raw_values)
        return _PreparedInput(
            frame=frame,
            raw_values=raw_values,
            prediction=self._extract_prediction(prediction_result),
        )

    def _validate_and_build_frame(self, user_input: dict[str, Any]) -> pd.DataFrame:
        if not isinstance(user_input, dict):
            raise InvalidInputSchemaError(f"Expected dict input, got {type(user_input).__name__}.")

        missing = [column for column in PREPROCESSOR_COLUMNS if column not in user_input]
        if missing:
            raise InvalidInputSchemaError(f"Missing required fields: {missing}")

        ordered = {column: [user_input[column]] for column in PREPROCESSOR_COLUMNS}
        return pd.DataFrame(ordered)

    def _prediction_result_to_raw_values(self, prediction_result: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(prediction_result, dict):
            raise InvalidInputSchemaError("prediction_result must be a dict when processed_features is used.")
        raw_values = prediction_result.get("raw_input") or prediction_result.get("user_input")
        if not isinstance(raw_values, dict):
            raise InvalidInputSchemaError(
                "processed_features requires prediction_result to include raw_input or user_input."
            )
        return raw_values

    @staticmethod
    def _extract_prediction(prediction_result: dict[str, Any] | None) -> str | None:
        if not isinstance(prediction_result, dict):
            return None
        prediction = prediction_result.get("prediction")
        return str(prediction) if prediction is not None else None

    # ------------------------------------------------------------------
    # Loading / explainer setup
    # ------------------------------------------------------------------

    def _load_artifacts(self) -> None:
        if not self.model_path.is_file():
            raise MissingModelError(f"Trained model not found at: {self.model_path}")
        if not self.preprocessor_path.is_file():
            raise MissingPreprocessorError(f"Preprocessor not found at: {self.preprocessor_path}")
        if not TARGET_ENCODER_PATH.is_file():
            raise MissingPreprocessorError(f"Target encoder not found at: {TARGET_ENCODER_PATH}")

        self._model = joblib.load(self.model_path)
        self._preprocessor = joblib.load(self.preprocessor_path)
        self._target_encoder = joblib.load(TARGET_ENCODER_PATH)

        if not isinstance(self._model, (shap.TreeExplainer,)) and not hasattr(self._model, "predict_proba"):
            raise UnsupportedModelError(
                f"Unsupported model type: {type(self._model).__name__}. Expected a probability-enabled tree model."
            )

        if not hasattr(self._preprocessor, "transform"):
            raise UnsupportedModelError(
                f"Unsupported preprocessor type: {type(self._preprocessor).__name__}."
            )

        self._feature_names = list(self._preprocessor.get_feature_names_out())
        self._raw_feature_names = self._build_raw_feature_names(self._feature_names)
        self._display_names_by_original = self._load_display_names()
        self._background_matrix = self._transform_frame(self._baseline_frame())

    def _build_explainer(self) -> None:
        try:
            self._explainer = shap.TreeExplainer(self._model)
        except Exception as exc:  # pragma: no cover - defensive, depends on SHAP internals
            raise UnsupportedModelError(
                f"Failed to construct SHAP explainer for {type(self._model).__name__}."
            ) from exc

    def _baseline_frame(self) -> pd.DataFrame:
        baseline: dict[str, Any] = {}
        for column in PREPROCESSOR_COLUMNS:
            if column.endswith("_score") or column in {"years_experience", "projects_completed", "certifications", "career_growth_score", "job_satisfaction", "salary_band", "work_hours_per_week"}:
                baseline[column] = 0
            elif column == "education_level":
                baseline[column] = "Bachelor"
            elif column == "remote_preference":
                baseline[column] = "Hybrid"
            elif column == "country":
                baseline[column] = "USA"
            elif column == "industry":
                baseline[column] = "Technology"
            elif column == "employment_type":
                baseline[column] = "Full-time"
            else:
                baseline[column] = 0
        return pd.DataFrame([baseline])

    # ------------------------------------------------------------------
    # Prediction / SHAP helpers
    # ------------------------------------------------------------------

    def _transform_frame(self, frame: pd.DataFrame) -> np.ndarray:
        transformed = self._preprocessor.transform(frame)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()
        return np.asarray(transformed)

    def _coerce_processed(self, processed_features: Any) -> np.ndarray:
        matrix = np.asarray(processed_features)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2:
            raise InvalidInputSchemaError(
                f"processed_features must be a 2D matrix, got shape {matrix.shape}."
            )
        return matrix

    def _get_shap_values(self, matrix: np.ndarray) -> np.ndarray:
        values = self._explainer.shap_values(matrix)
        array = np.asarray(values)
        if array.ndim == 2:
            array = array[:, :, np.newaxis]
        if array.ndim != 3:
            raise UnsupportedModelError(
                f"Unexpected SHAP output shape {array.shape} for {type(self._model).__name__}."
            )
        return array

    def _default_class_index(self) -> int:
        classes = getattr(self._model, "classes_", None)
        if classes is None:
            return 0
        try:
            return list(classes).index("ML Engineer")
        except ValueError:
            return 0

    def _resolve_prediction_index(self, prediction: str | None, matrix: np.ndarray) -> int:
        classes = list(getattr(self._model, "classes_", []))
        if prediction and prediction in classes:
            return classes.index(prediction)
        predicted = self._predict_label(matrix)
        if predicted in classes:
            return classes.index(predicted)
        return self._default_class_index()

    def _predict_label(self, matrix: np.ndarray) -> str:
        feature_frame = pd.DataFrame(matrix, columns=self._feature_names)
        prediction = self._model.predict(feature_frame)[0]
        if self._target_encoder is not None:
            try:
                return str(self._target_encoder.classes_[int(prediction)])
            except (TypeError, ValueError, IndexError):
                pass
        return str(prediction)

    def _select_class_values(self, shap_values: np.ndarray, class_index: int) -> np.ndarray:
        if class_index < 0 or class_index >= shap_values.shape[2]:
            class_index = self._default_class_index()
        return shap_values[0, :, class_index]

    def _aggregate_to_raw_features(self, class_shap: np.ndarray, raw_row: dict[str, Any]) -> list[dict[str, Any]]:
        grouped: dict[str, dict[str, Any]] = {}
        for transformed_name, raw_name, shap_value in zip(self._feature_names or [], self._raw_feature_names or [], class_shap, strict=False):
            if raw_name not in grouped:
                grouped[raw_name] = {
                    "feature": self._display_feature_name(raw_name),
                    "display_feature_name": self._display_feature_name(raw_name),
                    "original_feature_name": raw_name,
                    "processed_feature_name": transformed_name,
                    "value": raw_row.get(raw_name),
                    "shap_value": 0.0,
                    "processed_feature_contributions": [],
                }
            grouped[raw_name]["shap_value"] += float(shap_value)
            grouped[raw_name]["processed_feature_contributions"].append(
                {
                    "processed_feature_name": transformed_name,
                    "original_feature_name": raw_name,
                    "display_feature_name": self._display_feature_name(raw_name),
                    "shap_value": round(float(shap_value), 4),
                }
            )

        aggregated: list[dict[str, Any]] = []
        for item in grouped.values():
            dominant = max(
                item["processed_feature_contributions"],
                key=lambda record: abs(record["shap_value"]),
                default={"processed_feature_name": item["processed_feature_name"]},
            )
            aggregated.append(
                {
                    "feature": item["display_feature_name"],
                    "display_feature_name": item["display_feature_name"],
                    "original_feature_name": item["original_feature_name"],
                    "processed_feature_name": dominant["processed_feature_name"],
                    "value": item["value"],
                    "shap_value": round(float(item["shap_value"]), 4),
                    "processed_feature_contributions": item["processed_feature_contributions"],
                }
            )
        return aggregated

    def _top_contributors(self, aggregated: list[dict[str, Any]], raw_row: dict[str, Any], *, positive: bool) -> list[dict[str, Any]]:
        filtered = [item for item in aggregated if (item["shap_value"] >= 0) == positive]
        ordered = sorted(filtered, key=lambda item: item["shap_value"], reverse=positive)
        return [
            {
                "feature": item["feature"],
                "display_feature_name": item["display_feature_name"],
                "value": item["value"],
                "original_feature_name": item["original_feature_name"],
                "processed_feature_name": item["processed_feature_name"],
                "shap_value": round(float(item["shap_value"]), 4),
            }
            for item in ordered[:TOP_LOCAL_FEATURES]
        ]

    def _mean_absolute_shap_values(self, class_shap: np.ndarray) -> list[dict[str, Any]]:
        grouped: dict[str, list[float]] = defaultdict(list)
        for raw_name, shap_value in zip(self._raw_feature_names or [], class_shap, strict=False):
            grouped[raw_name].append(abs(float(shap_value)))

        summary = [
            {
                "feature": self._display_feature_name(feature_name),
                "display_feature_name": self._display_feature_name(feature_name),
                "original_feature_name": feature_name,
                "mean_abs_shap_value": round(float(np.mean(values)), 6),
            }
            for feature_name, values in grouped.items()
        ]
        summary.sort(key=lambda item: item["mean_abs_shap_value"], reverse=True)
        return summary[:TOP_FEATURES]

    def _build_raw_feature_names(self, transformed_names: Iterable[str]) -> list[str]:
        raw_names: list[str] = []
        for name in transformed_names:
            if "__" not in name:
                raw_names.append(name)
                continue
            prefix, remainder = name.split("__", 1)
            if prefix == "nominal":
                raw_names.append(remainder.rsplit("_", 1)[0])
            elif prefix in {"numerical", "ordinal"}:
                raw_names.append(remainder)
            else:
                raw_names.append(remainder)
        return raw_names

    def _load_display_names(self) -> dict[str, str]:
        if not DISPLAY_NAMES_PATH.is_file():
            return {}
        try:
            with DISPLAY_NAMES_PATH.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception as exc:  # pragma: no cover - defensive file read
            raise InvalidInputSchemaError(
                f"Unable to load display-name mapping from {DISPLAY_NAMES_PATH}."
            ) from exc
        if not isinstance(data, dict):
            raise InvalidInputSchemaError(
                f"Display-name mapping must be a JSON object: {DISPLAY_NAMES_PATH}."
            )
        return {str(key): str(value) for key, value in data.items()}

    def _display_feature_name(self, original_name: str) -> str:
        return self._display_names_by_original.get(original_name, original_name.replace("_", " ").title())

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _save_plots(self, aggregated_features: list[dict[str, Any]], raw_row: dict[str, Any], class_index: int) -> None:
        self._save_global_plots(aggregated_features)
        self._save_waterfall_plot(aggregated_features, raw_row, class_index)

    def _save_global_plots(self, aggregated_features: list[dict[str, Any]]) -> None:
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        feature_names = [item["display_feature_name"] for item in aggregated_features]
        values = np.array([item["shap_value"] for item in aggregated_features], dtype=float)

        shap.summary_plot(values.reshape(1, -1), feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(self.plots_dir / "shap_summary.png", bbox_inches="tight", dpi=150)
        plt.close()

        shap.summary_plot(values.reshape(1, -1), feature_names=feature_names, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(self.plots_dir / "shap_bar.png", bbox_inches="tight", dpi=150)
        plt.close()

        shap.summary_plot(values.reshape(1, -1), feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(self.plots_dir / "shap_beeswarm.png", bbox_inches="tight", dpi=150)
        plt.close()

    def _save_waterfall_plot(self, aggregated_features: list[dict[str, Any]], raw_row: dict[str, Any], class_index: int) -> None:
        base_values = getattr(self._explainer, "expected_value", 0.0)
        if isinstance(base_values, (list, tuple, np.ndarray)):
            base_values = float(np.asarray(base_values)[class_index])
        feature_names = [item["display_feature_name"] for item in aggregated_features]
        values = np.array([item["shap_value"] for item in aggregated_features], dtype=float)
        explanation = shap.Explanation(
            values=values,
            base_values=base_values,
            data=np.array([raw_row.get(name) for name in feature_names], dtype=object),
            feature_names=feature_names,
        )
        shap.plots.waterfall(explanation, max_display=20, show=False)
        plt.tight_layout()
        plt.savefig(self.plots_dir / "shap_waterfall.png", bbox_inches="tight", dpi=150)
        plt.close()

    def _background_feature_row(self) -> dict[str, Any]:
        baseline = self._baseline_frame().iloc[0]
        return {column: baseline[column] for column in PREPROCESSOR_COLUMNS}


__all__: Final[tuple[str, ...]] = (
    "SHAPEngine",
    "SHAPEngineError",
    "MissingModelError",
    "MissingPreprocessorError",
    "UnsupportedModelError",
    "InvalidInputSchemaError",
)
