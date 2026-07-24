"""
Random Forest Evaluation Module
Comprehensive evaluation of trained Random Forest model.
"""

import json
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving images
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ml import config

# Configure logging
logger: logging.Logger = logging.getLogger(__name__)


# =============================================================================
# Evaluation Configuration
# =============================================================================
class EvaluationConfig:
    """Configuration for model evaluation."""

    # Output filenames
    METRICS_FILENAME: str = "evaluation_metrics.json"
    CLASSIFICATION_REPORT_FILENAME: str = "classification_report.txt"
    CONFUSION_MATRIX_FILENAME: str = "confusion_matrix.png"

    # Plot settings
    FIGURE_SIZE: tuple[int, int] = (20, 16)
    DPI: int = 150
    FONT_SIZE: int = 8


# =============================================================================
# Evaluation Result
# =============================================================================
class EvaluationResult:
    """Results from model evaluation."""

    def __init__(
        self,
        accuracy: float,
        precision: float,
        recall: float,
        f1_score: float,
        classification_report: str,
        confusion_matrix: np.ndarray,
        class_names: list[str],
        predictions: np.ndarray,
        probabilities: np.ndarray | None = None,
    ) -> None:
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1_score = f1_score
        self.classification_report = classification_report
        self.confusion_matrix = confusion_matrix
        self.class_names = class_names
        self.predictions = predictions
        self.probabilities = probabilities
        self.timestamp: str = datetime.now().isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "metrics": {
                "accuracy": float(self.accuracy),
                "precision": float(self.precision),
                "recall": float(self.recall),
                "f1_score": float(self.f1_score),
            },
            "num_classes": len(self.class_names),
            "class_names": self.class_names,
            "confusion_matrix": self.confusion_matrix.tolist(),
        }

    def save_metrics(self, output_path: Path) -> Path:
        """Save evaluation metrics to JSON."""
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        return output_path

    def save_classification_report(self, output_path: Path) -> Path:
        """Save classification report to text file."""
        with open(output_path, "w") as f:
            f.write(self.classification_report)
        return output_path


# =============================================================================
# Model Evaluator
# =============================================================================
class RandomForestEvaluator:
    """
    Evaluate a trained Random Forest model.

    Computes comprehensive metrics and visualizations.

    Example:
        >>> evaluator = RandomForestEvaluator()
        >>> result = evaluator.evaluate()
        >>> print(f"Accuracy: {result.accuracy:.4f}")
    """

    def __init__(
        self,
        model_path: Path | None = None,
        data_path: Path | None = None,
    ) -> None:
        """
        Initialize evaluator.

        Args:
            model_path: Path to trained model (uses default if None)
            data_path: Path to dataset (uses default if None)
        """
        self.model_path = model_path or (config.MODELS_DIR / "random_forest_model.pkl")
        self.data_path = data_path or config.DATASET_CONFIG.synthetic_v2
        self.model: Any = None
        self.label_encoder: Any = None
        self.feature_names: list[str] = []
        self._logger = logging.getLogger(__name__)

    def evaluate(
        self,
        save_results: bool = True,
    ) -> EvaluationResult:
        """
        Evaluate the model on test set.

        Args:
            save_results: Whether to save results to disk

        Returns:
            EvaluationResult with all metrics
        """
        self._logger.info("=" * 60)
        self._logger.info("Random Forest Evaluation Pipeline")
        self._logger.info("=" * 60)

        # =========================================================================
        # Step 1: Load model and data
        # =========================================================================
        self._logger.info("\n[STEP 1] Loading model and data...")
        self._load_model()
        self._load_test_data()

        # =========================================================================
        # Step 2: Make predictions
        # =========================================================================
        self._logger.info("\n[STEP 2] Making predictions on test set...")
        predictions = self.model.predict(self.X_test)

        # Get probabilities if available
        probabilities = None
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(self.X_test)

        # =========================================================================
        # Step 3: Calculate metrics
        # =========================================================================
        self._logger.info("\n[STEP 3] Calculating metrics...")

        accuracy = accuracy_score(self.y_test, predictions)
        precision = precision_score(self.y_test, predictions, average="weighted")
        recall = recall_score(self.y_test, predictions, average="weighted")
        f1 = f1_score(self.y_test, predictions, average="weighted")

        self._logger.info(f"  Accuracy:  {accuracy:.4f}")
        self._logger.info(f"  Precision: {precision:.4f}")
        self._logger.info(f"  Recall:    {recall:.4f}")
        self._logger.info(f"  F1 Score:  {f1:.4f}")

        # Classification report
        class_report = classification_report(
            self.y_test,
            predictions,
            target_names=self.class_names,
            digits=4,
        )

        self._logger.info("\nClassification Report:")
        print(class_report)

        # Confusion matrix
        conf_matrix = confusion_matrix(self.y_test, predictions)

        # Create result
        result = EvaluationResult(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            classification_report=class_report,
            confusion_matrix=conf_matrix,
            class_names=self.class_names,
            predictions=predictions,
            probabilities=probabilities,
        )

        # =========================================================================
        # Step 4: Save results
        # =========================================================================
        if save_results:
            self._logger.info("\n[STEP 4] Saving results...")
            self._save_results(result)

        self._logger.info("\n" + "=" * 60)
        self._logger.info("Evaluation complete!")
        self._logger.info("=" * 60)

        return result

    def _load_model(self) -> None:
        """Load trained model and associated files."""
        # Load model
        with open(self.model_path, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data.get("model")
        self.label_encoder = model_data.get("label_encoder")

        # Load feature names
        feature_list_path = config.MODELS_DIR / "feature_list.json"
        if feature_list_path.exists():
            with open(feature_list_path, "r") as f:
                feature_data = json.load(f)
                self.feature_names = feature_data.get("feature_names", [])

        # Load class names
        if self.label_encoder:
            self.class_names = list(self.label_encoder.classes_)
        else:
            # Try to load from metadata
            metadata_path = config.MODELS_DIR / "training_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    self.class_names = metadata.get("preprocessing", {}).get("class_names", [])
            else:
                self.class_names = []

        self._logger.info(f"Loaded model from: {self.model_path.name}")
        self._logger.info(f"Classes: {len(self.class_names)}")

    def _load_test_data(self) -> None:
        """Load preprocessed test data."""
        # Load split data
        splits_dir = config.MODELS_DIR / "splits"

        X_test_path = splits_dir / "X_test.npy"
        y_test_path = splits_dir / "y_test.npy"

        self.X_test = np.load(X_test_path)
        self.y_test = np.load(y_test_path)

        self._logger.info(f"Test set: {len(self.X_test):,} samples")

    def _save_results(self, result: EvaluationResult) -> None:
        """Save evaluation results to disk."""
        # Save metrics
        metrics_path = config.MODELS_DIR / EvaluationConfig.METRICS_FILENAME
        result.save_metrics(metrics_path)
        self._logger.info(f"Metrics saved: {metrics_path.name}")

        # Save classification report
        report_path = config.MODELS_DIR / EvaluationConfig.CLASSIFICATION_REPORT_FILENAME
        result.save_classification_report(report_path)
        self._logger.info(f"Classification report saved: {report_path.name}")

        # Save confusion matrix plot
        self._plot_confusion_matrix(result)

    def _plot_confusion_matrix(self, result: EvaluationResult) -> None:
        """Create and save confusion matrix plot."""
        output_path = config.MODELS_DIR / EvaluationConfig.CONFUSION_MATRIX_FILENAME

        fig, ax = plt.subplots(figsize=EvaluationConfig.FIGURE_SIZE)

        # Plot confusion matrix
        im = ax.imshow(result.confusion_matrix, interpolation="nearest", cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        ax.set(
            xticks=np.arange(len(result.class_names)),
            yticks=np.arange(len(result.class_names)),
            xticklabels=result.class_names,
            yticklabels=result.class_names,
            ylabel="True Label",
            xlabel="Predicted Label",
            title="Confusion Matrix - Random Forest",
        )

        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add text annotations
        thresh = result.confusion_matrix.max() / 2.0
        for i in range(len(result.class_names)):
            for j in range(len(result.class_names)):
                ax.text(
                    j,
                    i,
                    format(result.confusion_matrix[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if result.confusion_matrix[i, j] > thresh else "black",
                    fontsize=6,
                )

        plt.tight_layout()
        plt.savefig(output_path, dpi=EvaluationConfig.DPI)
        plt.close()

        self._logger.info(f"Confusion matrix saved: {output_path.name}")


# =============================================================================
# Convenience Function
# =============================================================================
def evaluate_random_forest(
    model_path: Path | None = None,
    data_path: Path | None = None,
) -> EvaluationResult:
    """
    Convenience function to evaluate Random Forest.

    Args:
        model_path: Optional path to model
        data_path: Optional path to data

    Returns:
        EvaluationResult
    """
    evaluator = RandomForestEvaluator(model_path=model_path, data_path=data_path)
    return evaluator.evaluate()


# =============================================================================
# Main Entry Point
# =============================================================================
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format=config.LOG_FORMAT,
        datefmt=config.LOG_DATE_FORMAT,
    )

    print("=" * 60)
    print("Random Forest Evaluation")
    print("=" * 60)

    # Evaluate model
    result = evaluate_random_forest()

    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"Accuracy:  {result.accuracy:.4f}")
    print(f"Precision: {result.precision:.4f}")
    print(f"Recall:    {result.recall:.4f}")
    print(f"F1 Score:  {result.f1_score:.4f}")
    print("=" * 60)