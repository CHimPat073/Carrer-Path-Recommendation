"""
Random Forest Training Module
Train a Random Forest classifier using preprocessed data.
"""

import json
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ml import config
from ml.data import DataLoader, DataPreprocessor, DataSplitter

# Configure logging
logger: logging.Logger = logging.getLogger(__name__)


# =============================================================================
# Training Configuration
# =============================================================================
class TrainingConfig:
    """Configuration for Random Forest training."""

    # Model hyperparameters (using default as per requirement)
    MODEL_PARAMS: dict[str, Any] = {
        # Default parameters - can be overridden
        "n_estimators": 100,
        "max_depth": None,  # Unlimited
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": config.SPLIT_CONFIG.random_state,
        "n_jobs": -1,
        "class_weight": "balanced",  # Handle class imbalance
    }

    # Cross-validation settings
    CV_FOLDS: int = 5
    CV_SCORING: str = "f1_weighted"

    # Output settings
    MODEL_FILENAME: str = "random_forest_model.pkl"
    METRICS_FILENAME: str = "training_metrics.json"
    FEATURE_IMPORTANCE_FILENAME: str = "feature_importance.json"


# =============================================================================
# Training Result
# =============================================================================
class TrainingResult:
    """Results from model training."""

    def __init__(
        self,
        model: RandomForestClassifier,
        training_time: float,
        cv_scores: dict[str, list[float]],
        feature_importance: dict[str, float],
        class_names: list[str],
        config: dict[str, Any],
    ) -> None:
        self.model = model
        self.training_time = training_time
        self.cv_scores = cv_scores
        self.feature_importance = feature_importance
        self.class_names = class_names
        self.config = config
        self.timestamp: str = datetime.now().isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "training_time_seconds": self.training_time,
            "model_config": self.config,
            "cv_scores": {
                key: [float(v) for v in values]
                for key, values in self.cv_scores.items()
            },
            "cv_summary": {
                "accuracy_mean": float(np.mean(self.cv_scores["accuracy"])),
                "accuracy_std": float(np.std(self.cv_scores["accuracy"])),
                "f1_mean": float(np.mean(self.cv_scores["f1_weighted"])),
                "f1_std": float(np.std(self.cv_scores["f1_weighted"])),
            },
            "feature_importance": self.feature_importance,
            "class_names": self.class_names,
            "num_classes": len(self.class_names),
        }


# =============================================================================
# Random Forest Trainer
# =============================================================================
class RandomForestTrainer:
    """
    Train and evaluate a Random Forest classifier.

    Uses preprocessed data from the ml/data pipeline.

    Example:
        >>> trainer = RandomForestTrainer()
        >>> result = trainer.train()
        >>> print(f"Training complete in {result.training_time:.2f}s")
    """

    def __init__(
        self,
        model_params: dict[str, Any] | None = None,
        cv_folds: int = TrainingConfig.CV_FOLDS,
    ) -> None:
        """
        Initialize the trainer.

        Args:
            model_params: Optional custom model parameters
            cv_folds: Number of cross-validation folds
        """
        self.model_params = model_params or TrainingConfig.MODEL_PARAMS
        self.cv_folds = cv_folds
        self.model: RandomForestClassifier | None = None
        self.training_result: TrainingResult | None = None
        self._preprocessor = None  # Store preprocessor for saving
        self._feature_names: list[str] = []  # Store feature names
        self._class_names: list[str] = []  # Store class names
        self._logger = logging.getLogger(__name__)

    def train(
        self,
        data_path: Path | None = None,
        model_params: dict[str, Any] | None = None,
    ) -> TrainingResult:
        """
        Train Random Forest model on preprocessed data.

        Args:
            data_path: Path to dataset (uses default if None)
            model_params: Optional custom model parameters

        Returns:
            TrainingResult with model and metrics
        """
        self._logger.info("=" * 60)
        self._logger.info("Random Forest Training Pipeline")
        self._logger.info("=" * 60)

        # Use provided params or defaults
        params = model_params or self.model_params

        # =========================================================================
        # Step 1: Load and preprocess data
        # =========================================================================
        self._logger.info("\n[STEP 1] Loading data...")
        data_file = data_path or config.DATASET_CONFIG.synthetic_v2

        loader = DataLoader()
        df = loader.load(data_file)
        self._logger.info(f"Loaded {len(df):,} samples")

        # Preprocess
        self._logger.info("\n[STEP 2] Preprocessing...")
        preprocessor = DataPreprocessor()
        preprocessed = preprocessor.preprocess(df)
        self._preprocessor = preprocessor  # Store for later
        self._feature_names = preprocessed.feature_names
        self._class_names = list(preprocessed.label_encoder.classes_)
        self._logger.info(f"Features: {len(preprocessed.feature_names)}")
        self._logger.info(f"Classes: {len(preprocessed.label_encoder.classes_)}")

        # Split
        self._logger.info("\n[STEP 3] Splitting data (70/15/15)...")
        splitter = DataSplitter()
        splits = splitter.split(preprocessed.X, preprocessed.y)
        self._logger.info(f"Train: {splits.train_size:,}, Val: {splits.val_size:,}, Test: {splits.test_size:,}")

        # =========================================================================
        # Step 2: Train model
        # =========================================================================
        self._logger.info("\n[STEP 4] Training Random Forest...")
        self._logger.info(f"Parameters: {params}")

        # Use training set
        X_train = splits.X_train
        y_train = splits.y_train

        # Create and train model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        self.model = model
        self._logger.info("Training complete!")

        # =========================================================================
        # Step 3: Cross-validation
        # =========================================================================
        self._logger.info(f"\n[STEP 5] Cross-validation ({self.cv_folds}-fold)...")
        cv_scores = self._run_cross_validation(X_train, y_train)

        # =========================================================================
        # Step 4: Get feature importance
        # =========================================================================
        self._logger.info("\n[STEP 6] Extracting feature importance...")
        feature_importance = self._extract_feature_importance(
            preprocessed.feature_names
        )

        # =========================================================================
        # Step 5: Save model and results
        # =========================================================================
        self._logger.info("\n[STEP 7] Saving model and results...")

        # Create training result
        result = TrainingResult(
            model=model,
            training_time=0.0,  # Will be calculated
            cv_scores=cv_scores,
            feature_importance=feature_importance,
            class_names=list(preprocessed.label_encoder.classes_),
            config={
                "data_path": str(data_file),
                "train_size": int(splits.train_size),
                "val_size": int(splits.val_size),
                "test_size": int(splits.test_size),
                "model_params": params,
                "cv_folds": self.cv_folds,
            },
        )

        # Save components
        self._save_model(model)
        self._save_metrics(result)
        self._save_feature_importance(feature_importance, preprocessed.feature_names)

        # Save preprocessor for later use
        preprocessor.save(config.MODELS_DIR)

        self._logger.info("\n" + "=" * 60)
        self._logger.info("Training complete!")
        self._logger.info("=" * 60)

        self.training_result = result
        return result

    def _run_cross_validation(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> dict[str, list[float]]:
        """Run cross-validation and return scores."""
        from sklearn.model_selection import cross_val_score

        model = RandomForestClassifier(**self.model_params)

        # Calculate multiple metrics
        scoring_metrics = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]
        cv_results: dict[str, list[float]] = {}

        for metric in scoring_metrics:
            scores = cross_val_score(model, X, y, cv=self.cv_folds, scoring=metric, n_jobs=-1)
            cv_results[metric] = scores.tolist()
            self._logger.info(f"  {metric}: {scores.mean():.4f} (+/- {scores.std():.4f})")

        return cv_results

    def _extract_feature_importance(
        self,
        feature_names: list[str],
    ) -> dict[str, float]:
        """Extract feature importance from trained model."""
        if self.model is None:
            return {}

        importances = self.model.feature_importances_

        # Create feature importance dictionary
        importance_dict: dict[str, float] = {}
        for name, importance in zip(feature_names, importances):
            importance_dict[name] = float(importance)

        # Sort by importance
        importance_dict = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )

        # Log top 10
        self._logger.info("\nTop 10 Features:")
        for i, (name, importance) in enumerate(list(importance_dict.items())[:10]):
            display_name = config.FEATURE_DISPLAY_NAMES.get(name, name)
            self._logger.info(f"  {i+1}. {display_name}: {importance:.4f}")

        return importance_dict

    def _save_model(self, model: RandomForestClassifier) -> Path:
        """Save trained model to disk."""
        output_path = config.MODELS_DIR / TrainingConfig.MODEL_FILENAME
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save model with metadata
        model_data = {
            "model": model,
            "label_encoder": self._preprocessor.label_encoder if self._preprocessor else None,
            "feature_names": self._feature_names,
            "model_params": self.model_params,
            "class_names": self._class_names,
        }

        with open(output_path, "wb") as f:
            pickle.dump(model_data, f)

        self._logger.info(f"Model saved: {output_path.name}")
        return output_path

    def _save_metrics(self, result: TrainingResult) -> Path:
        """Save training metrics to JSON."""
        output_path = config.MODELS_DIR / TrainingConfig.METRICS_FILENAME
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

        self._logger.info(f"Metrics saved: {output_path.name}")
        return output_path

    def _save_feature_importance(
        self,
        importance: dict[str, float],
        feature_names: list[str],
    ) -> Path:
        """Save feature importance to JSON."""
        output_path = config.MODELS_DIR / TrainingConfig.FEATURE_IMPORTANCE_FILENAME
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Add display names
        data = {
            "features": [
                {
                    "name": name,
                    "display_name": config.FEATURE_DISPLAY_NAMES.get(name, name),
                    "importance": importance.get(name, 0.0),
                }
                for name in feature_names
            ],
            "top_10": [
                {
                    "name": name,
                    "display_name": config.FEATURE_DISPLAY_NAMES.get(name, name),
                    "importance": importance,
                }
                for name, importance in list(importance.items())[:10]
            ],
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        self._logger.info(f"Feature importance saved: {output_path.name}")
        return output_path


# =============================================================================
# Convenience Function
# =============================================================================
def train_random_forest(
    data_path: Path | None = None,
    model_params: dict[str, Any] | None = None,
) -> TrainingResult:
    """
    Convenience function to train Random Forest.

    Args:
        data_path: Optional path to dataset
        model_params: Optional model parameters

    Returns:
        TrainingResult
    """
    trainer = RandomForestTrainer(model_params=model_params)
    return trainer.train(data_path=data_path)


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
    print("Random Forest Training")
    print("=" * 60)

    # Train model
    result = train_random_forest()

    print("\nTraining complete!")
    print(f"Model saved to: {config.MODELS_DIR / TrainingConfig.MODEL_FILENAME}")