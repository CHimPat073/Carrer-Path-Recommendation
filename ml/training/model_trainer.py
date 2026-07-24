"""
Model Training Pipeline
Train Random Forest, XGBoost, and CatBoost classifiers.
Compare model performance and save the best model.
"""

import pickle
import time
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, learning_curve

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = PROJECT_ROOT / "ml" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Try to import optional dependencies
try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import catboost as cb

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


class ModelTrainer:
    """Train and evaluate multiple ML models."""

    def __init__(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        label_encoder: Any = None,
    ):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.label_encoder = label_encoder
        self.models: dict[str, Any] = {}
        self.results: dict[str, dict] = {}

    def train_random_forest(
        self,
        n_estimators: int = 200,
        max_depth: int = 20,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> dict[str, float]:
        """Train Random Forest classifier."""
        print("\n" + "=" * 60)
        print("Training Random Forest...")
        print("=" * 60)

        start_time = time.time()

        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs,
            class_weight="balanced",
        )

        rf.fit(self.X_train, self.y_train)
        train_time = time.time() - start_time

        # Predictions
        y_pred = rf.predict(self.X_test)
        y_pred_proba = rf.predict_proba(self.X_test)

        # Metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average="weighted")
        recall = recall_score(self.y_test, y_pred, average="weighted")
        f1 = f1_score(self.y_test, y_pred, average="weighted")

        # Cross-validation
        cv_scores = cross_val_score(rf, self.X_train, self.y_train, cv=5, scoring="f1_weighted")

        results = {
            "model": rf,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "cv_f1_mean": cv_scores.mean(),
            "cv_f1_std": cv_scores.std(),
            "train_time": train_time,
            "y_pred": y_pred,
            "y_pred_proba": y_pred_proba,
        }

        self.models["random_forest"] = rf
        self.results["random_forest"] = results

        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  CV F1:     {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"  Train Time: {train_time:.2f}s")

        return results

    def train_xgboost(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        use_label_encoder: bool = True,
        eval_metric: str = "mlogloss",
    ) -> dict[str, float]:
        """Train XGBoost classifier."""
        if not XGBOOST_AVAILABLE:
            print("XGBoost not installed. Skipping...")
            return {}

        print("\n" + "=" * 60)
        print("Training XGBoost...")
        print("=" * 60)

        start_time = time.time()

        xgb_model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            use_label_encoder=use_label_encoder,
            eval_metric=eval_metric,
            n_jobs=-1,
        )

        xgb_model.fit(self.X_train, self.y_train)
        train_time = time.time() - start_time

        # Predictions
        y_pred = xgb_model.predict(self.X_test)
        y_pred_proba = xgb_model.predict_proba(self.X_test)

        # Metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average="weighted")
        recall = recall_score(self.y_test, y_pred, average="weighted")
        f1 = f1_score(self.y_test, y_pred, average="weighted")

        # Cross-validation
        cv_scores = cross_val_score(xgb_model, self.X_train, self.y_train, cv=5, scoring="f1_weighted")

        results = {
            "model": xgb_model,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "cv_f1_mean": cv_scores.mean(),
            "cv_f1_std": cv_scores.std(),
            "train_time": train_time,
            "y_pred": y_pred,
            "y_pred_proba": y_pred_proba,
        }

        self.models["xgboost"] = xgb_model
        self.results["xgboost"] = results

        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  CV F1:     {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"  Train Time: {train_time:.2f}s")

        return results

    def train_catboost(
        self,
        iterations: int = 200,
        depth: int = 6,
        learning_rate: float = 0.1,
        random_state: int = 42,
        verbose: bool = False,
    ) -> dict[str, float]:
        """Train CatBoost classifier."""
        if not CATBOOST_AVAILABLE:
            print("CatBoost not installed. Skipping...")
            return {}

        print("\n" + "=" * 60)
        print("Training CatBoost...")
        print("=" * 60)

        start_time = time.time()

        cb_model = cb.CatBoostClassifier(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            random_state=random_state,
            verbose=verbose,
            task_type="CPU",
        )

        cb_model.fit(self.X_train, self.y_train)
        train_time = time.time() - start_time

        # Predictions
        y_pred = cb_model.predict(self.X_test)
        y_pred_proba = cb_model.predict_proba(self.X_test)

        # Metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average="weighted")
        recall = recall_score(self.y_test, y_pred, average="weighted")
        f1 = f1_score(self.y_test, y_pred, average="weighted")

        # Cross-validation
        cv_scores = cross_val_score(cb_model, self.X_train, self.y_train, cv=5, scoring="f1_weighted")

        results = {
            "model": cb_model,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "cv_f1_mean": cv_scores.mean(),
            "cv_f1_std": cv_scores.std(),
            "train_time": train_time,
            "y_pred": y_pred,
            "y_pred_proba": y_pred_proba,
        }

        self.models["catboost"] = cb_model
        self.results["catboost"] = results

        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  CV F1:     {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"  Train Time: {train_time:.2f}s")

        return results

    def train_all(self) -> dict[str, dict]:
        """Train all available models."""
        self.train_random_forest()

        if XGBOOST_AVAILABLE:
            self.train_xgboost()

        if CATBOOST_AVAILABLE:
            self.train_catboost()

        return self.results

    def compare_models(self) -> pd.DataFrame:
        """Compare all trained models."""
        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)

        comparison_data = []
        for name, results in self.results.items():
            comparison_data.append(
                {
                    "Model": name,
                    "Accuracy": results.get("accuracy", 0),
                    "Precision": results.get("precision", 0),
                    "Recall": results.get("recall", 0),
                    "F1 Score": results.get("f1_score", 0),
                    "CV F1 Mean": results.get("cv_f1_mean", 0),
                    "CV F1 Std": results.get("cv_f1_std", 0),
                    "Train Time (s)": results.get("train_time", 0),
                }
            )

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values("F1 Score", ascending=False)

        print(comparison_df.to_string(index=False))

        # Determine best model
        best_model_name = comparison_df.iloc[0]["Model"]
        print(f"\nBest Model: {best_model_name}")

        return comparison_df

    def get_best_model(self, metric: str = "f1_score") -> tuple[str, Any]:
        """Get the best performing model based on specified metric."""
        if not self.results:
            raise ValueError("No models trained yet")

        best_name = max(self.results, key=lambda k: self.results[k].get(metric, 0))
        return best_name, self.results[best_name]["model"]

    def save_model(self, model_name: str, path: Path | None = None) -> Path:
        """Save a trained model to disk."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")

        if path is None:
            path = MODELS_DIR / f"{model_name}_model.pkl"

        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self.models[model_name],
            "label_encoder": self.label_encoder,
            "results": self.results.get(model_name, {}),
        }

        with open(path, "wb") as f:
            pickle.dump(model_data, f)

        print(f"Model saved to: {path}")
        return path

    def save_best_model(self, metric: str = "f1_score") -> Path:
        """Save the best performing model."""
        best_name, _ = self.get_best_model(metric)
        return self.save_model(best_name)

    def get_classification_report(self, model_name: str) -> str:
        """Get detailed classification report for a model."""
        if model_name not in self.results:
            raise ValueError(f"Model '{model_name}' not found")

        y_pred = self.results[model_name]["y_pred"]

        if self.label_encoder:
            target_names = self.label_encoder.classes_
        else:
            target_names = None

        return classification_report(self.y_test, y_pred, target_names=target_names)


def train_pipeline(
    data_path: Path | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> ModelTrainer:
    """Complete training pipeline."""
    from .data_loader import load_training_data, preprocess_data, split_data

    # Load and preprocess data
    print("Loading data...")
    df = load_training_data(data_path)
    print(f"Loaded {len(df)} samples")

    X, y, label_encoder = preprocess_data(df)

    # Split data
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Number of classes: {len(label_encoder.classes_)}")

    # Train models
    trainer = ModelTrainer(X_train, X_test, y_train, y_test, label_encoder)
    trainer.train_all()

    # Compare models
    comparison = trainer.compare_models()

    # Save best model
    trainer.save_best_model()

    return trainer


if __name__ == "__main__":
    trainer = train_pipeline()
    print("\nTraining complete!")