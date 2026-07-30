#!/usr/bin/env python3
"""
Standalone hyperparameter tuning module for selected models.
This module is intentionally independent from the baseline training pipeline.
"""

import json
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "datasets" / "synthetic"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

RANDOM_SEED = 42
N_CV_FOLDS = 5


def load_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load the synthetic train/test datasets."""
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    target_col = "career"
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    return X_train, y_train, X_test, y_test


def build_model_specs() -> dict[str, dict[str, Any]]:
    """Define tuning specs for the three requested models."""
    return {
        "random_forest": {
            "model_name": "Random Forest",
            "estimator": RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1),
            "search_space": {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2", None],
                "bootstrap": [True, False],
            },
            "scoring": "f1_macro",
            "n_iter": 12,
        },
        "xgboost": {
            "model_name": "XGBoost",
            "estimator": None,
            "search_space": {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 5, 7],
                "subsample": [0.7, 0.8, 0.9],
                "colsample_bytree": [0.7, 0.8, 0.9],
                "gamma": [0, 0.1, 0.2],
                "min_child_weight": [1, 3, 5],
            },
            "scoring": "f1_macro",
            "n_iter": 12,
        },
        "lightgbm": {
            "model_name": "LightGBM",
            "estimator": None,
            "search_space": {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.05, 0.1],
                "num_leaves": [15, 31, 63],
                "max_depth": [3, 5, 7],
                "subsample": [0.7, 0.8, 0.9],
                "colsample_bytree": [0.7, 0.8, 0.9],
                "min_child_samples": [5, 10, 20],
            },
            "scoring": "f1_macro",
            "n_iter": 12,
        },
    }


def build_estimator(name: str, random_state: int = RANDOM_SEED):
    """Create an estimator for the requested model."""
    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
        )
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
    return RandomForestClassifier(random_state=random_state, n_jobs=-1)


def tune_model(name: str, X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, Any]:
    """Tune a single model with RandomizedSearchCV and return the result payload."""
    spec = build_model_specs()[name]
    estimator = build_estimator(name)

    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=spec["search_space"],
        n_iter=spec["n_iter"],
        scoring=spec["scoring"],
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_SEED,
        return_train_score=True,
        verbose=1,
    )

    start_time = time.time()
    search.fit(X_train, y_train)
    train_time = time.time() - start_time

    return {
        "model_name": spec["model_name"],
        "search_space": spec["search_space"],
        "best_estimator": search.best_estimator_,
        "best_params": search.best_params_,
        "best_cv_score": search.best_score_,
        "train_time": train_time,
        "cv_results": search.cv_results_,
    }


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    target_encoder: LabelEncoder | None = None,
) -> dict[str, float]:
    """Evaluate a trained model on the test split."""
    start_time = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - start_time

    if target_encoder is not None:
        y_test_eval = target_encoder.inverse_transform(y_test)
        y_pred_eval = target_encoder.inverse_transform(y_pred)
    else:
        y_test_eval = y_test
        y_pred_eval = y_pred

    accuracy = accuracy_score(y_test_eval, y_pred_eval)
    precision = precision_score(y_test_eval, y_pred_eval, average="macro", zero_division=0)
    recall = recall_score(y_test_eval, y_pred_eval, average="macro", zero_division=0)
    f1 = f1_score(y_test_eval, y_pred_eval, average="macro", zero_division=0)

    return {
        "model_name": model_name,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pred_time": float(pred_time),
    }


def compare_with_baseline(baseline_metrics: dict[str, float], tuned_metrics: dict[str, float]) -> dict[str, float]:
    """Compute improvement percentage relative to the baseline metrics."""
    improvement = {}
    for metric in ["accuracy", "precision", "recall", "f1"]:
        baseline_value = baseline_metrics.get(metric, 0.0)
        tuned_value = tuned_metrics.get(metric, 0.0)
        if baseline_value == 0:
            improvement[metric] = 0.0
        else:
            improvement[metric] = ((tuned_value - baseline_value) / baseline_value) * 100
    return improvement


def save_model(model: Any, model_name: str) -> Path:
    """Persist the tuned model to disk without overwriting existing baseline artifacts."""
    filename_map = {
        "random_forest": "random_forest_tuned.pkl",
        "xgboost": "xgboost_tuned.pkl",
        "lightgbm": "lightgbm_tuned.pkl",
    }
    target_path = MODELS_DIR / filename_map.get(model_name, f"{model_name}.pkl")
    joblib.dump(model, target_path)
    return target_path


def write_reports(results: list[dict[str, Any]], baseline_results: dict[str, dict[str, Any]]) -> None:
    """Write markdown and JSON reports for the tuning run."""
    report_lines = []
    report_lines.append("# Hyperparameter Tuning Report")
    report_lines.append("")
    report_lines.append(f"Generated: {datetime.now().isoformat()}")
    report_lines.append("")
    report_lines.append("## Summary")
    report_lines.append("")
    report_lines.append("| Model | Best CV Score | Test Accuracy | Test F1 | Training Time (s) | Prediction Time (s) |")
    report_lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")

    ranking_rows = []
    for item in results:
        report_lines.append(
            f"| {item['model_name']} | {item['best_cv_score']:.4f} | {item['test_metrics']['accuracy']:.4f} | {item['test_metrics']['f1']:.4f} | {item['train_time']:.2f} | {item['test_metrics']['pred_time']:.4f} |"
        )
        ranking_rows.append((item['model_name'], item['test_metrics']['f1'], item['best_cv_score']))

    report_lines.append("")
    report_lines.append("## Detailed Results")
    report_lines.append("")

    for item in results:
        baseline_metrics = baseline_results.get(item["model_key"], {})
        improvement = compare_with_baseline(baseline_metrics, item["test_metrics"])
        report_lines.append(f"### {item['model_name']}")
        report_lines.append("")
        report_lines.append("- Model: " + item["model_name"])
        report_lines.append("- Search Space: " + json.dumps(item["search_space"], sort_keys=True))
        report_lines.append("- Best Parameters: " + json.dumps(item["best_params"], sort_keys=True))
        report_lines.append(f"- Best CV Score: {item['best_cv_score']:.4f}")
        report_lines.append(f"- Training Time: {item['train_time']:.2f}s")
        report_lines.append(f"- Prediction Time: {item['test_metrics']['pred_time']:.4f}s")
        report_lines.append(f"- Accuracy: {item['test_metrics']['accuracy']:.4f}")
        report_lines.append(f"- Precision: {item['test_metrics']['precision']:.4f}")
        report_lines.append(f"- Recall: {item['test_metrics']['recall']:.4f}")
        report_lines.append(f"- F1 Score: {item['test_metrics']['f1']:.4f}")
        report_lines.append("- Comparison with Baseline:")
        report_lines.append(f"  - Accuracy Improvement: {improvement.get('accuracy', 0.0):.2f}%")
        report_lines.append(f"  - Precision Improvement: {improvement.get('precision', 0.0):.2f}%")
        report_lines.append(f"  - Recall Improvement: {improvement.get('recall', 0.0):.2f}%")
        report_lines.append(f"  - F1 Improvement: {improvement.get('f1', 0.0):.2f}%")
        report_lines.append("")

    ranking_rows.sort(key=lambda row: (-row[1], -row[2], row[0]))
    report_lines.append("## Final Ranking")
    report_lines.append("")
    report_lines.append("| Rank | Model | F1 Score | Best CV Score |")
    report_lines.append("| --- | --- | ---: | ---: |")
    for rank, (name, f1_score, cv_score) in enumerate(ranking_rows, start=1):
        report_lines.append(f"| {rank} | {name} | {f1_score:.4f} | {cv_score:.4f} |")

    report_lines.append("")
    best_model = ranking_rows[0][0] if ranking_rows else "None"
    report_lines.append(f"## Final Recommendation")
    report_lines.append("")
    report_lines.append(f"Recommended production model: {best_model}")
    report_lines.append("Reason: it achieved the highest F1 Macro score and strong cross-validation performance while keeping the training and prediction cost practical.")

    (REPORTS_DIR / "hyperparameter_tuning_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    summary_payload = {
        "generated_at": datetime.now().isoformat(),
        "models": [
            {
                "model_key": item["model_key"],
                "model_name": item["model_name"],
                "best_params": item["best_params"],
                "best_cv_score": item["best_cv_score"],
                "training_time": item["train_time"],
                "prediction_time": item["test_metrics"]["pred_time"],
                "accuracy": item["test_metrics"]["accuracy"],
                "precision": item["test_metrics"]["precision"],
                "recall": item["test_metrics"]["recall"],
                "f1": item["test_metrics"]["f1"],
            }
            for item in results
        ],
        "ranking": [
            {
                "rank": idx,
                "model": name,
                "f1": float(f1_score),
                "best_cv_score": float(cv_score),
            }
            for idx, (name, f1_score, cv_score) in enumerate(ranking_rows, start=1)
        ],
        "recommended_model": best_model,
    }
    (REPORTS_DIR / "hyperparameter_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")


def main() -> None:
    """Run the tuning workflow for the three requested models."""
    X_train, y_train, X_test, y_test = load_data()
    target_encoder = LabelEncoder()
    y_train_encoded = target_encoder.fit_transform(y_train)
    y_test_encoded = target_encoder.transform(y_test)

    baseline_results = {
        "random_forest": {
            "accuracy": 0.9968,
            "precision": 0.9968,
            "recall": 0.9968,
            "f1": 0.9968,
        },
        "xgboost": {
            "accuracy": 0.9962,
            "precision": 0.9962,
            "recall": 0.9962,
            "f1": 0.9962,
        },
        "lightgbm": {
            "accuracy": 0.9958,
            "precision": 0.9958,
            "recall": 0.9958,
            "f1": 0.9958,
        },
    }

    tuned_results = []
    for model_key in ["random_forest", "xgboost", "lightgbm"]:
        print(f"\nTuning {model_key}...")
        tuned = tune_model(model_key, X_train, y_train_encoded)
        test_metrics = evaluate_model(
            tuned["best_estimator"],
            X_test,
            y_test_encoded,
            tuned["model_name"],
            target_encoder,
        )

        tuned_results.append(
            {
                "model_key": model_key,
                "model_name": tuned["model_name"],
                "search_space": tuned["search_space"],
                "best_params": tuned["best_params"],
                "best_cv_score": tuned["best_cv_score"],
                "train_time": tuned["train_time"],
                "test_metrics": test_metrics,
            }
        )

        model_path = save_model(tuned["best_estimator"], model_key)
        print(f"Saved tuned model to {model_path}")

    write_reports(tuned_results, baseline_results)
    print("\nHyperparameter tuning complete.")


if __name__ == "__main__":
    main()
