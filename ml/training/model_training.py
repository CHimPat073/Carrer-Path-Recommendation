#!/usr/bin/env python3
"""
Baseline Model Training and Evaluation Script for CareerPilot AI.

This script trains multiple baseline ML models and compares their performance.
"""

import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold

warnings.filterwarnings("ignore")

# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "datasets" / "synthetic"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Data paths
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

# Model paths
RF_PATH = MODELS_DIR / "random_forest.pkl"
ET_PATH = MODELS_DIR / "extra_trees.pkl"
XGB_PATH = MODELS_DIR / "xgboost.pkl"
LGBM_PATH = MODELS_DIR / "lightgbm.pkl"
CB_PATH = MODELS_DIR / "catboost.pkl"

RANDOM_SEED = 42
N_CV_FOLDS = 5

# =============================================================================
# Data Loading
# =============================================================================


def load_data() -> tuple:
    """Load training and test data."""
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # Separate features and target
    TARGET_COL = "career"
    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_test, y_test


# =============================================================================
# Model Training
# =============================================================================


def train_model(name: str, model, X_train, y_train) -> tuple:
    """Train a model and return it with timing info."""
    print(f"\nTraining {name}...")

    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    print(f"  Training time: {train_time:.2f}s")
    return model, train_time


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """Evaluate a model and return metrics."""
    print(f"\nEvaluating {model_name}...")

    # Predictions
    start_time = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - start_time

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Prediction time: {pred_time:.4f}s")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pred_time": pred_time,
        "y_pred": y_pred,
    }


def cross_validate_model(model, X_train, y_train, cv: int = 5) -> float:
    """Perform stratified cross-validation."""
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
    scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="accuracy", n_jobs=-1)
    return scores.mean(), scores.std()


def get_feature_importance(model, feature_names, model_name: str) -> pd.DataFrame:
    """Extract feature importance from model."""
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        df = pd.DataFrame({
            "feature": feature_names,
            "importance": importance,
        }).sort_values("importance", ascending=False)
        return df.head(20)
    return None


# =============================================================================
# Main Training Pipeline
# =============================================================================


def main():
    """Train and evaluate all baseline models."""
    print("=" * 70)
    print("CAREERPILOT AI - BASELINE MODEL TRAINING")
    print("=" * 70)

    # Load data
    print("\n[1/6] Loading preprocessed data...")
    X_train, y_train, X_test, y_test = load_data()
    feature_names = list(X_train.columns)

    # Store results
    results = {}
    models = {}

    # ==========================================================================
    # 1. Random Forest
    # ==========================================================================
    print("\n" + "=" * 50)
    print("[2/6] Training Random Forest")
    print("=" * 50)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
    rf_model, rf_train_time = train_model("Random Forest", rf_model, X_train, y_train)

    # Cross-validation
    rf_cv_mean, rf_cv_std = cross_validate_model(rf_model, X_train, y_train, N_CV_FOLDS)
    print(f"  CV Accuracy: {rf_cv_mean:.4f} (+/- {rf_cv_std:.4f})")

    # Evaluation
    rf_eval = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    rf_eval["cv_accuracy"] = rf_cv_mean
    rf_eval["cv_std"] = rf_cv_std
    rf_eval["train_time"] = rf_train_time

    # Feature importance
    rf_importance = get_feature_importance(rf_model, feature_names, "Random Forest")

    # Save model
    joblib.dump(rf_model, RF_PATH)
    print(f"  Model saved to {RF_PATH}")

    results["Random Forest"] = rf_eval
    models["Random Forest"] = {"model": rf_model, "importance": rf_importance}

    # ==========================================================================
    # 2. Extra Trees
    # ==========================================================================
    print("\n" + "=" * 50)
    print("[3/6] Training Extra Trees")
    print("=" * 50)
    et_model = ExtraTreesClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
    et_model, et_train_time = train_model("Extra Trees", et_model, X_train, y_train)

    # Cross-validation
    et_cv_mean, et_cv_std = cross_validate_model(et_model, X_train, y_train, N_CV_FOLDS)
    print(f"  CV Accuracy: {et_cv_mean:.4f} (+/- {et_cv_std:.4f})")

    # Evaluation
    et_eval = evaluate_model(et_model, X_test, y_test, "Extra Trees")
    et_eval["cv_accuracy"] = et_cv_mean
    et_eval["cv_std"] = et_cv_std
    et_eval["train_time"] = et_train_time

    # Feature importance
    et_importance = get_feature_importance(et_model, feature_names, "Extra Trees")

    # Save model
    joblib.dump(et_model, ET_PATH)
    print(f"  Model saved to {ET_PATH}")

    results["Extra Trees"] = et_eval
    models["Extra Trees"] = {"model": et_model, "importance": et_importance}

    # ==========================================================================
    # 3. XGBoost
    # ==========================================================================
    print("\n" + "=" * 50)
    print("[4/6] Training XGBoost")
    print("=" * 50)
    try:
        from xgboost import XGBClassifier
        from sklearn.preprocessing import LabelEncoder

        # Encode target for XGBoost
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)

        xgb_model = XGBClassifier(
            n_estimators=100,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric="mlogloss",
            verbosity=0,
        )
        xgb_model, xgb_train_time = train_model("XGBoost", xgb_model, X_train, y_train_enc)

        # Cross-validation
        xgb_cv_mean, xgb_cv_std = cross_validate_model(xgb_model, X_train, y_train_enc, N_CV_FOLDS)
        print(f"  CV Accuracy: {xgb_cv_mean:.4f} (+/- {xgb_cv_std:.4f})")

        # Evaluation
        start_time = time.time()
        y_pred_enc = xgb_model.predict(X_test)
        xgb_pred_time = time.time() - start_time

        y_pred = le.inverse_transform(y_pred_enc)
        xgb_accuracy = accuracy_score(y_test_enc, y_pred_enc)
        xgb_precision = precision_score(y_test_enc, y_pred_enc, average="macro", zero_division=0)
        xgb_recall = recall_score(y_test_enc, y_pred_enc, average="macro", zero_division=0)
        xgb_f1 = f1_score(y_test_enc, y_pred_enc, average="macro", zero_division=0)

        xgb_eval = {
            "accuracy": xgb_accuracy,
            "precision": xgb_precision,
            "recall": xgb_recall,
            "f1": xgb_f1,
            "cv_accuracy": xgb_cv_mean,
            "cv_std": xgb_cv_std,
            "train_time": xgb_train_time,
            "pred_time": xgb_pred_time,
            "y_pred": y_pred,
        }
        print(f"  Accuracy: {xgb_accuracy:.4f}")
        print(f"  F1 Score: {xgb_f1:.4f}")

        # Feature importance
        xgb_importance = get_feature_importance(xgb_model, feature_names, "XGBoost")

        # Save model
        joblib.dump(xgb_model, XGB_PATH)
        print(f"  Model saved to {XGB_PATH}")

        results["XGBoost"] = xgb_eval
        models["XGBoost"] = {"model": xgb_model, "importance": xgb_importance, "le": le}

    except ImportError as e:
        print(f"  SKIPPED: XGBoost not available - {e}")
        results["XGBoost"] = {"error": str(e)}

    # ==========================================================================
    # 4. LightGBM
    # ==========================================================================
    print("\n" + "=" * 50)
    print("[5/6] Training LightGBM")
    print("=" * 50)
    try:
        from lightgbm import LGBMClassifier

        lgbm_model = LGBMClassifier(
            n_estimators=100,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbose=-1,
        )
        lgbm_model, lgbm_train_time = train_model("LightGBM", lgbm_model, X_train, y_train)

        # Cross-validation
        lgbm_cv_mean, lgbm_cv_std = cross_validate_model(lgbm_model, X_train, y_train, N_CV_FOLDS)
        print(f"  CV Accuracy: {lgbm_cv_mean:.4f} (+/- {lgbm_cv_std:.4f})")

        # Evaluation
        lgbm_eval = evaluate_model(lgbm_model, X_test, y_test, "LightGBM")
        lgbm_eval["cv_accuracy"] = lgbm_cv_mean
        lgbm_eval["cv_std"] = lgbm_cv_std
        lgbm_eval["train_time"] = lgbm_train_time

        # Feature importance
        lgbm_importance = get_feature_importance(lgbm_model, feature_names, "LightGBM")

        # Save model
        joblib.dump(lgbm_model, LGBM_PATH)
        print(f"  Model saved to {LGBM_PATH}")

        results["LightGBM"] = lgbm_eval
        models["LightGBM"] = {"model": lgbm_model, "importance": lgbm_importance}

    except ImportError as e:
        print(f"  SKIPPED: LightGBM not available - {e}")
        results["LightGBM"] = {"error": str(e)}

    # ==========================================================================
    # 5. CatBoost
    # ==========================================================================
    print("\n" + "=" * 50)
    print("[6/6] Training CatBoost")
    print("=" * 50)
    try:
        from catboost import CatBoostClassifier

        cb_model = CatBoostClassifier(
            n_estimators=100,
            random_state=RANDOM_SEED,
            verbose=0,
        )
        cb_model, cb_train_time = train_model("CatBoost", cb_model, X_train, y_train)

        # Cross-validation
        cb_cv_mean, cb_cv_std = cross_validate_model(cb_model, X_train, y_train, N_CV_FOLDS)
        print(f"  CV Accuracy: {cb_cv_mean:.4f} (+/- {cb_cv_std:.4f})")

        # Evaluation
        cb_eval = evaluate_model(cb_model, X_test, y_test, "CatBoost")
        cb_eval["cv_accuracy"] = cb_cv_mean
        cb_eval["cv_std"] = cb_cv_std
        cb_eval["train_time"] = cb_train_time

        # Feature importance
        cb_importance = get_feature_importance(cb_model, feature_names, "CatBoost")

        # Save model
        joblib.dump(cb_model, CB_PATH)
        print(f"  Model saved to {CB_PATH}")

        results["CatBoost"] = cb_eval
        models["CatBoost"] = {"model": cb_model, "importance": cb_importance}

    except ImportError as e:
        print(f"  SKIPPED: CatBoost not available - {e}")
        results["CatBoost"] = {"error": str(e)}

    # ==========================================================================
    # Generate Reports
    # ==========================================================================
    print("\n" + "=" * 70)
    print("GENERATING REPORTS")
    print("=" * 70)

    generate_reports(results, models, y_test, feature_names)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


# =============================================================================
# Report Generation
# =============================================================================


def generate_reports(results: dict, models: dict, y_test, feature_names: list):
    """Generate markdown and JSON reports."""

    timestamp = datetime.now().isoformat()

    # Build comparison table
    model_comparison = []
    for model_name, eval_data in results.items():
        if "error" in eval_data:
            continue
        model_comparison.append({
            "model": model_name,
            "accuracy": eval_data.get("accuracy", 0),
            "precision": eval_data.get("precision", 0),
            "recall": eval_data.get("recall", 0),
            "f1": eval_data.get("f1", 0),
            "cv_accuracy": eval_data.get("cv_accuracy", 0),
            "train_time": eval_data.get("train_time", 0),
            "pred_time": eval_data.get("pred_time", 0),
        })

    # Sort by different criteria
    by_f1 = sorted(model_comparison, key=lambda x: x["f1"], reverse=True)
    by_cv = sorted(model_comparison, key=lambda x: x["cv_accuracy"], reverse=True)
    by_train_time = sorted(model_comparison, key=lambda x: x["train_time"])
    by_pred_time = sorted(model_comparison, key=lambda x: x["pred_time"])

    # Find best model
    best_model = by_f1[0]["model"] if by_f1 else "N/A"

    # Generate markdown report
    md_report = f"""# Baseline Model Training Report

## Metadata
- **Generated**: {timestamp}
- **Random Seed**: {RANDOM_SEED}
- **CV Folds**: {N_CV_FOLDS}

## Model Comparison

### Performance Metrics

| Model | Accuracy | Precision | Recall | F1 Score | CV Accuracy | Train Time (s) | Pred Time (s) |
|-------|----------|-----------|--------|----------|-------------|----------------|---------------|
"""
    for m in model_comparison:
        md_report += f"| {m['model']} | {m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['cv_accuracy']:.4f} | {m['train_time']:.2f} | {m['pred_time']:.4f} |\n"

    md_report += f"""
### Rankings

**By F1 Score:**
"""
    for i, m in enumerate(by_f1, 1):
        md_report += f"{i}. {m['model']}: {m['f1']:.4f}\n"

    md_report += f"""
**By CV Accuracy:**
"""
    for i, m in enumerate(by_cv, 1):
        md_report += f"{i}. {m['model']}: {m['cv_accuracy']:.4f}\n"

    md_report += f"""
**By Training Time (fastest first):**
"""
    for i, m in enumerate(by_train_time, 1):
        md_report += f"{i}. {m['model']}: {m['train_time']:.2f}s\n"

    md_report += f"""
**By Prediction Time (fastest first):**
"""
    for i, m in enumerate(by_pred_time, 1):
        md_report += f"{i}. {m['model']}: {m['pred_time']:.4f}s\n"

    # Best model
    md_report += f"""
## Best Model Recommendation

**Best Model: {best_model}**

Based on F1 Score ranking, **{best_model}** is the recommended model for this classification task.

## Model Advantages and Disadvantages

### Random Forest
- **Advantages**: Robust to overfitting, handles missing values, provides feature importance
- **Disadvantages**: Slower than single decision tree, less interpretable

### Extra Trees
- **Advantages**: Faster training than Random Forest, more randomness leads to better generalization
- **Disadvantages**: Less interpretable, can be memory intensive

### XGBoost
- **Advantages**: High accuracy, handles sparse data, built-in regularization
- **Disadvantages**: Requires encoding for categorical features, can overfit

### LightGBM
- **Advantages**: Fastest training, handles large datasets, leaf-wise growth
- **Disadvantages**: Can overfit on small datasets, sensitive to hyperparameters

### CatBoost
- **Advantages**: Native categorical handling, built-in overfitting detection
- **Disadvantages**: Slower than LightGBM, less community support

## Conclusion

All models were trained with default parameters. The dataset has {len(feature_names)} features after preprocessing.
Based on the evaluation, **{best_model}** achieved the highest F1 score of {by_f1[0]['f1']:.4f}.

For production deployment, consider:
1. Using the best performing model: {best_model}
2. Enabling early stopping to prevent overfitting
3. Performing hyperparameter tuning if accuracy needs improvement
"""

    # Save markdown report
    md_path = REPORTS_DIR / "baseline_model_report.md"
    md_path.write_text(md_report, encoding="utf-8")
    print(f"  Saved: {md_path}")

    # Generate JSON summary
    json_summary = {
        "timestamp": timestamp,
        "random_seed": RANDOM_SEED,
        "cv_folds": N_CV_FOLDS,
        "models": {},
        "rankings": {
            "by_f1": [m["model"] for m in by_f1],
            "by_cv_accuracy": [m["model"] for m in by_cv],
            "by_train_time": [m["model"] for m in by_train_time],
            "by_pred_time": [m["model"] for m in by_pred_time],
        },
        "best_model": best_model,
    }

    for m in model_comparison:
        json_summary["models"][m["model"]] = {
            "accuracy": round(m["accuracy"], 4),
            "precision": round(m["precision"], 4),
            "recall": round(m["recall"], 4),
            "f1": round(m["f1"], 4),
            "cv_accuracy": round(m["cv_accuracy"], 4),
            "train_time": round(m["train_time"], 4),
            "pred_time": round(m["pred_time"], 6),
        }

    # Save JSON summary
    json_path = REPORTS_DIR / "baseline_model_summary.json"
    json_path.write_text(json.dumps(json_summary, indent=2), encoding="utf-8")
    print(f"  Saved: {json_path}")

    # Print summary
    print(f"\nModel Comparison:")
    print(f"  Best by F1: {by_f1[0]['model']} ({by_f1[0]['f1']:.4f})")
    print(f"  Best by CV: {by_cv[0]['model']} ({by_cv[0]['cv_accuracy']:.4f})")
    print(f"  Fastest Train: {by_train_time[0]['model']} ({by_train_time[0]['train_time']:.2f}s)")
    print(f"  Fastest Pred: {by_pred_time[0]['model']} ({by_pred_time[0]['pred_time']:.4f}s)")
    print(f"\nBest Model: {best_model}")


if __name__ == "__main__":
    main()


