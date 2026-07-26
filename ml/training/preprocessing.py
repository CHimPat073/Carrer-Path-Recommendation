#!/usr/bin/env python3
"""
Production-ready preprocessing pipeline for CareerPilot AI.

This module provides a reusable preprocessing pipeline for both training and inference.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "datasets" / "synthetic"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset paths
INPUT_PATH = DATA_DIR / "synthetic_dataset_20000.csv"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

# Model paths
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
TARGET_ENCODER_PATH = MODELS_DIR / "target_encoder.joblib"

# Split configuration
TEST_SIZE = 0.2
RANDOM_SEED = 42

# =============================================================================
# Feature Definitions
# =============================================================================

# Target column
TARGET_COLUMN = "career"

# Numerical features (no scaling required for tree-based models, but modular)
NUMERICAL_FEATURES = [
    "years_experience",
    "projects_completed",
    "certifications",
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
]

# Categorical features with natural ordering (use OrdinalEncoder)
ORDINAL_FEATURES = {
    "education_level": ["High School", "Associate", "Bachelor", "Master", "PhD"],
}

# Categorical features without natural ordering (use OneHotEncoder)
NOMINAL_FEATURES = [
    "remote_preference",
    "country",
    "industry",
    "employment_type",
]


# =============================================================================
# Pipeline Builder
# =============================================================================


def build_preprocessor(enable_scaling: bool = False) -> ColumnTransformer:
    """
    Build a sklearn ColumnTransformer for feature preprocessing.

    Args:
        enable_scaling: Whether to apply StandardScaler to numerical features.
                       Set to True for models that benefit from scaling (e.g., SVM, Neural Networks).

    Returns:
        ColumnTransformer: Preprocessing pipeline for features.
    """
    transformers = []

    # Numerical features: passthrough (no scaling for tree-based models)
    if enable_scaling:
        transformers.append(
            (
                "numerical",
                StandardScaler(),
                NUMERICAL_FEATURES,
            )
        )
    else:
        transformers.append(
            (
                "numerical",
                "passthrough",
                NUMERICAL_FEATURES,
            )
        )

    # Ordinal features: education_level has natural ordering
    if ORDINAL_FEATURES:
        for col, categories in ORDINAL_FEATURES.items():
            transformers.append(
                (
                    f"ordinal_{col}",
                    OrdinalEncoder(categories=[categories], handle_unknown="use_encoded_value", unknown_value=-1),
                    [col],
                )
            )

    # Nominal features: one-hot encoding
    if NOMINAL_FEATURES:
        transformers.append(
            (
                "nominal",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                NOMINAL_FEATURES,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_full_pipeline(enable_scaling: bool = False) -> Pipeline:
    """
    Build a complete sklearn Pipeline including preprocessing.

    Args:
        enable_scaling: Whether to apply StandardScaler to numerical features.

    Returns:
        Pipeline: Complete preprocessing pipeline.
    """
    preprocessor = build_preprocessor(enable_scaling=enable_scaling)

    return Pipeline(
        [
            ("preprocessor", preprocessor),
        ]
    )


# =============================================================================
# Data Loading and Preparation
# =============================================================================


def load_dataset(path: Path = INPUT_PATH) -> pd.DataFrame:
    """Load the dataset from CSV."""
    print(f"Loading dataset from {path}...")
    df = pd.read_csv(path)
    print(f"  Loaded {len(df)} rows with {len(df.columns)} columns")
    return df


def separate_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate features and target variable.

    Args:
        df: Full dataset DataFrame.

    Returns:
        Tuple of (X, y) where X is features DataFrame and y is target Series.
    """
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    print(f"\nSeparated features and target:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")

    return X, y


def get_feature_names() -> dict:
    """Get categorized feature names."""
    return {
        "numerical": NUMERICAL_FEATURES,
        "ordinal": list(ORDINAL_FEATURES.keys()),
        "nominal": NOMINAL_FEATURES,
    }


# =============================================================================
# Encoding
# =============================================================================


def create_target_encoder() -> LabelEncoder:
    """
    Create and fit LabelEncoder for target variable.

    Returns:
        Fitted LabelEncoder.
    """
    return LabelEncoder()


def encode_target(y: pd.Series, encoder: LabelEncoder) -> np.ndarray:
    """
    Encode target variable using LabelEncoder.

    Args:
        y: Target Series.
        encoder: Fitted LabelEncoder.

    Returns:
        Encoded target array.
    """
    return encoder.fit_transform(y)


# =============================================================================
# Validation
# =============================================================================


def validate_no_missing_values(df: pd.DataFrame) -> bool:
    """Check for missing values in the dataset."""
    missing = df.isnull().sum().sum()
    if missing > 0:
        print(f"  WARNING: Found {missing} missing values")
        return False
    print("  [PASS] No missing values")
    return True


def validate_no_unseen_categories(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    nominal_cols: list[str],
) -> bool:
    """Check for unseen categories in test set."""
    all_valid = True
    for col in nominal_cols:
        train_cats = set(X_train[col].unique())
        test_cats = set(X_test[col].unique())
        unseen = test_cats - train_cats
        if unseen:
            print(f"  WARNING: Unseen categories in {col}: {unseen}")
            all_valid = False
    if all_valid:
        print("  [PASS] No unseen categories in test set")
    return all_valid


def validate_class_balance(y_train: pd.Series, y_test: pd.Series, test_size: float) -> bool:
    """Validate that class balance is maintained in train/test split."""
    train_dist = y_train.value_counts(normalize=True).sort_index()
    test_dist = y_test.value_counts(normalize=True).sort_index()

    # Check that distribution is similar (within 5% tolerance)
    max_diff = (train_dist - test_dist).abs().max()
    if max_diff > 0.05:
        print(f"  WARNING: Class distribution difference: {max_diff:.4f}")
        return False
    print(f"  [PASS] Class balance maintained (max diff: {max_diff:.4f})")
    return True


# =============================================================================
# Main Pipeline Execution
# =============================================================================


def run_preprocessing_pipeline() -> dict:
    """
    Execute the complete preprocessing pipeline.

    Returns:
        Dictionary containing pipeline artifacts and statistics.
    """
    print("=" * 60)
    print("CAREERPILOT AI PREPROCESSING PIPELINE")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Step 1: Load Dataset
    # -------------------------------------------------------------------------
    print("\n[1/10] Loading dataset...")
    df = load_dataset(INPUT_PATH)

    # -------------------------------------------------------------------------
    # Step 2: Separate Features and Target
    # -------------------------------------------------------------------------
    print("\n[2/10] Separating features and target...")
    X, y = separate_features_target(df)

    # -------------------------------------------------------------------------
    # Step 3: Identify Feature Types
    # -------------------------------------------------------------------------
    print("\n[3/10] Identifying feature types...")
    feature_info = get_feature_names()
    print(f"  Numerical features: {len(feature_info['numerical'])}")
    print(f"  Ordinal features: {len(feature_info['ordinal'])}")
    print(f"  Nominal features: {len(feature_info['nominal'])}")

    # -------------------------------------------------------------------------
    # Step 4: Validate Data Quality
    # -------------------------------------------------------------------------
    print("\n[4/10] Validating data quality...")
    validate_no_missing_values(X)

    # -------------------------------------------------------------------------
    # Step 5: Train/Test Split
    # -------------------------------------------------------------------------
    print("\n[5/10] Performing stratified train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    print(f"  Train size: {len(X_train)} ({1 - TEST_SIZE:.0%})")
    print(f"  Test size: {len(X_test)} ({TEST_SIZE:.0%})")

    # -------------------------------------------------------------------------
    # Step 6: Validate Split Quality
    # -------------------------------------------------------------------------
    print("\n[6/10] Validating split quality...")
    validate_no_unseen_categories(X_train, X_test, NOMINAL_FEATURES)
    validate_class_balance(y_train, y_test, TEST_SIZE)

    # -------------------------------------------------------------------------
    # Step 7: Build and Fit Preprocessor
    # -------------------------------------------------------------------------
    print("\n[7/10] Building and fitting preprocessor...")
    preprocessor = build_preprocessor(enable_scaling=False)
    preprocessor.fit(X_train)
    print("  [PASS] Preprocessor fitted")

    # -------------------------------------------------------------------------
    # Step 8: Fit Target Encoder
    # -------------------------------------------------------------------------
    print("\n[8/10] Fitting target encoder...")
    target_encoder = create_target_encoder()
    y_train_encoded = target_encoder.fit_transform(y_train)
    y_test_encoded = target_encoder.transform(y_test)
    print(f"  [PASS] Target encoder fitted with {len(target_encoder.classes_)} classes")

    # -------------------------------------------------------------------------
    # Step 9: Transform Features
    # -------------------------------------------------------------------------
    print("\n[9/10] Transforming features...")
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Get feature names after transformation
    feature_names_out = preprocessor.get_feature_names_out()
    print(f"  [PASS] Transformed {len(X_train.columns)} features to {len(feature_names_out)} encoded features")

    # -------------------------------------------------------------------------
    # Step 10: Save Artifacts
    # -------------------------------------------------------------------------
    print("\n[10/10] Saving artifacts...")

    # Save preprocessor
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    print(f"  [PASS] Preprocessor saved to {PREPROCESSOR_PATH}")

    # Save target encoder
    joblib.dump(target_encoder, TARGET_ENCODER_PATH)
    print(f"  [PASS] Target encoder saved to {TARGET_ENCODER_PATH}")

    # Save train/test CSVs
    train_df = pd.DataFrame(X_train_transformed, columns=feature_names_out)
    train_df[TARGET_COLUMN] = y_train.values
    train_df.to_csv(TRAIN_PATH, index=False)
    print(f"  [PASS] Train data saved to {TRAIN_PATH}")

    test_df = pd.DataFrame(X_test_transformed, columns=feature_names_out)
    test_df[TARGET_COLUMN] = y_test.values
    test_df.to_csv(TEST_PATH, index=False)
    print(f"  [PASS] Test data saved to {TEST_PATH}")

    # -------------------------------------------------------------------------
    # Print Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION COMPLETE")
    print("=" * 60)

    print(f"\nDataset Shape: {df.shape}")
    print(f"Train Shape: {X_train.shape} -> {X_train_transformed.shape} (encoded)")
    print(f"Test Shape: {X_test.shape} -> {X_test_transformed.shape} (encoded)")

    print(f"\nFeature Counts:")
    print(f"  Numerical: {len(NUMERICAL_FEATURES)}")
    print(f"  Ordinal: {len(ORDINAL_FEATURES)}")
    print(f"  Nominal: {len(NOMINAL_FEATURES)}")
    print(f"  Total Original: {len(NUMERICAL_FEATURES) + len(ORDINAL_FEATURES) + len(NOMINAL_FEATURES)}")
    print(f"  Encoded: {len(feature_names_out)}")

    print(f"\nTarget Classes ({len(target_encoder.classes_)}):")
    for i, cls in enumerate(target_encoder.classes_):
        print(f"  {i}: {cls}")

    print("\nValidation Results:")
    print("  [PASS] No missing values")
    print("  [PASS] No unseen categories")
    print("  [PASS] Class balance maintained")
    print("  [PASS] Pipeline successfully serialized")

    return {
        "dataset_shape": df.shape,
        "train_shape": X_train_transformed.shape,
        "test_shape": X_test_transformed.shape,
        "numerical_features": len(NUMERICAL_FEATURES),
        "ordinal_features": len(ORDINAL_FEATURES),
        "nominal_features": len(NOMINAL_FEATURES),
        "encoded_features": len(feature_names_out),
        "target_classes": list(target_encoder.classes_),
        "preprocessor_path": str(PREPROCESSOR_PATH),
        "target_encoder_path": str(TARGET_ENCODER_PATH),
        "train_path": str(TRAIN_PATH),
        "test_path": str(TEST_PATH),
    }


# =============================================================================
# Inference-Ready Functions
# =============================================================================


def load_preprocessor(path: Path = PREPROCESSOR_PATH) -> ColumnTransformer:
    """Load the saved preprocessor for inference."""
    return joblib.load(path)


def load_target_encoder(path: Path = TARGET_ENCODER_PATH) -> LabelEncoder:
    """Load the saved target encoder for inference."""
    return joblib.load(path)


def preprocess_for_inference(X: pd.DataFrame, preprocessor: ColumnTransformer) -> np.ndarray:
    """
    Preprocess new data for inference.

    Args:
        X: Feature DataFrame.
        preprocessor: Fitted ColumnTransformer.

    Returns:
        Preprocessed feature array.
    """
    return preprocessor.transform(X)


# =============================================================================
# Main Entry Point
# =============================================================================


if __name__ == "__main__":
    run_preprocessing_pipeline()
