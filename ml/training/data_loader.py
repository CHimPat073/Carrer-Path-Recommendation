"""
ML Training Data Loader and Feature Engineering
Handles data loading, preprocessing, and feature transformation for model training.
"""

import json
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SYNTHETIC_DATA_PATH = PROJECT_ROOT / "datasets" / "synthetic" / "synthetic_dataset.csv"
KNOWLEDGE_BASE_PATH = PROJECT_ROOT / "knowledge_base"

# Debug: Print paths if needed
# print(f"PROJECT_ROOT: {PROJECT_ROOT}")
# print(f"SYNTHETIC_DATA_PATH: {SYNTHETIC_DATA_PATH}")

# Feature columns used for training
SKILL_FEATURES: Final[list[str]] = [
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
]

BASE_FEATURES: Final[list[str]] = [
    "years_experience",
    "projects_completed",
    "certifications",
]

CATEGORICAL_FEATURES: Final[list[str]] = [
    "education_level",
    "country",
    "industry",
    "employment_type",
    "remote_preference",
]

ALL_NUMERIC_FEATURES = BASE_FEATURES + SKILL_FEATURES


def load_training_data(data_path: Path | None = None) -> pd.DataFrame:
    """Load the synthetic dataset for training."""
    path = data_path or SYNTHETIC_DATA_PATH
    df = pd.read_csv(path, low_memory=False)
    return df


def encode_education(education: str) -> int:
    """Encode education level to numeric value."""
    mapping = {
        "High School": 0,
        "Associate": 1,
        "Bachelor": 2,
        "Master": 3,
        "PhD": 4,
    }
    return mapping.get(education, 2)


def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, LabelEncoder]:
    """
    Preprocess data for training.
    Returns: X (features), y (encoded labels), label_encoder
    """
    # Make a copy
    data = df.copy()

    # Handle missing values
    data[ALL_NUMERIC_FEATURES] = data[ALL_NUMERIC_FEATURES].fillna(0)
    data[CATEGORICAL_FEATURES] = data[CATEGORICAL_FEATURES].fillna("Unknown")

    # Encode education level
    data["education_level_encoded"] = data["education_level"].apply(encode_education)

    # Encode categorical features
    for col in CATEGORICAL_FEATURES:
        if col != "education_level":
            le = LabelEncoder()
            data[f"{col}_encoded"] = le.fit_transform(data[col].astype(str))

    # Build feature matrix
    feature_cols = [
        "years_experience",
        "projects_completed",
        "certifications",
        "education_level_encoded",
    ] + SKILL_FEATURES

    X = data[feature_cols].astype(float)

    # Encode target
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data["career"])

    return X, y, label_encoder


def get_feature_names() -> list[str]:
    """Return list of all feature names used in training."""
    return ["years_experience", "projects_completed", "certifications", "education_level"] + SKILL_FEATURES


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train and test sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Scale numeric features using StandardScaler."""
    scaler = StandardScaler()

    # Fit on training data only
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


def load_career_requirements() -> dict[str, dict[str, float]]:
    """Load career skill requirements from knowledge base."""
    requirements_path = KNOWLEDGE_BASE_PATH / "career_personas.json"

    if not requirements_path.exists():
        # Return default requirements
        return _get_default_requirements()

    try:
        with open(requirements_path, "r") as f:
            data = json.load(f)

        career_requirements = {}
        for career_data in data.get("career_personas", []):
            career_name = career_data.get("career")
            if not career_name:
                continue

            # Extract required skills with their importance levels
            required_skills = {}
            for skill in SKILL_FEATURES:
                skill_key = skill.replace("_score", "")
                # Default to medium importance (5) if not specified
                required_skills[skill] = career_data.get(skill_key, 5)

            career_requirements[career_name] = required_skills

        return career_requirements if career_requirements else _get_default_requirements()

    except Exception:
        return _get_default_requirements()


def _get_default_requirements() -> dict[str, dict[str, float]]:
    """Return default career skill requirements."""
    default: dict[str, dict[str, float]] = {}

    # Default requirements for each career
    career_defaults = {
        "Software Engineer": {"python_score": 8, "java_score": 7, "sql_score": 6, "problem_solving_score": 8},
        "Data Scientist": {"python_score": 9, "machine_learning_score": 9, "data_analysis_score": 8, "sql_score": 7},
        "ML Engineer": {"python_score": 9, "machine_learning_score": 9, "deep_learning_score": 8, "cloud_score": 6},
        "Backend Developer": {"python_score": 8, "java_score": 7, "database_score": 7, "sql_score": 8},
        "Frontend Developer": {"javascript_score": 9, "ui_design_score": 7, "communication_score": 6},
        "Full Stack Developer": {"python_score": 7, "javascript_score": 7, "sql_score": 6, "cloud_score": 6},
        "DevOps Engineer": {"cloud_score": 9, "devops_score": 9, "networking_score": 7, "problem_solving_score": 7},
        "Data Engineer": {"sql_score": 9, "python_score": 7, "database_score": 8, "cloud_score": 7},
    }

    for career, skills in career_defaults.items():
        default[career] = {skill: 5.0 for skill in SKILL_FEATURES}
        for skill, value in skills.items():
            if skill in default[career]:
                default[career][skill] = value

    return default


if __name__ == "__main__":
    # Quick test
    df = load_training_data()
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    print(f"Career distribution:\n{df['career'].value_counts()}")