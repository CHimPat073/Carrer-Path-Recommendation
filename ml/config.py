"""
ML Configuration Module
Centralized configuration for all ML pipeline settings.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final


# =============================================================================
# Project Paths
# =============================================================================
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "datasets"
MODELS_DIR: Path = PROJECT_ROOT / "ml" / "models"
LOGS_DIR: Path = PROJECT_ROOT / "ml" / "logs"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Dataset Paths
# =============================================================================
@dataclass
class DatasetConfig:
    """Configuration for dataset paths."""
    synthetic_v2: Path = DATA_DIR / "synthetic" / "synthetic_dataset_v2.csv"
    synthetic: Path = DATA_DIR / "synthetic" / "synthetic_dataset.csv"
    final: Path = DATA_DIR / "processed" / "final_dataset.csv"
    hybrid: Path = DATA_DIR / "processed" / "hybrid_dataset.csv"


DATASET_CONFIG: Final[DatasetConfig] = DatasetConfig()


# =============================================================================
# Data Split Configuration
# =============================================================================
@dataclass
class SplitConfig:
    """Configuration for data splitting."""
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_state: int = 42

    def __post_init__(self) -> None:
        """Validate split ratios."""
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not abs(total - 1.0) < 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")


SPLIT_CONFIG: Final[SplitConfig] = SplitConfig()


# =============================================================================
# Feature Configuration
# =============================================================================
# Base numeric features (non-skill features)
BASE_FEATURES: Final[list[str]] = [
    "years_experience",
    "projects_completed",
    "certifications",
]

# Skill score features (0-10 scale)
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

# All numeric features (for tree models - no scaling needed)
NUMERIC_FEATURES: Final[list[str]] = BASE_FEATURES + SKILL_FEATURES

# Categorical features to encode
CATEGORICAL_FEATURES: Final[list[str]] = [
    "education_level",
    "country",
    "industry",
    "employment_type",
    "remote_preference",
]

# Target variable
TARGET_COLUMN: Final[str] = "Career"

# All features to use
ALL_FEATURES: Final[list[str]] = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# =============================================================================
# Feature Display Names (for UI/Explainability)
# =============================================================================
FEATURE_DISPLAY_NAMES: Final[dict[str, str]] = {
    "years_experience": "Years of Experience",
    "projects_completed": "Projects Completed",
    "certifications": "Certifications",
    "python_score": "Python",
    "java_score": "Java",
    "javascript_score": "JavaScript",
    "sql_score": "SQL",
    "machine_learning_score": "Machine Learning",
    "deep_learning_score": "Deep Learning",
    "cloud_score": "Cloud Computing",
    "devops_score": "DevOps",
    "cybersecurity_score": "Cybersecurity",
    "data_analysis_score": "Data Analysis",
    "database_score": "Database",
    "networking_score": "Networking",
    "mobile_score": "Mobile Development",
    "game_dev_score": "Game Development",
    "testing_score": "Testing/QA",
    "business_analysis_score": "Business Analysis",
    "product_management_score": "Product Management",
    "ui_design_score": "UI Design",
    "ux_research_score": "UX Research",
    "communication_score": "Communication",
    "leadership_score": "Leadership",
    "problem_solving_score": "Problem Solving",
    "teamwork_score": "Teamwork",
    "agile_score": "Agile/Scrum",
    "research_score": "Research",
    "education_level": "Education Level",
    "country": "Country",
    "industry": "Industry",
    "employment_type": "Employment Type",
    "remote_preference": "Remote Preference",
}


# =============================================================================
# Education Level Encoding
# =============================================================================
EDUCATION_ENCODING: Final[dict[str, int]] = {
    "High School": 0,
    "Associate": 1,
    "Bachelor": 2,
    "Master": 3,
    "PhD": 4,
}


# =============================================================================
# Model Metadata
# =============================================================================
@dataclass
class ModelMetadata:
    """Metadata for trained models."""
    dataset_name: str
    num_samples: int
    num_features: int
    num_classes: int
    class_names: list[str]
    feature_names: list[str]
    categorical_features: list[str]
    numeric_features: list[str]
    split_config: SplitConfig
    created_at: str


# =============================================================================
# Logging Configuration
# =============================================================================
LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"


def setup_logging(name: str, log_file: Path | None = None) -> logging.Logger:
    """
    Setup logging with console and optional file output.

    Args:
        name: Logger name
        log_file: Optional path to log file

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    )
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
        )
        logger.addHandler(file_handler)

    return logger


# =============================================================================
# Validation Thresholds
# =============================================================================
MIN_SAMPLES_PER_CLASS: Final[int] = 10
MAX_MISSING_RATIO: Final[float] = 0.1  # 10% max missing values per column
MIN_CLASSES: Final[int] = 2


# =============================================================================
# Export Config
# =============================================================================
ENCODER_FILENAME: Final[str] = "label_encoder.pkl"
FEATURE_LIST_FILENAME: Final[str] = "feature_list.json"
METADATA_FILENAME: Final[str] = "training_metadata.json"
SCALER_FILENAME: Final[str] = "feature_scaler.pkl"
CATEGORY_MAPPING_FILENAME: Final[str] = "category_mappings.json"


if __name__ == "__main__":
    # Print configuration summary
    print("CareerPilot-AI ML Configuration")
    print("=" * 50)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Models Directory: {MODELS_DIR}")
    print(f"\nSplit Config:")
    print(f"  Train: {SPLIT_CONFIG.train_ratio * 100}%")
    print(f"  Validation: {SPLIT_CONFIG.val_ratio * 100}%")
    print(f"  Test: {SPLIT_CONFIG.test_ratio * 100}%")
    print(f"  Random State: {SPLIT_CONFIG.random_state}")
    print(f"\nFeatures:")
    print(f"  Base: {len(BASE_FEATURES)}")
    print(f"  Skills: {len(SKILL_FEATURES)}")
    print(f"  Categorical: {len(CATEGORICAL_FEATURES)}")
    print(f"  Total: {len(ALL_FEATURES)}")