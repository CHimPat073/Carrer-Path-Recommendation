"""
Data Preprocessing Module
Production-ready preprocessing with validation, encoding, and transformations.
"""

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ml import config

# Configure module logger
logger: logging.Logger = logging.getLogger(__name__)


# =============================================================================
# Custom Exceptions
# =============================================================================
class PreprocessingError(Exception):
    """Raised when preprocessing fails."""
    pass


class ValidationError(PreprocessingError):
    """Raised when data validation fails."""
    pass


class EncodingError(PreprocessingError):
    """Raised when encoding fails."""
    pass


# =============================================================================
# Validation Results
# =============================================================================
@dataclass
class ValidationResult:
    """Result of dataset validation."""
    is_valid: bool
    num_rows: int
    num_columns: int
    num_classes: int | None
    class_distribution: dict[str, int]
    missing_values: dict[str, int]
    duplicate_rows: int
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "is_valid": self.is_valid,
            "num_rows": self.num_rows,
            "num_columns": self.num_columns,
            "num_classes": self.num_classes,
            "class_distribution": {str(k): int(v) for k, v in self.class_distribution.items()},
            "missing_values": {str(k): int(v) for k, v in self.missing_values.items()},
            "duplicate_rows": self.duplicate_rows,
            "warnings": self.warnings,
            "errors": self.errors,
        }


# =============================================================================
# Preprocessing Result
# =============================================================================
@dataclass
class PreprocessingResult:
    """Result of data preprocessing."""
    X: pd.DataFrame
    y: np.ndarray
    label_encoder: LabelEncoder
    feature_names: list[str]
    categorical_encoders: dict[str, LabelEncoder]
    scaler: StandardScaler | None
    feature_info: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "X_shape": self.X.shape,
            "y_shape": self.y.shape,
            "num_features": len(self.feature_names),
            "num_classes": len(self.label_encoder.classes_),
            "class_names": list(self.label_encoder.classes_),
            "feature_names": self.feature_names,
        }


# =============================================================================
# Data Preprocessor Class
# =============================================================================
class DataPreprocessor:
    """
    Production-ready data preprocessor with validation and encoding.

    Handles:
    - Dataset validation
    - Missing value handling
    - Categorical encoding
    - Target encoding
    - Feature scaling
    - Metadata preservation

    Example:
        >>> preprocessor = DataPreprocessor()
        >>> result = preprocessor.preprocess(df)
        >>> X_train, X_val, X_test = result.splits["train", "val", "test"]
    """

    def __init__(
        self,
        target_column: str | None = None,
        numeric_features: list[str] | None = None,
        categorical_features: list[str] | None = None,
        scale_features: bool = False,
    ) -> None:
        """
        Initialize preprocessor with configuration.

        Args:
            target_column: Name of target column (default: from config)
            numeric_features: List of numeric feature names
            categorical_features: List of categorical feature names
            scale_features: Whether to create scaler for non-tree models
        """
        self.target_column = target_column or config.TARGET_COLUMN
        self.numeric_features = numeric_features or config.NUMERIC_FEATURES
        self.categorical_features = categorical_features or config.CATEGORICAL_FEATURES
        self.scale_features = scale_features

        # State
        self._label_encoder: LabelEncoder | None = None
        self._categorical_encoders: dict[str, LabelEncoder] = {}
        self._scaler: StandardScaler | None = None
        self._feature_names: list[str] = []
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        """Check if preprocessor has been fitted."""
        return self._is_fitted

    @property
    def label_encoder(self) -> LabelEncoder:
        """Get fitted label encoder."""
        if not self._is_fitted:
            raise PreprocessingError("Preprocessor not fitted yet")
        return self._label_encoder

    @property
    def feature_names(self) -> list[str]:
        """Get feature names."""
        return self._feature_names.copy()

    @property
    def class_names(self) -> list[str]:
        """Get class names from label encoder."""
        if self._label_encoder is None:
            return []
        return list(self._label_encoder.classes_)

    # =========================================================================
    # Validation Methods
    # =========================================================================
    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate dataset meets requirements.

        Args:
            df: DataFrame to validate

        Returns:
            ValidationResult with detailed information

        Example:
            >>> preprocessor = DataPreprocessor()
            >>> result = preprocessor.validate(df)
            >>> if not result.is_valid:
            ...     for error in result.errors:
            ...         print(f"Error: {error}")
        """
        warnings: list[str] = []
        errors: list[str] = []

        # Check basic properties
        if df is None or df.empty:
            errors.append("Dataset is empty")
            return ValidationResult(
                is_valid=False,
                num_rows=0,
                num_columns=0,
                num_classes=None,
                class_distribution={},
                missing_values={},
                duplicate_rows=0,
                errors=errors,
            )

        # Check target column exists
        if self.target_column not in df.columns:
            # Try to find similar column
            found_col = None
            for col in df.columns:
                if "career" in col.lower() or "target" in col.lower():
                    found_col = col
                    break

            if found_col:
                warnings.append(f"Target column '{self.target_column}' not found, using '{found_col}'")
                self.target_column = found_col
            else:
                errors.append(f"Target column '{self.target_column}' not found in dataset")

        # Check features exist
        missing_features = []
        for feat in self.numeric_features + self.categorical_features:
            if feat not in df.columns:
                missing_features.append(feat)

        if missing_features:
            warnings.append(f"Missing features (will be created with defaults): {missing_features[:5]}...")

        # Check class distribution
        class_distribution: dict[str, int] = {}
        num_classes: int | None = None

        if self.target_column in df.columns:
            class_distribution = df[self.target_column].value_counts().to_dict()
            num_classes = len(class_distribution)

            # Check minimum samples per class
            min_samples = min(class_distribution.values()) if class_distribution else 0
            if min_samples < config.MIN_SAMPLES_PER_CLASS:
                warnings.append(
                    f"Class '{min(class_distribution, key=class_distribution.get)}' has only "
                    f"{min_samples} samples (minimum: {config.MIN_SAMPLES_PER_CLASS})"
                )

            if num_classes < config.MIN_CLASSES:
                errors.append(f"Only {num_classes} classes found (minimum: {config.MIN_CLASSES})")

        # Check missing values
        missing_values = df.isna().sum().to_dict()
        total_missing = sum(missing_values.values())

        if total_missing > 0:
            # Check for columns with high missing rate
            high_missing = [
                col for col, count in missing_values.items()
                if count / len(df) > config.MAX_MISSING_RATIO
            ]
            if high_missing:
                warnings.append(f"Columns with high missing rate: {high_missing}")

        # Check duplicates
        duplicate_rows = int(df.duplicated().sum())

        if duplicate_rows > 0:
            warnings.append(f"Found {duplicate_rows} duplicate rows")

        is_valid = len(errors) == 0

        return ValidationResult(
            is_valid=is_valid,
            num_rows=len(df),
            num_columns=len(df.columns),
            num_classes=num_classes,
            class_distribution=class_distribution,
            missing_values=missing_values,
            duplicate_rows=duplicate_rows,
            warnings=warnings,
            errors=errors,
        )

    # =========================================================================
    # Preprocessing Methods
    # =========================================================================
    def preprocess(self, df: pd.DataFrame) -> PreprocessingResult:
        """
        Preprocess dataset with full pipeline.

        Args:
            df: Raw DataFrame to preprocess

        Returns:
            PreprocessingResult with processed data and metadata

        Example:
            >>> preprocessor = DataPreprocessor()
            >>> result = preprocessor.preprocess(df)
            >>> X = result.X
            >>> y = result.y
        """
        self._logger = logging.getLogger(__name__)
        self._logger.info(f"Starting preprocessing for {len(df):,} rows")

        # Validate first
        validation = self.validate(df)
        self._log_validation_results(validation)

        if not validation.is_valid:
            raise ValidationError(f"Validation failed: {validation.errors}")

        # Create working copy
        data = df.copy()

        # Handle missing columns with defaults
        data = self._handle_missing_features(data)

        # Handle missing values
        data = self._handle_missing_values(data)

        # Remove duplicates if any
        data = self._remove_duplicates(data)

        # Extract target and encode
        y = self._encode_target(data)

        # Process features
        X = self._process_features(data)

        # Create scaler if requested
        if self.scale_features:
            self._scaler = StandardScaler()
            self._scaler.fit(X[self.numeric_features])

        # Store feature names
        self._feature_names = list(X.columns)

        # Mark as fitted
        self._is_fitted = True

        self._logger.info(f"Preprocessing complete: {X.shape[1]} features, {len(y)} samples")

        return PreprocessingResult(
            X=X,
            y=y,
            label_encoder=self._label_encoder,
            feature_names=self._feature_names,
            categorical_encoders=self._categorical_encoders.copy(),
            scaler=self._scaler,
            feature_info=self._create_feature_info(X),
        )

    def _handle_missing_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create missing features with default values."""
        for feat in self.numeric_features + [self.target_column]:
            if feat not in data.columns:
                self._logger.warning(f"Creating missing column '{feat}' with defaults")
                if feat == self.target_column:
                    data[feat] = "Unknown"
                else:
                    data[feat] = 0

        return data

    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in dataset."""
        # Numeric: fill with 0
        for col in self.numeric_features:
            if col in data.columns:
                data[col] = data[col].fillna(0)

        # Categorical: fill with "Unknown"
        for col in self.categorical_features:
            if col in data.columns:
                data[col] = data[col].fillna("Unknown").astype(str)

        return data

    def _remove_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows."""
        before = len(data)
        data = data.drop_duplicates()
        after = len(data)

        if before > after:
            self._logger.info(f"Removed {before - after} duplicate rows")

        return data

    def _encode_target(self, data: pd.DataFrame) -> np.ndarray:
        """Encode target variable."""
        target = data[self.target_column].astype(str)

        self._label_encoder = LabelEncoder()
        y = self._label_encoder.fit_transform(target)

        self._logger.info(
            f"Encoded {len(self._label_encoder.classes_)} classes: "
            f"{list(self._label_encoder.classes_[:5])}..."
        )

        return y

    def _process_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process all features."""
        feature_list: list[str] = []

        # Add numeric features
        for col in self.numeric_features:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)
                feature_list.append(col)

        # Encode and add categorical features
        for col in self.categorical_features:
            if col in data.columns:
                data[col] = data[col].astype(str)

                # Create label encoder for this column
                encoder = LabelEncoder()
                data[col] = encoder.fit_transform(data[col])
                self._categorical_encoders[col] = encoder

                feature_list.append(col)

        # Select only the features we want
        X = data[feature_list].copy()

        self._logger.info(f"Processed {len(feature_list)} features")

        return X

    def _create_feature_info(self, X: pd.DataFrame) -> dict[str, Any]:
        """Create feature information dictionary."""
        return {
            "num_features": len(X.columns),
            "feature_names": list(X.columns),
            "numeric_features": [f for f in X.columns if f in self.numeric_features],
            "categorical_features": [f for f in X.columns if f in self.categorical_features],
            "dtypes": X.dtypes.astype(str).to_dict(),
            "value_ranges": {
                col: {
                    "min": float(X[col].min()),
                    "max": float(X[col].max()),
                    "mean": float(X[col].mean()),
                }
                for col in X.columns[:5]  # First 5 only
            },
        }

    def _log_validation_results(self, result: ValidationResult) -> None:
        """Log validation results."""
        if result.is_valid:
            logger.info(
                f"Validation passed: {result.num_rows:,} rows, "
                f"{result.num_classes} classes"
            )
        else:
            logger.error(f"Validation failed: {result.errors}")

        if result.warnings:
            for warning in result.warnings:
                logger.warning(warning)

    # =========================================================================
    # Transform Methods (for new data)
    # =========================================================================
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using fitted preprocessor.

        Args:
            X: DataFrame to transform

        Returns:
            Transformed numpy array

        Example:
            >>> preprocessor = DataPreprocessor()
            >>> preprocessor.fit(train_df)
            >>> X_test_transformed = preprocessor.transform(test_df)
        """
        if not self._is_fitted:
            raise PreprocessingError("Preprocessor not fitted")

        # Handle missing values
        X = self._handle_missing_values(X.copy())

        # Ensure all features exist
        for feat in self._feature_names:
            if feat not in X.columns:
                X[feat] = 0

        # Encode categorical features
        for col, encoder in self._categorical_encoders.items():
            if col in X.columns:
                # Handle unseen categories
                X[col] = X[col].astype(str)
                known_classes = set(encoder.classes_)
                X[col] = X[col].apply(lambda x: x if x in known_classes else encoder.classes_[0])
                X[col] = encoder.transform(X[col])

        # Select features in correct order
        X = X[self._feature_names]

        return X.values

    def inverse_transform_labels(self, y: np.ndarray) -> np.ndarray:
        """Inverse transform encoded labels to original class names."""
        if self._label_encoder is None:
            raise PreprocessingError("Label encoder not fitted")
        return self._label_encoder.inverse_transform(y)

    # =========================================================================
    # Save/Load Methods
    # =========================================================================
    def save(self, output_dir: Path) -> dict[str, Path]:
        """
        Save preprocessor components to disk.

        Args:
            output_dir: Directory to save files

        Returns:
            Dictionary mapping component names to file paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files: dict[str, Path] = {}

        # Save label encoder
        if self._label_encoder:
            path = output_dir / config.ENCODER_FILENAME
            with open(path, "wb") as f:
                pickle.dump(self._label_encoder, f)
            saved_files["label_encoder"] = path

        # Save feature list
        path = output_dir / config.FEATURE_LIST_FILENAME
        with open(path, "w") as f:
            json.dump({
                "feature_names": self._feature_names,
                "numeric_features": self.numeric_features,
                "categorical_features": self.categorical_features,
                "target_column": self.target_column,
            }, f, indent=2)
        saved_files["feature_list"] = path

        # Save categorical encoders
        if self._categorical_encoders:
            path = output_dir / config.CATEGORY_MAPPING_FILENAME
            encoders_dict = {
                col: list(encoder.classes_)
                for col, encoder in self._categorical_encoders.items()
            }
            with open(path, "w") as f:
                json.dump(encoders_dict, f, indent=2)
            saved_files["categorical_mappings"] = path

        # Save scaler
        if self._scaler:
            path = output_dir / config.SCALER_FILENAME
            with open(path, "wb") as f:
                pickle.dump(self._scaler, f)
            saved_files["scaler"] = path

        logger.info(f"Saved preprocessor components to {output_dir}")

        return saved_files

    @classmethod
    def load(cls, input_dir: Path) -> "DataPreprocessor":
        """
        Load preprocessor from disk.

        Args:
            input_dir: Directory containing saved files

        Returns:
            Loaded preprocessor instance
        """
        preprocessor = cls()

        # Load label encoder
        encoder_path = input_dir / config.ENCODER_FILENAME
        if encoder_path.exists():
            with open(encoder_path, "rb") as f:
                preprocessor._label_encoder = pickle.load(f)

        # Load feature list
        feature_path = input_dir / config.FEATURE_LIST_FILENAME
        if feature_path.exists():
            with open(feature_path, "r") as f:
                data = json.load(f)
                preprocessor._feature_names = data["feature_names"]

        preprocessor._is_fitted = True

        logger.info(f"Loaded preprocessor from {input_dir}")

        return preprocessor


# =============================================================================
# Convenience Functions
# =============================================================================
def preprocess_dataset(
    df: pd.DataFrame,
    target_column: str | None = None,
    scale_features: bool = False,
) -> PreprocessingResult:
    """
    Convenience function to preprocess dataset.

    Args:
        df: DataFrame to preprocess
        target_column: Name of target column
        scale_features: Whether to create scaler

    Returns:
        PreprocessingResult
    """
    preprocessor = DataPreprocessor(
        target_column=target_column,
        scale_features=scale_features,
    )
    return preprocessor.preprocess(df)


# =============================================================================
# Module Test
# =============================================================================
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format=config.LOG_FORMAT,
        datefmt=config.LOG_DATE_FORMAT,
    )

    from ml.data.load_data import DataLoader

    print("Preprocessing Module Test")
    print("=" * 50)

    # Load data
    loader = DataLoader()
    df = loader.load(config.DATASET_CONFIG.synthetic_v2)

    # Preprocess
    preprocessor = DataPreprocessor(scale_features=True)
    result = preprocessor.preprocess(df)

    print(f"\nPreprocessing Result:")
    print(f"  X shape: {result.X.shape}")
    print(f"  y shape: {result.y.shape}")
    print(f"  Features: {len(result.feature_names)}")
    print(f"  Classes: {len(result.label_encoder.classes_)}")
    print(f"  Class names: {result.label_encoder.classes_[:5]}...")
    print(f"  Scaler: {result.scaler is not None}")

    # Save components
    saved = preprocessor.save(config.MODELS_DIR)
    print(f"\nSaved files:")
    for name, path in saved.items():
        print(f"  {name}: {path.name}")