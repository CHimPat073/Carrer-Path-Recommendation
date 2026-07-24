"""
Data Split Module
Production-ready train/validation/test split with stratification.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml import config

# Configure module logger
logger: logging.Logger = logging.getLogger(__name__)


# =============================================================================
# Split Ratios
# =============================================================================
SPLIT_RATIOS: Final[dict[str, float]] = {
    "train": config.SPLIT_CONFIG.train_ratio,
    "val": config.SPLIT_CONFIG.val_ratio,
    "test": config.SPLIT_CONFIG.test_ratio,
}


# =============================================================================
# Split Result
# =============================================================================
@dataclass
class SplitResult:
    """Result of data split operation."""
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray

    # Additional metadata
    original_size: int = 0
    train_size: int = 0
    val_size: int = 0
    test_size: int = 0
    num_features: int = 0
    num_classes: int = 0
    class_distribution_train: dict[int, int] = field(default_factory=dict)
    class_distribution_val: dict[int, int] = field(default_factory=dict)
    class_distribution_test: dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "original_size": self.original_size,
            "train_size": self.train_size,
            "val_size": self.val_size,
            "test_size": self.test_size,
            "num_features": self.num_features,
            "num_classes": self.num_classes,
            "ratios": {
                "train": self.train_size / self.original_size if self.original_size else 0,
                "val": self.val_size / self.original_size if self.original_size else 0,
                "test": self.test_size / self.original_size if self.original_size else 0,
            },
            "class_distribution": {
                "train": {int(k): int(v) for k, v in self.class_distribution_train.items()},
                "val": {int(k): int(v) for k, v in self.class_distribution_val.items()},
                "test": {int(k): int(v) for k, v in self.class_distribution_test.items()},
            },
        }

    @property
    def splits(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Get all splits as dictionary."""
        return {
            "train": (self.X_train, self.y_train),
            "val": (self.X_val, self.y_val),
            "test": (self.X_test, self.y_test),
        }

    def get_split(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Get a specific split by name."""
        if name == "train":
            return self.X_train, self.y_train
        elif name == "val":
            return self.X_val, self.y_val
        elif name == "test":
            return self.X_test, self.y_test
        else:
            raise ValueError(f"Unknown split: {name}. Use 'train', 'val', or 'test'")

    def save(self, output_dir: Path) -> dict[str, Path]:
        """
        Save split data to numpy files.

        Args:
            output_dir: Directory to save files

        Returns:
            Dictionary of saved file paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        saved: dict[str, Path] = {}

        # Save arrays
        for name in ["train", "val", "test"]:
            X, y = self.get_split(name)
            X_path = output_dir / f"X_{name}.npy"
            y_path = output_dir / f"y_{name}.npy"

            np.save(X_path, X)
            np.save(y_path, y)

            saved[f"X_{name}"] = X_path
            saved[f"y_{name}"] = y_path

        # Save metadata
        meta_path = output_dir / "split_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        saved["metadata"] = meta_path

        logger.info(f"Saved split data to {output_dir}")

        return saved

    @classmethod
    def load(cls, input_dir: Path) -> "SplitResult":
        """
        Load split data from numpy files.

        Args:
            input_dir: Directory containing saved files

        Returns:
            Loaded SplitResult
        """
        splits = {}

        for name in ["train", "val", "test"]:
            X_path = input_dir / f"X_{name}.npy"
            y_path = input_dir / f"y_{name}.npy"

            splits[name] = (
                np.load(X_path),
                np.load(y_path),
            )

        # Load metadata
        meta_path = input_dir / "split_metadata.json"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)
        else:
            meta = {}

        return cls(
            X_train=splits["train"][0],
            X_val=splits["val"][0],
            X_test=splits["test"][0],
            y_train=splits["train"][1],
            y_val=splits["val"][1],
            y_test=splits["test"][1],
            train_indices=np.arange(len(splits["train"][1])),
            val_indices=np.arange(len(splits["val"][1])),
            test_indices=np.arange(len(splits["test"][1])),
            original_size=meta.get("original_size", 0),
            train_size=meta.get("train_size", 0),
            val_size=meta.get("val_size", 0),
            test_size=meta.get("test_size", 0),
        )


# =============================================================================
# Data Splitter Class
# =============================================================================
class DataSplitter:
    """
    Production-ready data splitter with stratification.

    Splits data into train/validation/test sets with:
    - Configurable ratios (default: 70/15/15)
    - Stratification by target variable
    - Deterministic random state
    - Comprehensive metadata

    Example:
        >>> splitter = DataSplitter()
        >>> result = splitter.split(X, y)
        >>> X_train, y_train = result.get_split("train")
    """

    def __init__(
        self,
        train_ratio: float = config.SPLIT_CONFIG.train_ratio,
        val_ratio: float = config.SPLIT_CONFIG.val_ratio,
        test_ratio: float = config.SPLIT_CONFIG.test_ratio,
        random_state: int = config.SPLIT_CONFIG.random_state,
        stratify: bool = True,
    ) -> None:
        """
        Initialize data splitter.

        Args:
            train_ratio: Ratio of training data (default: 0.70)
            val_ratio: Ratio of validation data (default: 0.15)
            test_ratio: Ratio of test data (default: 0.15)
            random_state: Random seed for reproducibility
            stratify: Whether to stratify by target
        """
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state
        self.stratify = stratify

        # Validate ratios
        total = train_ratio + val_ratio + test_ratio
        if not abs(total - 1.0) < 1e-6:
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total}"
            )

        self._logger = logging.getLogger(__name__)

    def split(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> SplitResult:
        """
        Split data into train/validation/test sets.

        Args:
            X: Feature matrix (DataFrame or ndarray)
            y: Target vector (Series or ndarray)

        Returns:
            SplitResult with all splits and metadata

        Example:
            >>> splitter = DataSplitter()
            >>> result = splitter.split(X, y)
            >>> X_train, y_train = result.get_split("train")
            >>> X_val, y_val = result.get_split("val")
            >>> X_test, y_test = result.get_split("test")
        """
        self._logger.info("Starting data split")

        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        original_size = len(X)
        num_features = X.shape[1] if len(X.shape) > 1 else 1
        num_classes = len(np.unique(y))

        self._logger.info(
            f"Splitting {original_size:,} samples into "
            f"{self.train_ratio:.0%} train / {self.val_ratio:.0%} val / {self.test_ratio:.0%} test"
        )

        # First split: train+val vs test (85% vs 15%)
        test_size = self.test_ratio

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y if self.stratify else None,
        )

        # Second split: train vs val (from 85%, get 70% train / 15% val = 70/15 split of total)
        # So val gets 15/85 = ~17.65% of train_val split
        val_size = self.val_ratio / (self.train_ratio + self.val_ratio)

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=val_size,
            random_state=self.random_state,
            stratify=y_train_val if self.stratify else None,
        )

        # Get indices for reference
        train_indices = np.arange(len(y_train))
        val_indices = np.arange(len(y_val))
        test_indices = np.arange(len(y_test))

        # Calculate class distributions
        class_dist_train = self._get_class_distribution(y_train)
        class_dist_val = self._get_class_distribution(y_val)
        class_dist_test = self._get_class_distribution(y_test)

        result = SplitResult(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            original_size=original_size,
            train_size=len(y_train),
            val_size=len(y_val),
            test_size=len(y_test),
            num_features=num_features,
            num_classes=num_classes,
            class_distribution_train=class_dist_train,
            class_distribution_val=class_dist_val,
            class_distribution_test=class_dist_test,
        )

        self._logger.info(
            f"Split complete: train={len(y_train):,}, "
            f"val={len(y_val):,}, test={len(y_test):,}"
        )

        # Log class distribution
        self._log_class_distribution(result)

        return result

    def _get_class_distribution(self, y: np.ndarray) -> dict[int, int]:
        """Get class distribution as counts."""
        unique, counts = np.unique(y, return_counts=True)
        return {int(u): int(c) for u, c in zip(unique, counts)}

    def _log_class_distribution(self, result: SplitResult) -> None:
        """Log class distribution for all splits."""
        self._logger.debug("Class distribution:")
        self._logger.debug(f"  Train: {result.class_distribution_train}")
        self._logger.debug(f"  Val:   {result.class_distribution_val}")
        self._logger.debug(f"  Test:  {result.class_distribution_test}")

    # =========================================================================
    # Convenience Methods
    # =========================================================================
    def split_dataframe(
        self,
        df: pd.DataFrame,
        target_column: str,
    ) -> SplitResult:
        """
        Split DataFrame directly using target column.

        Args:
            df: DataFrame to split
            target_column: Name of target column

        Returns:
            SplitResult
        """
        y = df[target_column].values
        X = df.drop(columns=[target_column])

        return self.split(X, y)

    def split_xy(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> SplitResult:
        """
        Split X and y directly.

        Args:
            X: Feature DataFrame
            y: Target Series

        Returns:
            SplitResult
        """
        return self.split(X, y)


# =============================================================================
# Convenience Functions
# =============================================================================
def split_data(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    train_ratio: float = config.SPLIT_CONFIG.train_ratio,
    val_ratio: float = config.SPLIT_CONFIG.val_ratio,
    test_ratio: float = config.SPLIT_CONFIG.test_ratio,
    random_state: int = config.SPLIT_CONFIG.random_state,
    stratify: bool = True,
) -> SplitResult:
    """
    Convenience function to split data.

    Args:
        X: Feature matrix
        y: Target vector
        train_ratio: Training ratio
        val_ratio: Validation ratio
        test_ratio: Test ratio
        random_state: Random seed
        stratify: Whether to stratify

    Returns:
        SplitResult
    """
    splitter = DataSplitter(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
        stratify=stratify,
    )
    return splitter.split(X, y)


def get_train_val_test(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
) -> tuple:
    """
    Convenience function returning splits as separate tuples.

    Args:
        X: Feature matrix
        y: Target vector

    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test)

    Example:
        >>> X_train, X_val, X_test, y_train, y_val, y_test = get_train_val_test(X, y)
    """
    result = split_data(X, y)
    return result.X_train, result.X_val, result.X_test, result.y_train, result.y_val, result.y_test


# =============================================================================
# Module Test
# =============================================================================
if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format=config.LOG_FORMAT,
        datefmt=config.LOG_DATE_FORMAT,
    )

    # Add parent to path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from ml.data.load_data import DataLoader
    from ml.data.preprocess import DataPreprocessor

    print("Split Module Test")
    print("=" * 50)

    # Load and preprocess data
    loader = DataLoader()
    df = loader.load(config.DATASET_CONFIG.synthetic_v2)

    preprocessor = DataPreprocessor()
    result = preprocessor.preprocess(df)

    # Split data
    splitter = DataSplitter()
    split_result = splitter.split(result.X, result.y)

    print(f"\nSplit Result:")
    print(f"  Original: {split_result.original_size:,}")
    print(f"  Train: {split_result.train_size:,} ({split_result.train_size/split_result.original_size*100:.1f}%)")
    print(f"  Val:   {split_result.val_size:,} ({split_result.val_size/split_result.original_size*100:.1f}%)")
    print(f"  Test:  {split_result.test_size:,} ({split_result.test_size/split_result.original_size*100:.1f}%)")

    # Test get_split
    X_train, y_train = split_result.get_split("train")
    print(f"\nUsing get_split('train'):")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  y_train shape: {y_train.shape}")

    # Save split data
    saved = split_result.save(config.MODELS_DIR / "splits")
    print(f"\nSaved files:")
    for name, path in saved.items():
        print(f"  {name}: {path.name}")