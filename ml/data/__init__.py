"""
Data Module
Production-ready data loading, preprocessing, and splitting.
"""

from .load_data import DataLoader, DatasetNotFoundError, load_dataset
from .preprocess import (
    DataPreprocessor,
    PreprocessingResult,
    ValidationError,
    ValidationResult,
    preprocess_dataset,
)
from .split import (
    DataSplitter,
    SplitResult,
    get_train_val_test,
    split_data,
)

__all__ = [
    # Load data
    "DataLoader",
    "DatasetNotFoundError",
    "load_dataset",
    # Preprocess
    "DataPreprocessor",
    "PreprocessingResult",
    "ValidationError",
    "ValidationResult",
    "preprocess_dataset",
    # Split
    "DataSplitter",
    "SplitResult",
    "split_data",
    "get_train_val_test",
]