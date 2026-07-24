"""
Data Loading Module
Production-ready data loading with validation and error handling.
"""

import json
import logging
from pathlib import Path
from typing import Any, Final

import pandas as pd

from ml import config

# Configure module logger
logger: logging.Logger = logging.getLogger(__name__)


# =============================================================================
# Supported File Extensions
# =============================================================================
SUPPORTED_EXTENSIONS: Final[set[str]] = {".csv", ".tsv", ".json"}


# =============================================================================
# Custom Exceptions
# =============================================================================
class DataLoadError(Exception):
    """Raised when data loading fails."""
    pass


class DatasetNotFoundError(DataLoadError):
    """Raised when dataset file is not found."""
    pass


class InvalidDatasetError(DataLoadError):
    """Raised when dataset format is invalid."""
    pass


# =============================================================================
# Data Loader Class
# =============================================================================
class DataLoader:
    """
    Production-ready data loader with validation and error handling.

    Handles loading from various file formats (CSV, TSV, JSON) with
    automatic encoding detection and comprehensive validation.

    Example:
        >>> loader = DataLoader()
        >>> df = loader.load("path/to/dataset.csv")
        >>> print(f"Loaded {len(df)} rows")
    """

    def __init__(self, log_level: int = logging.INFO) -> None:
        """
        Initialize DataLoader.

        Args:
            log_level: Logging level (default: INFO)
        """
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(log_level)
        self._last_load_info: dict[str, Any] = {}

    @property
    def last_load_info(self) -> dict[str, Any]:
        """Get information about last successful load."""
        return self._last_load_info.copy()

    def load(
        self,
        file_path: str | Path,
        encoding: str = "utf-8-sig",
        low_memory: bool = False,
    ) -> pd.DataFrame:
        """
        Load dataset from file with automatic format detection.

        Args:
            file_path: Path to the dataset file
            encoding: File encoding (default: utf-8-sig for CSV with BOM)
            low_memory: Whether to use low memory mode for pandas

        Returns:
            Loaded DataFrame

        Raises:
            DatasetNotFoundError: If file doesn't exist
            InvalidDatasetError: If file format is unsupported or empty

        Example:
            >>> loader = DataLoader()
            >>> df = loader.load("data.csv")
            >>> df = loader.load("data.csv", encoding="latin-1")
        """
        path = Path(file_path)

        # Validate file exists
        if not path.exists():
            self._logger.error(f"Dataset not found: {path}")
            raise DatasetNotFoundError(f"Dataset not found: {path}")

        self._logger.info(f"Loading dataset: {path.name}")

        # Try to detect format and load
        df = self._load_by_extension(path, encoding, low_memory)

        # Validate loaded data
        self._validate_dataframe(df, path.name)

        # Store load info
        self._last_load_info = {
            "file_path": str(path),
            "file_name": path.name,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
        }

        self._logger.info(
            f"Loaded {len(df):,} rows × {len(df.columns)} columns from {path.name}"
        )

        return df

    def _load_by_extension(
        self,
        path: Path,
        encoding: str,
        low_memory: bool,
    ) -> pd.DataFrame:
        """Load file based on extension."""
        suffix = path.suffix.lower()

        if suffix not in SUPPORTED_EXTENSIONS:
            raise InvalidDatasetError(
                f"Unsupported file format: {suffix}. "
                f"Supported: {SUPPORTED_EXTENSIONS}"
            )

        if suffix == ".json":
            return self._load_json(path)
        else:
            return self._load_csv(path, encoding, low_memory)

    def _load_csv(
        self,
        path: Path,
        encoding: str,
        low_memory: bool,
    ) -> pd.DataFrame:
        """Load CSV/TSV with encoding fallback."""
        delimiter = "," if path.suffix.lower() == ".csv" else "\t"

        # Try multiple encodings
        encodings = [encoding, "utf-8", "latin-1"]

        for enc in encodings:
            try:
                df = pd.read_csv(
                    path,
                    delimiter=delimiter,
                    encoding=enc,
                    low_memory=low_memory,
                    encoding_errors="replace",
                )
                self._logger.debug(f"Successfully loaded with encoding: {enc}")
                return df
            except UnicodeDecodeError:
                continue
            except Exception as e:
                self._logger.warning(f"Failed with encoding {enc}: {e}")
                continue

        raise InvalidDatasetError(f"Could not decode file with any encoding")

    def _load_json(self, path: Path) -> pd.DataFrame:
        """Load JSON with automatic structure detection."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Handle different JSON structures
            if isinstance(data, list):
                # Array of objects
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                # Check for common wrapper keys
                for key in ["data", "records", "items", "results"]:
                    if key in data and isinstance(data[key], list):
                        df = pd.DataFrame(data[key])
                        self._logger.debug(f"Extracted data from '{key}' key")
                        return df
                # Single object or unknown structure
                df = pd.DataFrame([data])
            else:
                raise InvalidDatasetError(f"Unexpected JSON structure: {type(data)}")

            return df

        except json.JSONDecodeError as e:
            raise InvalidDatasetError(f"Invalid JSON: {e}")

    def _validate_dataframe(self, df: pd.DataFrame, source_name: str) -> None:
        """Validate loaded DataFrame meets minimum requirements."""
        if df is None or df.empty:
            raise InvalidDatasetError(f"Dataset {source_name} is empty")

        if len(df.columns) == 0:
            raise InvalidDatasetError(f"Dataset {source_name} has no columns")

        self._logger.debug(f"Validation passed for {source_name}")

    def load_with_validation(
        self,
        file_path: str | Path,
        required_columns: list[str] | None = None,
        min_rows: int = 1,
    ) -> pd.DataFrame:
        """
        Load dataset with additional validation checks.

        Args:
            file_path: Path to dataset
            required_columns: List of columns that must exist
            min_rows: Minimum number of rows required

        Returns:
            Loaded and validated DataFrame

        Raises:
            InvalidDatasetError: If validation fails
        """
        df = self.load(file_path)

        # Check minimum rows
        if len(df) < min_rows:
            raise InvalidDatasetError(
                f"Dataset has {len(df)} rows, minimum required: {min_rows}"
            )

        # Check required columns
        if required_columns:
            missing = set(required_columns) - set(df.columns)
            if missing:
                raise InvalidDatasetError(
                    f"Missing required columns: {missing}"
                )

        self._logger.info(f"Validation passed for {Path(file_path).name}")

        return df

    def get_column_info(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Get detailed information about DataFrame columns.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary with column information
        """
        info: dict[str, Any] = {
            "total_columns": len(df.columns),
            "numeric_columns": [],
            "categorical_columns": [],
            "columns_with_missing": [],
        }

        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                info["numeric_columns"].append(col)
            else:
                info["categorical_columns"].append(col)

            if df[col].isna().any():
                missing_pct = (df[col].isna().sum() / len(df)) * 100
                info["columns_with_missing"].append({
                    "column": col,
                    "missing_count": int(df[col].isna().sum()),
                    "missing_percent": round(missing_pct, 2),
                })

        return info


# =============================================================================
# Convenience Functions
# =============================================================================
def load_dataset(
    dataset_path: str | Path | None = None,
    log_level: int = logging.INFO,
) -> pd.DataFrame:
    """
    Convenience function to load dataset.

    Args:
        dataset_path: Path to dataset (uses default if None)
        log_level: Logging level

    Returns:
        Loaded DataFrame
    """
    path = Path(dataset_path) if dataset_path else config.DATASET_CONFIG.synthetic_v2

    if not path.exists():
        raise DatasetNotFoundError(f"Dataset not found: {path}")

    loader = DataLoader(log_level=log_level)
    return loader.load(path)


def get_default_dataset_path() -> Path:
    """Get the default dataset path."""
    return config.DATASET_CONFIG.synthetic_v2


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

    print("DataLoader Module Test")
    print("=" * 50)

    # Test loading default dataset
    try:
        loader = DataLoader()
        df = loader.load(config.DATASET_CONFIG.synthetic_v2)

        print(f"\nDataset loaded successfully!")
        print(f"  Rows: {len(df):,}")
        print(f"  Columns: {len(df.columns)}")

        # Get column info
        info = loader.get_column_info(df)
        print(f"\nColumn Info:")
        print(f"  Numeric: {len(info['numeric_columns'])}")
        print(f"  Categorical: {len(info['categorical_columns'])}")

        # Check target column
        target = config.TARGET_COLUMN
        if target in df.columns:
            print(f"\nTarget column '{target}':")
            print(f"  Unique values: {df[target].nunique()}")
            print(f"  Distribution:\n{df[target].value_counts().head()}")
        else:
            # Try to find similar column
            for col in df.columns:
                if "career" in col.lower():
                    print(f"\nFound potential target: {col}")
                    print(f"  Unique values: {df[col].nunique()}")

    except Exception as e:
        print(f"Error: {e}")