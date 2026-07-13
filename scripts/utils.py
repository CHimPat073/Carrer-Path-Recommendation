"""Shared utility helpers for the CareerPilot AI knowledge base pipeline.

This module centralizes reusable logic for logging, text normalization,
path handling, and data validation so the pipeline components remain
consistent and easy to test.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def configure_logging(name: str = "careerpilot") -> logging.Logger:
    """Create and return a configured logger.

    Input:
        name (str): The logger name.

    Output:
        logging.Logger: A configured logger instance.

    Purpose:
        Ensure all pipeline modules emit structured logs with consistent
        formatting and severity levels.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def normalize_text(value: Any) -> str:
    """Normalize a string value into a consistent lowercase form.

    Input:
        value (Any): A string-like value.

    Output:
        str: Normalized text with whitespace collapsed.

    Purpose:
        Standardize free-form text before skill and career matching.
    """
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_career_name(value: Any) -> str:
    """Normalize a career name into a consistent title-style label.

    Input:
        value (Any): Career name or role label.

    Output:
        str: A normalized career string.

    Purpose:
        Reduce alias variations such as 'software engineer' and 'sw engineer'.
    """
    normalized = normalize_text(value)
    if not normalized:
        return ""
    aliases: dict[str, str] = {
        "software engineer": "Software Engineer",
        "backend developer": "Backend Developer",
        "frontend developer": "Frontend Developer",
        "full stack developer": "Full Stack Developer",
        "mobile app developer": "Mobile App Developer",
        "data scientist": "Data Scientist",
        "machine learning engineer": "ML Engineer",
        "ml engineer": "ML Engineer",
        "data engineer": "Data Engineer",
        "ai research engineer": "AI Research Engineer",
        "devops engineer": "DevOps Engineer",
        "cloud engineer": "Cloud Engineer",
        "cyber security analyst": "Cyber Security Analyst",
        "ethical hacker": "Ethical Hacker",
        "network engineer": "Network Engineer",
        "ui ux designer": "UI/UX Designer",
        "product manager": "Product Manager",
        "business analyst": "Business Analyst",
        "quality assurance engineer": "QA Engineer",
        "qa engineer": "QA Engineer",
        "database administrator": "Database Administrator",
        "game developer": "Game Developer",
    }
    return aliases.get(normalized, normalized.title())


def normalize_skill_name(value: Any) -> str:
    """Normalize a skill name into a canonical form.

    Input:
        value (Any): Raw skill text.

    Output:
        str: Normalized skill name.

    Purpose:
        Collapse variations and formatting differences in skill labels.
    """
    normalized = normalize_text(value)
    if not normalized:
        return ""
    return normalized.title()


def ensure_output_directory(path: str | Path) -> Path:
    """Create an output directory if it does not already exist.

    Input:
        path (str | Path): Directory path to ensure.

    Output:
        Path: Absolute path object to the created directory.

    Purpose:
        Simplify writing generated artifacts to disk.
    """
    output_path = Path(path).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def read_dataframe(file_path: str | Path) -> pd.DataFrame:
    """Read a CSV or Excel file into a pandas DataFrame.

    Input:
        file_path (str | Path): Dataset file path.

    Output:
        pd.DataFrame: Parsed dataset.

    Purpose:
        Provide a single point for dataset loading with consistent handling.
    """
    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file format: {suffix}")


def to_dict_list(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame to a list of dictionaries.

    Input:
        dataframe (pd.DataFrame): Input table.

    Output:
        list[dict[str, Any]]: Row-wise dictionary representation.

    Purpose:
        Make DataFrame outputs easier to compose into JSON-ready payloads.
    """
    return dataframe.where(pd.notna(dataframe), None).to_dict(orient="records")


def flatten_list(values: Iterable[Any]) -> list[str]:
    """Flatten nested values into a list of strings.

    Input:
        values (Iterable[Any]): A collection of values.

    Output:
        list[str]: Clean string values.

    Purpose:
        Support safe extraction of lists from mixed inputs.
    """
    return [normalize_text(str(item)) for item in values if item is not None]
