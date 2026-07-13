"""Skill extraction pipeline for CareerPilot AI.

This module provides reusable functions to clean and harmonize job-related
text from multiple data sources, then derive career-to-skill mappings.
The implementation is intentionally modular and avoids persisting any files.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from scripts.utils import configure_logging, normalize_career_name, normalize_skill_name

logger = configure_logging("extract_skills")


def extract_skills_from_dataframe(
    dataframe: pd.DataFrame,
    *,
    career_column: str | None = None,
    skill_column: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    """Extract and normalize skills from a tabular dataset.

    Input:
        dataframe (pd.DataFrame): Raw dataset with career and skill information.
        career_column (str | None): Column containing career labels.
        skill_column (str | None): Column containing skill text.

    Output:
        tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
            - Cleaned skills DataFrame
            - Aggregated career-skill mapping DataFrame
            - Dictionary mapping careers to normalized skills

    Purpose:
        Create reusable cleaning and normalization logic for multiple datasets.
    """
    if dataframe.empty:
        logger.warning("Received an empty DataFrame; returning empty outputs")
        return pd.DataFrame(), pd.DataFrame(), {}

    career_key = career_column or "career"
    skill_key = skill_column or "skills"

    if career_key not in dataframe.columns:
        raise KeyError(f"Career column '{career_key}' not found")
    if skill_key not in dataframe.columns:
        raise KeyError(f"Skill column '{skill_key}' not found")

    cleaned = dataframe[[career_key, skill_key]].copy()
    cleaned[career_key] = cleaned[career_key].apply(normalize_career_name)
    cleaned[skill_key] = cleaned[skill_key].apply(_clean_skill_text)

    skill_rows: list[dict[str, Any]] = []
    career_skill_map: dict[str, list[str]] = {}
    for _, row in cleaned.iterrows():
        career = str(row[career_key]).strip()
        skill_text = str(row[skill_key]).strip()
        if not career or not skill_text:
            continue

        skills = _split_skills(skill_text)
        for skill in skills:
            normalized_skill = normalize_skill_name(skill)
            if not normalized_skill:
                continue
            skill_rows.append({"career": career, "skill": normalized_skill})
            career_skill_map.setdefault(career, []).append(normalized_skill)

    skill_df = pd.DataFrame(skill_rows)
    if not skill_df.empty:
        skill_df = skill_df.drop_duplicates().reset_index(drop=True)

    aggregated_df = (
        skill_df.groupby("career", as_index=False)["skill"]
        .agg(lambda values: list(dict.fromkeys(values)))
        .rename(columns={"skill": "skills"})
    )

    return cleaned, aggregated_df, career_skill_map


def calculate_skill_frequency(skill_dataframe: pd.DataFrame) -> pd.DataFrame:
    """Calculate the frequency of each skill.

    Input:
        skill_dataframe (pd.DataFrame): DataFrame with at least a 'skill' column.

    Output:
        pd.DataFrame: A DataFrame with skill counts.

    Purpose:
        Highlight the most common skills across the extracted data.
    """
    if skill_dataframe.empty or "skill" not in skill_dataframe.columns:
        return pd.DataFrame(columns=["skill", "frequency"])

    frequencies = skill_dataframe["skill"].value_counts().reset_index()
    frequencies.columns = ["skill", "frequency"]
    return frequencies.sort_values("frequency", ascending=False).reset_index(drop=True)


def _clean_skill_text(value: Any) -> str:
    """Clean and normalize skill text.

    Input:
        value (Any): Raw skill text.

    Output:
        str: Cleaned skill text.

    Purpose:
        Remove punctuation and split combined text into meaningful tokens.
    """
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9\s,;]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _split_skills(text: str) -> list[str]:
    """Split a skill text block into individual skills.

    Input:
        text (str): Combined skill text.

    Output:
        list[str]: Extracted skills.

    Purpose:
        Support different delimiters and common formatting styles.
    """
    if not text:
        return []
    separators = [",", ";", "|", "/", " and "]
    for separator in separators:
        if separator in text:
            parts = [part.strip() for part in text.split(separator)]
            return [part for part in parts if part]
    return [text]
