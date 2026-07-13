import re
from pathlib import Path
from typing import Any

from .constants import CAREER_ALIASES, CAREER_LABEL_COLUMNS, STANDARD_CAREERS, TEXT_COLUMNS


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"[^a-z0-9]+", " ", str(value).strip().lower()).strip()


def normalize_career_label(value: Any) -> str:
    if value is None:
        return "Other"

    normalized = normalize_text(value)
    if not normalized:
        return "Other"

    for alias, career in CAREER_ALIASES.items():
        if normalized == normalize_text(alias):
            return career

    for career in STANDARD_CAREERS:
        if normalized == normalize_text(career):
            return career

    for alias, career in CAREER_ALIASES.items():
        if alias in normalized or normalized in alias:
            return career

    return "Other"


def detect_label_column(row: dict[str, Any]) -> str | None:
    for column in CAREER_LABEL_COLUMNS:
        if column in row and str(row.get(column, "")).strip():
            return column
    return None


def build_text(row: dict[str, Any]) -> str:
    text_chunks: list[str] = []
    for key, value in row.items():
        if key in {"source_file", "raw_label", "career_label", "text"}:
            continue
        if value is None:
            continue
        text_value = str(value).strip()
        if text_value:
            text_chunks.append(text_value)

    for column in TEXT_COLUMNS:
        value = row.get(column)
        if value is None:
            continue
        text_value = str(value).strip()
        if text_value:
            text_chunks.append(text_value)

    return " ".join(text_chunks).strip()


def standardize_row(row: dict[str, Any], source_file: str) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in row.items():
        if value is None:
            cleaned[key] = ""
        elif isinstance(value, (int, float)):
            cleaned[key] = value
        else:
            cleaned[key] = str(value).strip()

    label_column = detect_label_column(cleaned)
    raw_label = cleaned.get(label_column, "") if label_column else ""

    cleaned["source_file"] = source_file
    cleaned["raw_label"] = raw_label
    cleaned["career_label"] = normalize_career_label(raw_label)
    cleaned["text"] = build_text(cleaned)
    return cleaned


def clean_missing_values(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned_rows: list[dict[str, Any]] = []
    for row in rows:
        if not row:
            continue

        normalized_row = {}
        for key, value in row.items():
            if value is None:
                normalized_row[key] = ""
            elif isinstance(value, str):
                normalized_row[key] = value.strip()
            else:
                normalized_row[key] = value

        if not any(str(value).strip() for value in normalized_row.values()):
            continue

        if not normalized_row.get("career_label"):
            normalized_row["career_label"] = "Other"

        cleaned_rows.append(normalized_row)

    return cleaned_rows


def merge_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, str], ...]] = set()

    for row in rows:
        signature = tuple(sorted((str(key), str(value if value is not None else "")) for key, value in row.items()))
        if signature in seen:
            continue
        seen.add(signature)
        merged.append(row)

    return merged


def save_dataset(rows: list[dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base_columns = ["source_file", "raw_label", "career_label", "text"]
    extra_columns = [
        column
        for column in sorted({str(key) for row in rows for key in row.keys() if key is not None})
        if column not in base_columns
    ]
    fieldnames = base_columns + extra_columns

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = __import__("csv").DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    return output_path
