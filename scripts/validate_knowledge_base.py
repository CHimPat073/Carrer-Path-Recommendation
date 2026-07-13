"""Validation utilities for the CareerPilot AI knowledge base pipeline.

This module validates generated knowledge payloads before they are exported.
It focuses on structural correctness and sanity checks rather than dataset
inspection or model training.
"""

from __future__ import annotations

from typing import Any

from scripts.utils import configure_logging

logger = configure_logging("validate_knowledge_base")


def validate_knowledge_base(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a generated knowledge base payload.

    Input:
        payload (dict[str, Any]): Knowledge payload produced by the pipeline.

    Output:
        dict[str, Any]: Validation report describing pass/fail status.

    Purpose:
        Catch missing fields, duplicate entries, invalid ranges, and malformed overlap/correlation data.
    """
    report: dict[str, Any] = {
        "valid": True,
        "checks": {},
    }

    report["checks"]["missing_fields"] = _check_missing_fields(payload)
    report["checks"]["duplicate_careers"] = _check_duplicate_careers(payload)
    report["checks"]["duplicate_skills"] = _check_duplicate_skills(payload)
    report["checks"]["invalid_ranges"] = _check_invalid_ranges(payload)
    report["checks"]["invalid_overlaps"] = _check_invalid_overlaps(payload)
    report["checks"]["invalid_correlations"] = _check_invalid_correlations(payload)

    report["valid"] = all(
        check.get("valid", True) for check in report["checks"].values()
    )
    return report


def _check_missing_fields(payload: dict[str, Any]) -> dict[str, Any]:
    """Verify the core knowledge sections are present.

    Input:
        payload (dict[str, Any]): Knowledge payload.

    Output:
        dict[str, Any]: Check result.

    Purpose:
        Ensure required top-level sections are not missing.
    """
    required_sections = [
        "career_personas",
        "feature_ranges",
        "career_overlap",
        "user_personas",
        "noise_rules",
        "correlation_rules",
    ]
    missing = [section for section in required_sections if section not in payload]
    return {"valid": not missing, "missing": missing}


def _check_duplicate_careers(payload: dict[str, Any]) -> dict[str, Any]:
    """Ensure careers are unique within the persona section.

    Input:
        payload (dict[str, Any]): Knowledge payload.

    Output:
        dict[str, Any]: Check result.

    Purpose:
        Prevent duplicate career definitions.
    """
    personas = payload.get("career_personas", {})
    if isinstance(personas, dict):
        careers = list(personas.keys())
    else:
        careers = []
    duplicates = [career for career in careers if careers.count(career) > 1]
    return {"valid": not duplicates, "duplicates": duplicates}


def _check_duplicate_skills(payload: dict[str, Any]) -> dict[str, Any]:
    """Ensure skills are unique per career in the persona data.

    Input:
        payload (dict[str, Any]): Knowledge payload.

    Output:
        dict[str, Any]: Check result.

    Purpose:
        Avoid duplicate skills in persona definitions.
    """
    personas = payload.get("career_personas", {})
    duplicate_skills: list[dict[str, Any]] = []
    if isinstance(personas, dict):
        for career, profile in personas.items():
            if not isinstance(profile, dict):
                continue
            for field in ["required_technical_skills", "nice_to_have_skills", "soft_skills"]:
                values = profile.get(field, [])
                if isinstance(values, list):
                    if len(values) != len(set(values)):
                        duplicate_skills.append({"career": career, "field": field})
    return {"valid": not duplicate_skills, "duplicates": duplicate_skills}


def _check_invalid_ranges(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate feature range statistics.

    Input:
        payload (dict[str, Any]): Knowledge payload.

    Output:
        dict[str, Any]: Check result.

    Purpose:
        Ensure range values are numeric and have sensible bounds.
    """
    feature_ranges = payload.get("feature_ranges", {})
    invalid: list[dict[str, Any]] = []
    if isinstance(feature_ranges, dict):
        for career, values in feature_ranges.items():
            if not isinstance(values, dict):
                continue
            for feature_name, stats in values.get("features", {}).items():
                if not isinstance(stats, dict):
                    continue
                minimum = stats.get("min")
                maximum = stats.get("max")
                average = stats.get("average")
                if not isinstance(minimum, (int, float)) or not isinstance(maximum, (int, float)):
                    invalid.append({"career": career, "feature": feature_name, "reason": "non_numeric_bounds"})
                    continue
                if minimum > maximum:
                    invalid.append({"career": career, "feature": feature_name, "reason": "min_exceeds_max"})
                    continue
                if average is not None and not isinstance(average, (int, float)):
                    invalid.append({"career": career, "feature": feature_name, "reason": "non_numeric_average"})
    return {"valid": not invalid, "invalid": invalid}


def _check_invalid_overlaps(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate overlap structures.

    Input:
        payload (dict[str, Any]): Knowledge payload.

    Output:
        dict[str, Any]: Check result.

    Purpose:
        Ensure overlap values are within a valid similarity range.
    """
    overlap_payload = payload.get("career_overlap", {})
    invalid: list[dict[str, Any]] = []
    matrix = overlap_payload.get("similarity_matrix", {}) if isinstance(overlap_payload, dict) else {}
    for career, values in matrix.items():
        if not isinstance(values, dict):
            continue
        for other, score in values.items():
            if not isinstance(score, (int, float)):
                invalid.append({"career": career, "other": other, "reason": "non_numeric_score"})
            elif not 0 <= float(score) <= 1:
                invalid.append({"career": career, "other": other, "reason": "score_out_of_range"})
    return {"valid": not invalid, "invalid": invalid}


def _check_invalid_correlations(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate correlation rule payloads.

    Input:
        payload (dict[str, Any]): Knowledge payload.

    Output:
        dict[str, Any]: Check result.

    Purpose:
        Ensure correlation strengths are numeric and within expected bounds.
    """
    correlation_payload = payload.get("correlation_rules", {})
    invalid: list[dict[str, Any]] = []
    rules = correlation_payload.get("rules", []) if isinstance(correlation_payload, dict) else []
    for rule in rules:
        strength = rule.get("strength")
        if not isinstance(strength, (int, float)):
            invalid.append({"reason": "non_numeric_strength", "rule": rule})
            continue
        if not 0 <= float(strength) <= 1:
            invalid.append({"reason": "strength_out_of_range", "rule": rule})
    return {"valid": not invalid, "invalid": invalid}
