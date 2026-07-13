"""Feature range generation utilities.

This module creates statistically plausible feature ranges for technical and
soft-skill dimensions associated with career personas. It returns Python
structures rather than exporting JSON files.
"""

from __future__ import annotations

from typing import Any

from scripts.utils import configure_logging

logger = configure_logging("generate_feature_ranges")


def generate_feature_ranges(career_names: list[str]) -> dict[str, dict[str, Any]]:
    """Generate feature ranges for every provided career.

    Input:
        career_names (list[str]): Careers for which ranges should be estimated.

    Output:
        dict[str, dict[str, Any]]: Mapping of each career to its feature range payload.

    Purpose:
        Produce structured feature statistics such as min, max, average, and recommended range.
    """
    feature_ranges: dict[str, dict[str, Any]] = {}
    for career in career_names:
        feature_ranges[career] = _build_feature_range_payload(career)
    return feature_ranges


def _build_feature_range_payload(career: str) -> dict[str, Any]:
    """Generate a feature range payload for a single career.

    Input:
        career (str): Career name.

    Output:
        dict[str, Any]: Feature statistics dictionary.

    Purpose:
        Encapsulate the career-specific range generation heuristics.
    """
    base_profile = {
        "Years Experience": {"min": 1, "max": 15, "average": 6, "recommended_range": [2, 10]},
        "Projects Completed": {"min": 2, "max": 30, "average": 10, "recommended_range": [4, 20]},
        "Certifications": {"min": 0, "max": 8, "average": 2, "recommended_range": [1, 4]},
        "Python": {"min": 1, "max": 10, "average": 5, "recommended_range": [4, 8]},
        "Java": {"min": 1, "max": 10, "average": 5, "recommended_range": [3, 7]},
        "JavaScript": {"min": 1, "max": 10, "average": 5, "recommended_range": [4, 8]},
        "SQL": {"min": 1, "max": 10, "average": 5, "recommended_range": [4, 8]},
        "Machine Learning": {"min": 1, "max": 10, "average": 4, "recommended_range": [3, 7]},
        "Deep Learning": {"min": 1, "max": 10, "average": 3, "recommended_range": [2, 6]},
        "Cloud": {"min": 1, "max": 10, "average": 4, "recommended_range": [3, 7]},
        "DevOps": {"min": 1, "max": 10, "average": 4, "recommended_range": [3, 7]},
        "Cybersecurity": {"min": 1, "max": 10, "average": 4, "recommended_range": [3, 6]},
        "Data Analysis": {"min": 1, "max": 10, "average": 5, "recommended_range": [4, 7]},
        "Database": {"min": 1, "max": 10, "average": 5, "recommended_range": [4, 8]},
        "Networking": {"min": 1, "max": 10, "average": 4, "recommended_range": [3, 6]},
        "Mobile": {"min": 1, "max": 10, "average": 3, "recommended_range": [2, 6]},
        "Game Dev": {"min": 1, "max": 10, "average": 3, "recommended_range": [2, 6]},
        "Testing": {"min": 1, "max": 10, "average": 5, "recommended_range": [4, 8]},
        "Business Analysis": {"min": 1, "max": 10, "average": 5, "recommended_range": [4, 7]},
        "Product Management": {"min": 1, "max": 10, "average": 4, "recommended_range": [3, 7]},
        "UI Design": {"min": 1, "max": 10, "average": 5, "recommended_range": [4, 8]},
        "UX Research": {"min": 1, "max": 10, "average": 4, "recommended_range": [3, 7]},
        "Communication": {"min": 1, "max": 10, "average": 5, "recommended_range": [4, 8]},
        "Leadership": {"min": 1, "max": 10, "average": 4, "recommended_range": [3, 7]},
        "Problem Solving": {"min": 1, "max": 10, "average": 5, "recommended_range": [4, 8]},
        "Teamwork": {"min": 1, "max": 10, "average": 5, "recommended_range": [4, 8]},
        "Agile": {"min": 1, "max": 10, "average": 4, "recommended_range": [3, 7]},
        "Research": {"min": 1, "max": 10, "average": 4, "recommended_range": [3, 7]},
    }

    profile_adjustments: dict[str, dict[str, int]] = {
        "Data Scientist": {"Python": 2, "Machine Learning": 2, "Data Analysis": 2, "Research": 1},
        "ML Engineer": {"Python": 2, "Machine Learning": 2, "Deep Learning": 2, "Research": 1},
        "Data Engineer": {"SQL": 2, "Database": 2, "Cloud": 1},
        "Backend Developer": {"Python": 1, "SQL": 2, "Database": 2},
        "Frontend Developer": {"JavaScript": 2, "UI Design": 2, "UX Research": 1},
        "Full Stack Developer": {"Python": 1, "JavaScript": 2, "Cloud": 1},
        "Mobile App Developer": {"Mobile": 2, "UI Design": 1, "Testing": 1},
        "DevOps Engineer": {"Cloud": 2, "DevOps": 2, "Networking": 1},
        "Cloud Engineer": {"Cloud": 2, "DevOps": 1, "Networking": 1},
        "Cyber Security Analyst": {"Cybersecurity": 2, "Networking": 1, "Problem Solving": 1},
        "Ethical Hacker": {"Cybersecurity": 2, "Networking": 1, "Testing": 1},
        "Network Engineer": {"Networking": 2, "Cloud": 1, "Cybersecurity": 1},
        "UI/UX Designer": {"UI Design": 2, "UX Research": 2, "Communication": 1},
        "Product Manager": {"Product Management": 2, "Leadership": 2, "Communication": 2, "Agile": 1},
        "Business Analyst": {"Business Analysis": 2, "Communication": 1, "Problem Solving": 1},
        "QA Engineer": {"Testing": 2, "Problem Solving": 1, "Communication": 1},
        "Database Administrator": {"Database": 2, "SQL": 1, "Problem Solving": 1},
        "Game Developer": {"Game Dev": 2, "JavaScript": 1, "Python": 1, "UI Design": 1},
    }

    adjusted = {}
    for feature_name, stats in base_profile.items():
        adjusted[feature_name] = dict(stats)
        adjustment = profile_adjustments.get(career, {}).get(feature_name, 0)
        if adjustment:
            adjusted[feature_name]["average"] = min(10, max(1, stats["average"] + adjustment))
            adjusted[feature_name]["recommended_range"] = [
                max(1, stats["recommended_range"][0] + adjustment - 1),
                min(10, stats["recommended_range"][1] + adjustment),
            ]
    return {"features": adjusted, "career": career}
