"""Correlation rule generation utilities.

This module creates structured feature dependency rules that can be used to
simulate realistic relationships between technical and soft-skill features.
"""

from __future__ import annotations

from typing import Any

from scripts.utils import configure_logging

logger = configure_logging("generate_correlations")


def generate_correlation_rules() -> dict[str, Any]:
    """Create a dictionary of correlation rules.

    Input:
        None.

    Output:
        dict[str, Any]: Correlation payload with rule definitions.

    Purpose:
        Encapsulate common feature dependency heuristics for profile generation.
    """
    return {
        "description": "Realistic feature dependency rules for synthetic career profiles.",
        "rules": [
            {"feature_pair": ["Machine Learning", "Python"], "relationship": "strong_positive", "strength": 0.9},
            {"feature_pair": ["Leadership", "Years Experience"], "relationship": "positive", "strength": 0.84},
            {"feature_pair": ["Projects Completed", "Years Experience"], "relationship": "strong_positive", "strength": 0.91},
            {"feature_pair": ["Deep Learning", "Machine Learning"], "relationship": "strong_positive", "strength": 0.88},
            {"feature_pair": ["Communication", "Years Experience"], "relationship": "moderate_positive", "strength": 0.72},
            {"feature_pair": ["Cloud", "DevOps"], "relationship": "positive", "strength": 0.8},
            {"feature_pair": ["Cybersecurity", "Networking"], "relationship": "positive", "strength": 0.78},
            {"feature_pair": ["UI Design", "UX Research"], "relationship": "positive", "strength": 0.82},
            {"feature_pair": ["Database", "SQL"], "relationship": "strong_positive", "strength": 0.9},
            {"feature_pair": ["Problem Solving", "Years Experience"], "relationship": "positive", "strength": 0.76},
        ],
        "career_specific_rules": [
            {"career": "ML Engineer", "rules": [["Machine Learning", "Python", "strong_positive"]]},
            {"career": "Data Scientist", "rules": [["Machine Learning", "Python", "strong_positive"]]},
            {"career": "Backend Developer", "rules": [["Database", "SQL", "strong_positive"]]},
            {"career": "Frontend Developer", "rules": [["JavaScript", "UI Design", "positive"]]},
            {"career": "Product Manager", "rules": [["Communication", "Years Experience", "positive"]]},
        ],
    }
