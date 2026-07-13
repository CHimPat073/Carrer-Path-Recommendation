"""User persona generation utilities.

This module defines experience-level personas that can later be used to
simulate realistic candidate profiles for synthetic data generation.
"""

from __future__ import annotations

from typing import Any

from scripts.utils import configure_logging

logger = configure_logging("generate_user_personas")


def generate_user_personas() -> dict[str, dict[str, Any]]:
    """Generate persona templates for beginner through expert levels.

    Input:
        None.

    Output:
        dict[str, dict[str, Any]]: Persona dictionary keyed by persona name.

    Purpose:
        Create reusable templates for synthetic candidate generation.
    """
    return {
        "Beginner": {
            "experience": "0-1 years",
            "projects": 2,
            "certifications": 1,
            "leadership": 3,
            "communication": 5,
            "problem_solving": 4,
            "career_growth": 7,
        },
        "Junior": {
            "experience": "1-3 years",
            "projects": 4,
            "certifications": 2,
            "leadership": 4,
            "communication": 6,
            "problem_solving": 5,
            "career_growth": 8,
        },
        "Mid-Level": {
            "experience": "3-6 years",
            "projects": 8,
            "certifications": 3,
            "leadership": 5,
            "communication": 7,
            "problem_solving": 6,
            "career_growth": 8,
        },
        "Senior": {
            "experience": "6-10 years",
            "projects": 12,
            "certifications": 4,
            "leadership": 7,
            "communication": 8,
            "problem_solving": 7,
            "career_growth": 8,
        },
        "Expert": {
            "experience": "10+ years",
            "projects": 20,
            "certifications": 5,
            "leadership": 8,
            "communication": 8,
            "problem_solving": 8,
            "career_growth": 9,
        },
    }
