"""Noise-rule generation utilities.

This module defines realistic augmentation rules that can be used later to
inject noise into synthetic career profiles without producing impossible
combinations.
"""

from __future__ import annotations

from typing import Any

from scripts.utils import configure_logging

logger = configure_logging("generate_noise_rules")


def generate_noise_rules() -> dict[str, Any]:
    """Create realistic noise-generation rules.

    Input:
        None.

    Output:
        dict[str, Any]: A dictionary describing noise patterns and examples.

    Purpose:
        Provide a reusable rule set for later synthetic profile augmentation.
    """
    return {
        "description": "Realistic noise rules for synthetic career profile augmentation.",
        "rules": [
            {
                "type": "weak_skill",
                "description": "A candidate shows a weak skill in a non-core area.",
                "example": "ML Engineer weak communication",
            },
            {
                "type": "interest_mismatch",
                "description": "A candidate has a mild interest mismatch with their primary career.",
                "example": "Backend Developer interested in AI",
            },
            {
                "type": "average_soft_skill",
                "description": "A candidate exhibits average soft-skill performance.",
                "example": "Product Manager average leadership",
            },
            {
                "type": "lower_certification_level",
                "description": "A candidate has fewer certifications than expected.",
                "example": "Cloud Engineer low DevOps",
            },
            {
                "type": "limited_experience",
                "description": "A candidate appears less experienced than the role typically requires.",
                "example": "Senior career with limited experience",
            },
        ],
    }
