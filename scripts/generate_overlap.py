"""Career overlap generation utilities.

This module computes similarity between careers using both Jaccard and
cosine-style similarity metrics based on normalized skill profiles. The
functions return Python dictionaries and do not write JSON files.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from scripts.utils import configure_logging

logger = configure_logging("generate_overlap")


def compute_career_overlap(career_skill_map: dict[str, list[str]]) -> dict[str, Any]:
    """Compute overlap data for a set of careers.

    Input:
        career_skill_map (dict[str, list[str]]): Career-to-skill mapping.

    Output:
        dict[str, Any]: Dictionary containing similarity matrix and related careers.

    Purpose:
        Quantify how related different careers are based on shared skills.
    """
    careers = sorted(career_skill_map)
    similarity_matrix: dict[str, dict[str, float]] = {}
    related_careers: dict[str, list[dict[str, Any]]] = {}

    for career in careers:
        similarity_matrix[career] = {}
        skill_vector = set(career_skill_map.get(career, []))
        for other in careers:
            if career == other:
                similarity_matrix[career][other] = 1.0
                continue
            other_vector = set(career_skill_map.get(other, []))
            similarity_matrix[career][other] = _jaccard_similarity(skill_vector, other_vector)

        related = [
            {"career": other, "similarity_percentage": round(similarity_matrix[career][other] * 100, 2)}
            for other in careers
            if other != career and similarity_matrix[career][other] > 0
        ]
        related.sort(key=lambda item: item["similarity_percentage"], reverse=True)
        related_careers[career] = related[:5]

    return {"similarity_matrix": similarity_matrix, "related_careers": related_careers}


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    """Calculate Jaccard similarity for two skill sets.

    Input:
        left (set[str]): First skill set.
        right (set[str]): Second skill set.

    Output:
        float: Similarity score between 0 and 1.

    Purpose:
        Measure overlap between skill profiles.
    """
    if not left and not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def cosine_similarity(left: list[str], right: list[str]) -> float:
    """Calculate cosine similarity for two skill lists.

    Input:
        left (list[str]): First skill list.
        right (list[str]): Second skill list.

    Output:
        float: Cosine similarity score between 0 and 1.

    Purpose:
        Provide an alternative similarity metric for skill overlap.
    """
    left_counter = Counter(left)
    right_counter = Counter(right)
    numerator = sum(left_counter[token] * right_counter[token] for token in left_counter.keys() & right_counter.keys())
    left_norm = math.sqrt(sum(count * count for count in left_counter.values()))
    right_norm = math.sqrt(sum(count * count for count in right_counter.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)
