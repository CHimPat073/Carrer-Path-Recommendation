
"""
CareerPilot AI
Correlation Engine

Applies realistic relationships between generated features.
"""

from __future__ import annotations

from typing import Dict

from distributions import clamp


def apply_skill_correlations(profile: Dict[str, int]) -> Dict[str, int]:
    """
    Adjust related skills to maintain realistic consistency.
    """

    p = profile.copy()

    # Python -> Machine Learning
    if "python_score" in p and "machine_learning_score" in p:
        p["machine_learning_score"] = int(round(
            clamp(
                0.75 * p["machine_learning_score"] +
                0.25 * p["python_score"]
            )
        ))

    # Machine Learning -> Deep Learning
    if "machine_learning_score" in p and "deep_learning_score" in p:
        p["deep_learning_score"] = int(round(
            clamp(
                0.80 * p["deep_learning_score"] +
                0.20 * p["machine_learning_score"]
            )
        ))

    # Cloud -> DevOps
    if "cloud_score" in p and "devops_score" in p:
        p["devops_score"] = int(round(
            clamp(
                0.70 * p["devops_score"] +
                0.30 * p["cloud_score"]
            )
        ))

    return p


def apply_experience_correlations(
    profile: Dict[str, int]
) -> Dict[str, int]:
    """
    Experience influences projects, leadership and communication.
    """

    p = profile.copy()

    years = p.get("experience_years", 0)

    if "leadership_score" in p:
        p["leadership_score"] = int(round(
            clamp(
                p["leadership_score"] + years * 0.20
            )
        ))

    if "communication_score" in p:
        p["communication_score"] = int(round(
            clamp(
                p["communication_score"] + years * 0.10
            )
        ))

    p["projects_completed"] = max(
        p.get("projects_completed", 0),
        years * 3
    )

    return p


def apply_all_correlations(
    profile: Dict[str, int]
) -> Dict[str, int]:
    """
    Master correlation pipeline.
    """

    profile = apply_skill_correlations(profile)
    profile = apply_experience_correlations(profile)

    return profile


if __name__ == "__main__":

    sample = {
        "python_score": 9,
        "machine_learning_score": 8,
        "deep_learning_score": 7,
        "cloud_score": 7,
        "devops_score": 5,
        "leadership_score": 5,
        "communication_score": 6,
        "experience_years": 4,
        "projects_completed": 5,
    }

    print("Before")
    print(sample)

    updated = apply_all_correlations(sample)

    print("\nAfter")
    print(updated)
