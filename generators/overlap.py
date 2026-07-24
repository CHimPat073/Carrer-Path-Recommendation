
"""
CareerPilot AI
Career Overlap Engine

Introduces realistic overlap between similar careers so the
synthetic dataset is not perfectly separable.
"""

from __future__ import annotations

import random
from typing import Dict

from career_profiles import CAREER_PROFILES
from distributions import blend_scores

random.seed(42)


class CareerOverlapEngine:
    """Blend skill values between related careers."""

    def __init__(self):
        self.registry = CAREER_PROFILES

    def apply_overlap(
        self,
        career_name: str,
        profile: Dict[str, int],
    ) -> Dict[str, int]:
        """
        Blend a generated profile with one similar career
        based on overlap probabilities defined in career_profiles.py.
        """

        if career_name not in self.registry:
            return profile

        career = self.registry[career_name]

        if not career.overlaps:
            return profile

        # Select one similar career according to overlap weights
        similar_careers = list(career.overlaps.keys())
        weights = list(career.overlaps.values())

        selected = random.choices(
            similar_careers,
            weights=weights,
            k=1,
        )[0]

        # If the selected career is not implemented yet,
        # simply return the original profile.
        if selected not in self.registry:
            return profile

        similar_profile = self.registry[selected]

        updated = profile.copy()

        for skill, dist in similar_profile.technical_skills.items():

            if skill in updated:
                updated[skill] = blend_scores(
                    updated[skill],
                    int(round(dist.mean)),
                    weight=0.20,
                )

        for skill, dist in similar_profile.soft_skills.items():

            if skill in updated:
                updated[skill] = blend_scores(
                    updated[skill],
                    int(round(dist.mean)),
                    weight=0.15,
                )

        updated["similar_career"] = selected

        return updated


if __name__ == "__main__":

    engine = CareerOverlapEngine()

    sample = {
        "python_score": 9,
        "machine_learning_score": 10,
        "deep_learning_score": 9,
        "communication_score": 6,
        "leadership_score": 5,
    }

    print("Before")
    print(sample)

    print()

    updated = engine.apply_overlap(
        "ML Engineer",
        sample,
    )

    print("After")
    print(updated)
