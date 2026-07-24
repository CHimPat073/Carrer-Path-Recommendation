
"""
CareerPilot AI
Career Profile Generator

Generates synthetic career profiles using the knowledge base.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from career_profiles import CAREER_PROFILES, CareerProfile
from correlations import apply_all_correlations
from distributions import (
    sample_normal,
    sample_experience,
    projects_from_experience,
    salary_from_experience,
)
from metadata import METADATA_REGISTRY, generate_metadata


class CareerGenerator:
    """Generate realistic synthetic career profiles."""

    def __init__(self):
        self.profiles = CAREER_PROFILES

    def _sample_skills(self, profile: CareerProfile) -> Dict[str, int]:
        skills = {}

        for name, dist in profile.technical_skills.items():
            skills[name] = sample_normal(
                dist.mean,
                dist.std,
                dist.min_value,
                dist.max_value,
            )

        for name, dist in profile.soft_skills.items():
            skills[name] = sample_normal(
                dist.mean,
                dist.std,
                dist.min_value,
                dist.max_value,
            )

        return skills

    def generate_person(self, career: str) -> Dict:
        profile = self.profiles[career]

        record = self._sample_skills(profile)

        years = sample_experience("mid")
        record["experience_years"] = years
        record["projects_completed"] = projects_from_experience(years)
        record["salary"] = salary_from_experience(years)

        if career in METADATA_REGISTRY:
            record.update(generate_metadata(METADATA_REGISTRY[career]))

        record = apply_all_correlations(record)

        record["Career"] = career

        return record

    def generate_dataset(
        self,
        rows_per_career: int = 1000,
    ) -> pd.DataFrame:

        dataset: List[Dict] = []

        for career in self.profiles:
            for _ in range(rows_per_career):
                dataset.append(
                    self.generate_person(career)
                )

        df = pd.DataFrame(dataset)

        return df.sample(
            frac=1,
            random_state=42
        ).reset_index(drop=True)

    def save_dataset(
        self,
        output_path: str,
        rows_per_career: int = 1000,
    ) -> pd.DataFrame:

        df = self.generate_dataset(rows_per_career)

        df.to_csv(
            output_path,
            index=False
        )

        print(f"Dataset saved to {output_path}")
        print(df.shape)

        return df


if __name__ == "__main__":

    generator = CareerGenerator()

    df = generator.generate_dataset(
        rows_per_career=100
    )

    print(df.head())

    # Example:
    # generator.save_dataset(
    #     "datasets/synthetic/careerpilot_v3.csv",
    #     rows_per_career=1000,
    # )
