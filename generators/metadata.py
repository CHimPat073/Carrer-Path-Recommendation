
"""
CareerPilot AI
Metadata Generator

Generates realistic metadata for synthetic career profiles.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict

from distributions import weighted_choice, probability_event

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


@dataclass(frozen=True)
class MetadataProfile:
    education: Dict[str, float]
    industries: Dict[str, float]
    countries: Dict[str, float]
    employment_types: Dict[str, float]
    remote_probability: float


DEFAULT_COUNTRIES = {
    "India": 0.40,
    "USA": 0.20,
    "Canada": 0.10,
    "Germany": 0.08,
    "United Kingdom": 0.08,
    "Singapore": 0.07,
    "Australia": 0.07,
}

DEFAULT_EMPLOYMENT = {
    "Full-Time": 0.75,
    "Internship": 0.10,
    "Contract": 0.10,
    "Freelance": 0.05,
}


def generate_metadata(profile: MetadataProfile) -> dict:
    """
    Generate one realistic metadata record.

    Returns
    -------
    dict
        education_level
        industry
        country
        employment_type
        remote_preference
    """
    return {
        "education_level": weighted_choice(profile.education),
        "industry": weighted_choice(profile.industries),
        "country": weighted_choice(profile.countries),
        "employment_type": weighted_choice(profile.employment_types),
        "remote_preference": (
            "Remote"
            if probability_event(profile.remote_probability)
            else random.choice(["Hybrid", "On-site"])
        ),
    }


ML_ENGINEER_METADATA = MetadataProfile(
    education={
        "Bachelor": 0.45,
        "Master": 0.40,
        "PhD": 0.15,
    },
    industries={
        "Artificial Intelligence": 0.40,
        "Cloud": 0.25,
        "Healthcare": 0.15,
        "Finance": 0.10,
        "Automotive": 0.10,
    },
    countries=DEFAULT_COUNTRIES,
    employment_types=DEFAULT_EMPLOYMENT,
    remote_probability=0.75,
)

BACKEND_METADATA = MetadataProfile(
    education={
        "Diploma": 0.10,
        "Bachelor": 0.65,
        "Master": 0.25,
    },
    industries={
        "Finance": 0.25,
        "Cloud": 0.25,
        "E-Commerce": 0.25,
        "Healthcare": 0.25,
    },
    countries=DEFAULT_COUNTRIES,
    employment_types=DEFAULT_EMPLOYMENT,
    remote_probability=0.60,
)

PRODUCT_MANAGER_METADATA = MetadataProfile(
    education={
        "Bachelor": 0.45,
        "Master": 0.35,
        "MBA": 0.20,
    },
    industries={
        "EdTech": 0.25,
        "Finance": 0.25,
        "Healthcare": 0.25,
        "E-Commerce": 0.25,
    },
    countries=DEFAULT_COUNTRIES,
    employment_types=DEFAULT_EMPLOYMENT,
    remote_probability=0.65,
)

METADATA_REGISTRY = {
    "ML Engineer": ML_ENGINEER_METADATA,
    "Backend Developer": BACKEND_METADATA,
    "Product Manager": PRODUCT_MANAGER_METADATA,
}

if __name__ == "__main__":
    for career, profile in METADATA_REGISTRY.items():
        print(career)
        print(generate_metadata(profile))
        print("-" * 40)
