
"""
CareerPilot AI
Distribution Utilities

Reusable statistical sampling functions used by the synthetic
dataset generator.
"""

from __future__ import annotations

import random
from typing import Dict, Sequence

import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def clamp(value: float, minimum: float = 1.0, maximum: float = 10.0) -> float:
    """Clamp a numeric value within a range."""
    return max(minimum, min(maximum, value))


def sample_normal(
    mean: float,
    std: float,
    minimum: int,
    maximum: int,
    as_int: bool = True,
):
    """Sample from a truncated normal distribution."""
    value = np.random.normal(mean, std)
    value = clamp(value, minimum, maximum)
    return int(round(value)) if as_int else float(value)


def weighted_choice(probabilities: Dict[str, float]) -> str:
    """Return a weighted random choice."""
    return random.choices(
        population=list(probabilities.keys()),
        weights=list(probabilities.values()),
        k=1,
    )[0]


def random_choice(values: Sequence[str]) -> str:
    """Return a random element."""
    return random.choice(list(values))


def probability_event(probability: float) -> bool:
    """Return True with the specified probability."""
    return random.random() <= probability


def correlated_score(
    base_score: int,
    correlation: float = 0.8,
    noise_std: float = 0.5,
) -> int:
    """Generate a score correlated with another score."""
    value = (
        correlation * base_score
        + (1 - correlation) * 5
        + np.random.normal(0, noise_std)
    )
    return int(round(clamp(value)))


def blend_scores(
    score_a: int,
    score_b: int,
    weight: float = 0.2,
) -> int:
    """Blend two scores to simulate career overlap."""
    value = score_a * (1 - weight) + score_b * weight
    return int(round(clamp(value)))


def sample_experience(level: str = "mid") -> int:
    """Sample years of experience based on career level."""
    ranges = {
        "beginner": (0, 1),
        "junior": (1, 3),
        "mid": (3, 6),
        "senior": (6, 12),
    }
    low, high = ranges[level]
    return random.randint(low, high)


def projects_from_experience(years: int) -> int:
    """Estimate completed projects from experience."""
    return years * random.randint(2, 5) + random.randint(0, 4)


def salary_from_experience(
    years: int,
    base_salary: int = 400_000,
    annual_increment: int = 120_000,
) -> int:
    """Estimate salary with Gaussian noise."""
    noise = np.random.normal(0, 40_000)
    return int(base_salary + years * annual_increment + noise)


if __name__ == "__main__":
    print("Distribution module test")
    print(sample_normal(9.2, 0.5, 8, 10))
    print(weighted_choice({"Bachelor": 0.6, "Master": 0.3, "PhD": 0.1}))
    print(correlated_score(9))
