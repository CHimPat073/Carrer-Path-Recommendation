"""
Statistical Model for Career Profile Generation.

This module implements a principled probability distribution-based approach
to generating realistic synthetic career profiles.

Distributions Used:
- Truncated Normal: Technical skills, soft skills, experience metrics
- Zero-Inflated Negative Binomial: Certifications (count data with excess zeros)
- Weighted Categorical: Education levels
- Log-Normal: Salary bands
- Beta: Preferences (remote, career growth)
"""

import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Final, cast

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.preprocessing.role_normalizer import TARGET_CAREERS

RANDOM_SEED: Final[int] = 42
KNOWLEDGE_ROOT = PROJECT_ROOT / "knowledge_base"

# Feature names in snake_case format
FEATURE_NAMES: Final[list[str]] = [
    "years_experience",
    "education_level",
    "projects_completed",
    "certifications",
    "python_score",
    "java_score",
    "javascript_score",
    "sql_score",
    "machine_learning_score",
    "deep_learning_score",
    "cloud_score",
    "devops_score",
    "cybersecurity_score",
    "data_analysis_score",
    "database_score",
    "networking_score",
    "mobile_score",
    "game_dev_score",
    "testing_score",
    "business_analysis_score",
    "product_management_score",
    "ui_design_score",
    "ux_research_score",
    "communication_score",
    "leadership_score",
    "problem_solving_score",
    "teamwork_score",
    "agile_score",
    "research_score",
]

# Seniority levels with ordinal mapping
SENIORITY_LEVELS: Final[list[str]] = ["Beginner", "Junior", "Mid-Level", "Senior", "Expert"]
SENIORITY_WEIGHTS: Final[list[float]] = [0.15, 0.25, 0.30, 0.20, 0.10]
SENIORITY_TO_LEVEL: Final[dict[str, int]] = {
    "Beginner": 0,
    "Junior": 1,
    "Mid-Level": 2,
    "Senior": 3,
    "Expert": 4
}

EDUCATION_LEVELS: Final[list[str]] = ["High School", "Associate", "Bachelor", "Master", "PhD"]


# =============================================================================
# PROBABILITY DISTRIBUTION FUNCTIONS
# =============================================================================

def truncated_normal(mean: float, std_dev: float, lower: float, upper: float, rng: random.Random) -> float:
    """
    Generate a sample from a truncated normal distribution.

    Uses rejection sampling to ensure values fall within bounds.
    This is used for technical skills, soft skills, and experience metrics.

    Args:
        mean: Distribution mean
        std_dev: Standard deviation
        lower: Lower bound (inclusive)
        upper: Upper bound (inclusive)
        rng: Random number generator

    Returns:
        Sampled value within bounds
    """
    # For efficiency, use a limited number of attempts
    max_attempts = 100
    for _ in range(max_attempts):
        value = rng.gauss(mean, std_dev)
        if lower <= value <= upper:
            return value
    # Fallback: clamp the mean-adjacent value
    return max(lower, min(upper, mean))


def zero_inflated_negative_binomial(
    zero_prob: float,
    lambda_param: float,
    dispersion: float,
    rng: random.Random
) -> int:
    """
    Generate a sample from a Zero-Inflated Negative Binomial distribution.

    This distribution models count data with excess zeros - perfect for
    certifications where many professionals have none.

    Implementation uses proper Negative Binomial sampling via NumPy,
    avoiding the incorrect Gaussian noise approximation.

    Args:
        zero_prob: Probability of generating a zero (structural zero)
        lambda_param: Rate parameter for NB component (mean when not zero)
        dispersion: Dispersion parameter (higher = more variance)
        rng: Random number generator

    Returns:
        Count (0 or positive integer)
    """
    # Structural zero: inflate zeros based on zero_prob
    if rng.random() < zero_prob:
        return 0

    # Proper Negative Binomial using NumPy
    # NumPy uses parameterization: NB(n, p) where n = dispersion, p = success probability
    # Mean = n * (1-p) / p, so p = n / (n + lambda)
    n = dispersion
    p = n / (n + lambda_param)

    # Create a seeded numpy random state from the Python RNG state
    # This preserves reproducibility while using NumPy for proper distribution
    np_seed = rng.getrandbits(32)
    np_rng = np.random.RandomState(np_seed)

    # Sample from Negative Binomial
    # numpy.random.negative_binomial(n, p) returns number of successes
    # We want the count of failures (which is the NB random variable)
    count = np_rng.negative_binomial(n, p)

    # Ensure non-negative output
    return max(0, int(count))


def weighted_categorical(options: list[str], weights: list[float], rng: random.Random) -> str:
    """
    Generate a sample from a weighted categorical distribution.

    Used for education levels where different careers have different
    typical education distributions.

    Args:
        options: List of category values
        weights: Probability weights for each category
        rng: Random number generator

    Returns:
        Selected category
    """
    return rng.choices(options, weights=weights, k=1)[0]


def log_normal(mean: float, std_dev: float, lower: float, upper: float, rng: random.Random) -> float:
    """
    Generate a sample from a truncated log-normal distribution.

    Used for salary bands as salaries are strictly positive and
    right-skewed (more low salaries, long tail to high).

    Args:
        mean: Distribution mean (on log scale)
        std_dev: Standard deviation (on log scale)
        lower: Lower bound
        upper: Upper bound
        rng: Random number generator

    Returns:
        Sampled salary value
    """
    max_attempts = 100
    for _ in range(max_attempts):
        value = math.exp(rng.gauss(mean, std_dev))
        if lower <= value <= upper:
            return value
    return max(lower, min(upper, math.exp(mean)))


def beta_sample(alpha: float, beta_param: float, lower: float, upper: float, rng: random.Random) -> float:
    """
    Generate a sample from a scaled Beta distribution.

    Used for preferences (remote work, career growth) as they are
    bounded and can be skewed in either direction.

    Args:
        alpha: Beta distribution shape parameter
        beta_param: Beta distribution shape parameter
        lower: Lower bound of output range
        upper: Upper bound of output range
        rng: Random number generator

    Returns:
        Sampled preference value
    """
    # Use Beta distribution from random module
    value = rng.betavariate(alpha, beta_param)
    # Scale to desired range
    return lower + value * (upper - lower)


# =============================================================================
# CAREER-SPECIFIC PARAMETERS
# =============================================================================

# Maps feature names (without _score suffix) to standardized keys
FEATURE_KEY_MAP = {
    "python": "Python",
    "java": "Java",
    "javascript": "JavaScript",
    "sql": "SQL",
    "machine_learning": "Machine Learning",
    "deep_learning": "Deep Learning",
    "cloud": "Cloud",
    "devops": "DevOps",
    "cybersecurity": "Cybersecurity",
    "data_analysis": "Data Analysis",
    "database": "Database",
    "networking": "Networking",
    "mobile_development": "Mobile Development",
    "mobile": "Mobile Development",
    "game_development": "Game Development",
    "game_dev": "Game Development",
    "testing": "Testing",
    "business_analysis": "Business Analysis",
    "product_management": "Product Management",
    "ui_design": "UI Design",
    "ux_research": "UX Research",
    "communication": "Communication",
    "leadership": "Leadership",
    "problem_solving": "Problem Solving",
    "teamwork": "Teamwork",
    "agile": "Agile",
    "research": "Research",
}

# Education weights by career family
EDUCATION_WEIGHTS = {
    # Software Engineering family
    "software_engineering": {
        "High School": 0.03, "Associate": 0.08, "Bachelor": 0.60, "Master": 0.25, "PhD": 0.04
    },
    # Data/AI family
    "data_ai": {
        "High School": 0.02, "Associate": 0.05, "Bachelor": 0.40, "Master": 0.45, "PhD": 0.08
    },
    # Platform/Infrastructure family
    "platform_infra": {
        "High School": 0.05, "Associate": 0.15, "Bachelor": 0.55, "Master": 0.22, "PhD": 0.03
    },
    # Product/Design family
    "product_design": {
        "High School": 0.02, "Associate": 0.05, "Bachelor": 0.50, "Master": 0.38, "PhD": 0.05
    },
    # Mobile Development
    "mobile": {
        "High School": 0.03, "Associate": 0.10, "Bachelor": 0.60, "Master": 0.24, "PhD": 0.03
    },
}

# Map careers to their family
CAREER_TO_FAMILY = {
    "Software Engineer": "software_engineering",
    "Backend Developer": "software_engineering",
    "Frontend Developer": "software_engineering",
    "Full Stack Developer": "software_engineering",
    "QA Engineer": "software_engineering",
    "Game Developer": "software_engineering",
    "Data Scientist": "data_ai",
    "ML Engineer": "data_ai",
    "Data Engineer": "data_ai",
    "AI Research Engineer": "data_ai",
    "DevOps Engineer": "platform_infra",
    "Cloud Engineer": "platform_infra",
    "Network Engineer": "platform_infra",
    "Database Administrator": "platform_infra",
    "Cyber Security Analyst": "platform_infra",
    "Ethical Hacker": "platform_infra",
    "Product Manager": "product_design",
    "Business Analyst": "product_design",
    "UI/UX Designer": "product_design",
    "Mobile App Developer": "mobile",
}

# Seniority multipliers for technical skills
SENIORITY_TECH_SHIFT = {
    "Beginner": -1.5,
    "Junior": -0.5,
    "Mid-Level": 0.0,
    "Senior": 0.5,
    "Expert": 1.0,
}

SENIORITY_TECH_SCALE = {
    "Beginner": 1.3,
    "Junior": 1.1,
    "Mid-Level": 1.0,
    "Senior": 0.9,
    "Expert": 0.8,
}

# Seniority multipliers for soft skills
SENIORITY_SOFT_SHIFT = {
    "Beginner": {"leadership": -2.5, "communication": -1.5, "problem_solving": -2.0, "teamwork": -1.0},
    "Junior": {"leadership": -1.5, "communication": -0.5, "problem_solving": -1.0, "teamwork": -0.5},
    "Mid-Level": {"leadership": 0.0, "communication": 0.5, "problem_solving": 0.5, "teamwork": 0.0},
    "Senior": {"leadership": 1.5, "communication": 1.0, "problem_solving": 1.0, "teamwork": 0.5},
    "Expert": {"leadership": 2.5, "communication": 1.5, "problem_solving": 1.5, "teamwork": 1.0},
}

# Experience parameters by career (mean, std_dev)
EXPERIENCE_PARAMS = {
    "Software Engineer": (4.3, 2.8),
    "Backend Developer": (5.2, 2.8),
    "Frontend Developer": (4.0, 2.5),
    "Full Stack Developer": (5.0, 2.5),
    "Mobile App Developer": (4.1, 2.5),
    "Data Scientist": (5.4, 2.8),
    "ML Engineer": (6.2, 2.8),
    "Data Engineer": (6.1, 2.8),
    "AI Research Engineer": (7.8, 3.2),
    "DevOps Engineer": (6.0, 2.8),
    "Cloud Engineer": (6.1, 2.8),
    "Cyber Security Analyst": (4.7, 2.5),
    "Ethical Hacker": (5.7, 2.8),
    "Network Engineer": (4.9, 2.5),
    "UI/UX Designer": (3.6, 2.2),
    "Product Manager": (6.7, 2.8),
    "Business Analyst": (4.5, 2.5),
    "QA Engineer": (3.7, 2.2),
    "Database Administrator": (6.1, 2.8),
    "Game Developer": (3.8, 2.2),
}

# Certification parameters (zero_prob, lambda, dispersion)
CERT_PARAMS = {
    "Software Engineer": (0.50, 1.2, 1.5),
    "Backend Developer": (0.45, 1.4, 1.5),
    "Frontend Developer": (0.55, 0.9, 1.5),
    "Full Stack Developer": (0.50, 1.2, 1.5),
    "Mobile App Developer": (0.55, 0.9, 1.5),
    "Data Scientist": (0.40, 1.4, 1.5),
    "ML Engineer": (0.35, 1.7, 1.5),
    "Data Engineer": (0.30, 2.0, 1.5),
    "AI Research Engineer": (0.50, 1.2, 1.5),
    "DevOps Engineer": (0.25, 2.2, 1.5),
    "Cloud Engineer": (0.20, 2.3, 1.5),
    "Cyber Security Analyst": (0.20, 2.5, 1.5),
    "Ethical Hacker": (0.15, 3.4, 1.5),
    "Network Engineer": (0.30, 2.1, 1.5),
    "UI/UX Designer": (0.60, 0.8, 1.5),
    "Product Manager": (0.50, 1.1, 1.5),
    "Business Analyst": (0.50, 1.0, 1.5),
    "QA Engineer": (0.45, 1.1, 1.5),
    "Database Administrator": (0.25, 2.2, 1.5),
    "Game Developer": (0.65, 0.7, 1.5),
}

# Salary parameters (log_mean, log_std, min, max)
SALARY_PARAMS = {
    "Software Engineer": (11.73, 0.20, 85000, 160000),
    "Backend Developer": (11.85, 0.20, 90000, 180000),
    "Frontend Developer": (11.55, 0.20, 80000, 150000),
    "Full Stack Developer": (11.76, 0.20, 95000, 170000),
    "Mobile App Developer": (11.70, 0.20, 85000, 160000),
    "Data Scientist": (11.82, 0.20, 95000, 180000),
    "ML Engineer": (12.02, 0.20, 110000, 220000),
    "Data Engineer": (11.89, 0.20, 100000, 190000),
    "AI Research Engineer": (12.13, 0.20, 120000, 250000),
    "DevOps Engineer": (11.89, 0.20, 100000, 190000),
    "Cloud Engineer": (11.93, 0.20, 105000, 200000),
    "Cyber Security Analyst": (11.70, 0.20, 80000, 160000),
    "Ethical Hacker": (11.86, 0.20, 95000, 190000),
    "Network Engineer": (11.53, 0.20, 75000, 150000),
    "UI/UX Designer": (11.51, 0.20, 75000, 145000),
    "Product Manager": (12.02, 0.20, 110000, 220000),
    "Business Analyst": (11.34, 0.20, 70000, 135000),
    "QA Engineer": (11.31, 0.20, 70000, 130000),
    "Database Administrator": (11.70, 0.20, 85000, 160000),
    "Game Developer": (11.68, 0.25, 80000, 170000),
}


# =============================================================================
# CORRELATION ENGINE - SKILL-TO-SKILL DEPENDENCIES
# =============================================================================

# Correlation clusters define skill dependencies:
# source_feature -> [(target_feature, base_strength), ...]
# Strengths are multiplied by career-specific modifiers
CORRELATION_CLUSTERS: Final[dict[str, list[tuple[str, float]]]] = {
    # Programming Cluster: Python → ML → Deep Learning → AI Research
    "python_score": [
        ("machine_learning_score", 0.30),
        ("deep_learning_score", 0.20),
        ("data_analysis_score", 0.25),
        ("sql_score", 0.15),
    ],
    # Java → Backend → Database → System Design
    "java_score": [
        ("database_score", 0.25),
        ("sql_score", 0.20),
    ],
    # JavaScript → Frontend chain
    "javascript_score": [
        ("ui_design_score", 0.25),
        ("ux_research_score", 0.20),
    ],
    # Machine Learning → Deep Learning → AI Research
    "machine_learning_score": [
        ("deep_learning_score", 0.40),
        ("python_score", 0.15),
        ("data_analysis_score", 0.25),
        ("research_score", 0.20),
    ],
    "deep_learning_score": [
        ("machine_learning_score", 0.30),
        ("python_score", 0.20),
        ("research_score", 0.25),
    ],
    # Cloud → Docker → Kubernetes → DevOps → CI/CD
    "cloud_score": [
        ("devops_score", 0.35),
    ],
    "devops_score": [
        ("cloud_score", 0.25),
    ],
    # Networking → Cyber Security → Ethical Hacking → Incident Response
    "networking_score": [
        ("cybersecurity_score", 0.40),
    ],
    "cybersecurity_score": [
        ("networking_score", 0.25),
    ],
    # Communication → Leadership → Project Management → Business Analysis
    "communication_score": [
        ("leadership_score", 0.35),
        ("teamwork_score", 0.25),
    ],
    "leadership_score": [
        ("communication_score", 0.20),
        ("problem_solving_score", 0.15),
    ],
    # Experience Cluster: Years Exp → Projects → Leadership → Salary → Mentoring
    "years_experience": [
        ("projects_completed", 0.45),
        ("leadership_score", 0.20),
        ("certifications", 0.15),
    ],
    "projects_completed": [
        ("years_experience", 0.25),
        ("problem_solving_score", 0.15),
    ],
}

# Career-specific correlation strength modifiers
# Higher values = stronger correlations for that career
# Format: {career: {source_feature: {target: strength_modifier}}}
CAREER_CORRELATION_MODIFIERS: Final[dict[str, dict[str, dict[str, float]]]] = {
    # Data/AI careers: Strong Python→ML→DL chain
    "Data Scientist": {
        "python_score": {"machine_learning_score": 1.2, "deep_learning_score": 1.1},
        "machine_learning_score": {"deep_learning_score": 1.2},
    },
    "ML Engineer": {
        "python_score": {"machine_learning_score": 1.3, "deep_learning_score": 1.2},
        "machine_learning_score": {"deep_learning_score": 1.3},
    },
    "AI Research Engineer": {
        "python_score": {"machine_learning_score": 1.4, "deep_learning_score": 1.4},
        "machine_learning_score": {"deep_learning_score": 1.5, "research_score": 1.3},
        "deep_learning_score": {"research_score": 1.4},
    },
    # Backend: Strong Java→Database chain
    "Backend Developer": {
        "java_score": {"database_score": 1.3, "sql_score": 1.2},
    },
    "Database Administrator": {
        "java_score": {"database_score": 1.2},
        "sql_score": {"database_score": 1.3},
    },
    "Software Engineer": {
        "java_score": {"database_score": 1.1},
        "python_score": {"sql_score": 1.1},
    },
    # Frontend: Strong JS chain
    "Frontend Developer": {
        "javascript_score": {"ui_design_score": 1.3, "ux_research_score": 1.2},
    },
    "Full Stack Developer": {
        "python_score": {"machine_learning_score": 0.8},  # Less relevant for full stack
        "javascript_score": {"ui_design_score": 1.1},
    },
    # Cloud/DevOps: Strong cloud chain
    "DevOps Engineer": {
        "cloud_score": {"devops_score": 1.4},
    },
    "Cloud Engineer": {
        "cloud_score": {"devops_score": 1.3, "networking_score": 1.1},
    },
    # Security: Strong networking chain
    "Cyber Security Analyst": {
        "networking_score": {"cybersecurity_score": 1.4},
    },
    "Ethical Hacker": {
        "networking_score": {"cybersecurity_score": 1.3},
        "cybersecurity_score": {"testing_score": 1.2},
    },
    "Network Engineer": {
        "networking_score": {"cybersecurity_score": 1.1},
    },
    # Management: Strong communication→leadership chain
    "Product Manager": {
        "communication_score": {"leadership_score": 1.4, "business_analysis_score": 1.3},
        "leadership_score": {"problem_solving_score": 1.2},
    },
    "Business Analyst": {
        "communication_score": {"business_analysis_score": 1.3},
    },
    # Experience-driven correlations (stronger for senior roles)
    "Senior": {
        "years_experience": {"leadership_score": 1.3, "projects_completed": 1.2},
    },
    "Expert": {
        "years_experience": {"leadership_score": 1.5, "projects_completed": 1.3},
    },
}


# =============================================================================
# STATISTICAL MODEL GENERATOR
# =============================================================================

class StatisticalProfileGenerator:
    """
    Generate realistic synthetic career profiles using principled probability distributions.

    This replaces the simplistic random generation with:
    - Truncated Normal for skills and experience
    - Zero-Inflated Negative Binomial for certifications
    - Weighted Categorical for education
    - Log-Normal for salaries
    - Beta for preferences
    """

    def __init__(self, rows: int = 20000, seed: int = RANDOM_SEED) -> None:
        self.rows = rows
        self.seed = seed
        self.rng = random.Random(seed)
        self.knowledge_root = KNOWLEDGE_ROOT

        # Load knowledge base data
        self.feature_ranges = self._load_json(self.knowledge_root / "feature_ranges.json")
        self.career_personas = self._load_json(self.knowledge_root / "career_personas.json")
        self.overlap_data = self._load_json(self.knowledge_root / "career_overlap.json")
        self.noise_rules = self._load_json(self.knowledge_root / "noise_rules.json")
        self.correlation_rules = self._load_json(self.knowledge_root / "correlation_rules.json")
        self.domain_metadata = self._load_json(self.knowledge_root / "domain_metadata.json")

        # Create lookup dictionaries
        self.feature_lookup = {
            item.get("career"): item.get("features", {})
            for item in self.feature_ranges.get("career_feature_ranges", [])
            if item.get("career")
        }

        self.correlation_lookup = self.correlation_rules.get("correlation_rules", {}).get("rules", [])

        self.overlap_lookup = {
            item.get("career"): item.get("similar_careers", [])
            for item in self.overlap_data.get("career_similarity_matrix", [])
            if item.get("career")
        }

        self.noise_rule_list = self.noise_rules.get("noise_generation_rules", {}).get("rules", [])

        # Load domain metadata - these replace hardcoded constants
        self._load_domain_metadata()

    def _load_domain_metadata(self) -> None:
        """Load all domain metadata from knowledge base, with fallbacks for backward compatibility."""
        dm = self.domain_metadata

        # Education levels
        self.education_levels = dm.get("education_levels", {}).get("values", EDUCATION_LEVELS)
        self.education_ordinal = dm.get("education_levels", {}).get("ordinal_mapping", {})

        # Seniority levels
        self.seniority_levels = dm.get("seniority_levels", {}).get("values", SENIORITY_LEVELS)
        self.seniority_weights = dm.get("seniority_levels", {}).get("weights", SENIORITY_WEIGHTS)
        self.seniority_to_level = dm.get("seniority_levels", {}).get("ordinal_mapping", SENIORITY_TO_LEVEL)

        # Domain values
        self.countries = dm.get("countries", {}).get("values", ["USA", "Canada", "UK", "India", "Australia", "Germany"])
        self.industries = dm.get("industries", {}).get("values", ["Technology", "Finance", "Healthcare", "Education", "Retail", "Consulting", "Gaming"])
        self.work_modes = dm.get("work_modes", {}).get("values", ["Remote", "Hybrid", "On-site"])
        self.work_mode_weights = dm.get("work_modes", {}).get("weights", [0.35, 0.40, 0.25])
        self.employment_types = dm.get("employment_types", {}).get("values", ["Full-time", "Part-time", "Contract", "Freelance"])

        # Career families
        self.career_families = dm.get("career_families", {}).get("families", {})
        self.education_weights_by_family = dm.get("education_weights_by_family", EDUCATION_WEIGHTS)

        # Build career to family mapping
        self.career_to_family = {}
        for family_name, family_data in self.career_families.items():
            for career in family_data.get("careers", []):
                self.career_to_family[career] = family_name

        # Seniority modifiers
        self.seniority_tech_shift = dm.get("seniority_tech_shift", SENIORITY_TECH_SHIFT)
        self.seniority_tech_scale = dm.get("seniority_tech_scale", SENIORITY_TECH_SCALE)
        self.seniority_soft_shift = dm.get("seniority_soft_shift", SENIORITY_SOFT_SHIFT)

        # Career-specific parameters
        self.experience_params = dm.get("experience_params", dict(EXPERIENCE_PARAMS))
        self.cert_params = dm.get("cert_params", dict(CERT_PARAMS))
        self.salary_params = dm.get("salary_params", dict(SALARY_PARAMS))

        # Career growth Beta parameters
        self.career_growth_alpha_beta = dm.get("career_growth_alpha_beta", {})

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    @staticmethod
    def _bounded_int(value: float, low: int, high: int) -> int:
        return int(round(StatisticalProfileGenerator._clamp(value, low, high)))

    def _get_seniority_level(self) -> str:
        """Select seniority level using weights from domain metadata."""
        return self.rng.choices(self.seniority_levels, weights=self.seniority_weights, k=1)[0]

    def _get_feature_params(self, career: str, feature_key: str) -> dict[str, float]:
        """Get distribution parameters for a career-feature combination."""
        career_features = self.feature_lookup.get(career, {})

        # Try direct key first
        if feature_key in career_features:
            return career_features[feature_key]

        # Try with spaces
        feature_key_spaced = feature_key.replace("_", " ")
        if feature_key_spaced in career_features:
            return career_features[feature_key_spaced]

        # Try with snake_case conversion
        feature_key_snake = feature_key.replace(" ", "_")
        if feature_key_snake in career_features:
            return career_features[feature_key_snake]

        # Default parameters
        return {"min": 1, "max": 10, "average": 5}

    def _get_education_weights(self, career: str) -> dict[str, float]:
        """Get education weight distribution for a career from domain metadata."""
        family = self.career_to_family.get(career, "software_engineering")
        return self.education_weights_by_family.get(family, self.education_weights_by_family.get("software_engineering", {}))

    def _generate_technical_skill(
        self,
        career: str,
        feature_key: str,
        seniority: str,
        is_anchor: bool = False
    ) -> float:
        """Generate a technical skill score using truncated normal distribution."""
        params = self._get_feature_params(career, feature_key)

        base_mean = params.get("average", 5)
        base_std = max(0.8, (params.get("max", 10) - params.get("min", 1)) / 6)

        # Apply seniority modifiers from domain metadata
        mean = base_mean + self.seniority_tech_shift.get(seniority, 0)
        std = base_std * self.seniority_tech_scale.get(seniority, 1.0)

        # Boost anchor skills (core career skills)
        if is_anchor:
            mean += 0.5

        # Apply bounds
        lower = max(1, params.get("min", 1))
        upper = min(10, params.get("max", 10))

        return truncated_normal(mean, std, lower, upper, self.rng)

    def _generate_soft_skill(
        self,
        career: str,
        feature_key: str,
        seniority: str
    ) -> float:
        """Generate a soft skill score using truncated normal distribution."""
        # Base soft skill parameters (career-agnostic defaults)
        base_params = {
            "Communication": {"min": 6, "max": 9, "average": 7.5},
            "Leadership": {"min": 2, "max": 8, "average": 5.0},
            "Problem Solving": {"min": 6, "max": 9, "average": 7.5},
            "Teamwork": {"min": 6, "max": 9, "average": 7.5},
            "Research": {"min": 3, "max": 8, "average": 5.5},
            "Agile": {"min": 3, "max": 8, "average": 5.5},
        }

        params = base_params.get(feature_key, {"min": 4, "max": 8, "average": 6})

        base_mean = params.get("average", 6)
        base_std = 1.5

        # Apply seniority shifts from domain metadata
        shift = self.seniority_soft_shift.get(seniority, {}).get(feature_key.lower(), 0)
        mean = base_mean + shift
        std = base_std

        lower = params.get("min", 1)
        upper = params.get("max", 10)

        return truncated_normal(mean, std, lower, upper, self.rng)

    def _generate_years_experience(self, career: str, seniority: str) -> int:
        """Generate years of experience using truncated normal distribution."""
        # Get params from domain metadata with fallback
        params = self.experience_params.get(career, [4.0, 2.5])
        base_mean, base_std = params[0], params[1]

        # Adjust for seniority
        seniority_level = self.seniority_to_level.get(seniority, 2)
        mean = base_mean + (seniority_level - 2) * 2.5  # Shift around mid-level
        std = base_std

        return self._bounded_int(truncated_normal(mean, std, 0, 20, self.rng), 0, 20)

    def _generate_projects_completed(self, years_exp: int) -> int:
        """Generate projects completed (scales with experience)."""
        # Projects typically 2-5x years of experience
        mean = 2.5 * years_exp
        std = max(1.0, 0.4 * years_exp)

        return self._bounded_int(truncated_normal(mean, std, 0, 50, self.rng), 0, 50)

    def _generate_certifications(self, career: str) -> int:
        """Generate certification count using Zero-Inflated Negative Binomial."""
        # Get params from domain metadata with fallback
        params = self.cert_params.get(career, [0.5, 1.0, 1.5])
        zero_prob, lambda_param, dispersion = params[0], params[1], params[2]

        return zero_inflated_negative_binomial(zero_prob, lambda_param, dispersion, self.rng)

    def _generate_education(self, career: str) -> str:
        """Generate education level using weighted categorical distribution."""
        weights = self._get_education_weights(career)

        options = list(weights.keys())
        weight_values = list(weights.values())

        return weighted_categorical(options, weight_values, self.rng)

    def _generate_salary(self, career: str) -> int:
        """Generate salary using truncated log-normal distribution."""
        # Get params from domain metadata with fallback
        params = self.salary_params.get(career, [11.5, 0.2, 70000, 150000])
        log_mean, log_std, salary_min, salary_max = params[0], params[1], params[2], params[3]

        return self._bounded_int(
            log_normal(log_mean, log_std, salary_min, salary_max, self.rng),
            salary_min,
            salary_max
        )

    def _generate_remote_preference(self) -> str:
        """Generate remote preference using work modes from domain metadata."""
        return weighted_categorical(
            self.work_modes,
            self.work_mode_weights,
            self.rng
        )

    def _generate_career_growth(self, seniority: str) -> int:
        """Generate career growth score using Beta distribution from domain metadata."""
        # Get alpha/beta from domain metadata
        params = self.career_growth_alpha_beta.get(seniority, [6, 2])
        alpha, beta_param = params[0], params[1]

        value = beta_sample(alpha, beta_param, 1, 10, self.rng)

        return self._bounded_int(value, 1, 10)

    def _get_career_anchor_features(self, career: str) -> list[str]:
        """Get anchor (core) features for a career."""
        anchor_map = {
            "Software Engineer": ["python", "java", "javascript", "sql", "testing", "problem_solving"],
            "Backend Developer": ["python", "java", "sql", "database", "problem_solving"],
            "Frontend Developer": ["javascript", "ui_design", "ux_research", "communication"],
            "Full Stack Developer": ["python", "javascript", "sql", "cloud", "problem_solving"],
            "Mobile App Developer": ["mobile", "javascript", "ui_design", "testing"],
            "Data Scientist": ["python", "machine_learning", "data_analysis", "sql", "research"],
            "ML Engineer": ["python", "machine_learning", "deep_learning", "data_analysis", "research"],
            "Data Engineer": ["sql", "database", "python", "cloud", "devops"],
            "AI Research Engineer": ["python", "machine_learning", "deep_learning", "research"],
            "DevOps Engineer": ["cloud", "devops", "networking", "problem_solving"],
            "Cloud Engineer": ["cloud", "devops", "networking", "cybersecurity"],
            "Cyber Security Analyst": ["cybersecurity", "networking", "problem_solving", "research"],
            "Ethical Hacker": ["cybersecurity", "networking", "problem_solving", "testing"],
            "Network Engineer": ["networking", "cloud", "cybersecurity", "problem_solving"],
            "UI/UX Designer": ["ui_design", "ux_research", "communication", "research"],
            "Product Manager": ["product_management", "leadership", "communication", "agile", "business_analysis"],
            "Business Analyst": ["business_analysis", "communication", "problem_solving", "sql"],
            "QA Engineer": ["testing", "problem_solving", "communication", "agile"],
            "Database Administrator": ["database", "sql", "problem_solving", "cybersecurity"],
            "Game Developer": ["game_dev", "javascript", "python", "teamwork", "ui_design"],
        }
        return anchor_map.get(career, [])

    def _apply_correlation_rules(self, row: dict[str, Any]) -> None:
        """Apply correlation rules to ensure realistic feature relationships.

        Implements:
        - Skill-to-skill correlations (Python→ML, ML→DL, Cloud→DevOps, etc.)
        - Experience-based correlations (Years Exp→Projects→Leadership→Salary)
        - Career-specific correlation strengths

        Uses weighted blending: new = 0.80*original + 0.20*correlated
        to preserve randomness while enforcing realistic tendencies.
        """
        career = row.get("career", "")
        seniority = row.get("seniority", "Mid-Level")

        # Get career-specific modifiers
        career_modifiers = CAREER_CORRELATION_MODIFIERS.get(career, {})

        # Apply skill-to-skill correlations from CORRELATION_CLUSTERS
        for source_key, targets in CORRELATION_CLUSTERS.items():
            if source_key not in row:
                continue

            source_value = row[source_key]

            for target_key, base_strength in targets:
                if target_key not in row:
                    continue

                # Get career-specific modifier
                career_modifier = 1.0
                if source_key in career_modifiers:
                    source_mods = career_modifiers[source_key]
                    career_modifier = source_mods.get(target_key, 1.0)

                # Apply seniority modifier for experience correlations
                seniority_modifier = 1.0
                if source_key == "years_experience" and target_key == "leadership_score":
                    if seniority == "Senior":
                        seniority_modifier = 1.2
                    elif seniority == "Expert":
                        seniority_modifier = 1.4

                # Calculate final strength
                strength = base_strength * career_modifier * seniority_modifier
                strength = min(0.5, max(0.05, strength))  # Clamp to [0.05, 0.5]

                # Apply weighted blend: new = (1-strength)*original + strength*source
                original_target = row[target_key]
                blended_value = (1 - strength) * original_target + strength * source_value

                # Clamp to valid range
                if target_key in ("years_experience", "projects_completed", "certifications"):
                    row[target_key] = self._bounded_int(blended_value, 0, 20 if target_key != "projects_completed" else 50)
                else:
                    row[target_key] = self._bounded_int(blended_value, 1, 10)

        # Apply legacy experience correlation rules (for backwards compatibility)
        for rule in self.correlation_lookup:
            pair = rule.get("feature_pair", [])
            if len(pair) != 2:
                continue

            first, second = pair[0], pair[1]
            first_key = self._normalize_feature_name(first)
            second_key = self._normalize_feature_name(second)

            if first_key not in row or second_key not in row:
                continue

            strength = float(rule.get("strength", 0.5))

            # Handle years_experience → salary correlation
            if second_key == "salary_band":
                delta = max(0, int(row[first_key]) - 3) * 5000
                row[second_key] = self._bounded_int(
                    row[second_key] + delta * strength * 0.3,
                    50000, 300000
                )

    @staticmethod
    def _normalize_feature_name(value: str) -> str:
        """Normalize feature names to score format."""
        cleaned = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")

        aliases = {
            "python": "python_score",
            "java": "java_score",
            "javascript": "javascript_score",
            "sql": "sql_score",
            "machine_learning": "machine_learning_score",
            "deep_learning": "deep_learning_score",
            "cloud": "cloud_score",
            "devops": "devops_score",
            "cybersecurity": "cybersecurity_score",
            "data_analysis": "data_analysis_score",
            "database": "database_score",
            "networking": "networking_score",
            "mobile_development": "mobile_score",
            "mobile": "mobile_score",
            "game_development": "game_dev_score",
            "game_dev": "game_dev_score",
            "testing": "testing_score",
            "business_analysis": "business_analysis_score",
            "product_management": "product_management_score",
            "ui_design": "ui_design_score",
            "ux_research": "ux_research_score",
            "communication": "communication_score",
            "leadership": "leadership_score",
            "problem_solving": "problem_solving_score",
            "teamwork": "teamwork_score",
            "agile": "agile_score",
            "research": "research_score",
            "years_experience": "years_experience",
            "projects_completed": "projects_completed",
            "certifications": "certifications",
            "salary_band": "salary_band",
        }
        return aliases.get(cleaned, cleaned + "_score")

    def _apply_noise(self, row: dict[str, Any]) -> None:
        """Apply realistic noise to simulate imperfect profiles."""
        if self.rng.random() >= 0.08 or not self.noise_rule_list:
            return

        selected = self.rng.choice(self.noise_rule_list)
        noise_type = selected.get("type")

        if noise_type == "weak_skill":
            target = self.rng.choice([
                "python_score", "sql_score", "cloud_score",
                "communication_score", "testing_score", "database_score"
            ])
            row[target] = max(1, row.get(target, 5) - self.rng.randint(1, 2))

        elif noise_type == "interest_mismatch":
            row["machine_learning_score"] = min(10, row.get("machine_learning_score", 5) + 1)
            row["python_score"] = max(1, row.get("python_score", 5) - 1)

        elif noise_type == "average_soft_skill":
            target = self.rng.choice(["communication_score", "leadership_score", "teamwork_score"])
            row[target] = 5

        elif noise_type == "lower_certification_level":
            row["certifications"] = max(0, row.get("certifications", 2) - self.rng.randint(1, 2))

        elif noise_type == "limited_experience":
            row["years_experience"] = max(0, row.get("years_experience", 3) - self.rng.randint(1, 2))
            row["projects_completed"] = max(0, row.get("projects_completed", 8) - self.rng.randint(1, 3))

    def _apply_overlap(self, row: dict[str, Any], career: str) -> None:
        """Apply career overlap blending for cross-career profiles."""
        similar = self.overlap_lookup.get(career, [])
        if not similar or self.rng.random() >= 0.25:
            return

        selected = self.rng.choices(
            similar,
            weights=[s.get("similarity_percentage", 50) for s in similar],
            k=1
        )[0]

        if not selected.get("career"):
            return

        # Blend features from similar career
        blend_features = [
            "python_score", "sql_score", "javascript_score", "cloud_score",
            "devops_score", "machine_learning_score", "ui_design_score",
            "communication_score", "leadership_score"
        ]

        for feature in blend_features:
            if self.rng.random() < 0.35:
                row[feature] = self._bounded_int(
                    (row.get(feature, 5) + self.rng.randint(4, 7)) / 2, 1, 10
                )

    def _generate_row(self, career: str) -> dict[str, Any]:
        """Generate a single career profile using the statistical model."""
        seniority = self._get_seniority_level()
        anchors = self._get_career_anchor_features(career)

        row: dict[str, Any] = {}

        # Generate experience metrics
        row["years_experience"] = self._generate_years_experience(career, seniority)
        row["projects_completed"] = self._generate_projects_completed(row["years_experience"])
        row["certifications"] = self._generate_certifications(career)
        row["education_level"] = self._generate_education(career)

        # Generate technical skills
        technical_features = [
            "python", "java", "javascript", "sql",
            "machine_learning", "deep_learning", "cloud", "devops",
            "cybersecurity", "data_analysis", "database", "networking",
            "mobile", "game_dev", "testing", "business_analysis",
            "product_management", "ui_design", "ux_research"
        ]

        for feature in technical_features:
            score_key = f"{feature}_score"
            is_anchor = feature in anchors
            row[score_key] = self._bounded_int(
                self._generate_technical_skill(career, feature, seniority, is_anchor),
                1, 10
            )

        # Generate soft skills
        soft_features = [
            "communication", "leadership", "problem_solving",
            "teamwork", "agile", "research"
        ]

        for feature in soft_features:
            score_key = f"{feature}_score"
            row[score_key] = self._bounded_int(
                self._generate_soft_skill(career, feature, seniority),
                1, 10
            )

        # Store seniority for correlation engine
        row["seniority"] = seniority

        # Apply correlation constraints
        self._apply_correlation_rules(row)

        # Apply overlap blending
        self._apply_overlap(row, career)

        # Apply noise injection
        self._apply_noise(row)

        # Add metadata
        row["career"] = career
        row["salary_band"] = self._generate_salary(career)
        row["remote_preference"] = self._generate_remote_preference()
        row["career_growth_score"] = self._generate_career_growth(seniority)
        row["job_satisfaction"] = round(self.rng.uniform(3.2, 4.9), 2)
        row["work_hours_per_week"] = self.rng.randint(35, 60)
        # Use domain metadata for these fields
        row["country"] = self.rng.choice(self.countries)
        row["industry"] = self.rng.choice(self.industries)
        row["employment_type"] = self.rng.choice(self.employment_types)

        return row

    def generate(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Generate all synthetic profiles."""
        # Balance careers
        balanced_careers = [
            career for career in TARGET_CAREERS
            for _ in range(self.rows // len(TARGET_CAREERS))
        ]
        if len(balanced_careers) < self.rows:
            balanced_careers.extend(TARGET_CAREERS[:self.rows - len(balanced_careers)])
        self.rng.shuffle(balanced_careers)

        generated_rows = []
        for career in balanced_careers[:self.rows]:
            row = self._generate_row(career)
            generated_rows.append(row)

        # Generate report
        report = {
            "rows_generated": len(generated_rows),
            "career_distribution": dict(Counter(row["career"] for row in generated_rows)),
            "seniority_distribution": {},  # Would need to track during generation
            "feature_columns": FEATURE_NAMES + [
                "career", "salary_band", "remote_preference",
                "career_growth_score", "job_satisfaction",
                "work_hours_per_week", "country", "industry", "employment_type"
            ],
            "model_version": "2.0",
            "distributions_used": [
                "Truncated Normal",
                "Zero-Inflated Negative Binomial",
                "Weighted Categorical",
                "Log-Normal",
                "Beta"
            ]
        }

        return generated_rows, report


class StatisticalDatasetValidator:
    """Validate synthetic profiles against statistical model constraints."""

    def __init__(self, knowledge_root: Path | None = None) -> None:
        self.knowledge_root = knowledge_root or KNOWLEDGE_ROOT

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _load_feature_ranges(self) -> dict[str, dict[str, Any]]:
        payload = self._load_json(self.knowledge_root / "feature_ranges.json")
        ranges = {}
        for entry in payload.get("career_feature_ranges", []):
            career = entry.get("career")
            features = entry.get("features", {})
            if career:
                ranges[career] = features
        return ranges

    def validate_rows(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Validate generated rows."""
        feature_ranges = self._load_feature_ranges()

        issues = []

        for row in rows:
            career = row.get("career")
            if not career:
                continue

            career_range = feature_ranges.get(career, {})

            # Check numeric bounds
            for feature in FEATURE_NAMES:
                if feature in {"years_experience", "projects_completed", "certifications", "education_level"}:
                    continue

                if feature not in row:
                    continue

                # Normalize feature name
                feature_key = feature.replace("_score", "")
                feature_key_spaced = feature_key.replace("_", " ")

                feature_info = career_range.get(feature_key) or career_range.get(feature_key_spaced)

                if not isinstance(feature_info, dict):
                    continue

                minimum = feature_info.get("min")
                maximum = feature_info.get("max")
                value = row[feature]

                if minimum is not None and maximum is not None:
                    if value < minimum or value > maximum:
                        issues.append({
                            "career": career,
                            "feature": feature,
                            "value": value,
                            "expected_range": [minimum, maximum]
                        })

        quality_score = 100
        if issues:
            quality_score -= min(20, len(issues))

        return {
            "quality_score": quality_score,
            "warnings": ["Some values outside expected ranges"] if issues else [],
            "issues": issues[:20],
            "status": "pass" if not issues else "fail"
        }


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Generate synthetic dataset using statistical model."""
    print("=" * 60)
    print("STATISTICAL MODEL CAREER PROFILE GENERATOR")
    print("=" * 60)
    print()

    # Initialize generator
    generator = StatisticalProfileGenerator(rows=5000, seed=42)

    print("Generating career profiles using statistical model...")
    print("Distributions: Truncated Normal, Zero-Inflated NB, Weighted Categorical, Log-Normal, Beta")
    print()

    # Generate profiles
    profiles, report = generator.generate()

    print(f"✓ Generated {report['rows_generated']} career profiles")
    print()

    # Show distribution summary
    print("Career Distribution:")
    for career, count in sorted(report["career_distribution"].items()):
        print(f"  {career}: {count}")

    print()
    print("Model Information:")
    print(f"  Version: {report.get('model_version', 'N/A')}")
    print(f"  Distributions: {', '.join(report.get('distributions_used', []))}")

    # Validate
    print()
    print("Validating profiles...")
    validator = StatisticalDatasetValidator()
    validation = validator.validate_rows(profiles)

    print(f"Quality Score: {validation['quality_score']}/100")
    print(f"Status: {validation['status']}")

    if validation.get('warnings'):
        for warning in validation['warnings']:
            print(f"  Warning: {warning}")

    return profiles, report


if __name__ == "__main__":
    profiles, report = main()