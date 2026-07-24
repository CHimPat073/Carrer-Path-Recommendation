import json
import logging
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Final, Optional, Set, Tuple, cast

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.preprocessing.role_normalizer import TARGET_CAREERS

RANDOM_SEED: Final[int] = 42

KNOWLEDGE_ROOT = PROJECT_ROOT / "knowledge_base"

# Configure logging for feature lookup verification
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Canonical feature name mapping: lowercase_underscore -> JSON Title Case format
# This normalizes the feature names from feature_ranges.json to a common internal format
FEATURE_NAME_CANONICAL_MAP: Dict[str, str] = {
    # Technical skills
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
    # Soft skills
    "communication": "Communication",
    "leadership": "Leadership",
    "problem_solving": "Problem Solving",
    "teamwork": "Teamwork",
    "agile": "Agile",
    "research": "Research",
    # Metadata fields (already in correct format)
    "years_experience": "Years Experience",
    "projects_completed": "Projects Completed",
    "certifications": "Certifications",
    "education": "Education",
    "salary_band": "Salary Band",
    "remote_preference": "Remote Preference",
    "career_growth": "Career Growth",
}

# Reverse mapping: JSON Title Case -> lowercase_underscore
CANONICAL_TO_INTERNAL: Dict[str, str] = {v: k for k, v in FEATURE_NAME_CANONICAL_MAP.items()}

# All features that MUST have a lookup (no fallback allowed)
REQUIRED_FEATURES: Set[str] = {
    "python", "java", "javascript", "sql", "machine_learning", "deep_learning",
    "cloud", "devops", "cybersecurity", "data_analysis", "database", "networking",
    "mobile_development", "game_development", "testing", "business_analysis",
    "product_management", "ui_design", "ux_research", "communication", "leadership",
    "problem_solving", "teamwork", "agile", "research",
}

# Default fallback (should only be used for truly unknown features)
DEFAULT_FEATURE_RANGE: Dict[str, Any] = {"min": 1, "max": 10, "average": 5}

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

EDUCATION_LEVELS: Final[list[str]] = ["High School", "Associate", "Bachelor", "Master", "PhD"]
PERSONA_LEVELS: Final[list[str]] = ["Beginner", "Junior", "Mid-Level", "Senior", "Expert"]
PERSONA_WEIGHTS: Final[list[float]] = [0.15, 0.25, 0.30, 0.20, 0.10]


class SyntheticProfileGenerator:
    """Generate realistic synthetic career profiles from knowledge-base rules."""

    def __init__(self, rows: int = 20000, seed: int = RANDOM_SEED) -> None:
        self.rows = rows
        self.seed = seed
        self.random = random.Random(seed)
        self.knowledge_root = KNOWLEDGE_ROOT
        self.career_personas = self._load_json(self.knowledge_root / "career_personas.json").get("career_personas", [])
        self.feature_ranges = self._load_json(self.knowledge_root / "feature_ranges.json").get("career_feature_ranges", [])
        self.overlap_data = self._load_json(self.knowledge_root / "career_overlap.json").get("career_similarity_matrix", [])
        self.noise_rules = self._load_json(self.knowledge_root / "noise_rules.json").get("noise_generation_rules", {})
        self.correlation_rules = self._load_json(self.knowledge_root / "correlation_rules.json").get("correlation_rules", {}).get("rules", [])
        self.career_lookup = {item.get("career"): item for item in self.career_personas if item.get("career")}
        self.feature_lookup = self._normalize_feature_ranges()
        self._validate_all_features_have_ranges()
        self.overlap_lookup = {
            item.get("career"): item.get("similar_careers", [])
            for item in self.overlap_data
            if item.get("career")
        }
        self.user_personas = self._load_json(self.knowledge_root / "user_personas.json").get("user_personas", [])

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    @staticmethod
    def _bounded_int(value: float, low: int, high: int) -> int:
        return int(round(SyntheticProfileGenerator._clamp(value, low, high)))

    def _normalize_feature_ranges(self) -> Dict[str, Dict[str, Any]]:
        """Normalize feature ranges to use lowercase_underscore keys.

        This converts the Title Case keys from feature_ranges.json (e.g., "Machine Learning")
        to lowercase_underscore format (e.g., "machine_learning") for consistent internal use.
        """
        normalized: Dict[str, Dict[str, Any]] = {}

        for item in self.feature_ranges:
            career = item.get("career")
            if not career:
                continue

            features = item.get("features", {})
            normalized_features: Dict[str, Any] = {}

            for json_key, feature_info in features.items():
                # Find the internal key that maps to this JSON key
                internal_key = CANONICAL_TO_INTERNAL.get(json_key, json_key.lower().replace(" ", "_"))

                if internal_key not in normalized_features:
                    normalized_features[internal_key] = feature_info
                else:
                    # Merge if duplicate (shouldn't happen, but be safe)
                    logger.warning(f"Duplicate feature key for career {career}: {json_key} -> {internal_key}")

            normalized[career] = normalized_features

        return normalized

    def _validate_all_features_have_ranges(self) -> None:
        """Validate that all required features have ranges defined for each career.

        Raises AssertionError if any required feature is missing for any career.
        """
        missing_lookups: Dict[str, List[str]] = {}

        for career in TARGET_CAREERS:
            career_range = self.feature_lookup.get(career, {})
            missing_features = []

            for required_feature in REQUIRED_FEATURES:
                if required_feature not in career_range:
                    missing_features.append(required_feature)

            if missing_features:
                missing_lookups[career] = missing_features

        if missing_lookups:
            error_msg = "\n".join([
                f"  Career '{career}': missing features {features}"
                for career, features in missing_lookups.items()
            ])
            logger.error(f"Feature lookup validation failed:\n{error_msg}")
            raise AssertionError(
                f"Missing feature ranges for {len(missing_lookups)} careers. "
                f"Ensure all required features are defined in feature_ranges.json"
            )

        logger.info(f"Feature lookup validation passed: all {len(TARGET_CAREERS)} careers have all {len(REQUIRED_FEATURES)} features defined")

    def _resolve_feature_range(self, feature_name: str, career: str) -> Tuple[Dict[str, Any], bool]:
        """Resolve feature range with proper canonical mapping.

        Args:
            feature_name: Internal feature name (e.g., "python", "machine_learning")
            career: Career name for lookup

        Returns:
            Tuple of (feature_range_dict, was_found)
        """
        career_range = self.feature_lookup.get(career, {})

        # First try: direct lookup with lowercase_underscore
        if feature_name in career_range:
            return career_range[feature_name], True

        # Second try: canonical mapping (lowercase_underscore -> Title Case)
        canonical_key = FEATURE_NAME_CANONICAL_MAP.get(feature_name, feature_name)
        if canonical_key in career_range:
            return career_range[canonical_key], True

        # Check if this is a required feature (should not fall back)
        if feature_name in REQUIRED_FEATURES:
            logger.error(f"REQUIRED feature '{feature_name}' not found for career '{career}'")
            raise AssertionError(f"Missing required feature '{feature_name}' for career '{career}'")

        # Fallback for non-required features (should be rare)
        logger.warning(f"Feature '{feature_name}' not found for career '{career}', using default fallback")
        return DEFAULT_FEATURE_RANGE.copy(), False

    def _sample_from_range(self, range_info: dict[str, Any], default_mean: float, sigma: float | None = None) -> float:
        if not isinstance(range_info, dict):
            return float(default_mean)
        minimum = float(range_info.get("min", 1))
        maximum = float(range_info.get("max", 10))
        average = float(default_mean)
        spread = sigma if sigma is not None else max(0.6, (maximum - minimum) / 6)
        value = self.random.gauss(average, spread)
        return self._clamp(value, minimum, maximum)

    def _persona_level(self) -> str:
        return self.random.choices(PERSONA_LEVELS, weights=PERSONA_WEIGHTS, k=1)[0]

    def _persona_profile(self, persona_name: str) -> dict[str, Any]:
        for persona in self.user_personas:
            if persona.get("persona") == persona_name:
                return persona
        return {
            "years_of_experience": 3,
            "projects_completed": 8,
            "certifications": 2,
            "leadership": 5,
            "communication": 6,
            "problem_solving": 6,
            "expected_salary": 90000,
            "career_growth": 8,
        }

    def _education_from_text(self, education_text: str) -> str:
        lowered = education_text.lower()
        if "phd" in lowered:
            return "PhD"
        if "master" in lowered:
            return "Master"
        if "bachelor" in lowered:
            return "Bachelor"
        if "associate" in lowered:
            return "Associate"
        return "Bachelor"

    def _choose_education_level(self, career: str, persona_name: str, career_entry: dict[str, Any]) -> str:
        base_level = self._education_from_text(career_entry.get("typical_education", "Bachelor's degree"))
        options = [base_level]
        if base_level == "Bachelor":
            options.extend(["Associate", "Master"])
        elif base_level == "Master":
            options.extend(["Bachelor", "PhD"])
        elif base_level == "PhD":
            options.extend(["Master"])
        else:
            options.extend(["Bachelor", "Associate"])

        persona = self._persona_profile(persona_name)
        if persona_name == "Beginner":
            options = ["High School", "Associate", "Bachelor"]
        elif persona_name == "Junior":
            options = ["Associate", "Bachelor", "Master"]
        elif persona_name == "Mid-Level":
            options = ["Bachelor", "Master"]
        elif persona_name == "Senior":
            options = ["Bachelor", "Master", "PhD"]
        elif persona_name == "Expert":
            options = ["Master", "PhD"]

        return self.random.choice(options)

    def _career_anchor_features(self, career: str) -> list[str]:
        anchor_map = {
            "Software Engineer": ["python_score", "java_score", "javascript_score", "sql_score", "testing_score", "problem_solving_score"],
            "Backend Developer": ["python_score", "java_score", "sql_score", "database_score", "problem_solving_score"],
            "Frontend Developer": ["javascript_score", "ui_design_score", "ux_research_score", "communication_score"],
            "Full Stack Developer": ["python_score", "javascript_score", "sql_score", "cloud_score", "problem_solving_score"],
            "Mobile App Developer": ["mobile_score", "javascript_score", "ui_design_score", "testing_score"],
            "Data Scientist": ["python_score", "machine_learning_score", "data_analysis_score", "sql_score", "research_score"],
            "ML Engineer": ["python_score", "machine_learning_score", "deep_learning_score", "data_analysis_score", "research_score"],
            "Data Engineer": ["sql_score", "database_score", "python_score", "cloud_score", "devops_score"],
            "AI Research Engineer": ["python_score", "machine_learning_score", "deep_learning_score", "research_score"],
            "DevOps Engineer": ["cloud_score", "devops_score", "networking_score", "problem_solving_score"],
            "Cloud Engineer": ["cloud_score", "devops_score", "networking_score", "cybersecurity_score"],
            "Cyber Security Analyst": ["cybersecurity_score", "networking_score", "problem_solving_score", "research_score"],
            "Ethical Hacker": ["cybersecurity_score", "networking_score", "problem_solving_score", "testing_score"],
            "Network Engineer": ["networking_score", "cloud_score", "cybersecurity_score", "problem_solving_score"],
            "UI/UX Designer": ["ui_design_score", "ux_research_score", "communication_score", "research_score"],
            "Product Manager": ["product_management_score", "leadership_score", "communication_score", "agile_score", "business_analysis_score"],
            "Business Analyst": ["business_analysis_score", "communication_score", "problem_solving_score", "sql_score"],
            "QA Engineer": ["testing_score", "problem_solving_score", "communication_score", "agile_score"],
            "Database Administrator": ["database_score", "sql_score", "problem_solving_score", "cybersecurity_score"],
            "Game Developer": ["game_dev_score", "javascript_score", "python_score", "teamwork_score", "ui_design_score"],
        }
        return anchor_map.get(career, [])

    def _generate_base_row(self, career: str, persona_name: str, career_entry: dict[str, Any]) -> dict[str, Any]:
        persona = self._persona_profile(persona_name)
        anchors = self._career_anchor_features(career)
        career_range = self.feature_lookup.get(career, {})
        row: dict[str, Any] = {}

        # Use canonical keys (lowercase_underscore format) for lookup
        years_info = career_range.get("years_experience", {})
        projects_info = career_range.get("projects_completed", {})
        cert_info = career_range.get("certifications", {})

        years_floor = int(years_info.get("min", 0)) if isinstance(years_info, dict) else 0
        years_ceiling = int(years_info.get("max", 20)) if isinstance(years_info, dict) else 20
        projects_floor = int(projects_info.get("min", 0)) if isinstance(projects_info, dict) else 0
        projects_ceiling = int(projects_info.get("max", 30)) if isinstance(projects_info, dict) else 30
        cert_floor = int(cert_info.get("min", 0)) if isinstance(cert_info, dict) else 0
        cert_ceiling = int(cert_info.get("max", 6)) if isinstance(cert_info, dict) else 6

        experience_anchor = max(years_floor, min(years_ceiling, persona.get("years_of_experience", 3) + self.random.randint(-1, 2)))
        projects_anchor = max(projects_floor, min(projects_ceiling, persona.get("projects_completed", 8) + self.random.randint(-2, 3)))
        cert_anchor = max(cert_floor, min(cert_ceiling, persona.get("certifications", 2) + self.random.randint(-1, 1)))

        row["years_experience"] = self._bounded_int(self.random.gauss(experience_anchor, 1.6), years_floor, years_ceiling)
        row["projects_completed"] = self._bounded_int(self.random.gauss(projects_anchor, 2.2), projects_floor, projects_ceiling)
        row["certifications"] = self._bounded_int(self.random.gauss(cert_anchor, 1.0), cert_floor, cert_ceiling)
        row["education_level"] = self._choose_education_level(career, persona_name, career_entry)

        for feature in FEATURE_NAMES:
            if feature in {"years_experience", "projects_completed", "certifications", "education_level"}:
                continue

            feature_name = feature.replace("_score", "")

            # Use canonical resolution with proper mapping
            feature_info, was_found = self._resolve_feature_range(feature_name, career)

            mean_value = float(feature_info.get("average", 5))
            if feature in anchors:
                mean_value += 1.0
            if feature in {"communication_score", "leadership_score", "problem_solving_score", "teamwork_score", "research_score"}:
                mean_value += 0.2
            if feature == "communication_score":
                mean_value += max(0, (persona.get("communication", 6) - 6) * 0.25)
            if feature == "leadership_score":
                mean_value += max(0, (persona.get("leadership", 5) - 5) * 0.25)
            if feature == "problem_solving_score":
                mean_value += max(0, (persona.get("problem_solving", 6) - 6) * 0.25)

            sigma = max(0.5, (float(feature_info.get("max", 10)) - float(feature_info.get("min", 1))) / 4)
            sampled = self._sample_from_range(feature_info, default_mean=mean_value, sigma=sigma)
            row[feature] = self._bounded_int(sampled, int(feature_info.get("min", 1)), int(feature_info.get("max", 10)))

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Generated base row for {career}")
            for k,v in row.items():
                logger.debug(f"  {k}: {v}")
        return row

    def _apply_correlation_rules(self, row: dict[str, Any]) -> None:
        for rule in self.correlation_rules:
            pair = rule.get("feature_pair", [])
            if len(pair) != 2:
                continue
            first, second = pair[0], pair[1]
            first_key = self._normalize_feature_name(first)
            second_key = self._normalize_feature_name(second)
            if first_key not in row or second_key not in row:
                continue
            strength = float(rule.get("strength", 0.5))
            if second_key == "years_experience":
                delta = max(0, int(row[first_key]) - 5)
                row[second_key] = self._bounded_int(row[second_key] + delta * strength * 0.4, 0, 20)
            else:
                delta = max(0, int(row[first_key]) - 5)
                row[second_key] = self._bounded_int(row[second_key] + delta * strength * 0.3, 1, 10)

    @staticmethod
    def _normalize_feature_name(value: str) -> str:
        cleaned = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
        aliases = {
            "years_experience": "years_experience",
            "projects_completed": "projects_completed",
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
        }
        return aliases.get(cleaned, cleaned)

    def _apply_noise(self, row: dict[str, Any]) -> None:
        rules = self.noise_rules.get("rules", [])
        if self.random.random() >= 0.08 or not rules:
            return
        selected = self.random.choice(rules)
        noise_type = selected.get("type")
        if noise_type == "weak_skill":
            target_skill = self.random.choice(["python_score", "sql_score", "cloud_score", "communication_score", "testing_score", "database_score"])
            row[target_skill] = max(1, row.get(target_skill, 5) - self.random.randint(1, 2))
        elif noise_type == "interest_mismatch":
            row["machine_learning_score"] = min(10, row.get("machine_learning_score", 5) + 1)
            row["python_score"] = max(1, row.get("python_score", 5) - 1)
        elif noise_type == "average_soft_skill":
            soft_target = self.random.choice(["communication_score", "leadership_score", "teamwork_score"])
            row[soft_target] = 5
        elif noise_type == "lower_certification_level":
            row["certifications"] = max(0, row.get("certifications", 2) - self.random.randint(1, 2))
        elif noise_type == "limited_experience":
            row["years_experience"] = max(0, row.get("years_experience", 3) - self.random.randint(1, 2))
            row["projects_completed"] = max(0, row.get("projects_completed", 8) - self.random.randint(1, 3))

    def _apply_overlap(self, row: dict[str, Any], career: str) -> None:
        similar = self.overlap_lookup.get(career, [])
        if not similar or self.random.random() >= 0.25:
            return
        selected = self.random.choices(similar, weights=[item.get("similarity_percentage", 50) for item in similar], k=1)[0]
        if not selected.get("career"):
            return
        for feature in ["python_score", "sql_score", "javascript_score", "cloud_score", "devops_score", "machine_learning_score", "ui_design_score", "communication_score", "leadership_score"]:
            if self.random.random() < 0.35:
                row[feature] = self._bounded_int((row.get(feature, 5) + self.random.randint(4, 7)) / 2, 1, 10)

    def _finalize_row(self, row: dict[str, Any], career: str, career_entry: dict[str, Any]) -> dict[str, Any]:
        salary_min, salary_max = self._parse_salary_band(career_entry.get("salary_band", "$80k-$120k USD"))
        base_salary = self.random.randint(salary_min, salary_max)
        row["career"] = career
        row["salary_band"] = base_salary
        row["remote_preference"] = self.random.choice(["Remote", "Hybrid", "On-site"])
        row["career_growth_score"] = self._bounded_int(6 + self.random.random() * 3, 1, 10)
        row["job_satisfaction"] = round(self.random.uniform(3.2, 4.9), 2)
        row["work_hours_per_week"] = self.random.randint(35, 60)
        row["country"] = self.random.choice(["USA", "Canada", "UK", "India", "Australia", "Germany"])
        row["industry"] = self.random.choice(["Technology", "Finance", "Healthcare", "Education", "Retail", "Consulting", "Gaming"])
        row["employment_type"] = self.random.choice(["Full-time", "Part-time", "Contract", "Freelance"])
        return row

    @staticmethod
    def _parse_salary_band(band_text: str) -> tuple[int, int]:
        numbers = [int(n) for n in re.findall(r"\d+", band_text)]
        if len(numbers) >= 2:
            return numbers[0] * 1000, numbers[1] * 1000
        if numbers:
            return numbers[0] * 1000, numbers[0] * 1000
        return 60000, 120000

    def generate(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        balanced_careers = [career for career in TARGET_CAREERS for _ in range(self.rows // len(TARGET_CAREERS))]
        if len(balanced_careers) < self.rows:
            balanced_careers.extend(TARGET_CAREERS[: self.rows - len(balanced_careers)])
        self.random.shuffle(balanced_careers)

        generated_rows: list[dict[str, Any]] = []
        for career in balanced_careers[: self.rows]:
            persona_name = self._persona_level()
            row = self._generate_base_row(career, persona_name, self.career_lookup.get(career, {}))
            self._apply_correlation_rules(row)
            self._apply_overlap(row, career)
            self._apply_noise(row)
            row = self._finalize_row(row, career, self.career_lookup.get(career, {}))
            generated_rows.append(row)

        report = {
            "rows_generated": len(generated_rows),
            "career_distribution": dict(Counter(row["career"] for row in generated_rows)),
            "persona_distribution": dict(Counter(self._persona_level() for _ in generated_rows)),
            "feature_columns": FEATURE_NAMES + ["career", "salary_band", "remote_preference", "career_growth_score", "job_satisfaction", "work_hours_per_week", "country", "industry", "employment_type"],
        }
        return generated_rows, report


class DatasetValidator:
    """Validate synthetic profiles against career-specific ranges and data quality rules."""

    def __init__(self, knowledge_root: Path | None = None) -> None:
        self.knowledge_root = knowledge_root or KNOWLEDGE_ROOT
        self.feature_ranges = self._load_feature_ranges()
        self.career_personas = self._load_career_personas()

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _load_feature_ranges(self) -> dict[str, dict[str, Any]]:
        payload = self._load_json(self.knowledge_root / "feature_ranges.json")
        ranges: dict[str, dict[str, Any]] = {}
        for entry in payload.get("career_feature_ranges", []):
            career = entry.get("career")
            features = entry.get("features", {})
            if not career:
                continue
            converted: dict[str, Any] = {}
            for feature_name, feature_info in features.items():
                normalized = SyntheticProfileGenerator._normalize_feature_name(feature_name)
                if isinstance(feature_info, dict):
                    converted[normalized] = feature_info
            ranges[career] = converted
        return ranges

    def _load_career_personas(self) -> dict[str, dict[str, Any]]:
        payload = self._load_json(self.knowledge_root / "career_personas.json")
        personas: dict[str, dict[str, Any]] = {}
        for entry in payload.get("career_personas", []):
            career = entry.get("career")
            if career:
                personas[career] = entry
        return personas

    @staticmethod
    def _education_rank(education_level: Any) -> int:
        if not isinstance(education_level, str):
            return 0
        ranking = {"high school": 1, "associate": 2, "bachelor": 3, "master": 4, "phd": 5}
        lowered = education_level.strip().lower()
        for key, value in ranking.items():
            if lowered.startswith(key):
                return value
        return 0

    def _validate_required_metadata(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        required_fields = ["education_level", "remote_preference", "country", "industry", "employment_type"]
        missing = [field for field in required_fields if any(not row.get(field) for row in rows)]
        return {
            "status": "pass" if not missing else "fail",
            "missing_fields": missing,
        }

    def _validate_numeric_bounds(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        issues: list[dict[str, Any]] = []
        for row in rows:
            career = row.get("career")
            if not career:
                continue
            career_range = self.feature_ranges.get(career, {})
            for feature in ["years_experience", "projects_completed", "certifications", "python_score", "java_score", "javascript_score", "sql_score", "machine_learning_score", "deep_learning_score", "cloud_score", "devops_score", "cybersecurity_score", "data_analysis_score", "database_score", "networking_score", "mobile_score", "game_dev_score", "testing_score", "business_analysis_score", "product_management_score", "ui_design_score", "ux_research_score", "communication_score", "leadership_score", "problem_solving_score", "teamwork_score", "agile_score", "research_score"]:
                if feature not in row:
                    continue
                value = row[feature]
                if feature in {"years_experience", "projects_completed", "certifications"}:
                    feature_info = career_range.get("years experience" if feature == "years_experience" else "projects completed" if feature == "projects_completed" else "certifications")
                else:
                    feature_info = career_range.get(feature)
                if not isinstance(feature_info, dict):
                    continue
                minimum = feature_info.get("min")
                maximum = feature_info.get("max")
                if minimum is not None and maximum is not None and (value < minimum or value > maximum):
                    issues.append({"career": career, "feature": feature, "value": value, "expected_range": [minimum, maximum]})
        return {"status": "pass" if not issues else "fail", "issues": issues[:20]}

    def _validate_impossible_education(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        issues: list[dict[str, Any]] = []
        for row in rows:
            career = row.get("career")
            education_level = row.get("education_level")
            if not isinstance(career, str) or not isinstance(education_level, str):
                continue
            persona = self.career_personas.get(career, {})
            typical_education = str(persona.get("typical_education", "Bachelor's degree")).lower()
            required_rank = self._education_rank(typical_education)
            observed_rank = self._education_rank(education_level)
            if required_rank and observed_rank and observed_rank < required_rank:
                issues.append({"career": career, "education_level": education_level, "typical_education": persona.get("typical_education")})
        return {"status": "pass" if not issues else "fail", "issues": issues[:10]}

    def validate_rows(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        validation = {
            "required_metadata": self._validate_required_metadata(rows),
            "numeric_bounds": self._validate_numeric_bounds(rows),
            "education_consistency": self._validate_impossible_education(rows),
        }

        warnings = []
        if validation["required_metadata"]["status"] != "pass":
            warnings.append("Missing required metadata values")
        if validation["numeric_bounds"]["status"] != "pass":
            warnings.append("Some rows fall outside career-specific value ranges")
        if validation["education_consistency"]["status"] != "pass":
            warnings.append("Some education levels contradict the typical education for a career")

        quality_score = 100
        quality_score -= 15 if validation["required_metadata"]["status"] != "pass" else 0
        quality_score -= 15 if validation["numeric_bounds"]["status"] != "pass" else 0
        quality_score -= 10 if validation["education_consistency"]["status"] != "pass" else 0
        quality_score = max(0, min(100, quality_score))

        career_validation = {}
        careers: set[str] = {cast(str, row.get("career")) for row in rows if row.get("career")}
        for career in sorted(careers):
            career_rows = [row for row in rows if row.get("career") == career]
            career_validation[career] = {
                "row_count": len(career_rows),
                "average_experience": round(sum(row.get("years_experience", 0) for row in career_rows) / max(1, len(career_rows)), 2),
                "average_salary": round(sum(row.get("salary_band", 0) for row in career_rows) / max(1, len(career_rows)), 2),
            }

        return {
            "quality_score": quality_score,
            "warnings": warnings,
            "career_validation": career_validation,
            "checks": validation,
        }
