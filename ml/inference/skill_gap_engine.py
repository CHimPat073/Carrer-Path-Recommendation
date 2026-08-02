"""CareerPilot-AI Skill Gap Engine.

This module compares the user's current skill profile against the expected
profile of a recommended career. It produces a deterministic, explainable
gap analysis.

It does NOT:

* load ML models
* perform prediction
* compute SHAP values
* compute confidence scores
* recommend learning resources
* access any UI

Its sole responsibility is skill gap analysis and overall readiness scoring.

All career information is read from existing JSON files in ``knowledge_base/``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Iterable

from ml.config import setup_logging


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
KNOWLEDGE_BASE_DIR: Final[Path] = PROJECT_ROOT / "knowledge_base"
FEATURE_RANGES_PATH: Final[Path] = KNOWLEDGE_BASE_DIR / "feature_ranges.json"


# Skill scale expected across all numeric skill features.
SKILL_SCALE_MIN: Final[int] = 0
SKILL_SCALE_MAX: Final[int] = 10


# Default user skill score applied for any required skill the user did not
# report. Using zero is the safest lower-bound assumption - it never inflates
# the user's readiness.
_MISSING_USER_SCORE: Final[int] = 0


# Default readiness labels returned in the output payload.
READINESS_LABELS: Final[tuple[str, ...]] = (
    "Excellent",
    "Good",
    "Average",
    "Poor",
)


# Threshold pairs used for status and readiness categorization.
# Each tuple is (upper_inclusive, label).
_STATUS_THRESHOLDS: Final[tuple[tuple[float, str], ...]] = (
    (0.0, "Excellent"),
    (1.0, "Good"),
    (2.0, "Needs Improvement"),
)


# Improvement-priority map, ordered from most urgent to least urgent.
_STATUS_TO_PRIORITY: Final[dict[str, str]] = {
    "Critical": "High",
    "Needs Improvement": "Medium",
    "Good": "Low",
    "Excellent": "None",
}


# Readiness thresholds (overall score, ascending in severity).
_READINESS_THRESHOLDS: Final[tuple[tuple[float, str], ...]] = (
    (40.0, "Poor"),
    (60.0, "Average"),
    (80.0, "Good"),
)


logger: Final[logging.Logger] = setup_logging("ml.inference.skill_gap_engine")


class SkillGapEngineError(ValueError):
    """Base class for skill gap engine failures."""


class MissingRecommendationPayloadError(SkillGapEngineError):
    """Raised when ``recommendation_result`` is missing or malformed."""


class MissingUserInputError(SkillGapEngineError):
    """Raised when ``user_input`` is missing or malformed."""


class UnknownCareerError(SkillGapEngineError):
    """Raised when the recommended career cannot be located in the knowledge base."""


class MissingSkillDataError(SkillGapEngineError):
    """Raised when the knowledge base cannot supply required skill scores."""


@dataclass(frozen=True)
class SkillGap:
    """Per-skill gap analysis record."""

    skill: str
    current: int
    required: int
    gap: int
    gap_percentage: float
    status: str
    priority: str


@dataclass(frozen=True)
class SkillGapReport:
    """Structured gap analysis report returned to callers."""

    recommended_career: str
    overall_readiness: float
    readiness_level: str
    skills: list[SkillGap]
    strengths: list[str]
    critical_skills: list[str]
    improvement_order: list[str]
    education_path: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert the report to a JSON-serializable dictionary."""
        return {
            "recommended_career": self.recommended_career,
            "overall_readiness": round(self.overall_readiness, 2),
            "readiness_level": self.readiness_level,
            "skills": [
                {
                    "skill": item.skill,
                    "current": item.current,
                    "required": item.required,
                    "gap": item.gap,
                    "gap_percentage": round(item.gap_percentage, 2),
                    "status": item.status,
                    "priority": item.priority,
                }
                for item in self.skills
            ],
            "strengths": self.strengths,
            "critical_skills": self.critical_skills,
            "improvement_order": self.improvement_order,
            "education_path": self.education_path,
        }


class SkillGapEngine:
    """Compare user skills against the skill expectations of a recommended career.

    The engine is intentionally framework-agnostic. It accepts plain Python
    dictionaries and returns plain Python dictionaries, so the same module is
    reusable from Streamlit, Flask, FastAPI, or any other backend without
    change.
    """

    def __init__(self, knowledge_base_dir: Path | str | None = None) -> None:
        self.knowledge_base_dir = (
            Path(knowledge_base_dir) if knowledge_base_dir is not None else KNOWLEDGE_BASE_DIR
        )
        self.feature_ranges_path = self.knowledge_base_dir / FEATURE_RANGES_PATH.name
        self._feature_ranges = self._load_feature_ranges(self.feature_ranges_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def analyze(
        self,
        user_input: dict[str, Any] | None,
        recommendation_result: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Run the full skill gap analysis.

        Args:
            user_input: Dictionary containing the user's current skill
                scores. Keys should match the project's skill feature naming
                convention (e.g. ``python_score``) or the human-friendly
                display names (e.g. ``Python``).
            recommendation_result: Output of the recommendation engine. Must
                include ``recommended_career`` and ``required_skills``.

        Returns:
            A dictionary matching the canonical skill gap output schema.

        Raises:
            MissingRecommendationPayloadError: If the recommendation payload
                is missing or malformed.
            MissingUserInputError: If the user input is missing or malformed.
            UnknownCareerError: If the recommended career cannot be resolved
                against the knowledge base.
            MissingSkillDataError: If the knowledge base cannot supply the
                expected proficiency scores for the required skills.
        """
        recommended_career = self._validate_recommendation_result(recommendation_result)
        normalized_user_skills = self._normalize_user_skills(user_input)
        required_skills = self._validate_required_skills(recommendation_result)
        expected_scores = self._resolve_expected_scores(recommended_career, required_skills)

        gap_records: list[SkillGap] = []
        for skill in required_skills:
            required_value = expected_scores[skill]
            current_value = self._resolve_user_score(normalized_user_skills, skill)
            gap_value = required_value - current_value  # positive = user is lacking the skill.
            status = self._classify_status(gap_value)
            priority = _STATUS_TO_PRIORITY[status]
            gap_records.append(
                SkillGap(
                    skill=skill,
                    current=current_value,
                    required=required_value,
                    gap=gap_value,
                    gap_percentage=self._compute_gap_percentage(current_value, required_value),
                    status=status,
                    priority=priority,
                )
            )

        overall_readiness = self._compute_overall_readiness(gap_records)
        readiness_level = self._classify_readiness(overall_readiness)

        strengths = self._collect_strengths(gap_records)
        critical_skills = self._collect_critical_skills(gap_records)
        improvement_order = self._order_skill_improvements(gap_records)

        report = SkillGapReport(
            recommended_career=recommended_career,
            overall_readiness=overall_readiness,
            readiness_level=readiness_level,
            skills=gap_records,
            strengths=strengths,
            critical_skills=critical_skills,
            improvement_order=improvement_order,
            education_path=list(recommendation_result.get("education_path", []) or []),
        )

        logger.info(
            "Skill gap analysis completed for %s | readiness=%.2f (%s)",
            recommended_career,
            overall_readiness,
            readiness_level,
        )
        return report.to_dict()

    # ------------------------------------------------------------------
    # Knowledge base loading
    # ------------------------------------------------------------------
    @staticmethod
    def _load_feature_ranges(path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise MissingSkillDataError(f"Missing knowledge base file: {path}")
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise MissingSkillDataError(
                f"Feature ranges file must contain an object: {path}"
            )
        return data

    def _resolve_career_entry(self, career: str) -> dict[str, Any]:
        for entry in self._feature_ranges.get("career_feature_ranges", []):
            if not isinstance(entry, dict):
                continue
            if entry.get("career") == career:
                features = entry.get("features", {})
                if not isinstance(features, dict) or not features:
                    raise MissingSkillDataError(
                        f"Knowledge base is missing feature ranges for '{career}'."
                    )
                return features
        raise UnknownCareerError(
            f"Career '{career}' has no entry in feature_ranges.json."
        )

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_recommendation_result(
        recommendation_result: dict[str, Any] | None,
    ) -> str:
        if not isinstance(recommendation_result, dict):
            raise MissingRecommendationPayloadError(
                "recommendation_result must be provided as a dict."
            )
        recommended_career = recommendation_result.get("recommended_career")
        if not isinstance(recommended_career, str) or not recommended_career.strip():
            raise MissingRecommendationPayloadError(
                "recommendation_result must contain a non-empty 'recommended_career'."
            )
        return recommended_career.strip()

    @staticmethod
    def _validate_required_skills(recommendation_result: dict[str, Any]) -> list[str]:
        required_skills = recommendation_result.get("required_skills", [])
        if not isinstance(required_skills, list) or not required_skills:
            raise MissingRecommendationPayloadError(
                "recommendation_result must contain a non-empty 'required_skills' list."
            )
        cleaned: list[str] = []
        for skill in required_skills:
            if not isinstance(skill, str):
                continue
            value = skill.strip()
            if value and value not in cleaned:
                cleaned.append(value)
        if not cleaned:
            raise MissingRecommendationPayloadError(
                "recommendation_result 'required_skills' must contain skill names."
            )
        return cleaned

    @staticmethod
    def _normalize_user_skills(user_input: dict[str, Any] | None) -> dict[str, int]:
        if user_input is None:
            raise MissingUserInputError("user_input must be provided.")
        if not isinstance(user_input, dict):
            raise MissingUserInputError(
                f"Expected user_input to be a dict, got {type(user_input).__name__}."
            )
        normalized: dict[str, int] = {}
        for key, raw_value in user_input.items():
            if not isinstance(key, str):
                continue
            value = SkillGapEngine._coerce_skill_value(raw_value)
            if value is not None:
                normalized[key] = value
        return normalized

    @staticmethod
    def _coerce_skill_value(raw_value: Any) -> int | None:
        if isinstance(raw_value, bool):
            # ``bool`` is a subclass of ``int`` in Python - treat explicitly.
            return int(raw_value) if raw_value in (0, 1) else None
        if isinstance(raw_value, (int, float)):
            return SkillGapEngine._clamp_score(int(round(float(raw_value))))
        return None

    @staticmethod
    def _clamp_score(value: int) -> int:
        if value < SKILL_SCALE_MIN:
            return SKILL_SCALE_MIN
        if value > SKILL_SCALE_MAX:
            return SKILL_SCALE_MAX
        return value

    # ------------------------------------------------------------------
    # Score resolution
    # ------------------------------------------------------------------
    def _resolve_expected_scores(
        self, career: str, required_skills: Iterable[str]
    ) -> dict[str, int]:
        career_features = self._resolve_career_entry(career)
        expected: dict[str, int] = {}
        missing_required: list[str] = []
        for skill in required_skills:
            score = self._match_skill_to_required(career_features, skill)
            if score is None:
                missing_required.append(skill)
                continue
            expected[skill] = SkillGapEngine._clamp_score(score)
        if missing_required:
            raise MissingSkillDataError(
                f"Knowledge base is missing expected scores for {missing_required}."
            )
        return expected

    @staticmethod
    def _match_skill_to_required(
        career_features: dict[str, Any], skill: str
    ) -> int | None:
        # Direct hit on the KB feature key (``Python``).
        direct = career_features.get(skill)
        if isinstance(direct, dict):
            average = direct.get("average")
            if isinstance(average, (int, float)):
                return SkillGapEngine._round_required(average)
            maximum = direct.get("max")
            if isinstance(maximum, (int, float)):
                return SkillGapEngine._round_required(maximum)
        # Case-insensitive lookup as a fallback.
        for key, value in career_features.items():
            if isinstance(key, str) and key.lower() == skill.lower():
                if isinstance(value, dict):
                    average = value.get("average")
                    if isinstance(average, (int, float)):
                        return SkillGapEngine._round_required(average)
        return None

    @staticmethod
    def _round_required(value: float) -> int:
        # The knowledge base stores ``average``/``max`` floats; required
        # proficiency is exposed as an integer in the canonical schema.
        return int(round(float(value)))

    @staticmethod
    def _resolve_user_score(
        normalized_user_skills: dict[str, int], skill: str
    ) -> int:
        # Try several naming conventions to be friendly to caller payloads.
        candidates = (
            skill,
            skill.replace(" ", "_"),
            skill.lower(),
            skill.lower().replace(" ", "_"),
            f"{skill.lower().replace(' ', '_')}_score",
        )
        for key, value in normalized_user_skills.items():
            for candidate in candidates:
                if key == candidate:
                    return SkillGapEngine._clamp_score(value)
        return SkillGapEngine._clamp_score(_MISSING_USER_SCORE)

    # ------------------------------------------------------------------
    # Classification helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _classify_status(gap: int) -> str:
        if gap <= 0:
            return "Excellent"
        for upper, label in _STATUS_THRESHOLDS:
            if gap <= upper:
                return label
        return "Critical"

    @staticmethod
    def _classify_readiness(overall_readiness: float) -> str:
        for upper, label in _READINESS_THRESHOLDS:
            if overall_readiness <= upper:
                return label
        return "Excellent"

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_gap_percentage(current: int, required: int) -> float:
        if required <= 0:
            return 0.0
        ratio = (current - required) / float(required)
        # Bound to [-100, 100] to keep extreme gaps interpretable.
        return max(min(ratio * 100.0, 100.0), -100.0)

    @staticmethod
    def _compute_overall_readiness(records: Iterable[SkillGap]) -> float:
        # Weighted skill similarity: each skill is weighted by its expected
        # required score, so higher-impact skills dominate the overall
        # readiness without artificially favouring low-skill careers.
        total_weight = 0.0
        total_similarity = 0.0
        for record in records:
            required = max(record.required, 1)
            weight = float(required)
            # Cosine-style similarity: 0 when the user is at zero, 1 when the
            # user meets the requirement, capped at 1 even if they exceed it.
            similarity = min(record.current / float(required), 1.0)
            similarity = max(similarity, 0.0)
            total_weight += weight
            total_similarity += weight * similarity
        if total_weight <= 0:
            return 0.0
        readiness = (total_similarity / total_weight) * 100.0
        return max(min(readiness, 100.0), 0.0)

    @staticmethod
    def _collect_strengths(records: Iterable[SkillGap]) -> list[str]:
        return [record.skill for record in records if record.gap <= 0]

    @staticmethod
    def _collect_critical_skills(records: Iterable[SkillGap]) -> list[str]:
        return [record.skill for record in records if record.status == "Critical"]

    @staticmethod
    def _order_skill_improvements(records: Iterable[SkillGap]) -> list[str]:
        # Sort by descending gap (worst first); preserve listing order for
        # ties so the output is deterministic across runs.
        return [
            record.skill
            for record in sorted(records, key=lambda item: (-item.gap, item.skill))
            if record.gap > 0
        ]


__all__ = [
    "MissingRecommendationPayloadError",
    "MissingSkillDataError",
    "MissingUserInputError",
    "SkillGapEngine",
    "SkillGapEngineError",
    "SkillGapReport",
    "UnknownCareerError",
]
