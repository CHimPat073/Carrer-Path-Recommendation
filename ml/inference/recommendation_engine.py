"""CareerPilot-AI Recommendation Engine.

This module is a pure orchestration layer. It merges already computed
prediction, confidence, SHAP, and knowledge-base outputs into one final
recommendation object.

It does not:

* load ML models
* preprocess data
* perform prediction
* compute SHAP values
* compute confidence scores

The engine is intentionally framework-agnostic and explainable. All career
information comes from the existing JSON files in ``knowledge_base/``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from ml.config import setup_logging


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
KNOWLEDGE_BASE_DIR: Final[Path] = PROJECT_ROOT / "knowledge_base"
CAREERS_PATH: Final[Path] = KNOWLEDGE_BASE_DIR / "careers.json"
CAREER_PERSONAS_PATH: Final[Path] = KNOWLEDGE_BASE_DIR / "career_personas.json"
ROADMAPS_PATH: Final[Path] = KNOWLEDGE_BASE_DIR / "roadmaps.json"
DOMAIN_METADATA_PATH: Final[Path] = KNOWLEDGE_BASE_DIR / "domain_metadata.json"


logger: Final[logging.Logger] = setup_logging("ml.inference.recommendation_engine")


class RecommendationEngineError(ValueError):
    """Base class for recommendation engine failures."""


class MissingPredictionOutputError(RecommendationEngineError):
    """Raised when the prediction result payload is missing or incomplete."""


class MissingConfidenceOutputError(RecommendationEngineError):
    """Raised when the confidence result payload is missing or incomplete."""


class MissingSHAPOutputError(RecommendationEngineError):
    """Raised when the SHAP result payload is missing or incomplete."""


class MissingKnowledgeBaseEntryError(RecommendationEngineError):
    """Raised when the knowledge base cannot supply the requested career data."""


class UnknownCareerError(MissingKnowledgeBaseEntryError):
    """Raised when the predicted career does not exist in the knowledge base."""


class InvalidRecommendationInputError(RecommendationEngineError):
    """Raised when a payload has the wrong type or shape."""


@dataclass(frozen=True)
class _CareerKnowledge:
    name: str
    description: str
    career_family: str
    industry: list[str]
    salary_range: str
    required_skills: list[str]
    education_path: list[str]
    next_roles: list[str]


class RecommendationEngine:
    """Merge model outputs, SHAP explanations, and KB metadata."""

    def __init__(self, knowledge_base_dir: Path | str | None = None) -> None:
        self.knowledge_base_dir = (
            Path(knowledge_base_dir) if knowledge_base_dir is not None else KNOWLEDGE_BASE_DIR
        )

        self.careers_path = self.knowledge_base_dir / CAREERS_PATH.name
        self.career_personas_path = self.knowledge_base_dir / CAREER_PERSONAS_PATH.name
        self.roadmaps_path = self.knowledge_base_dir / ROADMAPS_PATH.name
        self.domain_metadata_path = self.knowledge_base_dir / DOMAIN_METADATA_PATH.name

        self._careers = self._load_json(self.careers_path)
        self._career_personas = self._load_json(self.career_personas_path)
        self._roadmaps = self._load_json(self.roadmaps_path)
        self._domain_metadata = self._load_json(self.domain_metadata_path)

        self._careers_by_name = {
            str(item["name"]): item for item in self._careers.get("careers", [])
        }
        self._personas_by_career = {
            str(item["career"]): item for item in self._career_personas.get("career_personas", [])
        }
        self._roadmaps_by_career = {
            str(item["career"]): item for item in self._roadmaps.get("roadmaps", [])
        }
        self._career_families = self._domain_metadata.get("career_families", {}).get("families", {})

    def recommend(
        self,
        prediction_result: dict[str, Any] | None,
        confidence_result: dict[str, Any] | None,
        shap_result: dict[str, Any] | None,
        user_input: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build one final recommendation object from already computed outputs."""
        self._validate_user_input(user_input)
        predicted_career = self._extract_predicted_career(prediction_result)
        normalized_top3 = self._normalize_top3(prediction_result)
        confidence = self._normalize_confidence(confidence_result, normalized_top3)
        strengths, improvement_areas = self._extract_shap_contributions(shap_result)
        career_knowledge = self._build_career_knowledge(predicted_career)

        recommendation = {
            "recommended_career": predicted_career,
            "confidence": confidence["confidence"],
            "confidence_level": confidence["confidence_level"],
            "prediction_reliability": confidence["prediction_reliability"],
            "top3_careers": normalized_top3,
            "career_information": {
                "description": career_knowledge.description,
                "career_family": career_knowledge.career_family,
                "industry": career_knowledge.industry,
                "salary_range": career_knowledge.salary_range,
            },
            "strengths": strengths,
            "improvement_areas": improvement_areas,
            "required_skills": career_knowledge.required_skills,
            "education_path": career_knowledge.education_path,
            "next_roles": career_knowledge.next_roles,
        }

        logger.info(
            "Recommendation assembled for %s with confidence %.2f",
            recommendation["recommended_career"],
            recommendation["confidence"],
        )
        return recommendation

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise MissingKnowledgeBaseEntryError(f"Missing knowledge base file: {path}")
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise MissingKnowledgeBaseEntryError(f"Knowledge base file must contain an object: {path}")
        return data

    @staticmethod
    def _validate_user_input(user_input: dict[str, Any] | None) -> None:
        if user_input is not None and not isinstance(user_input, dict):
            raise InvalidRecommendationInputError(
                f"Expected user_input to be a dict, got {type(user_input).__name__}."
            )

    def _extract_predicted_career(self, prediction_result: dict[str, Any] | None) -> str:
        if not isinstance(prediction_result, dict):
            raise MissingPredictionOutputError("prediction_result must be provided as a dict.")

        predicted_career = prediction_result.get("prediction") or prediction_result.get("predicted_career")
        if not isinstance(predicted_career, str) or not predicted_career.strip():
            raise MissingPredictionOutputError("prediction_result must contain a non-empty prediction.")

        if predicted_career not in self._personas_by_career:
            raise UnknownCareerError(f"Unknown career in prediction_result: {predicted_career}")

        if predicted_career not in self._careers_by_name:
            raise MissingKnowledgeBaseEntryError(
                f"Career '{predicted_career}' is missing from careers.json."
            )

        return predicted_career

    def _normalize_top3(self, prediction_result: dict[str, Any] | None) -> list[dict[str, Any]]:
        if not isinstance(prediction_result, dict):
            raise MissingPredictionOutputError("prediction_result must be provided as a dict.")

        top3 = prediction_result.get("top3") or prediction_result.get("top3_careers")
        if not isinstance(top3, list) or not top3:
            raise MissingPredictionOutputError("prediction_result must contain a non-empty top3 list.")

        normalized: list[dict[str, Any]] = []
        for entry in top3:
            if not isinstance(entry, dict):
                raise MissingPredictionOutputError("Each top3 entry must be a dictionary.")
            career = entry.get("career")
            probability = entry.get("probability")
            if not isinstance(career, str) or not career.strip():
                raise MissingPredictionOutputError("Each top3 entry must include a non-empty career name.")
            if not isinstance(probability, (int, float)):
                raise MissingPredictionOutputError(
                    f"Top3 probability for '{career}' must be numeric."
                )
            normalized.append(
                {
                    "career": career,
                    "probability": round(float(probability), 4),
                }
            )

        normalized.sort(key=lambda item: item["probability"], reverse=True)
        return normalized[:3]

    def _normalize_confidence(
        self,
        confidence_result: dict[str, Any] | None,
        normalized_top3: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not isinstance(confidence_result, dict):
            raise MissingConfidenceOutputError("confidence_result must be provided as a dict.")

        confidence_value = confidence_result.get("confidence")
        if confidence_value is None:
            confidence_value = confidence_result.get("confidence_score")
        if not isinstance(confidence_value, (int, float)):
            raise MissingConfidenceOutputError("confidence_result must include a numeric confidence value.")

        confidence_level = confidence_result.get("confidence_level")
        if not isinstance(confidence_level, str) or not confidence_level.strip():
            raise MissingConfidenceOutputError("confidence_result must include confidence_level.")

        prediction_reliability = confidence_result.get("prediction_reliability")
        if not isinstance(prediction_reliability, str) or not prediction_reliability.strip():
            raise MissingConfidenceOutputError("confidence_result must include prediction_reliability.")

        confidence_top3 = confidence_result.get("top3")
        if isinstance(confidence_top3, list) and confidence_top3:
            expected = [item["career"] for item in normalized_top3]
            observed = [item.get("career") for item in confidence_top3 if isinstance(item, dict)]
            if observed[: len(expected)] and observed[: len(expected)] != expected[: len(observed[: len(expected)])]:
                logger.debug("Confidence top3 ordering differs from prediction top3; normalized prediction top3 will be used.")

        return {
            "confidence": round(float(confidence_value), 2),
            "confidence_level": confidence_level,
            "prediction_reliability": prediction_reliability,
        }

    def _extract_shap_contributions(self, shap_result: dict[str, Any] | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not isinstance(shap_result, dict):
            raise MissingSHAPOutputError("shap_result must be provided as a dict.")

        positive = shap_result.get("top_positive_features") or shap_result.get("strengths")
        negative = shap_result.get("top_negative_features") or shap_result.get("improvement_areas")

        if not isinstance(positive, list) or not isinstance(negative, list):
            raise MissingSHAPOutputError(
                "shap_result must include top_positive_features and top_negative_features."
            )

        strengths = [self._format_shap_item(item, positive=True) for item in positive]
        improvement_areas = [self._format_shap_item(item, positive=False) for item in negative]
        return strengths, improvement_areas

    @staticmethod
    def _format_shap_item(item: Any, *, positive: bool) -> dict[str, Any]:
        if not isinstance(item, dict):
            raise MissingSHAPOutputError("Each SHAP contribution must be a dictionary.")

        feature = item.get("feature")
        value = item.get("value")
        shap_value = item.get("shap_value")
        if not isinstance(feature, str) or not feature.strip():
            raise MissingSHAPOutputError("Each SHAP contribution must include feature.")
        if shap_value is None or not isinstance(shap_value, (int, float)):
            raise MissingSHAPOutputError(f"SHAP contribution for '{feature}' must include numeric shap_value.")

        reason_prefix = "Positive SHAP contribution" if positive else "Negative SHAP contribution"
        return {
            "feature": feature,
            "value": value,
            "reason": f"{reason_prefix} ({float(shap_value):+.4f})",
        }

    def _build_career_knowledge(self, career: str) -> _CareerKnowledge:
        persona = self._personas_by_career.get(career)
        if not isinstance(persona, dict):
            raise MissingKnowledgeBaseEntryError(f"Missing career persona for '{career}'.")

        roadmap = self._roadmaps_by_career.get(career)
        if not isinstance(roadmap, dict):
            raise MissingKnowledgeBaseEntryError(f"Missing roadmap entry for '{career}'.")

        family_name = self._resolve_career_family(career)
        career_record = self._careers_by_name.get(career)
        if not isinstance(career_record, dict):
            raise MissingKnowledgeBaseEntryError(f"Missing career catalog entry for '{career}'.")

        description = str(persona.get("description", "")).strip()
        if not description:
            raise MissingKnowledgeBaseEntryError(f"Missing career description for '{career}'.")

        salary_range = str(persona.get("salary_band") or self._format_salary_range(career_record)).strip()
        if not salary_range:
            raise MissingKnowledgeBaseEntryError(f"Missing salary range for '{career}'.")

        required_skills = self._merge_unique_strings(
            persona.get("required_technical_skills", []),
            persona.get("required_soft_skills", []),
        )

        education_path = [str(persona.get("typical_education", "")).strip()]
        education_path = [item for item in education_path if item]

        next_roles = self._extract_next_roles(career, roadmap)
        industries = [str(item) for item in persona.get("typical_industries", []) if isinstance(item, str)]

        return _CareerKnowledge(
            name=career,
            description=description,
            career_family=family_name,
            industry=industries,
            salary_range=salary_range,
            required_skills=required_skills,
            education_path=education_path,
            next_roles=next_roles,
        )

    def _resolve_career_family(self, career: str) -> str:
        for family in self._career_families.values():
            careers = family.get("careers", [])
            if isinstance(careers, list) and career in careers:
                family_name = family.get("name")
                if isinstance(family_name, str) and family_name.strip():
                    return family_name
        raise MissingKnowledgeBaseEntryError(f"Missing career family mapping for '{career}'.")

    @staticmethod
    def _merge_unique_strings(*groups: Any) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()
        for group in groups:
            if not isinstance(group, list):
                continue
            for item in group:
                if not isinstance(item, str):
                    continue
                value = item.strip()
                if value and value not in seen:
                    seen.add(value)
                    merged.append(value)
        return merged

    def _extract_next_roles(self, career: str, roadmap: dict[str, Any]) -> list[str]:
        stages = roadmap.get("stages", [])
        if not isinstance(stages, list) or not stages:
            raise MissingKnowledgeBaseEntryError(f"Missing roadmap stages for '{career}'.")

        current_index = None
        for index, stage in enumerate(stages):
            if not isinstance(stage, dict):
                continue
            if stage.get("level") == career:
                current_index = index
                break

        if current_index is None:
            for index, stage in enumerate(stages):
                if isinstance(stage, dict) and stage.get("level") == f"{career}":
                    current_index = index
                    break

        if current_index is None:
            raise MissingKnowledgeBaseEntryError(f"Missing roadmap stage for '{career}'.")

        next_roles: list[str] = []
        for stage in stages[current_index + 1 :]:
            if not isinstance(stage, dict):
                continue
            level = stage.get("level")
            if isinstance(level, str) and level.strip():
                next_roles.append(level)
        return next_roles

    @staticmethod
    def _format_salary_range(career_record: dict[str, Any]) -> str:
        salary_range = career_record.get("salary_range")
        if not isinstance(salary_range, dict):
            return ""
        minimum = salary_range.get("min")
        maximum = salary_range.get("max")
        if not isinstance(minimum, (int, float)) or not isinstance(maximum, (int, float)):
            return ""
        return f"${int(minimum/1000)}k-${int(maximum/1000)}k USD"


__all__ = [
    "InvalidRecommendationInputError",
    "MissingConfidenceOutputError",
    "MissingKnowledgeBaseEntryError",
    "MissingPredictionOutputError",
    "MissingSHAPOutputError",
    "RecommendationEngine",
    "RecommendationEngineError",
    "UnknownCareerError",
]