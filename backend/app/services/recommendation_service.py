import json
from pathlib import Path
from typing import Any

from ml.inference.confidence_engine import ConfidenceEngine
from ml.inference.recommendation_engine import RecommendationEngine
from ml.inference.roadmap_engine import RoadmapEngine
from ml.inference.skill_gap_engine import SkillGapEngine

from backend.app.services.prediction_service import predict_profile


PROJECT_ROOT = Path(__file__).resolve().parents[3]
KNOWLEDGE_BASE = PROJECT_ROOT / "knowledge_base"

_confidence_engine: ConfidenceEngine | None = None
_shap_engine: Any | None = None
_recommendation_engine: RecommendationEngine | None = None
_skill_gap_engine: SkillGapEngine | None = None
_roadmap_engine: RoadmapEngine | None = None

SKILL_ALIASES = {
    "Software testing": "Testing",
    "Test case design": "Testing",
    "Automation tools": "Testing",
    "Bug tracking": "Testing",
    "Basic scripting": "Python",
    "Attention to detail": "Problem Solving",
    "Patience": "Communication",
    "Analytical thinking": "Data Analysis",
    "Programming fundamentals": "Python",
    "Version control (Git)": "DevOps",
    "RESTful APIs": "Database",
    "Data structures and algorithms": "Problem Solving",
}


def build_recommendation(profile: dict[str, Any]) -> dict[str, Any]:
    """Run the existing inference engines and combine their results."""
    prediction = predict_profile(profile)
    confidence = get_confidence_engine().evaluate(prediction)
    shap_result = get_shap_engine().explain(
        user_input=profile,
        prediction_result=prediction,
    )
    recommendation = get_recommendation_engine().recommend(
        prediction,
        confidence,
        shap_result,
        user_input=profile,
    )
    gap_input = dict(recommendation)
    gap_input["required_skills"] = [
        SKILL_ALIASES.get(skill, skill)
        for skill in recommendation["required_skills"]
        if SKILL_ALIASES.get(skill, skill) not in {"Software design and debugging"}
    ]
    skill_gap = get_skill_gap_engine().analyze(profile, gap_input)
    roadmap = get_roadmap_engine().build(skill_gap, recommendation)
    recommendation["skill_gap"] = skill_gap
    recommendation["learning_roadmap"] = roadmap
    return recommendation


def get_careers() -> list[dict[str, Any]]:
    data = load_json("careers.json")
    return data.get("careers", [])


def get_career(career_name: str) -> dict[str, Any] | None:
    return next(
        (career for career in get_careers() if career.get("name") == career_name),
        None,
    )


def get_roadmap(career_name: str) -> dict[str, Any] | None:
    data = load_json("roadmaps.json")
    return next(
        (roadmap for roadmap in data.get("roadmaps", []) if roadmap.get("career") == career_name),
        None,
    )


def load_json(filename: str) -> dict[str, Any]:
    with (KNOWLEDGE_BASE / filename).open("r", encoding="utf-8") as file:
        return json.load(file)


def get_confidence_engine() -> ConfidenceEngine:
    global _confidence_engine
    if _confidence_engine is None:
        _confidence_engine = ConfidenceEngine()
    return _confidence_engine


def get_shap_engine() -> Any:
    global _shap_engine
    if _shap_engine is None:
        from ml.inference.shap_engine import SHAPEngine

        _shap_engine = SHAPEngine()
    return _shap_engine


def get_recommendation_engine() -> RecommendationEngine:
    global _recommendation_engine
    if _recommendation_engine is None:
        _recommendation_engine = RecommendationEngine()
    return _recommendation_engine


def get_skill_gap_engine() -> SkillGapEngine:
    global _skill_gap_engine
    if _skill_gap_engine is None:
        _skill_gap_engine = SkillGapEngine()
    return _skill_gap_engine


def get_roadmap_engine() -> RoadmapEngine:
    global _roadmap_engine
    if _roadmap_engine is None:
        _roadmap_engine = RoadmapEngine()
    return _roadmap_engine