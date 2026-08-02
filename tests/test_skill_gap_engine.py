"""Tests for the Skill Gap Engine."""

from __future__ import annotations

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

from ml.inference.skill_gap_engine import (
    MissingRecommendationPayloadError,
    MissingSkillDataError,
    MissingUserInputError,
    SkillGapEngine,
    UnknownCareerError,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"


def _ml_engineer_recommendation(
    *,
    career: str = "ML Engineer",
    required_skills: list[str] | None = None,
) -> dict[str, Any]:
    """Return a recommendation payload shaped like the production engine output."""
    return {
        "recommended_career": career,
        "confidence": 99.78,
        "confidence_level": "Very High",
        "prediction_reliability": "Reliable",
        "top3_careers": [
            {"career": "ML Engineer", "probability": 0.9978},
        ],
        "career_information": {
            "description": "Builds and deploys machine learning systems.",
            "career_family": "Data & AI",
        },
        "strengths": [],
        "improvement_areas": [],
        "required_skills": required_skills or [
            "Python",
            "Machine Learning",
            "SQL",
            "Cloud",
            "Deep Learning",
        ],
        "education_path": [
            "Bachelor's or Master's degree in Computer Science, AI, or related field"
        ],
        "next_roles": ["Senior ML Engineer", "ML Platform Engineer"],
    }


def _build_user_input(scores: dict[str, int]) -> dict[str, Any]:
    """Wrap raw scores using the project's score feature naming convention."""
    return {f"{name}_score": value for name, value in scores.items()}


class SkillGapEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = SkillGapEngine()
        self.recommendation = _ml_engineer_recommendation()

    # ------------------------------------------------------------------
    # Happy path - all status / priority categories
    # ------------------------------------------------------------------
    def test_perfect_match(self) -> None:
        # gap = required - current; negative gap => user >= required.
        # ML Engineer required averages (rounded): Py=9, ML=9, SQL=7, Cloud=6, DL=7.
        # All user inputs >= required -> every gap <= 0.
        user_input = _build_user_input(
            {
                "python": 10,
                "machine_learning": 10,
                "sql": 8,
                "cloud": 7,
                "deep_learning": 8,
            }
        )

        result = self.engine.analyze(user_input, self.recommendation)

        self.assertEqual(result["recommended_career"], "ML Engineer")
        self.assertEqual(result["skills"][0]["skill"], "Python")
        self.assertGreaterEqual(result["overall_readiness"], 90.0)
        self.assertEqual(result["readiness_level"], "Excellent")
        # All gaps <= 0 -> all strengths, no improvement_order.
        self.assertEqual(
            set(result["strengths"]),
            {"Python", "Machine Learning", "SQL", "Cloud", "Deep Learning"},
        )
        self.assertEqual(result["critical_skills"], [])
        self.assertEqual(result["improvement_order"], [])

    def test_minor_gap_yields_needs_improvement(self) -> None:
        # ML Engineer required averages (rounded): Py=9, ML=9, SQL=7, Cloud=6, DL=7.
        # Designed gaps: Python=2, ML=2, SQL=2, Cloud=0, DL=2.
        user_input = _build_user_input(
            {
                "python": 7,        # required 9 -> gap=2 -> Needs Improvement
                "machine_learning": 7,  # required 9 -> gap=2 -> Needs Improvement
                "sql": 5,           # required 7 -> gap=2 -> Needs Improvement
                "cloud": 6,         # required 6 -> gap=0 -> Excellent
                "deep_learning": 5, # required 7 -> gap=2 -> Needs Improvement
            }
        )

        result = self.engine.analyze(user_input, self.recommendation)

        statuses_by_skill = {item["skill"]: item["status"] for item in result["skills"]}
        priorities_by_skill = {item["skill"]: item["priority"] for item in result["skills"]}

        # gap=2 -> Needs Improvement across the board except Cloud.
        for skill in ("Python", "Machine Learning", "SQL", "Deep Learning"):
            self.assertEqual(statuses_by_skill[skill], "Needs Improvement", skill)
            self.assertEqual(priorities_by_skill[skill], "Medium", skill)

        # Cloud meets requirement -> Excellent / None priority.
        self.assertEqual(statuses_by_skill["Cloud"], "Excellent")
        self.assertEqual(priorities_by_skill["Cloud"], "None")

        # Strengths list reflects Excellent-rated skills only.
        self.assertIn("Cloud", result["strengths"])
        self.assertNotIn("Machine Learning", result["strengths"])

        # No critical gaps -> overall readiness in Average/Good band.
        self.assertEqual(result["critical_skills"], [])
        self.assertIn(result["readiness_level"], {"Average", "Good"})

    def test_critical_gap(self) -> None:
        # Force positive gap > 2 for multiple skills.
        # ML Engineer required averages: Py=9, ML=9, SQL=7, Cloud=6, DL=7.
        user_input = _build_user_input(
            {
                "python": 4,         # required 9 -> gap=5 -> Critical
                "machine_learning": 2,  # required 9 -> gap=7 -> Critical (worst)
                "sql": 2,            # required 7 -> gap=5 -> Critical
                "cloud": 2,          # required 6 -> gap=4 -> Critical
                "deep_learning": 1,  # required 7 -> gap=6 -> Critical
            }
        )

        result = self.engine.analyze(user_input, self.recommendation)

        # All gaps > 2 -> every skill must be Critical / High.
        for skill_record in result["skills"]:
            self.assertEqual(skill_record["status"], "Critical", skill_record["skill"])
            self.assertEqual(skill_record["priority"], "High", skill_record["skill"])
            self.assertGreater(skill_record["gap"], 2)

        self.assertEqual(
            result["critical_skills"],
            ["Python", "Machine Learning", "SQL", "Cloud", "Deep Learning"],
        )
        # improvement_order must list critical skills first, worst gap first.
        self.assertEqual(result["improvement_order"][0], "Machine Learning")
        self.assertEqual(result["improvement_order"][1], "Deep Learning")
        self.assertIn("Cloud", result["improvement_order"])

        # Overall readiness severely low.
        self.assertLess(result["overall_readiness"], 40.0)
        self.assertEqual(result["readiness_level"], "Poor")
        self.assertEqual(result["strengths"], [])

    # ------------------------------------------------------------------
    # Output schema + canonical example
    # ------------------------------------------------------------------
    def test_output_schema_matches_spec(self) -> None:
        user_input = _build_user_input(
            {"python": 9, "docker": 2, "git": 4, "kubernetes": 3}
        )
        # Software Engineer required averages: Py=6, DevOps=5.
        recommendation = {
            "recommended_career": "Software Engineer",
            "required_skills": ["Python", "DevOps"],
            "education_path": ["Bachelor's"],
        }

        result = self.engine.analyze(user_input, recommendation)

        for key in (
            "recommended_career",
            "overall_readiness",
            "readiness_level",
            "skills",
            "strengths",
            "critical_skills",
            "improvement_order",
        ):
            self.assertIn(key, result)

        for skill_record in result["skills"]:
            self.assertIn("skill", skill_record)
            self.assertIn("current", skill_record)
            self.assertIn("required", skill_record)
            self.assertIn("gap", skill_record)
            self.assertIn("status", skill_record)
            self.assertIn("priority", skill_record)

    # ------------------------------------------------------------------
    # Canonical spec example
    # ------------------------------------------------------------------
    def test_canonical_spec_example(self) -> None:
        """Mirror the exact example from the specification."""
        # Spec example shows user >= required -> gap negative -> Excellent.
        user_input = _build_user_input({"python": 9})
        recommendation = {
            "recommended_career": "Software Engineer",
            "required_skills": ["Python"],  # required ~6 -> gap=-3 Excellent
        }
        result = self.engine.analyze(user_input, recommendation)
        python_record = next(item for item in result["skills"] if item["skill"] == "Python")
        self.assertEqual(python_record["current"], 9)
        # Required value depends on rounding; the assertion is the status/priority mapping.
        self.assertEqual(python_record["status"], "Excellent")
        self.assertEqual(python_record["priority"], "None")
        self.assertEqual(result["strengths"], ["Python"])
        self.assertEqual(result["critical_skills"], [])

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------
    def test_unknown_career(self) -> None:
        with self.assertRaises(UnknownCareerError):
            self.engine.analyze(
                {"python_score": 7},
                _ml_engineer_recommendation(career="Astronaut"),
            )

    def test_missing_user_input(self) -> None:
        with self.assertRaises(MissingUserInputError):
            self.engine.analyze(None, self.recommendation)
        with self.assertRaises(MissingUserInputError):
            self.engine.analyze([], self.recommendation)  # type: ignore[arg-type]

    def test_invalid_user_input_dict_payload(self) -> None:
        # The engine treats ``user_input`` as optional values, but if it is
        # the wrong type entirely it must raise.
        with self.assertRaises(MissingUserInputError):
            self.engine.analyze("not a dict", self.recommendation)  # type: ignore[arg-type]

    def test_invalid_recommendation_payload(self) -> None:
        with self.assertRaises(MissingRecommendationPayloadError):
            self.engine.analyze({"python_score": 7}, None)
        with self.assertRaises(MissingRecommendationPayloadError):
            self.engine.analyze({"python_score": 7}, {})
        payload = _ml_engineer_recommendation()
        payload["required_skills"] = []
        with self.assertRaises(MissingRecommendationPayloadError):
            self.engine.analyze({"python_score": 7}, payload)

    def test_missing_skill_in_knowledge_base(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_kb = Path(temp_dir)
            shutil.copy2(
                KNOWLEDGE_BASE_DIR / "feature_ranges.json", temp_kb / "feature_ranges.json"
            )

            feature_path = temp_kb / "feature_ranges.json"
            data = json.loads(feature_path.read_text(encoding="utf-8"))
            for entry in data["career_feature_ranges"]:
                if entry["career"] == "ML Engineer":
                    entry["features"] = {}  # blank out skill data.
                    break
            feature_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

            engine = SkillGapEngine(knowledge_base_dir=temp_kb)
            with self.assertRaises(MissingSkillDataError):
                engine.analyze(
                    {"python_score": 7},
                    _ml_engineer_recommendation(),
                )

    def test_partial_user_profile_still_produces_output(self) -> None:
        # Missing skills default to zero (gap = required) - engine still returns
        # a complete report.
        result = self.engine.analyze({"python_score": 9}, self.recommendation)
        self.assertEqual(
            len(result["skills"]), len(self.recommendation["required_skills"])
        )
        for skill_record in result["skills"]:
            self.assertIn(
                skill_record["status"],
                {"Excellent", "Good", "Needs Improvement", "Critical"},
            )

    # ------------------------------------------------------------------
    # Status / priority thresholds
    # ------------------------------------------------------------------
    def test_status_classification_thresholds(self) -> None:
        # Engineer user values that hit every band of the classify table:
        # Py required=9, ML required=9, SQL required=7, Cloud required=6, DL required=7.
        user_input = _build_user_input(
            {
                "python": 9,         # gap=0 -> Excellent / None
                "machine_learning": 8,  # gap=1 -> Good / Low
                "sql": 5,            # gap=2 -> Needs Improvement / Medium
                "cloud": 3,          # gap=3 -> Critical / High
                "deep_learning": 5,  # gap=2 -> Needs Improvement / Medium
            }
        )
        result = self.engine.analyze(user_input, self.recommendation)
        mapping = {item["skill"]: item for item in result["skills"]}

        self.assertEqual(mapping["Python"]["status"], "Excellent")
        self.assertEqual(mapping["Python"]["priority"], "None")
        self.assertEqual(mapping["Machine Learning"]["status"], "Good")
        self.assertEqual(mapping["Machine Learning"]["priority"], "Low")
        self.assertEqual(mapping["SQL"]["status"], "Needs Improvement")
        self.assertEqual(mapping["SQL"]["priority"], "Medium")
        self.assertEqual(mapping["Cloud"]["status"], "Critical")
        self.assertEqual(mapping["Cloud"]["priority"], "High")
        self.assertEqual(mapping["Deep Learning"]["status"], "Needs Improvement")
        self.assertEqual(mapping["Deep Learning"]["priority"], "Medium")

    def test_improvement_order_sorted_by_gap(self) -> None:
        # User values chosen to produce descending gaps among required skills.
        # Required: Py=9, ML=9, SQL=7, Cloud=6, DL=7.
        user_input = _build_user_input(
            {
                "python": 5,         # gap=4
                "machine_learning": 3,   # gap=6 (worst)
                "sql": 7,            # gap=0 (meets)
                "cloud": 6,          # gap=0 (meets)
                "deep_learning": 4,  # gap=3
            }
        )
        result = self.engine.analyze(user_input, self.recommendation)
        # Worst-first: ML > Python > Deep Learning; SQL/Cloud skipped (gap=0).
        self.assertEqual(result["improvement_order"][0], "Machine Learning")
        self.assertEqual(result["improvement_order"][1], "Python")
        self.assertEqual(result["improvement_order"][2], "Deep Learning")
        self.assertNotIn("SQL", result["improvement_order"])
        self.assertNotIn("Cloud", result["improvement_order"])

    # ------------------------------------------------------------------
    # Reusability across framework boundaries
    # ------------------------------------------------------------------
    def test_returns_plain_dict_consumable_by_any_framework(self) -> None:
        result = self.engine.analyze(
            _build_user_input(
                {"python": 8, "machine_learning": 7, "sql": 6, "cloud": 5, "deep_learning": 5}
            ),
            self.recommendation,
        )
        # Plain dict -> serializable, framework agnostic.
        self.assertIsInstance(result, dict)
        self.assertIsInstance(json.dumps(result), str)

    def test_engine_does_not_mutate_inputs(self) -> None:
        user_input = _build_user_input(
            {"python": 8, "machine_learning": 7, "sql": 6}
        )
        user_input_snapshot = copy.deepcopy(user_input)
        recommendation_snapshot = copy.deepcopy(self.recommendation)

        self.engine.analyze(user_input, self.recommendation)

        self.assertEqual(user_input, user_input_snapshot)
        self.assertEqual(self.recommendation, recommendation_snapshot)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
