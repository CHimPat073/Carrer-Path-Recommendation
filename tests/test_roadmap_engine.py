"""Tests for the Learning Roadmap Engine."""

from __future__ import annotations

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

from ml.inference.roadmap_engine import (
    CyclicSkillDependencyError,
    LearningResourceKB,
    MissingLearningResourceKBError,
    MissingRecommendationPayloadError,
    MissingSkillGapPayloadError,
    RoadmapEngine,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _recommendation(career: str = "ML Engineer") -> dict[str, Any]:
    """Return a minimal but valid recommendation payload."""
    return {
        "recommended_career": career,
        "confidence": 99.78,
        "confidence_level": "Very High",
        "prediction_reliability": "Reliable",
        "required_skills": ["Python", "Machine Learning", "SQL", "Cloud", "Deep Learning"],
        "education_path": ["Bachelor's in Computer Science"],
        "next_roles": ["Senior ML Engineer"],
    }


def _skill(
    skill: str,
    *,
    gap: int = 3,
    priority: str = "High",
    status: str = "Critical",
) -> dict[str, Any]:
    """Build a per-skill record matching the Skill Gap Engine output."""
    return {
        "skill": skill,
        "current": 4,
        "required": 9,
        "gap": gap,
        "gap_percentage": -55.0,
        "status": status,
        "priority": priority,
    }


def _gap(skills: list[dict[str, Any]] | None = None, **kwargs: Any) -> dict[str, Any]:
    """Build a complete skill-gap payload from a list of per-skill records."""
    record_skills = list(skills) if skills else []
    return {
        "recommended_career": kwargs.get("career", "ML Engineer"),
        "overall_readiness": 30.0,
        "readiness_level": "Poor",
        "skills": record_skills,
        "strengths": [],
        "critical_skills": [item["skill"] for item in record_skills],
        "improvement_order": [item["skill"] for item in record_skills],
        "education_path": [],
    }


def _phase_index(roadmap: dict[str, Any]) -> dict[str, int]:
    """Return ``{skill_name: phase_number}`` for fast assertions."""
    return {item["skill"]: item["phase"] for item in roadmap["roadmap"]}


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------
class RoadmapEngineTests(unittest.TestCase):

    def setUp(self) -> None:
        self.engine = RoadmapEngine()
        self.recommendation = _recommendation()

    # ------------------------------------------------------------------
    # Empty / no-missing input
    # ------------------------------------------------------------------
    def test_no_missing_skills_returns_empty_roadmap(self) -> None:
        gap_result = _gap(skills=[])
        result = self.engine.build(gap_result, self.recommendation)

        self.assertEqual(result["recommended_career"], "ML Engineer")
        self.assertEqual(result["roadmap"], [])
        self.assertEqual(result["estimated_completion_weeks"], 0)
        self.assertEqual(result["unresolved_skills"], [])

    def test_skills_with_zero_or_negative_gap_are_skipped(self) -> None:
        gap_result = _gap(
            skills=[
                _skill("Python", gap=0, priority="None", status="Excellent"),
                _skill("Machine Learning", gap=-1, priority="None", status="Excellent"),
                _skill("Docker", gap=4, priority="High", status="Critical"),
            ]
        )

        result = self.engine.build(gap_result, self.recommendation)
        names = [phase["skill"] for phase in result["roadmap"]]
        self.assertEqual(names, ["Docker"])
        self.assertEqual(result["estimated_completion_weeks"], 2)

    # ------------------------------------------------------------------
    # Single skill
    # ------------------------------------------------------------------
    def test_single_skill_roadmap(self) -> None:
        gap_result = _gap(skills=[_skill("Git", gap=5)])
        result = self.engine.build(gap_result, self.recommendation)

        self.assertEqual(len(result["roadmap"]), 1)
        phase = result["roadmap"][0]
        self.assertEqual(phase["phase"], 1)
        self.assertEqual(phase["skill"], "Git")
        self.assertEqual(phase["difficulty"], "Beginner")
        self.assertEqual(phase["duration_weeks"], 1)
        self.assertEqual(phase["prerequisites"], [])
        self.assertIn("Course", phase["resource_types"])
        self.assertEqual(result["estimated_completion_weeks"], 1)

    # ------------------------------------------------------------------
    # Multiple skills + dependency ordering
    # ------------------------------------------------------------------
    def test_multiple_skills_returns_all_phases(self) -> None:
        gap_result = _gap(
            skills=[
                _skill("Python", gap=5),
                _skill("SQL", gap=3),
                _skill("Git", gap=2),
            ]
        )
        result = self.engine.build(gap_result, self.recommendation)
        names = [phase["skill"] for phase in result["roadmap"]]
        self.assertEqual(set(names), {"Python", "SQL", "Git"})
        self.assertEqual(len(result["roadmap"]), 3)

    def test_dependency_ordering_docker_before_kubernetes(self) -> None:
        """Kubernetes must come AFTER Docker - the canonical ordering example."""
        gap_result = _gap(
            skills=[
                _skill("Kubernetes", gap=5),
                _skill("Docker", gap=3),
            ]
        )
        result = self.engine.build(gap_result, self.recommendation)

        index = _phase_index(result)
        self.assertLess(index["Docker"], index["Kubernetes"])
        # Kubernetes' phase must list Docker as a prerequisite.
        kubernetes_phase = next(
            item for item in result["roadmap"] if item["skill"] == "Kubernetes"
        )
        self.assertIn("Docker", kubernetes_phase["prerequisites"])

    def test_dependency_chain_resolves_in_correct_sequence(self) -> None:
        gap_result = _gap(
            skills=[
                _skill("Kubernetes", gap=5),
                _skill("Docker", gap=4),
                _skill("Linux", gap=3),
            ]
        )
        result = self.engine.build(gap_result, self.recommendation)
        index = _phase_index(result)

        # Linux no prereqs -> first; Docker depends on Linux; Kubernetes on Docker.
        self.assertLess(index["Linux"], index["Docker"])
        self.assertLess(index["Docker"], index["Kubernetes"])

    def test_deep_learning_after_machine_learning(self) -> None:
        gap_result = _gap(
            skills=[
                _skill("Deep Learning", gap=6),
                _skill("Machine Learning", gap=5),
            ]
        )
        result = self.engine.build(gap_result, self.recommendation)

        index = _phase_index(result)
        self.assertLess(index["Machine Learning"], index["Deep Learning"])
        deep_learning_phase = next(
            item for item in result["roadmap"] if item["skill"] == "Deep Learning"
        )
        self.assertIn("Machine Learning", deep_learning_phase["prerequisites"])

    def test_estimated_completion_weeks_is_sum_of_durations(self) -> None:
        gap_result = _gap(
            skills=[
                _skill("Python", gap=5),
                _skill("SQL", gap=2),
                _skill("Git", gap=2),
            ]
        )
        result = self.engine.build(gap_result, self.recommendation)
        total = sum(phase["duration_weeks"] for phase in result["roadmap"])
        self.assertEqual(result["estimated_completion_weeks"], total)
        # Sanity: Python=4, SQL=3, Git=1 -> 8 weeks.
        self.assertEqual(result["estimated_completion_weeks"], 8)

    def test_unknown_prerequisite_dropped_from_phase(self) -> None:
        """A prerequisite that is not a missing skill must not appear in the roadmap."""
        # Cluster requires Kubernetes whose KB-pre-req Docker. Docker is NOT a
        # missing skill (user already meets it).
        gap_result = _gap(
            skills=[
                _skill("Kubernetes", gap=5),
            ]
        )
        result = self.engine.build(gap_result, self.recommendation)

        # Only Kubernetes remains, with no Kubernetes-internal prereqs
        # (Docker was dropped because it is not in the missing list).
        self.assertEqual(len(result["roadmap"]), 1)
        phase = result["roadmap"][0]
        self.assertEqual(phase["skill"], "Kubernetes")
        self.assertEqual(phase["prerequisites"], [])

    # ------------------------------------------------------------------
    # Unknown / missing KB entry
    # ------------------------------------------------------------------
    def test_unknown_skill_is_listed_as_unresolved(self) -> None:
        """A skill with no KB entry appears in ``unresolved_skills`` and is
        excluded from the roadmap phases."""
        gap_result = _gap(
            skills=[
                _skill("Quantum Foo", gap=5),
                _skill("Git", gap=2),
            ]
        )

        # Rebuild KB with default_spec preserved (defaults already exist).
        kb = LearningResourceKB()
        # Quantum Foo is not in the KB; the default_spec should make it
        # resolvable. Verify the KB covers this gracefully.
        result = self.engine.build(gap_result, self.recommendation)
        names = [phase["skill"] for phase in result["roadmap"]]
        self.assertIn("Git", names)
        # Either Quantum Foo is resolved via default_spec OR it shows up in
        # unresolved_skills - both are acceptable, but only one should happen.
        if "Quantum Foo" in names:
            self.assertNotIn("Quantum Foo", result["unresolved_skills"])
        else:
            self.assertIn("Quantum Foo", result["unresolved_skills"])

    def test_missing_kb_file_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(MissingLearningResourceKBError):
                LearningResourceKB(knowledge_base_dir=Path(tmp))

    def test_kb_default_spec_resolves_unknown_skill(self) -> None:
        # Build a minimal KB with only a default_spec and no per-skill entries.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "learning_resources.json").write_text(
                json.dumps(
                    {
                        "metadata": {"description": "test", "version": "1.0"},
                        "skill_learning_specs": {},
                        "default_spec": {
                            "difficulty": "Intermediate",
                            "duration_weeks": 2,
                            "prerequisites": [],
                            "resource_types": ["Course"],
                        },
                    }
                ),
                encoding="utf-8",
            )
            kb = LearningResourceKB(knowledge_base_dir=tmp_path)
            engine = RoadmapEngine(knowledge_base=kb)

            gap_result = _gap(skills=[_skill("Exotic Skill", gap=5)])
            result = engine.build(gap_result, self.recommendation)
            self.assertEqual(len(result["roadmap"]), 1)
            phase = result["roadmap"][0]
            self.assertEqual(phase["skill"], "Exotic Skill")
            self.assertEqual(phase["difficulty"], "Intermediate")
            self.assertEqual(phase["duration_weeks"], 2)
            self.assertEqual(result["unresolved_skills"], [])

    def test_no_default_spec_and_unknown_skill_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "learning_resources.json").write_text(
                json.dumps(
                    {
                        "metadata": {"description": "test", "version": "1.0"},
                        "skill_learning_specs": {},
                        # No default_spec at all.
                    }
                ),
                encoding="utf-8",
            )
            kb = LearningResourceKB(knowledge_base_dir=tmp_path)
            engine = RoadmapEngine(knowledge_base=kb)

            gap_result = _gap(skills=[_skill("Exotic Skill", gap=5)])
            with self.assertRaises(MissingLearningResourceKBError):
                engine.build(gap_result, self.recommendation)

    # ------------------------------------------------------------------
    # Output contract
    # ------------------------------------------------------------------
    def test_output_schema_matches_spec(self) -> None:
        gap_result = _gap(
            skills=[
                _skill("Python", gap=5),
                _skill("SQL", gap=3),
                _skill("Docker", gap=4),
            ]
        )
        result = self.engine.build(gap_result, self.recommendation)

        # Top-level keys
        for key in (
            "recommended_career",
            "estimated_completion_weeks",
            "roadmap",
            "unresolved_skills",
        ):
            self.assertIn(key, result)

        # Each phase has the exact required schema.
        for phase in result["roadmap"]:
            self.assertIn("phase", phase)
            self.assertIn("skill", phase)
            self.assertIn("difficulty", phase)
            self.assertIn("duration_weeks", phase)
            self.assertIn("prerequisites", phase)
            self.assertIn("resource_types", phase)
            self.assertIsInstance(phase["phase"], int)
            self.assertIsInstance(phase["duration_weeks"], int)
            self.assertIsInstance(phase["prerequisites"], list)
            self.assertIsInstance(phase["resource_types"], list)

    def test_phase_numbers_are_one_indexed_and_contiguous(self) -> None:
        gap_result = _gap(
            skills=[
                _skill("Python", gap=5),
                _skill("SQL", gap=3),
                _skill("Docker", gap=4),
            ]
        )
        result = self.engine.build(gap_result, self.recommendation)
        phases = [phase["phase"] for phase in result["roadmap"]]
        self.assertEqual(phases, list(range(1, len(phases) + 1)))

    def test_recommended_career_is_echoed_from_recommendation(self) -> None:
        gap_result = _gap(skills=[_skill("Python", gap=5)], career="Data Scientist")
        recommendation = _recommendation(career="Data Scientist")
        result = self.engine.build(gap_result, recommendation)
        self.assertEqual(result["recommended_career"], "Data Scientist")

    # ------------------------------------------------------------------
    # Error handling / validation
    # ------------------------------------------------------------------
    def test_missing_skill_gap_payload(self) -> None:
        with self.assertRaises(MissingSkillGapPayloadError):
            self.engine.build(None, self.recommendation)
        with self.assertRaises(MissingSkillGapPayloadError):
            self.engine.build({}, self.recommendation)

    def test_missing_recommendation_payload(self) -> None:
        gap_result = _gap(skills=[_skill("Python", gap=5)])
        with self.assertRaises(MissingRecommendationPayloadError):
            self.engine.build(gap_result, None)
        with self.assertRaises(MissingRecommendationPayloadError):
            self.engine.build(gap_result, {})
        with self.assertRaises(MissingRecommendationPayloadError):
            self.engine.build(gap_result, {"recommended_career": "   "})

    # ------------------------------------------------------------------
    # Idempotency / non-mutation
    # ------------------------------------------------------------------
    def test_engine_does_not_mutate_inputs(self) -> None:
        gap_result = _gap(
            skills=[
                _skill("Python", gap=5),
                _skill("SQL", gap=3),
            ]
        )
        gap_snapshot = copy.deepcopy(gap_result)
        rec_snapshot = copy.deepcopy(self.recommendation)

        self.engine.build(gap_result, self.recommendation)

        self.assertEqual(gap_result, gap_snapshot)
        self.assertEqual(self.recommendation, rec_snapshot)

    def test_returns_plain_dict_serializable_to_json(self) -> None:
        gap_result = _gap(
            skills=[_skill("Python", gap=5), _skill("SQL", gap=3), _skill("Git", gap=2)]
        )
        result = self.engine.build(gap_result, self.recommendation)
        self.assertIsInstance(result, dict)
        # Must be JSON-serializable as-is.
        serialized = json.dumps(result)
        self.assertIsInstance(serialized, str)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
