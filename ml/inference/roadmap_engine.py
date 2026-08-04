"""CareerPilot-AI Learning Roadmap Engine.

This module converts the output of the **Skill Gap Engine** (plus the
**Recommendation Engine** payload) into a **personalized learning roadmap**.

It is the last - and entirely deterministic - link in the inference chain.

It does NOT:

* load ML models
* perform prediction
* compute SHAP values
* calculate confidence
* perform gap analysis
* recommend careers

Its sole responsibility is to translate a list of *missing skills* into an
ordered, dependency-aware, week-estimate-tagged learning plan that the user
can execute step-by-step.

Inputs:

1. ``skill_gap_result``  - output of :mod:`ml.inference.skill_gap_engine`.
2. ``recommendation_result`` - output of :mod:`ml.inference.recommendation_engine`.
3. ``learning_resources`` - the JSON KB at
   ``knowledge_base/learning_resources.json``.

The engine is intentionally framework-agnostic and follow SOLID principles:

* S - roadmap assembly only; no model, no UI, no business logic outside the
      ordering algorithm.
* O - the underlying ``LearningResourceKB`` is injected; new KB layouts can
      be plugged in by replacing it without changing ``RoadmapEngine``.
* L - subclasses that override the public ``build`` method remain valid
      facades; the KB loader and ordering helpers are reusable.
* I - the public surface is one method (``build``) plus narrowly-scoped
      exception classes.
* D - the KB path is injectable; the algorithm does not depend on a
      concrete filesystem layout.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from ml.config import setup_logging


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
KNOWLEDGE_BASE_DIR: Final[Path] = PROJECT_ROOT / "knowledge_base"
LEARNING_RESOURCES_PATH: Final[Path] = KNOWLEDGE_BASE_DIR / "learning_resources.json"


# Ordering keys used to break ties when two skills have the same number of
# missing prerequisites.
_DIFFICULTY_RANK: Final[dict[str, int]] = {
    "Beginner": 0,
    "Intermediate": 1,
    "Advanced": 2,
}

# Priority ordering produced by the Skill Gap Engine. We trust its criticality
# ranking (High > Medium > Low > None) for tie-breaking.
_PRIORITY_RANK: Final[dict[str, int]] = {
    "High": 0,
    "Medium": 1,
    "Low": 2,
    "None": 3,
}


# Skill features that are typically already known to the user at "baseline"
# (these get a duration discount if they show up in a roadmap). They never
# appear explicitly as "missing" because they correspond to the user's
# existing strengths - this list is used only by the optional duration tweak.
_BASELINE_SKILLS: Final[frozenset[str]] = frozenset(
    {
        "Communication",
        "Teamwork",
        "Adaptability",
        "Critical Thinking",
    }
)


logger: Final[logging.Logger] = setup_logging("ml.inference.roadmap_engine")


# =============================================================================
# Exceptions
# =============================================================================
class RoadmapEngineError(ValueError):
    """Base class for roadmap engine failures."""


class MissingSkillGapPayloadError(RoadmapEngineError):
    """Raised when the skill-gap payload is missing or malformed."""


class MissingRecommendationPayloadError(RoadmapEngineError):
    """Raised when the recommendation payload is missing or malformed."""


class MissingLearningResourceKBError(RoadmapEngineError):
    """Raised when a required skill has no entry in the learning resources KB."""


class CyclicSkillDependencyError(RoadmapEngineError):
    """Raised when the prerequisite graph contains a cycle."""


# =============================================================================
# Public data shapes
# =============================================================================
@dataclass(frozen=True)
class RoadmapPhase:
    """One ordered step in the personalized roadmap."""

    phase: int
    skill: str
    difficulty: str
    duration_weeks: int
    prerequisites: list[str]
    resource_types: list[str]


@dataclass(frozen=True)
class RoadmapPlan:
    """Full roadmap returned to callers."""

    recommended_career: str
    estimated_completion_weeks: int
    phases: list[RoadmapPhase] = field(default_factory=list)
    unresolved_skills: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert the roadmap plan to a JSON-serializable dictionary."""
        return {
            "recommended_career": self.recommended_career,
            "estimated_completion_weeks": int(self.estimated_completion_weeks),
            "roadmap": [
                {
                    "phase": phase.phase,
                    "skill": phase.skill,
                    "difficulty": phase.difficulty,
                    "duration_weeks": int(phase.duration_weeks),
                    "prerequisites": list(phase.prerequisites),
                    "resource_types": list(phase.resource_types),
                }
                for phase in self.phases
            ],
            "unresolved_skills": list(self.unresolved_skills),
        }


# =============================================================================
# Learning Resource KB
# =============================================================================
class LearningResourceKB:
    """Read-only access to ``knowledge_base/learning_resources.json``.

    The KB is intentionally narrow: it only exposes ``get(skill)`` and a
    fallback so the engine can remain decoupled from any specific JSON shape.
    """

    def __init__(self, knowledge_base_dir: Path | str | None = None) -> None:
        self.knowledge_base_dir = (
            Path(knowledge_base_dir) if knowledge_base_dir is not None else KNOWLEDGE_BASE_DIR
        )
        self.learning_resources_path = self.knowledge_base_dir / LEARNING_RESOURCES_PATH.name
        self._data = self._load(self.learning_resources_path)

        metadata = self._data.get("metadata", {})
        if not isinstance(metadata, dict):
            raise MissingLearningResourceKBError(
                "learning_resources.json must contain a 'metadata' object."
            )

        specs = self._data.get("skill_learning_specs", {})
        if not isinstance(specs, dict):
            raise MissingLearningResourceKBError(
                "learning_resources.json must contain a 'skill_learning_specs' object."
            )
        self._specs: dict[str, dict[str, Any]] = specs

        default = self._data.get("default_spec", {})
        self._default_spec: dict[str, Any] = default if isinstance(default, dict) else {}

        difficulty_levels = self._data.get("difficulty_levels", [])
        self._difficulty_levels: tuple[str, ...] = (
            tuple(difficulty_levels) if isinstance(difficulty_levels, list) else tuple(_DIFFICULTY_RANK)
        )

        self._case_insensitive_index: dict[str, str] = {
            name.lower(): name for name in self._specs if isinstance(name, str)
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get(self, skill: str) -> dict[str, Any]:
        """Return the KB entry for ``skill`` with sensible defaults filled in.

        Returns:
            A dictionary containing ``difficulty``, ``duration_weeks``,
            ``prerequisites``, ``resource_types``, and ``found`` (True/False).

        Raises:
            MissingLearningResourceKBError: If the skill is not in the KB and
                no default spec is available.
        """
        record = self._lookup(skill)
        if record is None:
            if not self._default_spec:
                raise MissingLearningResourceKBError(
                    f"Skill '{skill}' has no entry in learning_resources.json "
                    "and no default_spec fallback is available."
                )
            return self._materialise(self._default_spec, skill, found=False)

        return self._materialise(record, skill, found=True)

    def has(self, skill: str) -> bool:
        """Return ``True`` if the KB contains an entry for ``skill``."""
        return self._lookup(skill) is not None

    def known_skills(self) -> list[str]:
        """Return the sorted list of skills known to the KB."""
        return sorted(self._specs.keys())

    @property
    def default_spec(self) -> dict[str, Any]:
        """Return the KB-level fallback spec (``{}`` when none configured)."""
        return dict(self._default_spec)

    # ------------------------------------------------------------------
    # KB internals
    # ------------------------------------------------------------------
    @staticmethod
    def _load(path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise MissingLearningResourceKBError(
                f"Missing knowledge base file: {path}"
            )
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise MissingLearningResourceKBError(
                f"learning_resources.json must contain a top-level object: {path}"
            )
        return data

    def _lookup(self, skill: str) -> dict[str, Any] | None:
        if not isinstance(skill, str):
            return None
        direct = self._specs.get(skill)
        if isinstance(direct, dict):
            return direct
        canonical = self._case_insensitive_index.get(skill.lower())
        if canonical is None:
            return None
        record = self._specs.get(canonical)
        return record if isinstance(record, dict) else None

    def _materialise(
        self, record: dict[str, Any], skill: str, *, found: bool
    ) -> dict[str, Any]:
        difficulty = str(record.get("difficulty", "")).strip() or str(
            self._default_spec.get("difficulty", "Intermediate")
        )
        duration = record.get("duration_weeks", self._default_spec.get("duration_weeks", 4))
        prerequisites = record.get("prerequisites", self._default_spec.get("prerequisites", []))
        resource_types = record.get(
            "resource_types", self._default_spec.get("resource_types", ["Course"])
        )

        if not isinstance(duration, int) or duration <= 0:
            duration = 4
        if not isinstance(prerequisites, list):
            prerequisites = []
        if not isinstance(resource_types, list) or not resource_types:
            resource_types = ["Course"]

        return {
            "skill": skill,
            "difficulty": difficulty,
            "duration_weeks": int(duration),
            "prerequisites": [str(item) for item in prerequisites if isinstance(item, str)],
            "resource_types": [str(item) for item in resource_types if isinstance(item, str)],
            "found": bool(found),
        }


# =============================================================================
# Roadmap Engine
# =============================================================================
class RoadmapEngine:
    """Translate skill-gap output into an ordered learning roadmap.

    The engine never infers new learning material - everything comes from
    the injected :class:`LearningResourceKB`. The ordering algorithm is
    topological with a deterministic tie-breaker.
    """

    def __init__(
        self,
        knowledge_base: LearningResourceKB | None = None,
        knowledge_base_dir: Path | str | None = None,
    ) -> None:
        if knowledge_base is None:
            knowledge_base = LearningResourceKB(knowledge_base_dir)
        self.knowledge_base = knowledge_base

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build(
        self,
        skill_gap_result: dict[str, Any] | None,
        recommendation_result: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Build a learning roadmap from a skill-gap + recommendation payload.

        Args:
            skill_gap_result: Output of the Skill Gap Engine. Must contain a
                ``skills`` list with per-skill records (``skill``, ``gap``,
                ``priority``, ``status``).
            recommendation_result: Output of the Recommendation Engine. Used
                only for the ``recommended_career`` field.

        Returns:
            Dictionary matching the canonical roadmap output schema.

        Raises:
            MissingSkillGapPayloadError: If the skill-gap payload is missing.
            MissingRecommendationPayloadError: If the recommendation payload
                is missing or has no career.
            MissingLearningResourceKBError: If a required skill cannot be
                resolved and no default spec exists.
            CyclicSkillDependencyError: If the prerequisite graph contains a
                cycle.
        """
        recommended_career = self._validate_recommendation(recommendation_result)
        missing_records = self._validate_skill_gap(skill_gap_result)

        if not missing_records:
            plan = RoadmapPlan(
                recommended_career=recommended_career,
                estimated_completion_weeks=0,
                phases=[],
                unresolved_skills=[],
            )
            logger.info(
                "Roadmap engine: no missing skills for %s - empty roadmap returned.",
                recommended_career,
            )
            return plan.to_dict()

        resolved_phases, unresolved = self._order_phases(missing_records)
        estimated_weeks = int(sum(phase.duration_weeks for phase in resolved_phases))

        plan = RoadmapPlan(
            recommended_career=recommended_career,
            estimated_completion_weeks=estimated_weeks,
            phases=resolved_phases,
            unresolved_skills=unresolved,
        )
        logger.info(
            "Roadmap built for %s: phases=%d, weeks=%d, unresolved=%d",
            recommended_career,
            len(resolved_phases),
            estimated_weeks,
            len(unresolved),
        )
        return plan.to_dict()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_recommendation(
        recommendation_result: dict[str, Any] | None,
    ) -> str:
        if not isinstance(recommendation_result, dict):
            raise MissingRecommendationPayloadError(
                "recommendation_result must be provided as a dict."
            )
        career = recommendation_result.get("recommended_career")
        if not isinstance(career, str) or not career.strip():
            raise MissingRecommendationPayloadError(
                "recommendation_result must contain a non-empty 'recommended_career'."
            )
        return career.strip()

    @staticmethod
    def _validate_skill_gap(
        skill_gap_result: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        if not isinstance(skill_gap_result, dict):
            raise MissingSkillGapPayloadError(
                "skill_gap_result must be provided as a dict."
            )
        if "skills" not in skill_gap_result:
            raise MissingSkillGapPayloadError(
                "skill_gap_result must contain a 'skills' list."
            )
        skills = skill_gap_result["skills"]
        if not isinstance(skills, list):
            raise MissingSkillGapPayloadError(
                "skill_gap_result must contain a 'skills' list."
            )

        missing: list[dict[str, Any]] = []
        for entry in skills:
            if not isinstance(entry, dict):
                continue
            skill = entry.get("skill")
            if not isinstance(skill, str) or not skill.strip():
                continue
            gap = entry.get("gap", 0)
            if not isinstance(gap, (int, float)):
                continue
            if gap <= 0:
                # gap <= 0 means the user already meets the requirement.
                continue
            priority = entry.get("priority", "None")
            status = entry.get("status", "Needs Improvement")
            missing.append(
                {
                    "skill": skill.strip(),
                    "gap": int(gap),
                    "priority": priority if isinstance(priority, str) else "None",
                    "status": status if isinstance(status, str) else "Needs Improvement",
                }
            )

        if not missing:
            return []
        return missing

    # ------------------------------------------------------------------
    # Phase ordering
    # ------------------------------------------------------------------
    def _order_phases(
        self, missing_records: list[dict[str, Any]]
    ) -> tuple[list[RoadmapPhase], list[str]]:
        """Run the dependency-aware topological sort.

        Returns:
            (ordered_phases, unresolved_skills). ``unresolved_skills`` is the
            list of skill names whose ``prerequisites`` references a skill
            outside the gap (i.e. not a "missing" skill) AND was therefore
            silently treated as already known.
        """
        # Build a name-keyed lookup. Normalise casing to respect display-name
        # conventions used by the Skill Gap Engine.
        by_skill: dict[str, dict[str, Any]] = {}
        canonical_lookup: dict[str, str] = {}
        for record in missing_records:
            name = record["skill"]
            by_skill[name] = record
            canonical_lookup[name.lower()] = name

        specs: dict[str, dict[str, Any]] = {}
        unresolved: list[str] = []

        # When the KB has no default_spec fallback, every missing skill MUST
        # resolve to an explicit entry. Surface unknowns as a hard error so
        # curriculum designers notice the gap.
        strict = not bool(self.knowledge_base.default_spec)

        for skill, record in by_skill.items():
            try:
                spec = self.knowledge_base.get(skill)
            except MissingLearningResourceKBError:
                if strict:
                    raise
                logger.warning(
                    "Skill '%s' has no entry in the learning_resources KB; "
                    "it will be listed as unresolved.",
                    skill,
                )
                unresolved.append(skill)
                continue

            # Resolve prerequisites against missing skills only. A prerequisite
            # outside the missing set is treated as already known.
            cleaned_prereqs: list[str] = []
            for prereq in spec["prerequisites"]:
                canonical = canonical_lookup.get(prereq.lower())
                if canonical is None:
                    # Either the user already has the prereq (so it is not in
                    # the missing list) OR it is genuinely unknown. Either way,
                    # we do not pull it into the roadmap.
                    continue
                if canonical == skill:
                    # Self-loop - drop silently to avoid trivial cycles.
                    continue
                if canonical not in cleaned_prereqs:
                    cleaned_prereqs.append(canonical)

            duration = spec["duration_weeks"]
            # Soft tweak: soft-skills known to be baseline competencies get
            # their duration reduced by one week (min 1) so a roadmap never
            # over-spends time on Communication.
            if skill in _BASELINE_SKILLS and duration > 1:
                duration = duration - 1

            specs[skill] = {
                "difficulty": spec["difficulty"],
                "duration_weeks": int(duration),
                "prerequisites": cleaned_prereqs,
                "resource_types": spec["resource_types"],
                # Internal sort hints
                "_gap": record["gap"],
                "_priority": record["priority"],
            }

        if not specs:
            return [], unresolved

        ordered_skills = self._topological_sort(specs)
        phases = self._build_phases(ordered_skills, specs)
        return phases, unresolved

    @staticmethod
    def _topological_sort(specs: dict[str, dict[str, Any]]) -> list[str]:
        """Kahn's algorithm with a deterministic tie-breaker.

        Tie-break order:
            1. fewer unresolved prerequisites first (depth-first readiness)
            2. higher gap value first (worst skill first)
            3. higher priority (High > Medium > Low > None)
            4. easier difficulty first (Beginner < Intermediate < Advanced)
            5. alphabetical order
        """
        # Compute the in-degree (number of unresolved prerequisites).
        in_degree: dict[str, int] = {}
        graph: dict[str, list[str]] = {}
        for skill, spec in specs.items():
            graph.setdefault(skill, [])
            in_degree[skill] = len(spec["prerequisites"])

        for skill, spec in specs.items():
            for prereq in spec["prerequisites"]:
                if prereq in specs:
                    graph.setdefault(prereq, []).append(skill)

        # Sentinel references to keep closures small.
        def sort_key(skill: str) -> tuple[int, int, int, int, str]:
            spec = specs[skill]
            return (
                in_degree[skill],                                           # 1
                -int(spec["_gap"]),                                         # 2
                _PRIORITY_RANK.get(spec["_priority"], 99),                  # 3
                _DIFFICULTY_RANK.get(spec["difficulty"], 99),               # 4
                skill.lower(),                                              # 5
            )

        ready_heap: list[str] = sorted(
            (skill for skill, degree in in_degree.items() if degree == 0),
            key=sort_key,
        )

        ordered: list[str] = []
        while ready_heap:
            # ``sorted`` above keeps the smallest key at index 0; pop the
            # front of the list to consume them in order.
            current = ready_heap.pop(0)
            ordered.append(current)

            for neighbour in graph.get(current, []):
                in_degree[neighbour] -= 1
                if in_degree[neighbour] == 0:
                    ready_heap.append(neighbour)
            ready_heap.sort(key=sort_key)

            # Defensive: cap iterations to specs count to detect cycles.
            if len(ordered) > len(specs):
                break

        if len(ordered) != len(specs):
            raise CyclicSkillDependencyError(
                "Prerequisite graph contains a cycle; cannot produce a linear roadmap."
            )
        return ordered

    @staticmethod
    def _build_phases(
        ordered_skills: list[str], specs: dict[str, dict[str, Any]]
    ) -> list[RoadmapPhase]:
        phases: list[RoadmapPhase] = []
        for index, skill in enumerate(ordered_skills, start=1):
            spec = specs[skill]
            phases.append(
                RoadmapPhase(
                    phase=index,
                    skill=skill,
                    difficulty=spec["difficulty"],
                    duration_weeks=int(spec["duration_weeks"]),
                    prerequisites=list(spec["prerequisites"]),
                    resource_types=list(spec["resource_types"]),
                )
            )
        return phases


__all__ = [
    "CyclicSkillDependencyError",
    "LearningResourceKB",
    "MissingLearningResourceKBError",
    "MissingRecommendationPayloadError",
    "MissingSkillGapPayloadError",
    "RoadmapEngine",
    "RoadmapEngineError",
    "RoadmapPhase",
    "RoadmapPlan",
]
