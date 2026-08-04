# Learning Roadmap Engine — Architecture & Report

**Module:** `ml/inference/roadmap_engine.py`
**Status:** Production-ready
**Last Updated:** 2026-08-04
**Engineer:** Senior AI Solutions Architect

---

## 1. Purpose & Scope

The Learning Roadmap Engine is the **final, deterministic link** in the
CareerPilot-AI inference chain. It converts the output of the
**Skill Gap Engine** (combined with the **Recommendation Engine** payload)
into a **personalized, dependency-aware learning roadmap**.

The engine is the *only* module that constructs learning plans. Every other
inference component either predicts, explains, or analyzes — none of them
suggests how to close a gap.

The Roadmap Engine is **strictly read-only** with respect to ML assets and
business logic:

* It does **NOT** load any trained model.
* It does **NOT** perform prediction.
* It does **NOT** compute SHAP values.
* It does **NOT** calculate confidence.
* It does **NOT** perform gap analysis.
* It does **NOT** recommend careers.
* It does **NOT** generate or store any new learning material.

All learning-resource metadata is read from the **dedicated knowledge
base** at `knowledge_base/learning_resources.json`.

---

## 2. Architecture

```
                          +------------------------+
                          |   skill_gap_result     |
                          |   (from Skill Gap Eng) |
                          |                        |
                          |   recommendation_      |
                          |   result               |
                          |   (from Recommendation |
                          |    Engine)             |
                          +-----------+------------+
                                      |
                                      v
                          +------------------------+
                          |      RoadmapEngine     |
                          |   (stateless facade)   |
                          +-----------+------------+
                                      |
                                      v
                          +------------------------+
                          |  LearningResourceKB    |
                          |  (JSON loader + cache) |
                          +-----------+------------+
                                      |
                                      v
                          +------------------------+
                          |  Ordered Roadmap Plan  |
                          |  - phase               |
                          |  - skill               |
                          |  - difficulty          |
                          |  - duration_weeks      |
                          |  - prerequisites       |
                          |  - resource_types      |
                          +------------------------+
```

### 2.1 Modules

| Module | Responsibility |
|--------|---------------|
| `RoadmapEngine` | Public facade — accepts `skill_gap_result` and `recommendation_result`, returns the roadmap dictionary. |
| `LearningResourceKB` | Loads and serves `learning_resources.json`. Exposes a narrow `get(skill)` interface with default-spec fallback. |
| `RoadmapPlan` (dataclass) | Structured, framework-agnostic representation of the final roadmap. |
| `RoadmapPhase` (dataclass) | One ordered step (skill, difficulty, weeks, prereqs, resources). |
| `_topological_sort` | Kahn's algorithm with a deterministic tie-breaker. |
| `_order_phases` | Resolves KB specs, prunes external prerequisites, calls the topological sort. |
| `_validate_*` | Strict input validation, raising narrowly-scoped exceptions. |

### 2.2 SOLID Conformance

| Principle | Application |
|-----------|-------------|
| **S**ingle Responsibility | The engine only orders learning material; it does not predict, explain, or analyze. |
| **O**pen/Closed | The KB is injected (`LearningResourceKB`); new KB layouts can be plugged in without modifying `build()`. |
| **L**iskov Substitution | `RoadmapEngine` is interchangeable across callers; tests confirm a single public API. |
| **I**nterface Segregation | Public surface is one method (`build`) plus four narrowly-scoped exception classes. |
| **D**ependency Inversion | KB path is injectable; the ordering algorithm does not depend on a concrete filesystem layout. |

---

## 3. Input Contract

### 3.1 `skill_gap_result`

The output of the Skill Gap Engine. The Roadmap Engine only consumes the
`skills` list; it ignores `overall_readiness`, `critical_skills`, etc.

```python
skill_gap_result = {
    "recommended_career": "ML Engineer",
    "skills": [
        {"skill": "Docker",       "gap": 6, "priority": "High", "status": "Critical"},
        {"skill": "Kubernetes",   "gap": 8, "priority": "High", "status": "Critical"},
        {"skill": "Git",          "gap": 6, "priority": "High", "status": "Critical"},
        {"skill": "Machine Learning", "gap": 6, "priority": "High", "status": "Critical"},
        {"skill": "Deep Learning",    "gap": 7, "priority": "High", "status": "Critical"},
    ],
    ...
}
```

Only skills with `gap > 0` (i.e. skills the user still needs) feed the
roadmap.

### 3.2 `recommendation_result`

The output of the Recommendation Engine. The Roadmap Engine uses **only**
the `recommended_career` field, which is echoed into the output.

```python
recommendation_result = {
    "recommended_career": "ML Engineer",
    # ... other fields are ignored
}
```

---

## 4. Knowledge Base — `learning_resources.json`

This file is the **single source of truth** for:

* `difficulty` (`Beginner`, `Intermediate`, `Advanced`)
* `duration_weeks` (estimated weeks to learn the skill)
* `prerequisites` (other skill names)
* `resource_types` (Course, Book, Hands-on Project, Documentation, ...)

Every skill known to CareerPilot AI has an entry, plus a `default_spec`
fallback for graceful handling of brand-new skills.

```json
{
  "skill_learning_specs": {
    "Python": {
      "difficulty": "Beginner",
      "duration_weeks": 4,
      "prerequisites": [],
      "resource_types": ["Course", "Book", "Hands-on Project"]
    },
    "Kubernetes": {
      "difficulty": "Intermediate",
      "duration_weeks": 4,
      "prerequisites": ["Docker", "Linux"],
      "resource_types": ["Course", "Documentation", "Hands-on Lab"]
    }
  },
  "default_spec": {
    "difficulty": "Intermediate",
    "duration_weeks": 4,
    "prerequisites": [],
    "resource_types": ["Course", "Documentation"]
  }
}
```

The engine **never** hardcodes learning content. If a skill is not in the
KB, the engine either:

* resolves it via `default_spec` and flags it via `unresolved_skills`, **or**
* (when `default_spec` is absent) raises `MissingLearningResourceKBError`
  so curriculum designers notice the gap.

---

## 5. Roadmap Generation Algorithm

The algorithm is a **Kahn topological sort with a deterministic
tie-breaker**. It is a pure, reproducible transformation — the same input
always produces the same output.

### 5.1 Step 1 — Filter Missing Skills

```
missing_skills = {s for s in skill_gap_result.skills if s.gap > 0}
```

Any skill with `gap <= 0` is already met by the user and is skipped.

### 5.2 Step 2 — Resolve KB Specs

For each missing skill `s`:

1. Look up the KB entry (`difficulty`, `duration_weeks`, `prerequisites`,
   `resource_types`).
2. Prune any prerequisite that is **not** itself a missing skill. A
   prerequisite outside the missing set means the user already has it.
3. Drop self-loops (defensive — should never happen in a clean KB).

### 5.3 Step 3 — Topological Sort with Tie-Breakers

The Kahn algorithm processes skills in topological order. When multiple
skills become "ready" at the same step, ties are broken deterministically:

```
key(skill) =
    (
        in_degree[skill],                # 1. Fewer unresolved prereqs first
        -gap,                            # 2. Higher gap first (worst skill)
        priority_rank,                   # 3. High > Medium > Low > None
        difficulty_rank,                 # 4. Beginner < Intermediate < Advanced
        skill.lower()                    # 5. Alphabetical (deterministic)
    )
```

This guarantees:

* **Docker is always taught before Kubernetes.**
* **Machine Learning is always taught before Deep Learning.**
* Skills with the worst gap appear early (user addresses pain points first).
* Within the same priority band, easier skills appear first.

### 5.4 Step 4 — Number Phases and Sum Weeks

```
for index, skill in enumerate(ordered_skills, start=1):
    phase[index].skill            = skill
    phase[index].difficulty       = specs[skill].difficulty
    phase[index].duration_weeks   = specs[skill].duration_weeks
    phase[index].prerequisites    = specs[skill].prerequisites
    phase[index].resource_types   = specs[skill].resource_types

estimated_completion_weeks = sum(phase.duration_weeks for phase in phases)
```

---

## 6. Dependency Resolution

The engine treats prerequisites **literally**:

* A prerequisite is *resolved* if and only if it is also a missing skill.
* A prerequisite outside the missing set is silently dropped — the user
  either already has the skill, or it is not on this roadmap.
* The topological sort refuses to emit a cycle and raises
  `CyclicSkillDependencyError` so authors notice contradictory edges in
  the KB.

Example chain (canonical from the spec):

```
Linux  ──►  Docker  ──►  Kubernetes
                  ▲
                  │   ML pipeline
Machine Learning ─┘
```

Result: `Linux → Docker → Kubernetes`, with `Docker` appearing in the
`prerequisites` of the `Kubernetes` phase.

---

## 7. Output Schema

```json
{
  "recommended_career": "ML Engineer",
  "estimated_completion_weeks": 23,
  "roadmap": [
    {
      "phase": 1,
      "skill": "Docker",
      "difficulty": "Beginner",
      "duration_weeks": 2,
      "prerequisites": [],
      "resource_types": ["Course", "Documentation", "Hands-on Lab"]
    },
    {
      "phase": 2,
      "skill": "Kubernetes",
      "difficulty": "Intermediate",
      "duration_weeks": 4,
      "prerequisites": ["Docker"],
      "resource_types": ["Course", "Documentation", "Hands-on Lab"]
    },
    {
      "phase": 3,
      "skill": "Git",
      "difficulty": "Beginner",
      "duration_weeks": 1,
      "prerequisites": [],
      "resource_types": ["Course", "Documentation"]
    },
    {
      "phase": 4,
      "skill": "Machine Learning",
      "difficulty": "Intermediate",
      "duration_weeks": 8,
      "prerequisites": [],
      "resource_types": ["Course", "Book", "Hands-on Project"]
    },
    {
      "phase": 5,
      "skill": "Deep Learning",
      "difficulty": "Advanced",
      "duration_weeks": 8,
      "prerequisites": ["Machine Learning"],
      "resource_types": ["Course", "Book", "Research Papers"]
    }
  ],
  "unresolved_skills": []
}
```

The `unresolved_skills` list surfaces any skill whose KB entry was missing
**and** no `default_spec` was available. UI layers can flag those to the
user.

---

## 8. Ordering Guarantees

| Guarantee | How It Is Enforced |
|-----------|-------------------|
| Kubernetes never appears before Docker. | Topological sort + KB prerequisites. |
| Deep Learning never appears before Machine Learning. | Topological sort + KB prerequisites. |
| Critical skills surface first when no prerequisites compete. | Tie-breaker rank 2 (descending gap). |
| Same-gig skills break ties deterministically. | Tie-breaker rank 5 (alphabetical). |
| Roadmap is reproducible across runs. | Pure function + stable sort. |

---

## 9. Error Handling

| Exception | Trigger |
|-----------|---------|
| `MissingSkillGapPayloadError` | `skill_gap_result` is `None`, not a dict, or has no `skills` list. |
| `MissingRecommendationPayloadError` | `recommendation_result` is `None`, not a dict, or has no `recommended_career`. |
| `MissingLearningResourceKBError` | A required skill has no KB entry **and** `default_spec` is empty. |
| `CyclicSkillDependencyError` | The prerequisite graph contains a cycle. |

All exceptions inherit from `RoadmapEngineError(ValueError)`, so callers
may catch them generically or specifically.

---

## 10. Framework-Agnostic Usage

The engine exposes **only** a single public method (`build`) that takes
and returns plain dictionaries. It therefore plugs into:

```python
# Streamlit
plan = RoadmapEngine().build(st.session_state["skill_gap"], recommendation)

# Flask / FastAPI
@app.post("/roadmap")
def roadmap(req: RoadmapRequest) -> dict[str, Any]:
    return RoadmapEngine().build(req.skill_gap, req.recommendation)
```

No code change is required across frameworks.

---

## 11. Test Coverage

`tests/test_roadmap_engine.py` — **20 tests**, all passing:

* No missing skills → empty roadmap, `estimated_completion_weeks == 0`.
* Skills with `gap <= 0` are skipped.
* Single skill → phase 1, no prereqs.
* Multiple skills → every missing skill appears exactly once.
* **Dependency ordering — Docker before Kubernetes (canonical example).**
* Dependency chain — `Linux → Docker → Kubernetes` resolves correctly.
* Deep Learning ordered after Machine Learning.
* `estimated_completion_weeks` equals the sum of phase durations.
* Prerequisites that are not missing are silently pruned.
* Unknown skills resolve via `default_spec` when present.
* Unknown skills with no `default_spec` raise `MissingLearningResourceKBError`.
* Output schema conformance to the spec.
* Phase numbers are 1-indexed and contiguous.
* Recommended career is echoed from the recommendation payload.
* `MissingSkillGapPayloadError`, `MissingRecommendationPayloadError` paths.
* Engine does **not** mutate its inputs.
* Output is JSON-serializable.

Run:

```bash
.venv\Scripts\python.exe -m unittest tests.test_roadmap_engine -v
# Ran 20 tests in 0.018s — OK
```

---

## 12. Future Improvements

1. **Per-user weekly hour budget** — scale durations by the user's stated
   availability (e.g. 5 h/week vs. 20 h/week) so week counts become
   personally meaningful.
2. **Skill weight overrides** — let curriculum designers boost / dampen
   specific skills via a `skill_weights.json` KB without touching the
   algorithm.
3. **Multi-track roadmaps** — emit parallel tracks (Foundational,
   Specialised) so users with limited time can drop a track.
4. **Localization** — translate skill display names per locale via
   `display_names.json`.
5. **Audit metadata** — log `engine_version`, `kb_version`, and input
   hashes for compliance reporting.
6. **Graph visualisation** — export the prerequisite DAG as DOT/Graphviz
   for curriculum reviewers.

---

## 13. File Manifest

| Path | Purpose |
|------|---------|
| `ml/inference/roadmap_engine.py` | Engine implementation. |
| `knowledge_base/learning_resources.json` | Learning specs KB (difficulty, weeks, prereqs, resources). |
| `tests/test_roadmap_engine.py` | Unit tests (20 cases, all passing). |
| `reports/roadmap_engine_report.md` | This document. |
| `reports/roadmap_summary.json` | Latest run summary JSON. |