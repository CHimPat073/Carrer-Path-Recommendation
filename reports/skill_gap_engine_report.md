# Skill Gap Engine — Architecture & Report

**Module:** `ml/inference/skill_gap_engine.py`
**Status:** Production-ready
**Last Updated:** 2026-08-02
**Engineer:** Senior ML / AI Solutions Architect

---

## 1. Purpose & Scope

The Skill Gap Engine compares a user's **current skill profile** against the
**expected skill profile** of the recommended career produced by the
recommendation pipeline. It produces a deterministic, explainable gap analysis
that downstream surfaces (Streamlit, Flask, FastAPI) can render without any
additional processing.

The engine is **strictly read-only** with respect to ML assets:

* It does **NOT** load any trained model.
* It does **NOT** perform prediction.
* It does **NOT** compute SHAP values.
* It does **NOT** compute confidence.
* It does **NOT** access the UI.
* It does **NOT** recommend learning resources (handled by a later module).

All career and skill expectations are read from the existing JSON files in
`knowledge_base/`. No career information is duplicated or hardcoded.

---

## 2. Architecture

```
                    +----------------------+
                    |   user_input (dict)  |
                    |  recommendation_     |
                    |  result (dict)       |
                    +----------+-----------+
                               |
                               v
                    +----------------------+
                    |  SkillGapEngine      |
                    |  (stateless facade)  |
                    +----------+-----------+
                               |
        +----------------------+----------------------+
        |                                             |
        v                                             v
+------------------+                       +----------------------+
| feature_ranges.  |                       |  report.to_dict()    |
| json (KB)        |                       |                      |
+------------------+                       +----------+-----------+
                                                          |
                                                          v
                                          +------------------------------+
                                          | Strengths / Critical / Order  |
                                          | Overall Readiness (0-100)     |
                                          | Status / Priority per skill   |
                                          +------------------------------+
```

### 2.1 Modules

| Module | Responsibility |
|--------|---------------|
| `SkillGapEngine` | Public facade — accepts `user_input` and `recommendation_result`, returns the gap-analysis report dictionary. |
| `SkillGapReport` (dataclass) | Structured, framework-agnostic representation of the analysis. |
| `SkillGap` (dataclass) | Per-skill record: current, required, gap, status, priority. |
| `_coerce_skill_value` | Tolerant numeric coercion & clamping of user-reported scores. |
| `_resolve_user_score` | Resolves a user score by name (supports `python`, `python_score`, `Python`, etc.). |
| `_resolve_expected_scores` | Pulls the expected proficiency from `feature_ranges.json` for each required skill. |
| `_classify_status` / `_classify_readiness` | Threshold logic — never altered at runtime. |
| `_compute_overall_readiness` | Weighted skill-similarity aggregation. |

### 2.2 SOLID Conformance

| Principle | Application |
|-----------|-------------|
| **S**ingle Responsibility | Engine performs only gap analysis — no prediction, no SHAP, no UI calls. |
| **O**pen/Closed | New threshold tables or aggregation strategies can be added by subclassing or passing in custom helpers without editing `analyze()`. |
| **L**iskov Substitution | `SkillGapEngine` is interchangeable in any caller; tests confirm a single public API. |
| **I**nterface Segregation | Public surface is one method (`analyze`) plus four narrowly-scoped exception classes. |
| **D**ependency Inversion | Knowledge-base path is injectable through `knowledge_base_dir`, decoupling file location from business logic. |

---

## 3. Input Contract

### 3.1 `user_input`

A dictionary of current skill scores. The engine accepts three naming
conventions and silently ignores non-numeric values:

| Convention | Example |
|------------|---------|
| `<skill>_score` (project native) | `python_score`, `machine_learning_score` |
| `<skill>` (display name) | `Python`, `Machine Learning` |
| lowercase + underscore | `machine_learning` |

Scores outside `[0, 10]` are clamped. Missing skills default to `0` (worst
case) so the user is never given false credit.

### 3.2 `recommendation_result`

Must contain:

* `recommended_career: str`
* `required_skills: list[str]`
* `education_path: list[str]` (optional, surfaced in the report)

The recommendation engine already provides this contract — the Skill Gap
Engine never calls `Predictor`.

---

## 4. Gap Calculation

For every required skill `s`:

```
required[s]   = average proficiency from knowledge_base.feature_ranges
current[s]    = user-reported score (clamped to [0,10], defaults to 0)
gap[s]        = required[s] - current[s]
gap_pct[s]    = (current[s] - required[s]) / required[s] * 100  (clamped to [-100,100])
```

**Sign convention:** `gap > 0` means the user is **lacking** the skill. This
matches the spec example: `current=9, required=8 → gap=-1 → Excellent`.

---

## 5. Status Rules

| Gap value | Status | Priority |
|-----------|--------|----------|
| `gap <= 0` | Excellent | None |
| `gap <= 1` | Good | Low |
| `gap <= 2` | Needs Improvement | Medium |
| `gap > 2`  | Critical | High |

Priority order from most urgent to least urgent:

```
Critical       → High
Needs Improvement → Medium
Good           → Low
Excellent      → None
```

---

## 6. Readiness Formula

Overall readiness is computed as a **weighted skill similarity**:

```
For each required skill s:
    weight   = max(required[s], 1)              # higher-impact skills dominate
    sim_s    = clamp(current[s] / required[s], 0, 1)

readiness = (Σ weight * sim_s) / Σ weight * 100   # bounded to [0, 100]
```

Readiness thresholds:

| Score | Level |
|-------|-------|
| `[0, 40]`   | Poor |
| `(40, 60]`  | Average |
| `(60, 80]`  | Good |
| `(80, 100]` | Excellent |

The weighted formulation keeps the score robust to careers with many
"low-stakes" required skills (e.g. soft skills).

---

## 7. Output Schema

```json
{
  "recommended_career": "ML Engineer",
  "overall_readiness": 68.42,
  "readiness_level": "Good",
  "skills": [
    {
      "skill": "Python",
      "current": 9,
      "required": 9,
      "gap": 0,
      "gap_percentage": 0.0,
      "status": "Excellent",
      "priority": "None"
    },
    {
      "skill": "Machine Learning",
      "current": 5,
      "required": 9,
      "gap": 4,
      "gap_percentage": -44.44,
      "status": "Critical",
      "priority": "High"
    }
  ],
  "strengths": ["Python"],
  "critical_skills": ["Machine Learning", "Cloud", "Deep Learning"],
  "improvement_order": ["Cloud", "Machine Learning", "Deep Learning", "SQL"],
  "education_path": ["Bachelor's or Master's degree in Computer Science, AI, or related field"]
}
```

* `strengths`: skills where `gap <= 0` (user meets or exceeds requirement).
* `critical_skills`: skills where `status == "Critical"` (gap > 2).
* `improvement_order`: all skills with `gap > 0`, sorted by descending gap
  (worst-first); ties broken alphabetically for deterministic output.

---

## 8. Priority Logic (Worked Example)

User with `python_score=9, machine_learning_score=5, sql_score=6,
cloud_score=2, deep_learning_score=4` versus ML Engineer
(`required: Py=9, ML=9, SQL=7, Cloud=6, DL=7`):

| Skill | current | required | gap | status | priority |
|-------|---------|----------|-----|--------|----------|
| Python | 9 | 9 | 0 | Excellent | None |
| Machine Learning | 5 | 9 | 4 | Critical | High |
| SQL | 6 | 7 | 1 | Good | Low |
| Cloud | 2 | 6 | 4 | Critical | High |
| Deep Learning | 4 | 7 | 3 | Critical | High |

Overall readiness: `68.42 / 100` → **Good**.

---

## 9. Error Handling

| Exception | Trigger |
|-----------|---------|
| `MissingRecommendationPayloadError` | `recommendation_result` is `None`, malformed, or missing `recommended_career`/`required_skills`. |
| `MissingUserInputError` | `user_input` is `None` or not a dict. |
| `UnknownCareerError` | `recommended_career` is not present in `feature_ranges.json`. |
| `MissingSkillDataError` | A required skill cannot be mapped to the knowledge base. |

All exceptions inherit from `SkillGapEngineError(ValueError)`, so callers
may catch them generically or specifically.

---

## 10. Framework-Agnostic Usage

The engine exposes **only** a single public method (`analyze`) that takes
and returns plain dictionaries. It therefore plugs into:

```python
# Streamlit
gap = SkillGapEngine().analyze(st.session_state["user_input"], recommendation)

# Flask / FastAPI
@app.post("/gap-analysis")
def gap_analysis(req: GapRequest) -> dict[str, Any]:
    return SkillGapEngine().analyze(req.user_input, req.recommendation)
```

No code change is required across frameworks.

---

## 11. Test Coverage

`tests/test_skill_gap_engine.py` — 15 tests covering:

* Perfect match (all strengths, no improvement order)
* Minor gap (Needs Improvement band)
* Critical gap (gap > 2 → High priority, worst-first ordering)
* Canonical spec example (`current=9, required=8 → gap=-1 → Excellent`)
* Output schema conformance
* Unknown career → `UnknownCareerError`
* Missing user input → `MissingUserInputError`
* Invalid recommendation payload → `MissingRecommendationPayloadError`
* Missing skill in KB → `MissingSkillDataError`
* Partial user profile (defaults to 0)
* All status classification thresholds (`Excellent / Good / Needs Improvement / Critical`)
* `improvement_order` sort ordering & determinism
* Plain-dict reusability across frameworks
* Input immutability (`analyze` never mutates caller payloads)

All 15 tests pass.

---

## 12. Future Improvements

1. **Skill alias resolution** — fold `skill_taxonomy.json` aliases into
   `_match_skill_to_required` so personas that list "ML" resolve to
   `machine_learning`.
2. **Configurable threshold tables** — expose status and readiness thresholds
   via `dataclass` to allow A/B testing different cutoffs.
3. **Skill weight overrides** — let curriculum designers override default
   weights for specific careers via a new KB file (`skill_weights.json`).
4. **Confidence-weighted readiness** — multiply the readiness by the
   recommendation engine's confidence score for users with low model certainty.
5. **Time-to-readiness estimate** — project a "months to readiness" estimate
   using historical learning curves from a new KB file.
6. **Localization** — translate skill display names per locale via
   `display_names.json` so non-technical audiences see localized labels.

---

## 13. File Manifest

| Path | Purpose |
|------|---------|
| `ml/inference/skill_gap_engine.py` | Engine implementation. |
| `tests/test_skill_gap_engine.py` | Unit tests (15 cases, all passing). |
| `reports/skill_gap_engine_report.md` | This document. |
| `reports/skill_gap_summary.json` | Latest run summary JSON. |
