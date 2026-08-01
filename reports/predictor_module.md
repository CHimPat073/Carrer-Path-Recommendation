# CareerPilot-AI – Prediction Engine (`ml.inference.predictor`)

> Standalone, framework-agnostic inference module.
> **Predicts only.** No UI, no SHAP, no recommendations, no knowledge-base
> lookups, no explanations, no learning-roadmap logic.

---

## 1. Module Architecture

```
CareerPilot-AI/
├── models/
│   ├── best_model.pkl          # Tuned production model (Random Forest)
│   ├── preprocessor.joblib    # Fitted ColumnTransformer
│   └── target_encoder.joblib  # Fitted LabelEncoder (20 careers)
│
└── ml/
    └── inference/
        └── predictor.py        # ← THIS MODULE
            ├── Constants
            │   ├── PROJECT_ROOT, MODELS_DIR, artefact paths
            │   ├── PREPROCESSOR_COLUMNS    (schema, frozen tuple)
            │   ├── NUMERIC_RANGES          (validation)
            │   └── CATEGORICAL_VALUES      (validation)
            ├── Exceptions
            │   ├── PredictorError           (base)
            │   ├── MissingFieldError
            │   ├── InvalidValueError
            │   ├── UnknownCategoryError    ⊂ InvalidValueError
            │   └── OutOfRangeError         ⊂ InvalidValueError
            └── Predictor
                ├── __init__()           → loads artefacts
                ├── predict(user_input)  → full pipeline, Top-3 result
                ├── _load_artifacts()
                ├── _validate_input()
                ├── _validate_numeric()
                ├── _validate_category()
                └── _to_dataframe()
```

### Dependency direction (one-way)

```
UI / API layer
   │
   │  from ml.inference.predictor import Predictor
   │
   └──▶ Predictor ──▶ joblib.load() ──▶ models/*.pkl
```

The predictor never imports from the UI layer, never from the knowledge base,
never from the recommendation engine. It can be invoked from any framework
without code changes.

---

## 2. Prediction Flow

A single call to `predictor.predict(user_input)` runs the following steps:

```
                   ┌──────────────────────────────────┐
                   │  Predictor.predict(user_input)    │
                   └────────────────┬─────────────────┘
                                    │
            ┌───────────────────────┼────────────────────────┐
            ▼                       ▼                        ▼
   ┌────────────────┐      ┌────────────────┐      ┌────────────────────┐
   │ 1. Validate    │      │ 2. DataFrame   │      │ 3. Apply saved     │
   │    • presence  │ ───▶ │    (single row │ ───▶ │    preprocessor    │
   │    • ranges    │      │     in correct │      │    (ColumnTrans-   │
   │    • categories│      │     schema)    │      │     former)        │
   └────────────────┘      └────────────────┘      └────────┬───────────┘
                                                            │
                                                            ▼
                                               ┌────────────────────────┐
                                               │ 4. model.predict()      │
                                               │    → encoded label      │
                                               │ 5. model.predict_proba  │
                                               │    → prob. vector       │
                                               └───────────┬────────────┘
                                                           │
                                                           ▼
                                               ┌────────────────────────┐
                                               │ 6. target_encoder.     │
                                               │    inverse_transform() │
                                               │    → career name        │
                                               │ 7. Sort + slice Top-3   │
                                               │ 8. Round probabilities  │
                                               └───────────┬────────────┘
                                                           │
                                                           ▼
                                            ┌──────────────────────────┐
                                            │ Return dict              │
                                            │ { prediction, top3[] }   │
                                            └──────────────────────────┘
```

### Step-by-step

| # | Step | Responsibility |
|---|------|----------------|
| 1 | `_validate_input` | check required fields, numeric ranges and categorical domains |
| 2 | `_to_dataframe`  | build a single-row `pd.DataFrame` in the exact column order the fitted preprocessor expects |
| 3 | `preprocessor.transform` | run the persisted `ColumnTransformer` (numeric passthrough, ordinal-encoded education, one-hot-encoded nominal features) |
| 4 | `model.predict`     | obtain the encoded career label |
| 5 | `model.predict_proba` | obtain the full class probability vector |
| 6 | `target_encoder.classes_` lookup | decode the encoded label back to a human-readable career name |
| 7 | sort (`reverse=True`) + slice [:3] | produce a top-3 list |
| 8 | `round(prob, 4)` | format probabilities to four decimal places |

---

## 3. Input Schema

The input dictionary must follow the **original** dataset schema, i.e. the
schema used **before** the `ColumnTransformer` is applied.

### 3.1 Required columns (`PREPROCESSOR_COLUMNS`)

**Numeric (32 columns)**

| Field | Type | Range |
|---|---|---|
| `years_experience`              | `int`/`float` | 0 – 60 |
| `projects_completed`            | `int`/`float` | 0 – 500 |
| `certifications`                | `int`/`float` | 0 – 50 |
| `python_score` .. `research_score` (26 columns) | `int`/`float` | 0 – 10 |
| `salary_band`                   | `float` | 0 – ∞ |
| `career_growth_score`           | `int`/`float` | 0 – 10 |
| `job_satisfaction`              | `int`/`float` | 0 – 10 |
| `work_hours_per_week`           | `int`/`float` | 0 – 80 |

**Categorical (5 columns)**

| Field | Allowed values |
|---|---|
| `education_level`  | `High School`, `Associate`, `Bachelor`, `Master`, `PhD` |
| `remote_preference`| `Hybrid`, `On-site`, `Remote` |
| `country`          | `Australia`, `Canada`, `Germany`, `India`, `UK`, `USA` |
| `industry`         | `Consulting`, `Education`, `Finance`, `Gaming`, `Healthcare`, `Retail`, `Technology` |
| `employment_type`  | `Contract`, `Freelance`, `Full-time`, `Part-time` |

### 3.2 Example

```python
user_input = {
    "years_experience": 3,
    "education_level": "Bachelor",
    "projects_completed": 5,
    "certifications": 2,
    "python_score": 9, "java_score": 4, "javascript_score": 3, "sql_score": 7,
    "machine_learning_score": 8, "deep_learning_score": 6, "cloud_score": 5,
    "devops_score": 3, "cybersecurity_score": 2, "data_analysis_score": 8,
    "database_score": 6, "networking_score": 3, "mobile_score": 1,
    "game_dev_score": 0, "testing_score": 4, "business_analysis_score": 3,
    "product_management_score": 2, "ui_design_score": 1, "ux_research_score": 1,
    "communication_score": 6, "leadership_score": 4, "problem_solving_score": 8,
    "teamwork_score": 7, "agile_score": 5, "research_score": 6,
    "salary_band": 70000.0, "career_growth_score": 7,
    "job_satisfaction": 8, "work_hours_per_week": 40,
    "remote_preference": "Hybrid",
    "country": "USA",
    "industry": "Technology",
    "employment_type": "Full-time",
}
```

---

## 4. Output Schema

```python
{
    "prediction": "ML Engineer",      # str – top-1 decoded career
    "top3": [
        {"career": "ML Engineer",     "probability": 0.9920},
        {"career": "Data Scientist",  "probability": 0.8710},
        {"career": "Data Engineer",   "probability": 0.7440},
    ],
}
```

Contract:

* `prediction` ∈ `string`, non-empty.
* `top3` is a `list` of 1 – 3 entries, sorted by `probability` in
  **descending** order.
* `probability` ∈ `float`, rounded to **4 decimal places**, in `[0.0, 1.0]`.
* `career` ∉ string – duplicates cannot occur (probabilities come from a
  probability vector over distinct classes).

---

## 5. Error Handling

The module raises meaningful, typed exceptions so that callers can react
programmatically. **All** errors inherit from `PredictorError`
(`ValueError`).

| Exception | When | `.field` / `.missing` |
|-----------|------|------------------------|
| `MissingFieldError`   | one or more required fields are absent from the input dict | `.missing: list[str]` lists every missing column |
| `UnknownCategoryError`| categorical column receives a string not in the allowed set, or a non-string value | `.field: str` |
| `OutOfRangeError`     | numeric column is below `min` or above `max` of its declared range | `.field: str` |
| `InvalidValueError`   | base class – also raised when a numeric value cannot be coerced to `float`, or the input is not a dict at all | `.field: str` (when applicable) |
| `PredictorError`      | base class for any other predictor-level failure (e.g. class/probability length mismatch) | – |
| `FileNotFoundError`   | any of the three artefact files is missing from `models/` | – |

Example caller code:

```python
from ml.inference.predictor import (
    Predictor, MissingFieldError, UnknownCategoryError, OutOfRangeError,
)

predictor = Predictor()

try:
    result = predictor.predict(user_input)
except MissingFieldError as exc:
    print("Missing fields:", exc.missing)
except UnknownCategoryError as exc:
    print(f"Invalid category for {exc.field}: {exc}")
except OutOfRangeError as exc:
    print(f"Out-of-range value for {exc.field}: {exc}")
```

---

## 6. Logging

The module configures a logger via `ml.config.setup_logging`:

* Name: `ml.inference.predictor`
* Level: `DEBUG`
* Handlers: `StreamHandler` (console, `INFO+`); optional file handler when
  configured globally.
* Format: `%(asctime)s | %(levelname)-8s | %(name)s | %(message)s`

`print()` is **never** used.

---

## 7. Reusable Across Front-Ends

Because the predictor exposes only a single Python entry point
(`Predictor.predict`) and has no framework dependencies, it can be dropped
into any of the following without code changes:

| Front-end | Wrapper sketch |
|-----------|----------------|
| **Streamlit** | `predictor = Predictor(); st.json(predictor.predict(payload))` |
| **Flask**     | `@app.post("/predict"); def predict(): return predictor.predict(request.json)` |
| **FastAPI**   | `@app.post("/predict"); def predict(payload: dict): return predictor.predict(payload)` |
| **Plain script** | `print(Predictor().predict(user_dict))` |
| **Celery / RPC** | wrap in a task – the function is pure and thread-safe-per-call |

> Note: heavy artefacts (~115 MB for the model) are loaded **once**, on
> `Predictor.__init__`, and reused for the lifetime of the process. In a
> web server that means **one load per worker**.

---

## 8. Future Extension Points

These are intentionally **out of scope** for this module but the design
leaves room for them without refactoring the predictor:

| Future module | Where it would plug in |
|---------------|-----------------------|
| **SHAP explainer**     | call `self._model` after `predict()`; pass the preprocessed feature matrix. |
| **Recommendation engine** | consume `result["top3"]` and attach learning-roadmap / skill-gap data from the knowledge base. |
| **Confidence engine** | consume `result["top3"][0]["probability"]` and surface "high / medium / low". |
| **Skill gap analysis** | compare `user_input` skill scores with the prototype required-skill vector for each candidate career. |
| **Model registry / versioning** | add a constructor flag (e.g. `model_version="v3"`) that selects among multiple `best_model_v*.pkl` artefacts. |
| **Online learning / re-training** | expose a separate `Predictor.update(...)` method that re-fits the underlying estimator and re-persists artefacts – the public API of `predict()` stays unchanged. |
| **Streaming / batch input** | add `predict_batch(list[dict])` that loops over `self.predict()` and returns a list of dicts. |
| **Schema auto-sync** | derive `PREPROCESSOR_COLUMNS` automatically from `preprocessor.feature_names_in_` at load time, instead of a hard-coded tuple. |

---

## 9. Test Coverage

`tests/test_predictor.py` – 21 tests:

* **Valid prediction** – schema, types, ordering integrity.
* **Missing feature** – one, multiple, all fields missing.
* **Invalid category** – unknown education, country; non-string value; error message lists allowed values.
* **Invalid range** – above maximum, below minimum, non-numeric value.
* **Top-3 ordering** – descending sort, uniqueness, prediction present.
* **Reusability** – deterministic; does not mutate input.
* **Edge cases** – non-dict input.

```bash
$ python -m unittest tests.test_predictor -v
...
Ran 21 tests in 8.395s

OK
```

---

## 10. Quick Start

```python
from ml.inference.predictor import Predictor

predictor = Predictor()  # artefacts loaded once

result = predictor.predict(user_input)

print(result["prediction"])
for cand in result["top3"]:
    print(f"  {cand['career']:<25} {cand['probability']:.4f}")
```

Sample output:

```
ML Engineer
  ML Engineer               0.4678
  Data Scientist            0.1733
  AI Research Engineer      0.1594
```

---

*Generated for the CareerPilot-AI research-paper deliverable.*
