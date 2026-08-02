# Recommendation Engine Report

## Architecture

The Recommendation Engine is a pure orchestration layer. It does not load models, preprocess data, run inference, compute SHAP values, or calculate confidence. It only merges already computed outputs with metadata from the existing knowledge base.

The engine is intentionally framework-agnostic so future Streamlit, Flask, and FastAPI integrations can call the same service without duplicating business logic.

## Responsibilities

The engine performs four jobs only:

1. Validate the incoming payloads.
2. Merge prediction, confidence, SHAP, and KB information.
3. Extract explainable strengths and improvement areas from SHAP values.
4. Return one normalized recommendation object.

It does not contain career heuristics, model logic, or LLM-generated text.

## Input Contract

The engine expects already-computed dictionaries:

```python
prediction_result = {
    "prediction": "ML Engineer",
    "top3": [
        {"career": "ML Engineer", "probability": 0.9978},
        {"career": "Data Scientist", "probability": 0.8621},
        {"career": "Data Engineer", "probability": 0.7412},
    ],
}

confidence_result = {
    "confidence_score": 99.78,
    "confidence_level": "Very High",
    "prediction_reliability": "Reliable",
}

shap_result = {
    "top_positive_features": [...],
    "top_negative_features": [...],
}
```

`user_input` is accepted for orchestration boundaries, but the engine never uses it to compute model outputs.

## Output Contract

The final object contains:

```python
{
    "recommended_career": "ML Engineer",
    "confidence": 99.78,
    "confidence_level": "Very High",
    "prediction_reliability": "Reliable",
    "top3_careers": [...],
    "career_information": {
        "description": "...",
        "career_family": "Data & AI",
        "industry": ["AI startups", "Technology", "Autonomous systems", "E-commerce"],
        "salary_range": "$110k-$220k USD",
    },
    "strengths": [...],
    "improvement_areas": [...],
    "required_skills": [...],
    "education_path": [...],
    "next_roles": [...],
}
```

## Knowledge Base Integration

The engine reads only the existing JSON files in `knowledge_base/`:

* `careers.json` for salary bounds and catalog presence
* `career_personas.json` for descriptions, skills, education, and industries
* `roadmaps.json` for progression stages and next roles
* `domain_metadata.json` for career family mapping

The engine never duplicates this metadata in code and never hardcodes career lists.

## SHAP Handling

Positive SHAP contributions become `strengths` and negative SHAP contributions become `improvement_areas`.

Reasons are derived directly from the SHAP value itself, so the engine stays explainable and does not invent new justifications.

## Error Handling

The engine raises explicit exceptions for:

* missing prediction output
* missing confidence output
* missing SHAP output
* unknown careers
* missing knowledge-base entries
* invalid input payloads

## Future Extensions

Possible next steps without changing the orchestration contract:

* API adapters for Streamlit, Flask, and FastAPI
* response schema versioning
* caching of loaded KB files
* localization for career descriptions
* audit metadata for traceability