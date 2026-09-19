# SHAP Engine Report

## Architecture

The SHAP Engine is a standalone explainability module that operates on the trained production model and saved preprocessor. It is intentionally isolated from UI frameworks, recommendation logic, confidence scoring, and knowledge-base access.

The engine loads the persisted artifacts once, constructs one SHAP explainer once, and reuses that explainer for later requests. It supports both local explanations for a single prediction and global explanation summaries for the model as a whole.

## SHAP Workflow

1. Load `models/best_model.pkl` and `models/preprocessor.joblib`.
2. Validate input shape.
3. Transform raw user input with the persisted preprocessor when needed.
4. Create a SHAP `TreeExplainer` for the production tree model.
5. Compute SHAP values for the selected class.
6. Aggregate transformed contributions back to raw feature names.
7. Return structured explanation data and save plots to disk.

## Global Explanation

Global explainability is provided through:

- global feature importance ranking
- SHAP summary values
- mean absolute SHAP values

These outputs are suitable for dashboards, batch reporting, and downstream analysis.

## Local Explanation

Local explainability focuses on one prediction and returns:

- the predicted career label
- top positive contributors
- top negative contributors

The response is structured data, not a plot-only artifact, so it can be consumed by Streamlit, Flask, FastAPI, or a recommendation layer later.

## Output Schema

Example local response:

```python
{
    "prediction": "ML Engineer",
    "top_positive_features": [
        {"feature": "python_score", "value": 9, "shap_value": 2.41},
        {"feature": "machine_learning_score", "value": 8, "shap_value": 1.93}
    ],
    "top_negative_features": [
        {"feature": "communication_score", "value": 4, "shap_value": -0.42}
    ]
}
```

## Performance Considerations

- The model, preprocessor, and explainer are loaded once per engine instance.
- SHAP computation is the expensive step, so the engine avoids rebuilding explainers on each request.
- Global summaries should typically be computed on a bounded sample set for large batches.
- Plot generation writes to disk only and does not display figures interactively.

## Limitations

- The engine explains the selected tree model only.
- It does not compute confidence.
- It does not recommend careers or learning paths.
- It does not access the knowledge base.
- It does not retrain models.
- Plot generation depends on the availability of SHAP and matplotlib in the runtime environment.

## Future Extensions

Potential follow-up improvements:

- batch explanation APIs
- cached background datasets for faster global summaries
- class-specific explanation selection strategies
- richer visualization templates for notebooks and apps
- model registry integration for multiple production versions
