# SHAP Display Layer

## Purpose

The SHAP Engine keeps the model logic unchanged and only changes how feature names are presented to users.

The engine still works on the saved transformed feature matrix internally, but every visible label is translated through a display-name mapping file before being returned to the caller or drawn in a plot.

## Mapping System

Human-readable labels are stored in `knowledge_base/display_names.json`.

Example mappings:

- `python_score` -> `Python Skill`
- `machine_learning_score` -> `Machine Learning`
- `salary_band` -> `Expected Salary`
- `years_experience` -> `Years of Experience`

The SHAP engine loads this JSON once and applies it only at the presentation layer.

## Internal vs Display Names

The engine preserves original feature names internally for SHAP computation, aggregation, and traceability.

For every surfaced contribution it includes:

- `original_feature_name` for the internal dataset column
- `display_feature_name` for the human-readable label
- `processed_feature_name` for the transformed feature lineage when available

This keeps the explanation output readable without losing provenance.

## Plot Labeling

The following plots use display names instead of internal feature names:

- `shap_summary.png`
- `shap_bar.png`
- `shap_beeswarm.png`
- `shap_waterfall.png`

The waterfall plot is configured to show at least 20 features so that more context is visible in a single view.

## Design Rules

- Do not change the Predictor.
- Do not change preprocessing.
- Do not change the model.
- Do not change SHAP values.
- Only replace the labels shown to the user.

## Fallback Behavior

If a feature is not present in `display_names.json`, the engine falls back to a readable title-cased version of the internal column name.

## Outcome

This display layer makes the SHAP outputs easier to read in Streamlit, Flask, FastAPI, reports, and notebooks while keeping the underlying explanation values unchanged.
