# Confidence Engine

## Architecture

The Confidence Engine is a pure post-processing layer that consumes the structured output from `Predictor.predict()` and converts it into a confidence report.

It does not:

- load models
- preprocess input data
- perform inference
- access the knowledge base
- use SHAP
- generate recommendations or explanations

The module is intended to stay framework-agnostic so it can be reused by Streamlit, Flask, FastAPI, notebooks, and CLI workflows.

## Input Contract

The engine expects a dictionary shaped like the `Predictor.predict()` output:

```python
{
    "prediction": "ML Engineer",
    "top3": [
        {"career": "ML Engineer", "probability": 0.9923},
        {"career": "Data Scientist", "probability": 0.8734},
        {"career": "Data Engineer", "probability": 0.7121}
    ]
}
```

The engine validates that the payload is well-formed, that probabilities are finite numbers in the range `[0, 1]`, and that at least one ranked result is present.

## Confidence Calculation

The first ranked probability becomes the confidence score.

- `confidence_score = top1_probability * 100`
- `confidence_level` is derived from the score band:
  - `>= 95` -> `Very High`
  - `90-95` -> `High`
  - `75-90` -> `Medium`
  - `< 75` -> `Low`

The engine also reports:

- `margin_from_second`: difference between the first and second probabilities, expressed as percentage points
- `entropy`: Shannon entropy of the ranked probability slice

The top-3 list is preserved for downstream rendering, with probabilities rounded to four decimal places.

## Reliability Rules

A prediction is marked `Reliable` when all three conditions hold:

- confidence score is at least `90%`
- margin from second place is at least `5 percentage points`
- entropy is at most `1.0`

Otherwise, the prediction is marked `Uncertain`.

This rule combines strength of the top prediction with how concentrated the ranked distribution is.

## Error Handling

The engine raises explicit errors for:

- non-dictionary inputs
- missing `prediction` or `top3` keys
- empty prediction lists
- invalid probability values
- non-numeric, NaN, or infinite probabilities

The module is intentionally strict about input shape, but it does not require the top-3 probabilities to sum to exactly one, because the predictor only returns a ranked slice of the full class distribution.

## Future Extensions

Possible future additions without changing the core contract:

- configurable confidence thresholds
- richer reliability bands
- calibration-aware scoring
- framework adapters for direct API response formatting
- optional audit metadata for logging and observability
