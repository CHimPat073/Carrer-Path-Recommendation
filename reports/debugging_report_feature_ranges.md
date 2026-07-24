# Debugging Report: Feature Ranges Not Being Applied

## Summary

**ROOT CAUSE IDENTIFIED**: Key mismatch between `feature_ranges.json` and the generator code in `production_synthetic_pipeline.py`.

---

## Evidence

### 1. Key Format Mismatch

| Source | Key Format | Example |
|--------|------------|---------|
| `feature_ranges.json` | Title Case with spaces | `"Python"`, `"Machine Learning"`, `"UI Design"` |
| Generator code | lowercase with underscores | `"python"`, `"machine_learning"`, `"ui_design"` |

### 2. Lookup Failure Confirmed

```
career_range.get("python"): NOT FOUND (falls back to default)
career_range.get("java"): NOT FOUND (falls back to default)
career_range.get("javascript"): NOT FOUND (falls back to default)
career_range.get("machine_learning"): NOT FOUND (falls back to default)
...
```

### 3. Fallback Default Values

When lookup fails (line 214-215 in `production_synthetic_pipeline.py`):

```python
if feature_info is None:
    feature_info = {"min": 1, "max": 10, "average": 5}
```

This means **all skill scores** default to random values centered around 5 with range 1-10, regardless of career.

### 4. Generated vs Expected Comparison

| Career | Feature | Expected Range | Generated | Match? |
|--------|---------|----------------|-----------|--------|
| Software Engineer | Python | 4-8 | 3 | NO |
| Software Engineer | Machine Learning | 1-5 | 4 | YES (by chance) |
| Software Engineer | Cloud | 3-8 | 4 | YES (by chance) |
| Backend Developer | Python | 4-9 | 4 | YES (by chance) |
| Frontend Developer | JavaScript | 7-10 | varies | NO |

Values appear to match occasionally due to random variation, but the career-specific distributions are NOT being applied.

---

## Root Cause Analysis

### Location: `ml/preprocessing/production_synthetic_pipeline.py`

**Function**: `_generate_base_row`  
**Lines**: 212-215

```python
feature_name = feature.replace("_score", "")  # "python_score" -> "python"
feature_info = career_range.get(feature_name) if feature_name in career_range else None
if feature_info is None:
    feature_info = {"min": 1, "max": 10, "average": 5}  # DEFAULT FALLBACK
```

### The Bug

1. `feature_ranges.json` stores keys as: `"Python"`, `"Machine Learning"`, `"Cybersecurity"`
2. Generator computes: `"python"`, `"machine_learning"`, `"cybersecurity"`
3. `career_range.get("python")` returns `None` (key doesn't exist)
4. Code falls back to default `{"min": 1, "max": 10, "average": 5}`
5. All careers generate with same distribution (centered on 5)

---

## Impact on Validation Metrics

This explains the validation report findings:

| Metric | Observed Value | Explanation |
|--------|---------------|-------------|
| CV Accuracy | ~4.8% (random) | Features have no career-specific signal |
| Mutual Information | 0.009 | Near-zero discriminative power |
| ANOVA Significant | 1/26 | Almost no feature varies by career |
| Skill Cluster | Around 5 | Default fallback value |

---

## Additional Issues Found

### 1. Underscore vs Space Handling

The generator uses underscores:
- `"machine_learning"` 
- `"deep_learning"`

But JSON has spaces:
- `"Machine Learning"`
- `"Deep Learning"`

This is a double mismatch (case + underscore vs space).

### 2. Special Case: "years_experience"

This key happens to work because:
- JSON: `"years_experience"` (snake_case)
- Generator: `"years_experience"` (same)

But JSON also has Title Case versions: `"Years Experience"`, which also fail.

---

## Bug Classification

| Category | Details |
|----------|---------|
| **Type** | Key lookup mismatch |
| **Severity** | Critical - prevents career-specific differentiation |
| **Root Cause** | Inconsistent key naming between JSON and code |
| **Fix Location** | `ml/preprocessing/production_synthetic_pipeline.py:212-215` |

---

## Fix Required (NOT IMPLEMENTED)

The code needs to normalize keys before lookup:

1. Convert generator's lowercase with underscores to Title Case with spaces, OR
2. Convert JSON keys to lowercase with underscores when building the lookup

Example fix approach:
```python
# Convert "python" -> "Python"
# Convert "machine_learning" -> "Machine Learning"
normalized_key = feature_name.replace('_', ' ').title()
# But this has edge cases...
```

Or better:
```python
# Map lowercase keys to Title Case
KEY_MAP = {
    'python': 'Python',
    'java': 'Java',
    'javascript': 'JavaScript',
    'machine_learning': 'Machine Learning',
    # ... etc
}
feature_info = career_range.get(KEY_MAP.get(feature_name, feature_name))
```

---

## Recommendation

1. **DO NOT modify the JSON** - it uses a consistent naming convention
2. **DO modify the generator** to normalize keys before lookup
3. Add unit tests to verify career-specific ranges are actually applied
4. Re-run validation to confirm discriminative power improves