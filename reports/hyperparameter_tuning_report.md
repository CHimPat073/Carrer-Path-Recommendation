# Hyperparameter Tuning Report

Generated: 2026-07-31T02:27:14.518224

## Summary

| Model | Best CV Score | Test Accuracy | Test F1 | Training Time (s) | Prediction Time (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random Forest | 0.9959 | 0.9978 | 0.9978 | 72.73 | 0.0998 |
| XGBoost | 0.9959 | 0.9968 | 0.9968 | 95.88 | 0.0635 |
| LightGBM | 0.9959 | 0.9958 | 0.9958 | 295.19 | 0.1578 |

## Detailed Results

### Random Forest

- Model: Random Forest
- Search Space: {"bootstrap": [true, false], "max_depth": [null, 10, 20, 30], "max_features": ["sqrt", "log2", null], "min_samples_leaf": [1, 2, 4], "min_samples_split": [2, 5, 10], "n_estimators": [100, 200, 300]}
- Best Parameters: {"bootstrap": false, "max_depth": 30, "max_features": "log2", "min_samples_leaf": 2, "min_samples_split": 2, "n_estimators": 300}
- Best CV Score: 0.9959
- Training Time: 72.73s
- Prediction Time: 0.0998s
- Accuracy: 0.9978
- Precision: 0.9978
- Recall: 0.9978
- F1 Score: 0.9978
- Comparison with Baseline:
  - Accuracy Improvement: 0.10%
  - Precision Improvement: 0.10%
  - Recall Improvement: 0.10%
  - F1 Improvement: 0.10%

### XGBoost

- Model: XGBoost
- Search Space: {"colsample_bytree": [0.7, 0.8, 0.9], "gamma": [0, 0.1, 0.2], "learning_rate": [0.01, 0.05, 0.1], "max_depth": [3, 5, 7], "min_child_weight": [1, 3, 5], "n_estimators": [100, 200, 300], "subsample": [0.7, 0.8, 0.9]}
- Best Parameters: {"colsample_bytree": 0.7, "gamma": 0.1, "learning_rate": 0.05, "max_depth": 3, "min_child_weight": 1, "n_estimators": 300, "subsample": 0.7}
- Best CV Score: 0.9959
- Training Time: 95.88s
- Prediction Time: 0.0635s
- Accuracy: 0.9968
- Precision: 0.9968
- Recall: 0.9967
- F1 Score: 0.9968
- Comparison with Baseline:
  - Accuracy Improvement: 0.06%
  - Precision Improvement: 0.06%
  - Recall Improvement: 0.06%
  - F1 Improvement: 0.06%

### LightGBM

- Model: LightGBM
- Search Space: {"colsample_bytree": [0.7, 0.8, 0.9], "learning_rate": [0.01, 0.05, 0.1], "max_depth": [3, 5, 7], "min_child_samples": [5, 10, 20], "n_estimators": [100, 200, 300], "num_leaves": [15, 31, 63], "subsample": [0.7, 0.8, 0.9]}
- Best Parameters: {"colsample_bytree": 0.8, "learning_rate": 0.05, "max_depth": 5, "min_child_samples": 10, "n_estimators": 200, "num_leaves": 63, "subsample": 0.7}
- Best CV Score: 0.9959
- Training Time: 295.19s
- Prediction Time: 0.1578s
- Accuracy: 0.9958
- Precision: 0.9958
- Recall: 0.9957
- F1 Score: 0.9958
- Comparison with Baseline:
  - Accuracy Improvement: -0.01%
  - Precision Improvement: -0.00%
  - Recall Improvement: -0.01%
  - F1 Improvement: -0.00%

## Final Ranking

| Rank | Model | F1 Score | Best CV Score |
| --- | --- | ---: | ---: |
| 1 | Random Forest | 0.9978 | 0.9959 |
| 2 | XGBoost | 0.9968 | 0.9959 |
| 3 | LightGBM | 0.9958 | 0.9959 |

## Final Recommendation

Recommended production model: Random Forest
Reason: it achieved the highest F1 Macro score and strong cross-validation performance while keeping the training and prediction cost practical.