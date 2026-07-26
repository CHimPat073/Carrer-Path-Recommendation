# Baseline Model Training Report

## Metadata
- **Generated**: 2026-07-26T19:47:39.530499
- **Random Seed**: 42
- **CV Folds**: 5

## Model Comparison

### Performance Metrics

| Model | Accuracy | Precision | Recall | F1 Score | CV Accuracy | Train Time (s) | Pred Time (s) |
|-------|----------|-----------|--------|----------|-------------|----------------|---------------|
| Random Forest | 0.9968 | 0.9968 | 0.9968 | 0.9968 | 0.9951 | 0.66 | 0.0795 |
| Extra Trees | 0.9960 | 0.9960 | 0.9960 | 0.9960 | 0.9959 | 0.51 | 0.0831 |
| XGBoost | 0.9962 | 0.9963 | 0.9963 | 0.9963 | 0.9950 | 3.49 | 0.0370 |
| LightGBM | 0.9958 | 0.9958 | 0.9957 | 0.9958 | 0.9954 | 3.46 | 0.2216 |
| CatBoost | 0.9930 | 0.9930 | 0.9930 | 0.9930 | 0.9928 | 9.30 | 0.0129 |

### Rankings

**By F1 Score:**
1. Random Forest: 0.9968
2. XGBoost: 0.9963
3. Extra Trees: 0.9960
4. LightGBM: 0.9958
5. CatBoost: 0.9930

**By CV Accuracy:**
1. Extra Trees: 0.9959
2. LightGBM: 0.9954
3. Random Forest: 0.9951
4. XGBoost: 0.9950
5. CatBoost: 0.9928

**By Training Time (fastest first):**
1. Extra Trees: 0.51s
2. Random Forest: 0.66s
3. LightGBM: 3.46s
4. XGBoost: 3.49s
5. CatBoost: 9.30s

**By Prediction Time (fastest first):**
1. CatBoost: 0.0129s
2. XGBoost: 0.0370s
3. Random Forest: 0.0795s
4. Extra Trees: 0.0831s
5. LightGBM: 0.2216s

## Best Model Recommendation

**Best Model: Random Forest**

Based on F1 Score ranking, **Random Forest** is the recommended model for this classification task.

## Model Advantages and Disadvantages

### Random Forest
- **Advantages**: Robust to overfitting, handles missing values, provides feature importance
- **Disadvantages**: Slower than single decision tree, less interpretable

### Extra Trees
- **Advantages**: Faster training than Random Forest, more randomness leads to better generalization
- **Disadvantages**: Less interpretable, can be memory intensive

### XGBoost
- **Advantages**: High accuracy, handles sparse data, built-in regularization
- **Disadvantages**: Requires encoding for categorical features, can overfit

### LightGBM
- **Advantages**: Fastest training, handles large datasets, leaf-wise growth
- **Disadvantages**: Can overfit on small datasets, sensitive to hyperparameters

### CatBoost
- **Advantages**: Native categorical handling, built-in overfitting detection
- **Disadvantages**: Slower than LightGBM, less community support

## Conclusion

All models were trained with default parameters. The dataset has 53 features after preprocessing.
Based on the evaluation, **Random Forest** achieved the highest F1 score of 0.9968.

For production deployment, consider:
1. Using the best performing model: Random Forest
2. Enabling early stopping to prevent overfitting
3. Performing hyperparameter tuning if accuracy needs improvement
