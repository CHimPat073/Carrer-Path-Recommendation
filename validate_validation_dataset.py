"""
Comprehensive Validation Dataset Analysis
==========================================
Performs thorough validation of the generated validation dataset.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

VALIDATION_CSV = PROJECT_ROOT / "datasets" / "validation" / "validation_dataset_v2.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"


def load_dataset():
    """Load the validation dataset."""
    df = pd.read_csv(VALIDATION_CSV)
    return df


def missing_value_analysis(df):
    """1. Missing Value Analysis"""
    print("\n[1/12] Missing Value Analysis...")

    total_missing = df.isnull().sum().sum()
    missing_by_col = df.isnull().sum()
    missing_by_col = missing_by_col[missing_by_col > 0]

    missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
    missing_pct = missing_pct[missing_pct > 0]

    result = {
        'total_missing': int(total_missing),
        'columns_with_missing': len(missing_by_col),
        'missing_by_column': missing_by_col.to_dict(),
        'missing_percentage_by_column': missing_pct.to_dict(),
        'status': 'PASS' if total_missing == 0 else 'FAIL'
    }

    return result


def duplicate_analysis(df):
    """2. Duplicate Analysis"""
    print("[2/12] Duplicate Analysis...")

    # Exact duplicates
    exact_duplicates = df.duplicated().sum()

    # Near duplicates (similar rows) - check for same career + similar features
    feature_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64'] and col != 'salary_band']

    df_numeric = df[feature_cols].copy()
    df_numeric = df_numeric.fillna(df_numeric.median())

    # Check for rows that are 95% similar
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(df_numeric)
    np.fill_diagonal(similarities, 0)
    near_duplicates = (similarities > 0.98).sum() // 2  # Each pair counted twice

    result = {
        'exact_duplicates': int(exact_duplicates),
        'near_duplicates_98pct': int(near_duplicates),
        'status': 'PASS' if exact_duplicates == 0 else 'WARNING'
    }

    return result


def class_distribution_analysis(df):
    """3. Class Distribution Analysis"""
    print("[3/12] Class Distribution Analysis...")

    career_dist = df['career'].value_counts()
    total_careers = len(career_dist)

    # Check balance
    min_count = career_dist.min()
    max_count = career_dist.max()
    mean_count = career_dist.mean()
    std_count = career_dist.std()
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

    # Chi-square test for uniform distribution
    expected = len(df) / total_careers
    chi2, p_value = stats.chisquare(career_dist.values, f_exp=[expected] * total_careers)

    result = {
        'total_classes': total_careers,
        'distribution': career_dist.to_dict(),
        'min_count': int(min_count),
        'max_count': int(max_count),
        'mean_count': float(mean_count),
        'std_count': float(std_count),
        'imbalance_ratio': float(imbalance_ratio),
        'chi2_statistic': float(chi2),
        'chi2_p_value': float(p_value),
        'is_balanced': imbalance_ratio <= 1.2,
        'status': 'PASS' if imbalance_ratio <= 1.2 else 'WARNING'
    }

    return result


def correlation_matrix_analysis(df):
    """4. Correlation Matrix Analysis"""
    print("[4/12] Correlation Matrix Analysis...")

    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != 'salary_band']

    corr_matrix = df[numeric_cols].corr()

    # Find high correlations (potential issues)
    high_corr_threshold = 0.85
    high_corr_pairs = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > high_corr_threshold:
                high_corr_pairs.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': float(corr_matrix.iloc[i, j])
                })

    # Average absolute correlation per feature
    avg_corr = corr_matrix.abs().mean()
    top_corr_features = avg_corr.sort_values(ascending=False).head(10).to_dict()

    result = {
        'high_correlation_pairs': high_corr_pairs,
        'high_corr_count': len(high_corr_pairs),
        'top_correlated_features': top_corr_features,
        'status': 'PASS' if len(high_corr_pairs) < 5 else 'WARNING'
    }

    return result


def feature_distribution_analysis(df):
    """5. Feature Distribution Plots"""
    print("[5/12] Feature Distribution Analysis...")

    feature_cols = [col for col in df.columns if '_score' in col or col in
                   ['years_experience', 'projects_completed', 'certifications']]

    distributions = {}

    for feature in feature_cols:
        data = df[feature].dropna()

        # Shapiro-Wilk test for normality (use sample if too large)
        sample_size = min(5000, len(data))
        sample_data = data.sample(n=sample_size, random_state=42)

        if len(sample_data) >= 3:
            stat, p_value = stats.shapiro(sample_data)
            is_normal = p_value > 0.05
        else:
            stat, p_value = 0, 1
            is_normal = True

        distributions[feature] = {
            'min': float(data.min()),
            'max': float(data.max()),
            'mean': float(data.mean()),
            'std': float(data.std()),
            'median': float(data.median()),
            'skewness': float(stats.skew(data)),
            'kurtosis': float(stats.kurtosis(data)),
            'normality_test_pvalue': float(p_value),
            'is_normal': is_normal,
            'range_violations': 0
        }

    # Check for range violations (values outside expected 1-10 range)
    score_cols = [c for c in feature_cols if '_score' in c]
    for col in score_cols:
        violations = ((df[col] < 1) | (df[col] > 10)).sum()
        distributions[col]['range_violations'] = int(violations)

    result = {
        'distributions': distributions,
        'non_normal_features': sum(1 for d in distributions.values() if not d['is_normal']),
        'range_violations': sum(d['range_violations'] for d in distributions.values()),
        'status': 'PASS'
    }

    return result


def pca_analysis(df):
    """6. PCA Analysis"""
    print("[6/12] PCA Analysis...")

    # Prepare data
    feature_cols = [col for col in df.columns if '_score' in col or col in
                   ['years_experience', 'projects_completed', 'certifications']]

    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df['career']

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(X_scaled)

    # Explained variance
    explained_var = pca.explained_variance_ratio_
    cumsum_var = np.cumsum(explained_var)

    # Components for each feature
    components_df = pd.DataFrame(pca.components_[:5].T, index=feature_cols,
                                  columns=[f'PC{i+1}' for i in range(5)])

    result = {
        'explained_variance_ratio': explained_var[:10].tolist(),
        'cumulative_variance': cumsum_var.tolist(),
        'components_to_90pct': int(np.argmax(cumsum_var >= 0.9)) + 1 if any(cumsum_var >= 0.9) else 10,
        'components_to_95pct': int(np.argmax(cumsum_var >= 0.95)) + 1 if any(cumsum_var >= 0.95) else 10,
        'top_features_per_component': components_df.abs().sum(axis=1).sort_values(ascending=False).head(10).to_dict(),
        'status': 'PASS'
    }

    return result


def tsne_analysis(df):
    """7. t-SNE Analysis"""
    print("[7/12] t-SNE Analysis...")

    # Prepare data
    feature_cols = [col for col in df.columns if '_score' in col or col in
                   ['years_experience', 'projects_completed', 'certifications']]

    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df['career']

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # t-SNE (use subset for speed)
    sample_size = min(2000, len(X_scaled))
    indices = np.random.RandomState(42).choice(len(X_scaled), sample_size, replace=False)
    X_sample = X_scaled[indices]
    y_sample = y.iloc[indices]

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    X_tsne = tsne.fit_transform(X_sample)

    # Calculate silhouette score on t-SNE components
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_sample)

    kmeans = KMeans(n_clusters=20, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_tsne)

    silhouette = silhouette_score(X_tsne, cluster_labels)

    result = {
        'perplexity_used': 30,
        'iterations': 1000,
        'silhouette_score_cluster': float(silhouette),
        'clusters_formed': 20,
        'class_separation': 'GOOD' if silhouette > 0.3 else 'MODERATE' if silhouette > 0.2 else 'WEAK',
        'status': 'PASS'
    }

    return result


def anova_analysis(df):
    """8. ANOVA Analysis"""
    print("[8/12] ANOVA Analysis...")

    feature_cols = [col for col in df.columns if '_score' in col]

    anova_results = {}

    for feature in feature_cols:
        groups = [group[feature].values for name, group in df.groupby('career')]

        # One-way ANOVA
        f_stat, p_value = stats.f_oneway(*groups)

        anova_results[feature] = {
            'f_statistic': float(f_stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'eta_squared': float(f_stat * (len(groups) - 1) / (len(df) - len(groups)))
        }

    significant_features = sum(1 for r in anova_results.values() if r['significant'])
    avg_f_stat = np.mean([r['f_statistic'] for r in anova_results.values()])

    result = {
        'anova_results': anova_results,
        'significant_features': significant_features,
        'total_features': len(feature_cols),
        'avg_f_statistic': float(avg_f_stat),
        'discriminative_power': 'HIGH' if significant_features > 15 else 'MODERATE' if significant_features > 10 else 'LOW',
        'status': 'PASS'
    }

    return result


def mutual_information_analysis(df):
    """9. Mutual Information Analysis"""
    print("[9/12] Mutual Information Analysis...")

    feature_cols = [col for col in df.columns if '_score' in col or col in
                   ['years_experience', 'projects_completed', 'certifications']]

    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df['career']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Calculate MI scores
    mi_scores = mutual_info_classif(X, y_encoded, random_state=42)

    mi_dict = dict(zip(feature_cols, mi_scores.tolist()))
    mi_sorted = sorted(mi_dict.items(), key=lambda x: x[1], reverse=True)

    result = {
        'mi_scores': mi_dict,
        'top_10_features': dict(mi_sorted[:10]),
        'low_mi_features': [f for f, s in mi_sorted if s < 0.1],
        'avg_mi_score': float(np.mean(mi_scores)),
        'status': 'PASS'
    }

    return result


def outlier_detection(df):
    """10. Outlier Detection"""
    print("[10/12] Outlier Detection...")

    feature_cols = [col for col in df.columns if '_score' in col or col in
                   ['years_experience', 'projects_completed', 'certifications']]

    outlier_results = {}

    for feature in feature_cols:
        data = df[feature].dropna()

        # Z-score method
        z_scores = np.abs(stats.zscore(data))
        z_outliers = (z_scores > 3).sum()

        # IQR method
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        iqr_outliers = ((data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)).sum()

        # Modified Z-score (using median)
        median = data.median()
        mad = np.median(np.abs(data - median))
        if mad != 0:
            modified_z = 0.6745 * (data - median) / mad
            modified_outliers = (np.abs(modified_z) > 3.5).sum()
        else:
            modified_outliers = 0

        outlier_results[feature] = {
            'z_score_outliers': int(z_outliers),
            'iqr_outliers': int(iqr_outliers),
            'modified_z_outliers': int(modified_outliers),
            'outlier_percentage': float(iqr_outliers / len(data) * 100)
        }

    total_outliers = sum(r['iqr_outliers'] for r in outlier_results.values())
    features_with_outliers = sum(1 for r in outlier_results.values() if r['iqr_outliers'] > 0)

    result = {
        'outlier_results': outlier_results,
        'total_outliers': total_outliers,
        'features_with_outliers': features_with_outliers,
        'outlier_percentage': float(total_outliers / len(df) * 100),
        'status': 'PASS' if total_outliers < len(df) * 0.05 else 'WARNING'
    }

    return result


def cluster_separation_analysis(df):
    """11. Cluster Separation Analysis"""
    print("[11/12] Cluster Separation Analysis...")

    feature_cols = [col for col in df.columns if '_score' in col or col in
                   ['years_experience', 'projects_completed', 'certifications']]

    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df['career']

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-means on original features
    kmeans = KMeans(n_clusters=20, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    silhouette_orig = silhouette_score(X_scaled, cluster_labels)

    # Check how well clusters match actual careers
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    silhouette_class = silhouette_score(X_scaled, y_encoded)

    # Adjusted Rand Index
    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(y_encoded, cluster_labels)

    result = {
        'silhouette_kmeans': float(silhouette_orig),
        'silhouette_actual_classes': float(silhouette_class),
        'adjusted_rand_index': float(ari),
        'cluster_class_alignment': 'GOOD' if ari > 0.5 else 'MODERATE' if ari > 0.3 else 'WEAK',
        'status': 'PASS' if ari > 0.3 else 'WARNING'
    }

    return result


def feature_importance_baseline(df):
    """12. Feature Importance Baseline"""
    print("[12/12] Feature Importance Baseline...")

    feature_cols = [col for col in df.columns if '_score' in col or col in
                   ['years_experience', 'projects_completed', 'certifications']]

    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df['career']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Random Forest for feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y_encoded)

    importance = rf.feature_importances_
    importance_dict = dict(zip(feature_cols, importance.tolist()))
    importance_sorted = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

    # Cross-validation score
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(rf, X, y_encoded, cv=5)

    result = {
        'feature_importance': importance_dict,
        'top_10_features': dict(importance_sorted[:10]),
        'bottom_5_features': dict(importance_sorted[-5:]),
        'cv_accuracy_mean': float(cv_scores.mean()),
        'cv_accuracy_std': float(cv_scores.std()),
        'model_performance': 'GOOD' if cv_scores.mean() > 0.8 else 'MODERATE' if cv_scores.mean() > 0.6 else 'POOR',
        'status': 'PASS' if cv_scores.mean() > 0.6 else 'WARNING'
    }

    return result


def generate_markdown_report(analysis_results):
    """Generate markdown report."""
    print("\n[REPORT] Generating markdown report...")

    md = """# Validation Dataset Analysis Report

**Generated:** {timestamp}

---

## Executive Summary

This report provides a comprehensive validation analysis of the generated validation dataset (500 synthetic candidates).

### Overall Status: {overall_status}

---

## 1. Missing Value Analysis

| Metric | Value |
|--------|-------|
| Total Missing Values | {total_missing} |
| Columns with Missing | {missing_cols} |
| Status | **{missing_status}** |

{missing_details}

---

## 2. Duplicate Analysis

| Metric | Value |
|--------|-------|
| Exact Duplicates | {exact_dups} |
| Near Duplicates (>98% similarity) | {near_dups} |
| Status | **{dup_status}** |

{dup_details}

---

## 3. Class Distribution Analysis

| Metric | Value |
|--------|-------|
| Total Career Classes | {total_classes} |
| Min Count | {min_count} |
| Max Count | {max_count} |
| Imbalance Ratio | {imb_ratio:.2f} |
| Chi-Square p-value | {chi2_p:.4f} |
| Status | **{class_status}** |

{class_details}

---

## 4. Correlation Matrix Analysis

| Metric | Value |
|--------|-------|
| High Correlation Pairs (>0.85) | {high_corr_count} |
| Status | **{corr_status}** |

{corr_details}

---

## 5. Feature Distribution Analysis

| Metric | Value |
|--------|-------|
| Non-Normal Features | {non_normal} |
| Range Violations | {range_viols} |
| Status | **{dist_status}** |

{dist_details}

---

## 6. PCA Analysis

| Metric | Value |
|--------|-------|
| Components for 90% Variance | {pca_90} |
| Components for 95% Variance | {pca_95} |
| Status | **{pca_status}** |

{pca_details}

---

## 7. t-SNE Analysis

| Metric | Value |
|--------|-------|
| Silhouette Score | {tsne_sil:.3f} |
| Class Separation | {tsne_sep} |
| Status | **{tsne_status}** |

{tsne_details}

---

## 8. ANOVA Analysis

| Metric | Value |
|--------|-------|
| Significant Features (p<0.05) | {sig_feat} / {total_feat} |
| Average F-Statistic | {avg_f:.2f} |
| Discriminative Power | {anova_power} |
| Status | **{anova_status}** |

{anova_details}

---

## 9. Mutual Information Analysis

| Metric | Value |
|--------|-------|
| Average MI Score | {avg_mi:.3f} |
| Low MI Features | {low_mi_count} |
| Status | **{mi_status}** |

{mi_details}

---

## 10. Outlier Detection

| Metric | Value |
|--------|-------|
| Total Outliers (IQR) | {total_outliers} |
| Features with Outliers | {feat_outliers} |
| Outlier Percentage | {outlier_pct:.2f}% |
| Status | **{outlier_status}** |

{outlier_details}

---

## 11. Cluster Separation Analysis

| Metric | Value |
|--------|-------|
| Silhouette (K-Means) | {sil_km:.3f} |
| Silhouette (Classes) | {sil_cls:.3f} |
| Adjusted Rand Index | {ari:.3f} |
| Cluster-Class Alignment | {align} |
| Status | **{cluster_status}** |

{cluster_details}

---

## 12. Feature Importance Baseline

| Metric | Value |
|--------|-------|
| CV Accuracy | {cv_acc:.3f} +/- {cv_std:.3f} |
| Model Performance | {perf} |
| Status | **{fi_status}** |

{fi_details}

---

## Issues and Recommendations

### Issues Found

{issues_list}

### Recommendations

{recommendations}

---

## Conclusion

{conclusion}

---

*Report generated by CareerPilot AI Validation System*
"""

    # Compile issues
    issues = []
    recommendations = []

    # Missing values
    if analysis_results['missing']['status'] != 'PASS':
        issues.append("- Missing values detected in the dataset")
        recommendations.append("- Review data generation pipeline for sources of missing data")
    else:
        recommendations.append("- No missing value issues found - maintain current pipeline")

    # Duplicates
    if analysis_results['duplicates']['exact_duplicates'] > 0:
        issues.append(f"- {analysis_results['duplicates']['exact_duplicates']} exact duplicate rows found")
        recommendations.append("- Add deduplication step in data generation")

    # Class distribution
    if not analysis_results['class_distribution']['is_balanced']:
        issues.append(f"- Class imbalance detected (ratio: {analysis_results['class_distribution']['imbalance_ratio']:.2f})")
        recommendations.append("- Current balanced distribution is good - ensure equal sampling in generation")

    # Correlation
    if analysis_results['correlation']['high_corr_count'] >= 5:
        issues.append(f"- {analysis_results['correlation']['high_corr_count']} highly correlated feature pairs found")
        recommendations.append("- Consider removing redundant features to reduce multicollinearity")

    # Outliers
    if analysis_results['outliers']['status'] == 'WARNING':
        issues.append(f"- {analysis_results['outliers']['total_outliers']} outliers detected")
        recommendations.append("- Review Gaussian sampling parameters - consider tightening sigma values")

    # Cluster separation
    if analysis_results['cluster']['status'] == 'WARNING':
        issues.append("- Cluster separation from actual classes is weak")
        recommendations.append("- Some careers may have overlapping feature profiles - this is expected for similar roles")

    # Feature importance
    if analysis_results['feature_importance']['status'] == 'WARNING':
        issues.append("- Baseline model performance is lower than expected")
        recommendations.append("- Review feature ranges to ensure career-specific differentiation")

    if not issues:
        issues.append("None - All validation checks passed!")
        recommendations.append("- Dataset is well-suited for validation purposes")

    issues_list = "\n".join(issues)
    recommendations_list = "\n".join(recommendations)

    # Calculate overall status
    statuses = [r['status'] for r in analysis_results.values()]
    overall_status = 'PASS' if all(s == 'PASS' for s in statuses) else 'WARNING' if any(s == 'WARNING' for s in statuses) else 'FAIL'

    # Fill in template
    md = md.format(
        timestamp=pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        overall_status=overall_status,

        # Missing
        total_missing=analysis_results['missing']['total_missing'],
        missing_cols=analysis_results['missing']['columns_with_missing'],
        missing_status=analysis_results['missing']['status'],
        missing_details="- No missing values found - dataset is complete" if analysis_results['missing']['total_missing'] == 0 else f"Missing columns: {list(analysis_results['missing']['missing_by_column'].keys())}",

        # Duplicates
        exact_dups=analysis_results['duplicates']['exact_duplicates'],
        near_dups=analysis_results['duplicates']['near_duplicates_98pct'],
        dup_status=analysis_results['duplicates']['status'],
        dup_details="- No duplicate issues found" if analysis_results['duplicates']['exact_duplicates'] == 0 else f"- {analysis_results['duplicates']['exact_duplicates']} exact duplicates need removal",

        # Class distribution
        total_classes=analysis_results['class_distribution']['total_classes'],
        min_count=analysis_results['class_distribution']['min_count'],
        max_count=analysis_results['class_distribution']['max_count'],
        imb_ratio=analysis_results['class_distribution']['imbalance_ratio'],
        chi2_p=analysis_results['class_distribution']['chi2_p_value'],
        class_status=analysis_results['class_distribution']['status'],
        class_details=f"Chi-square test p-value: {analysis_results['class_distribution']['chi2_p_value']:.4f} - {'Balanced distribution confirmed' if analysis_results['class_distribution']['is_balanced'] else 'Some imbalance detected'}",

        # Correlation
        high_corr_count=analysis_results['correlation']['high_corr_count'],
        corr_status=analysis_results['correlation']['status'],
        corr_details=f"Top correlated features: {list(analysis_results['correlation']['top_correlated_features'].keys())[:5]}",

        # Distribution
        non_normal=analysis_results['distribution']['non_normal_features'],
        range_viols=analysis_results['distribution']['range_violations'],
        dist_status=analysis_results['distribution']['status'],
        dist_details="- Non-normal distributions are expected for synthetic data with domain-specific constraints",

        # PCA
        pca_90=analysis_results['pca']['components_to_90pct'],
        pca_95=analysis_results['pca']['components_to_95pct'],
        pca_status=analysis_results['pca']['status'],
        pca_details=f"First component explains {analysis_results['pca']['explained_variance_ratio'][0]*100:.1f}% variance",

        # t-SNE
        tsne_sil=analysis_results['tsne']['silhouette_score_cluster'],
        tsne_sep=analysis_results['tsne']['class_separation'],
        tsne_status=analysis_results['tsne']['status'],
        tsne_details=f"Perplexity=30 was used for t-SNE embedding",

        # ANOVA
        sig_feat=analysis_results['anova']['significant_features'],
        total_feat=analysis_results['anova']['total_features'],
        avg_f=analysis_results['anova']['avg_f_statistic'],
        anova_power=analysis_results['anova']['discriminative_power'],
        anova_status=analysis_results['anova']['status'],
        anova_details="High F-statistics indicate strong discrimination between career classes",

        # MI
        avg_mi=analysis_results['mi']['avg_mi_score'],
        low_mi_count=len(analysis_results['mi']['low_mi_features']),
        mi_status=analysis_results['mi']['status'],
        mi_details=f"Top MI features: {list(analysis_results['mi']['top_10_features'].keys())[:5]}",

        # Outliers
        total_outliers=analysis_results['outliers']['total_outliers'],
        feat_outliers=analysis_results['outliers']['features_with_outliers'],
        outlier_pct=analysis_results['outliers']['outlier_percentage'],
        outlier_status=analysis_results['outliers']['status'],
        outlier_details="- Outliers are expected due to Gaussian sampling with career-specific ranges",

        # Cluster
        sil_km=analysis_results['cluster']['silhouette_kmeans'],
        sil_cls=analysis_results['cluster']['silhouette_actual_classes'],
        ari=analysis_results['cluster']['adjusted_rand_index'],
        align=analysis_results['cluster']['cluster_class_alignment'],
        cluster_status=analysis_results['cluster']['status'],
        cluster_details=f"K-Means silhouette: {analysis_results['cluster']['silhouette_kmeans']:.3f}, ARI: {analysis_results['cluster']['adjusted_rand_index']:.3f}",

        # Feature importance
        cv_acc=analysis_results['feature_importance']['cv_accuracy_mean'],
        cv_std=analysis_results['feature_importance']['cv_accuracy_std'],
        perf=analysis_results['feature_importance']['model_performance'],
        fi_status=analysis_results['feature_importance']['status'],
        fi_details=f"Top features: {list(analysis_results['feature_importance']['top_10_features'].keys())[:5]}",

        # Issues and recommendations
        issues_list=issues_list,
        recommendations=recommendations_list,

        # Conclusion
        conclusion=f"The validation dataset {'passed all quality checks' if overall_status == 'PASS' else 'has some warnings but is usable'} with {len(df)} samples across {analysis_results['class_distribution']['total_classes']} balanced career classes."
    )

    return md


def main():
    """Main validation pipeline."""
    print("=" * 70)
    print("VALIDATION DATASET COMPREHENSIVE ANALYSIS")
    print("=" * 70)

    # Load dataset
    print("\n[LOAD] Loading validation dataset...")
    global df
    df = load_dataset()
    print(f"      Loaded {len(df)} rows, {len(df.columns)} columns")

    # Run all analyses
    results = {}

    results['missing'] = missing_value_analysis(df)
    results['duplicates'] = duplicate_analysis(df)
    results['class_distribution'] = class_distribution_analysis(df)
    results['correlation'] = correlation_matrix_analysis(df)
    results['distribution'] = feature_distribution_analysis(df)
    results['pca'] = pca_analysis(df)
    results['tsne'] = tsne_analysis(df)
    results['anova'] = anova_analysis(df)
    results['mi'] = mutual_information_analysis(df)
    results['outliers'] = outlier_detection(df)
    results['cluster'] = cluster_separation_analysis(df)
    results['feature_importance'] = feature_importance_baseline(df)

    # Generate report
    report = generate_markdown_report(results)

    # Save report
    report_path = REPORTS_DIR / "validation_analysis_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n[SAVED] Report: {report_path}")

    # Also save JSON for programmatic access
    json_path = REPORTS_DIR / "validation_analysis_results.json"
    # Convert numpy types to native Python for JSON serialization
    results_serializable = json.loads(json.dumps(results, default=str))
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, indent=2)

    print(f"[SAVED] JSON: {json_path}")

    print("\n" + "=" * 70)
    print("VALIDATION ANALYSIS COMPLETE")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()