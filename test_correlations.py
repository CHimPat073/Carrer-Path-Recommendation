"""
Test script to validate correlation engine implementation.
"""

import sys
from pathlib import Path
from statistics import mean, stdev

PROJECT_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.preprocessing.statistical_model import StatisticalProfileGenerator


def calculate_correlation(rows, key1, key2):
    """Calculate Pearson correlation between two features."""
    values1 = [r.get(key1, 0) for r in rows if key1 in r and key2 in r]
    values2 = [r.get(key2, 0) for r in rows if key1 in r and key2 in r]

    if len(values1) < 2:
        return 0

    # Simple correlation
    n = len(values1)
    sum_x = sum(values1)
    sum_y = sum(values2)
    sum_xy = sum(values1[i] * values2[i] for i in range(n))
    sum_x2 = sum(v ** 2 for v in values1)
    sum_y2 = sum(v ** 2 for v in values2)

    numerator = n * sum_xy - sum_x * sum_y
    denominator = ((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2)) ** 0.5

    if denominator == 0:
        return 0
    return numerator / denominator


def test_correlations():
    """Test that correlations are working as expected."""
    print("=" * 60)
    print("CORRELATION ENGINE VALIDATION")
    print("=" * 60)

    # Generate test data
    generator = StatisticalProfileGenerator(rows=1000, seed=42)
    profiles, _ = generator.generate()

    print(f"\nGenerated {len(profiles)} profiles\n")

    # Test correlations
    correlations = [
        ("python_score", "machine_learning_score", "Python -> ML"),
        ("machine_learning_score", "deep_learning_score", "ML -> DL"),
        ("cloud_score", "devops_score", "Cloud -> DevOps"),
        ("years_experience", "salary_band", "Experience -> Salary"),
        ("years_experience", "leadership_score", "Experience -> Leadership"),
        ("years_experience", "projects_completed", "Experience -> Projects"),
        ("communication_score", "leadership_score", "Communication -> Leadership"),
        ("networking_score", "cybersecurity_score", "Networking -> Cyber Security"),
        ("java_score", "database_score", "Java -> Database"),
        ("javascript_score", "ui_design_score", "JavaScript -> UI Design"),
    ]

    print("Correlation Test Results:")
    print("-" * 50)

    all_passed = True
    for key1, key2, label in correlations:
        corr = calculate_correlation(profiles, key1, key2)

        # Determine if correlation is positive and significant
        passed = corr > 0.05
        status = "[PASS]" if passed else "[FAIL]"

        if not passed:
            all_passed = False

        print(f"{label:35s} r = {corr:+.3f}  {status}")

    print("-" * 50)

    # Test career-specific correlations
    print("\nCareer-Specific Correlation Test:")
    print("-" * 50)

    # Filter by career
    ml_profiles = [p for p in profiles if p.get("career") == "ML Engineer"]
    be_profiles = [p for p in profiles if p.get("career") == "Backend Developer"]
    fe_profiles = [p for p in profiles if p.get("career") == "Frontend Developer"]
    devops_profiles = [p for p in profiles if p.get("career") == "DevOps Engineer"]

    career_tests = [
        (ml_profiles, "python_score", "machine_learning_score", "ML Engineer: Python->ML"),
        (be_profiles, "java_score", "database_score", "Backend: Java->Database"),
        (fe_profiles, "javascript_score", "ui_design_score", "Frontend: JS->UI"),
        (devops_profiles, "cloud_score", "devops_score", "DevOps: Cloud->DevOps"),
    ]

    for career_profiles, key1, key2, label in career_tests:
        if len(career_profiles) < 10:
            print(f"{label:35s} - Insufficient data")
            continue

        corr = calculate_correlation(career_profiles, key1, key2)
        passed = corr > 0.05
        status = "[PASS]" if passed else "[FAIL]"

        if not passed:
            all_passed = False

        print(f"{label:35s} r = {corr:+.3f}  {status}")

    print("-" * 50)

    # Test seniority impact
    print("\nSeniority Impact Test (Leadership):")
    print("-" * 50)

    senior_profiles = [p for p in profiles if p.get("seniority") == "Senior"]
    expert_profiles = [p for p in profiles if p.get("seniority") == "Expert"]
    junior_profiles = [p for p in profiles if p.get("seniority") == "Junior"]

    leadership_tests = [
        (junior_profiles, "Junior"),
        (senior_profiles, "Senior"),
        (expert_profiles, "Expert"),
    ]

    for profs, label in leadership_tests:
        if len(profs) < 10:
            print(f"{label}: Insufficient data")
            continue

        avg_leadership = mean(p.get("leadership_score", 0) for p in profs)
        avg_exp = mean(p.get("years_experience", 0) for p in profs)
        print(f"{label}: Leadership = {avg_leadership:.2f}, Experience = {avg_exp:.1f} yrs")

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL CORRELATION TESTS PASSED")
    else:
        print("SOME CORRELATION TESTS FAILED")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = test_correlations()
    sys.exit(0 if success else 1)