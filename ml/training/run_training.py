#!/usr/bin/env python
"""
Main ML Training Script
Run the complete ML pipeline: train models, add SHAP explainability, and test skill-gap engine.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.training.data_loader import load_training_data, preprocess_data, split_data, get_feature_names
from ml.training.model_trainer import ModelTrainer
from ml.training.explainer import create_explainer
from ml.training.skill_gap_engine import SkillGapEngine, quick_analyze


def main():
    parser = argparse.ArgumentParser(description="ML Training Pipeline for CareerPilot-AI")
    parser.add_argument("--data", type=str, default=None, help="Path to training data CSV")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set ratio")
    parser.add_argument("--models", type=str, default="all", help="Models to train: rf, xgb, catboost, all")
    parser.add_argument("--skip-shap", action="store_true", help="Skip SHAP explainability")
    parser.add_argument("--skip-skill-gap", action="store_true", help="Skip skill-gap engine test")
    parser.add_argument("--sample", type=int, default=None, help="Use sample of data for quick testing")

    args = parser.parse_args()

    print("=" * 70)
    print("CareerPilot-AI ML Training Pipeline")
    print("=" * 70)

    # ============================================================
    # STEP 1: Load and preprocess data
    # ============================================================
    print("\n[STEP 1] Loading data...")
    data_path = Path(args.data) if args.data else None
    df = load_training_data(data_path)

    if args.sample:
        df = df.head(args.sample)
        print(f"  Using sample of {args.sample} rows")

    print(f"  Loaded {len(df)} samples")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Careers: {df['career'].nunique()}")

    # Preprocess
    print("\n[STEP 2] Preprocessing data...")
    X, y, label_encoder = preprocess_data(df)
    feature_names = get_feature_names()

    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Number of classes: {len(label_encoder.classes_)}")
    print(f"  Classes: {list(label_encoder.classes_)}")

    # Split
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=args.test_size)
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    # ============================================================
    # STEP 2: Train Models
    # ============================================================
    print("\n[STEP 3] Training models...")
    trainer = ModelTrainer(X_train, X_test, y_train, y_test, label_encoder)

    # Train Random Forest
    if args.models in ("rf", "all"):
        trainer.train_random_forest(n_estimators=100)

    # Train XGBoost
    if args.models in ("xgb", "all"):
        try:
            trainer.train_xgboost(n_estimators=100)
        except Exception as e:
            print(f"  XGBoost not available: {e}")

    # Train CatBoost
    if args.models in ("catboost", "all"):
        try:
            trainer.train_catboost(iterations=100)
        except Exception as e:
            print(f"  CatBoost not available: {e}")

    # Compare models
    print("\n[STEP 4] Model comparison:")
    comparison = trainer.compare_models()

    # Save best model
    print("\n[STEP 5] Saving best model...")
    trainer.save_best_model()

    # ============================================================
    # STEP 3: SHAP Explainability
    # ============================================================
    if not args.skip_shap:
        print("\n[STEP 6] SHAP Explainability...")
        try:
            best_name, best_model = trainer.get_best_model()
            explainer = create_explainer(best_model, best_name)

            # Explain a sample instance
            sample_instance = X_test.iloc[:1].values
            explanation = explainer.explain_instance(sample_instance)

            print("  Top 10 important features:")
            for i, (feature, importance) in enumerate(list(explanation.items())[:10]):
                print(f"    {i+1}. {feature}: {importance:.4f}")

            explainer.save_explanation()

        except Exception as e:
            print(f"  SHAP explainability failed: {e}")

    # ============================================================
    # STEP 4: Skill Gap Engine Test
    # ============================================================
    if not args.skip_skill_gap:
        print("\n[STEP 7] Testing Skill-Gap Engine...")
        print("-" * 50)

        # Test profile
        test_profile = {
            "years_experience": 3,
            "projects_completed": 8,
            "certifications": 2,
            "python_score": 8,
            "java_score": 5,
            "javascript_score": 6,
            "sql_score": 7,
            "machine_learning_score": 4,
            "deep_learning_score": 3,
            "cloud_score": 5,
            "devops_score": 4,
            "data_analysis_score": 6,
            "problem_solving_score": 7,
            "communication_score": 6,
            "leadership_score": 5,
            "teamwork_score": 7,
        }

        result = quick_analyze(test_profile)

        print("\nUser Profile Analysis:")
        print(f"  Years Experience: {result['overall_assessment']['experience_years']}")
        print(f"  Projects Completed: {result['overall_assessment']['projects_completed']}")
        print(f"  Average Skill Level: {result['overall_assessment']['average_skill_level']}")

        print("\nTop Career Recommendations:")
        for i, rec in enumerate(result["top_recommendations"]):
            print(f"  {i+1}. {rec['career']}")
            print(f"     Match Score: {rec['match_score']}%")
            print(f"     Time to Ready: {rec['time_to_ready_months']} months")
            print(f"     Readiness: {rec['readiness_level']}")

        top_career = result["top_recommendations"][0]["career"]
        gaps = result["skill_gaps"].get(top_career, {})

        print(f"\nSkill Gaps for {top_career}:")
        for gap in gaps.get("critical_skills", [])[:5]:
            print(f"  - {gap['skill']}: {gap['current']}/10 -> {gap['required']}/10 (gap: {gap['gap']})")

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)

    # Return comparison for programmatic use
    return trainer, comparison


if __name__ == "__main__":
    main()