"""
SHAP Explainability Module
Provides model-agnostic explanations using SHAP values.
"""

import json
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = PROJECT_ROOT / "ml" / "models"

# Try to import SHAP
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Feature display names for better readability
FEATURE_DISPLAY_NAMES: Final[dict[str, str]] = {
    "years_experience": "Years of Experience",
    "projects_completed": "Projects Completed",
    "certifications": "Certifications",
    "education_level_encoded": "Education Level",
    "python_score": "Python",
    "java_score": "Java",
    "javascript_score": "JavaScript",
    "sql_score": "SQL",
    "machine_learning_score": "Machine Learning",
    "deep_learning_score": "Deep Learning",
    "cloud_score": "Cloud Computing",
    "devops_score": "DevOps",
    "cybersecurity_score": "Cybersecurity",
    "data_analysis_score": "Data Analysis",
    "database_score": "Database",
    "networking_score": "Networking",
    "mobile_score": "Mobile Development",
    "game_dev_score": "Game Development",
    "testing_score": "Testing/QA",
    "business_analysis_score": "Business Analysis",
    "product_management_score": "Product Management",
    "ui_design_score": "UI Design",
    "ux_research_score": "UX Research",
    "communication_score": "Communication",
    "leadership_score": "Leadership",
    "problem_solving_score": "Problem Solving",
    "teamwork_score": "Teamwork",
    "agile_score": "Agile/Scrum",
    "research_score": "Research",
}


class ModelExplainer:
    """SHAP-based model explainer for career recommendations."""

    def __init__(self, model: Any, feature_names: list[str], model_type: str = "tree"):
        self.model = model
        self.feature_names = feature_names
        self.model_type = model_type
        self.explainer: Any = None
        self.shap_values: Any = None

        if not SHAP_AVAILABLE:
            print("SHAP not installed. Install with: pip install shap")
            return

        self._init_explainer()

    def _init_explainer(self):
        """Initialize SHAP explainer based on model type."""
        try:
            if self.model_type == "xgboost" or self.model_type == "catboost":
                # Tree-based models
                self.explainer = shap.TreeExplainer(self.model)
            elif self.model_type == "random_forest":
                self.explainer = shap.TreeExplainer(self.model)
            else:
                # Fallback to KernelExplainer
                self.explainer = shap.KernelExplainer(self.model.predict_proba)

            print(f"Initialized {self.model_type} SHAP explainer")
        except Exception as e:
            print(f"Failed to initialize explainer: {e}")
            self.explainer = None

    def explain_instance(self, instance: np.ndarray) -> dict[str, float]:
        """Explain a single prediction instance."""
        if self.explainer is None:
            return self._fallback_explain(instance)

        try:
            if hasattr(self.explainer, "shap_values"):
                # For TreeExplainer
                shap_vals = self.explainer.shap_values(instance)
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[0]
            else:
                # For KernelExplainer
                shap_vals = self.explainer.shap_values(instance)

            # Convert to dict
            feature_importance = {}
            for i, fname in enumerate(self.feature_names):
                display_name = FEATURE_DISPLAY_NAMES.get(fname, fname)
                feature_importance[display_name] = float(abs(shap_vals[0][i]) if len(shap_vals.shape) > 1 else abs(shap_vals[i]))

            # Sort by importance
            feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

            return feature_importance

        except Exception as e:
            print(f"SHAP explanation failed: {e}")
            return self._fallback_explain(instance)

    def explain_prediction(self, X: np.ndarray, class_names: list[str] | None = None) -> dict[str, Any]:
        """Explain predictions for multiple instances."""
        if self.explainer is None:
            return {"error": "SHAP not available"}

        try:
            # Calculate SHAP values
            if hasattr(self.explainer, "shap_values"):
                self.shap_values = self.explainer.shap_values(X)
            else:
                self.shap_values = self.explainer.shap_values(X)

            # Feature importance (mean absolute SHAP values)
            feature_importance = {}
            for i, fname in enumerate(self.feature_names):
                display_name = FEATURE_DISPLAY_NAMES.get(fname, fname)
                if isinstance(self.shap_values, list):
                    feature_importance[display_name] = float(np.mean(np.abs(self.shap_values[0][:, i])))
                else:
                    feature_importance[display_name] = float(np.mean(np.abs(self.shap_values[:, i])))

            # Sort by importance
            feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

            # Class-level importance
            class_importance = {}
            if class_names and isinstance(self.shap_values, list):
                for idx, class_name in enumerate(class_names):
                    class_importance[class_name] = float(np.mean(np.abs(self.shap_values[idx])))

            return {
                "feature_importance": feature_importance,
                "class_importance": class_importance,
                "shap_values_shape": self.shap_values.shape if hasattr(self.shap_values, "shape") else None,
            }

        except Exception as e:
            return {"error": str(e)}

    def get_top_features(self, n: int = 10) -> list[tuple[str, float]]:
        """Get top N most important features."""
        if self.shap_values is None:
            return []

        # Calculate mean absolute SHAP values
        if isinstance(self.shap_values, list):
            mean_abs = np.mean(np.abs(self.shap_values[0]), axis=0)
        else:
            mean_abs = np.mean(np.abs(self.shap_values), axis=0)

        # Get top features
        top_indices = np.argsort(mean_abs)[-n:][::-1]

        top_features = []
        for idx in top_indices:
            fname = self.feature_names[idx]
            display_name = FEATURE_DISPLAY_NAMES.get(fname, fname)
            top_features.append((display_name, float(mean_abs[idx])))

        return top_features

    def _fallback_explain(self, instance: np.ndarray) -> dict[str, float]:
        """Fallback explanation using model feature importances."""
        feature_importance = {}

        # Try to get feature importances from model
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            for i, fname in enumerate(self.feature_names):
                display_name = FEATURE_DISPLAY_NAMES.get(fname, fname)
                feature_importance[display_name] = float(importances[i])

        # Sort by importance
        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

        return feature_importance

    def get_career_explanation(
        self,
        user_profile: dict[str, float],
        predicted_career: str,
        all_careers: list[str],
        top_n: int = 5,
    ) -> dict[str, Any]:
        """Generate detailed explanation for a career prediction."""
        # Convert profile to feature vector
        feature_vector = np.array([[user_profile.get(fname, 0) for fname in self.feature_names]])

        # Get prediction explanation
        explanation = self.explain_instance(feature_vector)

        # Get top contributing factors
        top_factors = list(explanation.items())[:top_n]

        # Generate recommendations
        recommendations = self._generate_recommendations(user_profile, predicted_career)

        return {
            "predicted_career": predicted_career,
            "top_contributing_features": top_factors,
            "explanation": explanation,
            "recommendations": recommendations,
            "model_type": self.model_type,
        }

    def _generate_recommendations(self, user_profile: dict[str, float], target_career: str) -> list[str]:
        """Generate skill improvement recommendations."""
        recommendations = []

        # Get career requirements (would load from knowledge base in production)
        career_requirements = self._get_career_requirements(target_career)

        # Compare user skills with requirements
        for skill_key, required_level in career_requirements.items():
            user_level = user_profile.get(skill_key, 0)
            display_name = FEATURE_DISPLAY_NAMES.get(skill_key, skill_key)

            if user_level < required_level - 2:
                recommendations.append(
                    f"Increase {display_name} skills (currently {user_level:.1f}/10, aim for {required_level:.1f}/10)"
                )
            elif user_level >= required_level:
                recommendations.append(f"{display_name} skills are strong at {user_level:.1f}/10")

        return recommendations[:5]

    def _get_career_requirements(self, career: str) -> dict[str, float]:
        """Get skill requirements for a career."""
        # Default requirements (would load from knowledge base)
        defaults = {
            "Software Engineer": {"python_score": 8, "java_score": 7, "problem_solving_score": 8},
            "Data Scientist": {"python_score": 9, "machine_learning_score": 9, "data_analysis_score": 8},
            "ML Engineer": {"python_score": 9, "machine_learning_score": 9, "deep_learning_score": 8},
            "Backend Developer": {"python_score": 8, "java_score": 7, "database_score": 7},
            "Frontend Developer": {"javascript_score": 9, "ui_design_score": 7},
            "Full Stack Developer": {"python_score": 7, "javascript_score": 7, "sql_score": 6},
            "DevOps Engineer": {"cloud_score": 9, "devops_score": 9, "networking_score": 7},
            "Data Engineer": {"sql_score": 9, "python_score": 7, "database_score": 8},
        }

        return defaults.get(career, {skill: 5.0 for skill in FEATURE_DISPLAY_NAMES.keys()})

    def save_explanation(self, output_path: Path | None = None) -> Path:
        """Save explanation results to file."""
        if output_path is None:
            output_path = MODELS_DIR / "explanation_results.json"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "model_type": self.model_type,
            "top_features": self.get_top_features(),
            "feature_names": self.feature_names,
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Explanation saved to: {output_path}")
        return output_path


def create_explainer(model: Any, model_type: str = "random_forest") -> ModelExplainer:
    """Create a SHAP explainer for a trained model."""
    from .data_loader import get_feature_names

    feature_names = get_feature_names()
    return ModelExplainer(model, feature_names, model_type)


if __name__ == "__main__":
    # Example usage
    print("SHAP Explainability Module")
    print("=" * 40)
    print(f"SHAP available: {SHAP_AVAILABLE}")
    print(f"Features: {len(FEATURE_DISPLAY_NAMES)}")