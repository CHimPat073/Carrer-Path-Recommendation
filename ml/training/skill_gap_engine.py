"""
Skill Gap Analysis Engine
The core brain for career recommendations and skill gap analysis.
Analyzes user profiles, calculates skill gaps, and recommends career paths.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = PROJECT_ROOT / "ml" / "models"
KNOWLEDGE_BASE = PROJECT_ROOT / "knowledge_base"

# Skill feature names
SKILL_FEATURES: Final[list[str]] = [
    "python_score",
    "java_score",
    "javascript_score",
    "sql_score",
    "machine_learning_score",
    "deep_learning_score",
    "cloud_score",
    "devops_score",
    "cybersecurity_score",
    "data_analysis_score",
    "database_score",
    "networking_score",
    "mobile_score",
    "game_dev_score",
    "testing_score",
    "business_analysis_score",
    "product_management_score",
    "ui_design_score",
    "ux_research_score",
    "communication_score",
    "leadership_score",
    "problem_solving_score",
    "teamwork_score",
    "agile_score",
    "research_score",
]

# Feature display names
FEATURE_DISPLAY_NAMES: Final[dict[str, str]] = {
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
    "years_experience": "Years of Experience",
    "projects_completed": "Projects Completed",
    "certifications": "Certifications",
}


class SkillGapEngine:
    """
    Core engine for career recommendations and skill gap analysis.
    Analyzes user profiles, identifies skill gaps, and recommends career paths.
    """

    def __init__(self, model_path: Path | None = None):
        self.model = None
        self.label_encoder = None
        self.career_requirements = self._load_career_requirements()
        self.model_loaded = False

        if model_path and model_path.exists():
            self.load_model(model_path)

    def load_model(self, model_path: Path) -> bool:
        """Load trained model from disk."""
        try:
            with open(model_path, "rb") as f:
                model_data = pickle.load(f)

            self.model = model_data.get("model")
            self.label_encoder = model_data.get("label_encoder")
            self.model_loaded = self.model is not None

            print(f"Model loaded from: {model_path}")
            return self.model_loaded

        except Exception as e:
            print(f"Failed to load model: {e}")
            return False

    def _load_career_requirements(self) -> dict[str, dict[str, float]]:
        """Load career skill requirements from knowledge base."""
        try:
            req_path = KNOWLEDGE_BASE / "career_personas.json"
            if req_path.exists():
                with open(req_path, "r") as f:
                    data = json.load(f)

                requirements = {}
                for career_data in data.get("career_personas", []):
                    career = career_data.get("career")
                    if not career:
                        continue

                    # Map skill scores
                    skills = {}
                    for skill in SKILL_FEATURES:
                        key = skill.replace("_score", "")
                        skills[skill] = float(career_data.get(key, 5.0))

                    requirements[career] = skills

                if requirements:
                    return requirements

        except Exception as e:
            print(f"Could not load career requirements: {e}")

        # Return default requirements
        return self._get_default_requirements()

    def _get_default_requirements(self) -> dict[str, dict[str, float]]:
        """Get default career skill requirements."""
        return {
            "Software Engineer": {
                "python_score": 8, "java_score": 7, "javascript_score": 6,
                "sql_score": 6, "problem_solving_score": 8, "teamwork_score": 7,
                "testing_score": 6, "communication_score": 6,
            },
            "Data Scientist": {
                "python_score": 9, "machine_learning_score": 9, "data_analysis_score": 8,
                "sql_score": 7, "deep_learning_score": 7, "research_score": 7,
                "communication_score": 6,
            },
            "ML Engineer": {
                "python_score": 9, "machine_learning_score": 9, "deep_learning_score": 8,
                "cloud_score": 6, "data_analysis_score": 7, "research_score": 7,
            },
            "Backend Developer": {
                "python_score": 8, "java_score": 7, "sql_score": 8,
                "database_score": 7, "problem_solving_score": 7, "api_design_score": 6,
            },
            "Frontend Developer": {
                "javascript_score": 9, "ui_design_score": 7, "ux_research_score": 6,
                "css_score": 7, "communication_score": 6, "teamwork_score": 6,
            },
            "Full Stack Developer": {
                "python_score": 7, "javascript_score": 7, "sql_score": 6,
                "cloud_score": 6, "problem_solving_score": 7, "database_score": 6,
            },
            "DevOps Engineer": {
                "cloud_score": 9, "devops_score": 9, "networking_score": 7,
                "problem_solving_score": 7, "scripting_score": 8, "automation_score": 8,
            },
            "Data Engineer": {
                "sql_score": 9, "python_score": 7, "database_score": 8,
                "cloud_score": 7, "data_analysis_score": 7, "etl_score": 7,
            },
            "AI Research Engineer": {
                "python_score": 9, "machine_learning_score": 9, "deep_learning_score": 9,
                "research_score": 9, "math_score": 8, "publication_score": 6,
            },
            "Cyber Security Analyst": {
                "cybersecurity_score": 9, "networking_score": 8, "problem_solving_score": 7,
                "compliance_score": 6, "incident_response_score": 7,
            },
            "Ethical Hacker": {
                "cybersecurity_score": 9, "penetration_testing_score": 9,
                "networking_score": 8, "scripting_score": 7, "problem_solving_score": 8,
            },
            "Cloud Engineer": {
                "cloud_score": 9, "devops_score": 8, "networking_score": 7,
                "cybersecurity_score": 6, "automation_score": 7,
            },
            "Product Manager": {
                "product_management_score": 9, "communication_score": 8,
                "leadership_score": 7, "business_analysis_score": 7,
                "agile_score": 6, "stakeholder_management_score": 7,
            },
            "Business Analyst": {
                "business_analysis_score": 9, "communication_score": 8,
                "sql_score": 6, "problem_solving_score": 7, "documentation_score": 7,
            },
            "UI/UX Designer": {
                "ui_design_score": 9, "ux_research_score": 8,
                "prototyping_score": 7, "communication_score": 7, "visual_design_score": 8,
            },
            "QA Engineer": {
                "testing_score": 9, "problem_solving_score": 7,
                "automation_score": 6, "communication_score": 6, "agile_score": 6,
            },
            "Database Administrator": {
                "database_score": 9, "sql_score": 9,
                "backup_recovery_score": 8, "performance_tuning_score": 7,
                "security_score": 6,
            },
            "Game Developer": {
                "game_dev_score": 9, "python_score": 6, "c_plus_plus_score": 7,
                "graphics_programming_score": 7, "problem_solving_score": 7,
            },
        }

    def analyze_profile(self, user_profile: dict[str, float]) -> dict[str, Any]:
        """
        Analyze a user profile and return detailed analysis.
        """
        # Ensure all skill features are present
        profile = self._normalize_profile(user_profile)

        # Calculate career match scores
        career_matches = self._calculate_career_matches(profile)

        # Sort by match score
        sorted_careers = sorted(career_matches.items(), key=lambda x: x[1], reverse=True)

        # Get top recommendations
        top_careers = self._get_top_recommendations(profile, sorted_careers)

        # Generate skill gap analysis
        skill_gaps = self._analyze_skill_gaps(profile, top_careers)

        return {
            "user_profile": profile,
            "career_matches": career_matches,
            "top_recommendations": top_careers,
            "skill_gaps": skill_gaps,
            "overall_assessment": self._generate_assessment(profile, top_careers),
        }

    def _normalize_profile(self, profile: dict[str, float]) -> dict[str, float]:
        """Ensure profile has all required features with default values."""
        normalized = {}

        # Base features
        normalized["years_experience"] = float(profile.get("years_experience", 0))
        normalized["projects_completed"] = float(profile.get("projects_completed", 0))
        normalized["certifications"] = float(profile.get("certifications", 0))

        # Skill features (default to 5 if not present)
        for skill in SKILL_FEATURES:
            normalized[skill] = float(profile.get(skill, 5.0))

        return normalized

    def _calculate_career_matches(self, profile: dict[str, float]) -> dict[str, float]:
        """Calculate match scores for all careers."""
        career_scores = {}

        for career, requirements in self.career_requirements.items():
            score = self._calculate_match_score(profile, requirements)
            career_scores[career] = score

        return career_scores

    def _calculate_match_score(
        self,
        profile: dict[str, float],
        requirements: dict[str, float],
    ) -> float:
        """Calculate how well a profile matches career requirements."""
        if not requirements:
            return 0.0

        total_weight = 0.0
        weighted_score = 0.0

        for skill, required_level in requirements.items():
            user_level = profile.get(skill, 5.0)

            # Weight based on required level (higher requirements = more weight)
            weight = required_level / 10.0

            # Score: how close user is to required level (capped at 1.0)
            if user_level >= required_level:
                score = 1.0
            else:
                score = user_level / required_level

            weighted_score += weight * score
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return weighted_score / total_weight * 100.0

    def _get_top_recommendations(
        self,
        profile: dict[str, float],
        sorted_careers: list[tuple[str, float]],
        top_n: int = 5,
    ) -> list[dict[str, Any]]:
        """Get top N career recommendations."""
        recommendations = []

        for career, score in sorted_careers[:top_n]:
            # Calculate time to career readiness
            requirements = self.career_requirements.get(career, {})
            time_to_ready = self._estimate_time_to_readiness(profile, requirements)

            recommendations.append(
                {
                    "career": career,
                    "match_score": round(score, 1),
                    "time_to_ready_months": time_to_ready,
                    "readiness_level": self._get_readiness_level(score),
                }
            )

        return recommendations

    def _analyze_skill_gaps(
        self,
        profile: dict[str, float],
        recommendations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Analyze skill gaps for top recommendations."""
        gaps = {}

        for rec in recommendations:
            career = rec["career"]
            requirements = self.career_requirements.get(career, {})

            skill_gaps = []
            for skill, required in requirements.items():
                user_level = profile.get(skill, 5.0)
                gap = required - user_level

                if gap > 0:
                    skill_gaps.append(
                        {
                            "skill": FEATURE_DISPLAY_NAMES.get(skill, skill),
                            "current": user_level,
                            "required": required,
                            "gap": gap,
                            "priority": "high" if gap > 3 else "medium" if gap > 1 else "low",
                        }
                    )

            # Sort by gap size
            skill_gaps.sort(key=lambda x: x["gap"], reverse=True)

            gaps[career] = {
                "total_gaps": len(skill_gaps),
                "critical_skills": [s for s in skill_gaps if s["priority"] == "high"],
                "all_gaps": skill_gaps,
            }

        return gaps

    def _estimate_time_to_readiness(
        self,
        profile: dict[str, float],
        requirements: dict[str, float],
    ) -> int:
        """Estimate months to reach career readiness."""
        total_gap = 0

        for skill, required in requirements.items():
            user_level = profile.get(skill, 5.0)
            gap = max(0, required - user_level)
            total_gap += gap

        # Rough estimate: 1 month per 2 points of gap
        return max(1, int(total_gap / 2))

    def _get_readiness_level(self, score: float) -> str:
        """Get readiness level description."""
        if score >= 80:
            return "Ready Now"
        elif score >= 60:
            return "Almost Ready"
        elif score >= 40:
            return "In Progress"
        elif score >= 20:
            return "Early Stage"
        else:
            return "Needs Foundation"

    def _generate_assessment(
        self,
        profile: dict[str, float],
        recommendations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Generate overall career assessment."""
        top_career = recommendations[0] if recommendations else None

        # Calculate average skill level
        avg_skill = np.mean([profile.get(s, 5.0) for s in SKILL_FEATURES])

        # Count strong skills (7+)
        strong_skills = sum(1 for s in SKILL_FEATURES if profile.get(s, 5.0) >= 7)

        # Count weak skills (below 5)
        weak_skills = sum(1 for s in SKILL_FEATURES if profile.get(s, 5.0) < 5)

        return {
            "top_career": top_career["career"] if top_career else "Unknown",
            "top_career_score": top_career["match_score"] if top_career else 0,
            "average_skill_level": round(avg_skill, 1),
            "strong_skills_count": strong_skills,
            "weak_skills_count": weak_skills,
            "experience_years": profile.get("years_experience", 0),
            "projects_completed": profile.get("projects_completed", 0),
            "certifications": profile.get("certifications", 0),
        }

    def predict_career(self, user_profile: dict[str, float]) -> dict[str, Any]:
        """
        Predict career recommendation using trained model (if available)
        or rule-based analysis.
        """
        if self.model_loaded and self.model:
            return self._model_predict(user_profile)
        else:
            return self._rule_based_predict(user_profile)

    def _model_predict(self, user_profile: dict[str, float]) -> dict[str, Any]:
        """Use trained ML model for prediction."""
        # Convert profile to feature vector
        from .data_loader import get_feature_names

        feature_names = get_feature_names()
        feature_vector = np.array([[user_profile.get(f, 5.0) for f in feature_names]])

        # Get prediction
        prediction = self.model.predict(feature_vector)[0]
        probabilities = self.model.predict_proba(feature_vector)[0]

        # Get class name
        career = self.label_encoder.inverse_transform([prediction])[0]

        # Get confidence
        confidence = float(probabilities[prediction])

        return {
            "predicted_career": career,
            "confidence": confidence,
            "all_probabilities": {
                self.label_encoder.classes_[i]: float(prob)
                for i, prob in enumerate(probabilities)
            },
        }

    def _rule_based_predict(self, user_profile: dict[str, float]) -> dict[str, Any]:
        """Use rule-based analysis for prediction."""
        analysis = self.analyze_profile(user_profile)
        top = analysis["top_recommendations"][0] if analysis["top_recommendations"] else None

        return {
            "predicted_career": top["career"] if top else "Unknown",
            "confidence": top["match_score"] / 100.0 if top else 0,
            "analysis": analysis,
        }

    def get_learning_path(
        self,
        target_career: str,
        user_profile: dict[str, float],
    ) -> list[dict[str, Any]]:
        """Generate a learning path to reach target career."""
        requirements = self.career_requirements.get(target_career, {})
        profile = self._normalize_profile(user_profile)

        learning_path = []

        # Sort skills by gap size
        skill_gaps = []
        for skill, required in requirements.items():
            user_level = profile.get(skill, 5.0)
            gap = required - user_level

            if gap > 0:
                skill_gaps.append(
                    {
                        "skill": FEATURE_DISPLAY_NAMES.get(skill, skill),
                        "skill_key": skill,
                        "current": user_level,
                        "target": required,
                        "gap": gap,
                    }
                )

        skill_gaps.sort(key=lambda x: x["gap"], reverse=True)

        # Create learning phases
        for i, gap in enumerate(skill_gaps):
            learning_path.append(
                {
                    "phase": i + 1,
                    "skill": gap["skill"],
                    "current_level": gap["current"],
                    "target_level": gap["target"],
                    "improvement_needed": gap["gap"],
                    "estimated_weeks": gap["gap"] * 2,
                    "priority": "high" if gap["gap"] > 3 else "medium",
                }
            )

        return learning_path

    def save_analysis(self, analysis: dict, output_path: Path) -> Path:
        """Save analysis results to JSON."""
        with open(output_path, "w") as f:
            json.dump(analysis, f, indent=2, default=str)

        return output_path


# Convenience function for quick analysis
def quick_analyze(profile: dict[str, float]) -> dict[str, Any]:
    """Quickly analyze a user profile."""
    engine = SkillGapEngine()
    return engine.analyze_profile(profile)


def quick_predict(profile: dict[str, float]) -> dict[str, Any]:
    """Quickly predict best career."""
    engine = SkillGapEngine()
    return engine.predict_career(profile)


if __name__ == "__main__":
    # Example usage
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
    }

    print("Skill Gap Engine Demo")
    print("=" * 50)

    result = quick_analyze(test_profile)

    print(f"\nTop Recommendation: {result['top_recommendations'][0]['career']}")
    print(f"Match Score: {result['top_recommendations'][0]['match_score']}%")
    print(f"Time to Ready: {result['top_recommendations'][0]['time_to_ready_months']} months")

    print("\nSkill Gaps:")
    for gap in result["skill_gaps"][result["top_recommendations"][0]["career"]]["critical_skills"][:5]:
        print(f"  - {gap['skill']}: {gap['current']} -> {gap['required']} (gap: {gap['gap']})")