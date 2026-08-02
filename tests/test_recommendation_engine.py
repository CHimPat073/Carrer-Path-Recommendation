"""Tests for the Recommendation Engine."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from ml.inference.recommendation_engine import (
    MissingConfidenceOutputError,
    MissingKnowledgeBaseEntryError,
    MissingPredictionOutputError,
    MissingSHAPOutputError,
    RecommendationEngine,
    UnknownCareerError,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"


def _valid_recommendation_payloads() -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    prediction_result = {
        "prediction": "ML Engineer",
        "top3": [
            {"career": "ML Engineer", "probability": 0.9978},
            {"career": "Data Scientist", "probability": 0.8621},
            {"career": "Data Engineer", "probability": 0.7412},
        ],
    }
    confidence_result = {
        "predicted_career": "ML Engineer",
        "confidence_score": 99.78,
        "confidence_level": "Very High",
        "prediction_reliability": "Reliable",
        "top3": prediction_result["top3"],
    }
    shap_result = {
        "prediction": "ML Engineer",
        "top_positive_features": [
            {"feature": "python_score", "value": 9, "shap_value": 2.41},
            {"feature": "machine_learning_score", "value": 8, "shap_value": 1.93},
        ],
        "top_negative_features": [
            {"feature": "communication_score", "value": 5, "shap_value": -0.42},
        ],
    }
    user_input = {"years_experience": 4}
    return prediction_result, confidence_result, shap_result, user_input


class RecommendationEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = RecommendationEngine()

    def test_valid_recommendation(self) -> None:
        prediction_result, confidence_result, shap_result, user_input = _valid_recommendation_payloads()

        result = self.engine.recommend(prediction_result, confidence_result, shap_result, user_input)

        self.assertEqual(result["recommended_career"], "ML Engineer")
        self.assertEqual(result["confidence"], 99.78)
        self.assertEqual(result["confidence_level"], "Very High")
        self.assertEqual(result["prediction_reliability"], "Reliable")
        self.assertEqual(result["top3_careers"][0]["career"], "ML Engineer")
        self.assertEqual(result["career_information"]["career_family"], "Data & AI")
        self.assertIn("Builds and deploys machine learning systems", result["career_information"]["description"])
        self.assertIn("Python", result["required_skills"])
        self.assertTrue(result["education_path"])
        self.assertIn("Senior ML Engineer", result["next_roles"])
        self.assertEqual(result["strengths"][0]["feature"], "python_score")
        self.assertIn("Positive SHAP contribution", result["strengths"][0]["reason"])
        self.assertEqual(result["improvement_areas"][0]["feature"], "communication_score")
        self.assertIn("Negative SHAP contribution", result["improvement_areas"][0]["reason"])

    def test_missing_career_raises(self) -> None:
        prediction_result, confidence_result, shap_result, user_input = _valid_recommendation_payloads()
        prediction_result = dict(prediction_result)
        prediction_result["prediction"] = "Astronaut"

        with self.assertRaises(UnknownCareerError):
            self.engine.recommend(prediction_result, confidence_result, shap_result, user_input)

    def test_missing_kb_entry_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_kb = Path(temp_dir)
            for name in ("careers.json", "career_personas.json", "roadmaps.json", "domain_metadata.json"):
                shutil.copy2(KNOWLEDGE_BASE_DIR / name, temp_kb / name)

            domain_metadata_path = temp_kb / "domain_metadata.json"
            domain_metadata = json.loads(domain_metadata_path.read_text(encoding="utf-8"))
            families = domain_metadata["career_families"]["families"]
            families["data_ai"]["careers"] = [career for career in families["data_ai"]["careers"] if career != "ML Engineer"]
            domain_metadata_path.write_text(json.dumps(domain_metadata, indent=2), encoding="utf-8")

            engine = RecommendationEngine(knowledge_base_dir=temp_kb)
            prediction_result, confidence_result, shap_result, user_input = _valid_recommendation_payloads()

            with self.assertRaises(MissingKnowledgeBaseEntryError):
                engine.recommend(prediction_result, confidence_result, shap_result, user_input)

    def test_missing_shap_raises(self) -> None:
        prediction_result, confidence_result, shap_result, user_input = _valid_recommendation_payloads()

        with self.assertRaises(MissingSHAPOutputError):
            self.engine.recommend(prediction_result, confidence_result, None, user_input)

    def test_missing_confidence_raises(self) -> None:
        prediction_result, confidence_result, shap_result, user_input = _valid_recommendation_payloads()

        with self.assertRaises(MissingConfidenceOutputError):
            self.engine.recommend(prediction_result, None, shap_result, user_input)

    def test_missing_prediction_raises(self) -> None:
        _, confidence_result, shap_result, user_input = _valid_recommendation_payloads()

        with self.assertRaises(MissingPredictionOutputError):
            self.engine.recommend(None, confidence_result, shap_result, user_input)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()