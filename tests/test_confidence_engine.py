"""Tests for the Confidence Engine (ml.inference.confidence_engine)."""

from __future__ import annotations

import unittest

from ml.inference.confidence_engine import (
    ConfidenceEngine,
    ConfidenceEngineError,
    EmptyPredictionError,
    InvalidProbabilityError,
    InvalidPredictionStructureError,
    ProbabilitiesSumError,
)


class ConfidenceEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = ConfidenceEngine()

    def test_very_high_confidence(self) -> None:
        result = self.engine.evaluate(
            {
                "prediction": "ML Engineer",
                "top3": [
                    {"career": "ML Engineer", "probability": 0.9923},
                    {"career": "Data Scientist", "probability": 0.8734},
                    {"career": "Data Engineer", "probability": 0.7121},
                ],
            }
        )

        self.assertEqual(result["predicted_career"], "ML Engineer")
        self.assertEqual(result["confidence_level"], "Very High")
        self.assertEqual(result["prediction_reliability"], "Reliable")
        self.assertGreaterEqual(result["confidence_score"], 99.0)
        self.assertGreater(result["margin_from_second"], 10.0)
        self.assertGreaterEqual(result["top3"][0]["probability"], result["top3"][1]["probability"])

    def test_low_confidence(self) -> None:
        result = self.engine.evaluate(
            {
                "prediction": "Data Scientist",
                "top3": [
                    {"career": "Data Scientist", "probability": 0.41},
                    {"career": "ML Engineer", "probability": 0.39},
                    {"career": "Data Engineer", "probability": 0.20},
                ],
            }
        )

        self.assertEqual(result["confidence_level"], "Low")
        self.assertEqual(result["prediction_reliability"], "Uncertain")
        self.assertEqual(result["predicted_career"], "Data Scientist")
        self.assertAlmostEqual(result["margin_from_second"], 2.0, places=2)

    def test_equal_probabilities(self) -> None:
        result = self.engine.evaluate(
            {
                "prediction": "Data Engineer",
                "top3": [
                    {"career": "Data Engineer", "probability": 0.3333},
                    {"career": "Data Scientist", "probability": 0.3333},
                    {"career": "ML Engineer", "probability": 0.3334},
                ],
            }
        )

        self.assertEqual(result["confidence_level"], "Low")
        self.assertEqual(result["prediction_reliability"], "Uncertain")
        self.assertAlmostEqual(result["margin_from_second"], 0.0, places=2)
        self.assertGreater(result["entropy"], 1.5)

    def test_invalid_probabilities(self) -> None:
        with self.assertRaises(InvalidProbabilityError):
            self.engine.evaluate(
                {
                    "prediction": "ML Engineer",
                    "top3": [
                        {"career": "ML Engineer", "probability": 1.2},
                        {"career": "Data Scientist", "probability": -0.1},
                    ],
                }
            )

    def test_empty_predictions_raise(self) -> None:
        with self.assertRaises(EmptyPredictionError):
            self.engine.evaluate({"prediction": "ML Engineer", "top3": []})

    def test_non_dict_input_raises(self) -> None:
        with self.assertRaises(InvalidPredictionStructureError):
            self.engine.evaluate([])  # type: ignore[arg-type]

    def test_missing_top3_raises(self) -> None:
        with self.assertRaises(InvalidPredictionStructureError):
            self.engine.evaluate({"prediction": "ML Engineer"})

    def test_probabilities_do_not_need_to_sum_to_one(self) -> None:
        result = self.engine.evaluate(
            {
                "prediction": "ML Engineer",
                "top3": [
                    {"career": "ML Engineer", "probability": 0.5},
                    {"career": "Data Scientist", "probability": 0.3},
                    {"career": "Data Engineer", "probability": 0.1},
                ],
            }
        )

        self.assertEqual(result["predicted_career"], "ML Engineer")
        self.assertEqual(result["confidence_score"], 50.0)
        self.assertEqual(result["confidence_level"], "Low")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
