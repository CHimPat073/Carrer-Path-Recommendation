"""Tests for the SHAP Explainability Engine."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ml.inference.shap_engine import (
    InvalidInputSchemaError,
    SHAPEngine,
)
from tests.test_predictor import _valid_payload


class SHAPEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine = SHAPEngine()

    def test_model_loading(self) -> None:
        self.assertIsNotNone(self.engine._model)
        self.assertIsNotNone(self.engine._preprocessor)
        self.assertIsNotNone(self.engine._explainer)

    def test_global_importance(self) -> None:
        result = self.engine.global_feature_importance()
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertIn("feature", result[0])
        self.assertIn("mean_abs_shap_value", result[0])
        self.assertGreaterEqual(result[0]["mean_abs_shap_value"], 0.0)

    def test_local_explanation(self) -> None:
        result = self.engine.explain(_valid_payload())
        self.assertEqual(result["prediction"], "ML Engineer")
        self.assertIn("top_positive_features", result)
        self.assertIn("top_negative_features", result)
        self.assertGreater(len(result["top_positive_features"]), 0)
        self.assertGreater(len(result["top_negative_features"]), 0)
        self.assertIn("feature", result["top_positive_features"][0])
        self.assertIn("shap_value", result["top_positive_features"][0])

    def test_invalid_input_raises(self) -> None:
        with self.assertRaises(InvalidInputSchemaError):
            self.engine.explain({"prediction": "ML Engineer"})

    def test_plots_are_saved(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            engine = SHAPEngine(plots_dir=temp_dir)
            engine.explain(_valid_payload())
            self.assertTrue(Path(temp_dir, "shap_summary.png").is_file())
            self.assertTrue(Path(temp_dir, "shap_bar.png").is_file())
            self.assertTrue(Path(temp_dir, "shap_waterfall.png").is_file())
            self.assertTrue(Path(temp_dir, "shap_beeswarm.png").is_file())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
