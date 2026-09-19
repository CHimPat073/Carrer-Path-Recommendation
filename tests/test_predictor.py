"""Tests for the Prediction Engine (`ml.inference.predictor`).

Covered scenarios:

* valid prediction – returns the expected output schema.
* missing feature – raises ``MissingFieldError`` listing the gaps.
* invalid category – raises ``UnknownCategoryError``.
* invalid range – raises ``OutOfRangeError`` (numeric out of bounds).
* top-3 ordering – probabilities are sorted in descending order.
"""

from __future__ import annotations

import copy
import unittest

from ml.inference.predictor import (
    CATEGORICAL_VALUES,
    NUMERIC_RANGES,
    PREPROCESSOR_COLUMNS,
    InvalidValueError,
    MissingFieldError,
    OutOfRangeError,
    Predictor,
    UnknownCategoryError,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def _valid_payload() -> dict:
    """Build a fully-valid input record (one of each preprocessor column)."""
    payload: dict = {
        # base numeric
        "years_experience": 3,
        "projects_completed": 5,
        "certifications": 2,
        # skill scores (0–10)
        "python_score": 9,
        "java_score": 4,
        "javascript_score": 3,
        "sql_score": 7,
        "machine_learning_score": 8,
        "deep_learning_score": 6,
        "cloud_score": 5,
        "devops_score": 3,
        "cybersecurity_score": 2,
        "data_analysis_score": 8,
        "database_score": 6,
        "networking_score": 3,
        "mobile_score": 1,
        "game_dev_score": 0,
        "testing_score": 4,
        "business_analysis_score": 3,
        "product_management_score": 2,
        "ui_design_score": 1,
        "ux_research_score": 1,
        "communication_score": 6,
        "leadership_score": 4,
        "problem_solving_score": 8,
        "teamwork_score": 7,
        "agile_score": 5,
        "research_score": 6,
        # extra numeric (not skill score)
        "salary_band": 70_000.0,
        "career_growth_score": 7,
        "job_satisfaction": 8,
        "work_hours_per_week": 40,
        # categorical
        "education_level": "Bachelor",
        "remote_preference": "Hybrid",
        "country": "USA",
        "industry": "Technology",
        "employment_type": "Full-time",
    }

    # Sanity check – every required column must be present.
    missing = [c for c in PREPROCESSOR_COLUMNS if c not in payload]
    assert not missing, f"Test fixture incomplete, missing: {missing}"

    return payload


class PredictorBaseTests(unittest.TestCase):
    """Shared setup for cases that need an initialised predictor."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.predictor = Predictor()
        cls.valid = _valid_payload()


# ---------------------------------------------------------------------------
# 1. Valid prediction
# ---------------------------------------------------------------------------

class ValidPredictionTests(PredictorBaseTests):
    def test_predict_returns_expected_schema(self) -> None:
        result = self.predictor.predict(self.valid)

        self.assertIsInstance(result, dict)
        self.assertIn("prediction", result)
        self.assertIn("top3", result)

        self.assertIsInstance(result["prediction"], str)
        self.assertGreater(len(result["prediction"]), 0)

        self.assertIsInstance(result["top3"], list)
        self.assertGreater(len(result["top3"]), 0)
        self.assertLessEqual(len(result["top3"]), 3)

        for entry in result["top3"]:
            self.assertIsInstance(entry, dict)
            self.assertIn("career", entry)
            self.assertIn("probability", entry)
            self.assertIsInstance(entry["career"], str)
            self.assertIsInstance(entry["probability"], float)
            # Probability rounded to 4 decimal places.
            self.assertEqual(
                round(entry["probability"], 4), entry["probability"]
            )
            self.assertGreaterEqual(entry["probability"], 0.0)
            self.assertLessEqual(entry["probability"], 1.0)

    def test_top_prediction_matches_top3_leader(self) -> None:
        result = self.predictor.predict(self.valid)
        self.assertEqual(result["prediction"], result["top3"][0]["career"])


# ---------------------------------------------------------------------------
# 2. Missing feature
# ---------------------------------------------------------------------------

class MissingFieldTests(PredictorBaseTests):
    def test_missing_single_field_raises(self) -> None:
        payload = copy.deepcopy(self.valid)
        payload.pop("python_score")

        with self.assertRaises(MissingFieldError) as ctx:
            self.predictor.predict(payload)

        self.assertIn("python_score", ctx.exception.missing)
        self.assertEqual(ctx.exception.missing, ["python_score"])

    def test_missing_multiple_fields_raises_with_all(self) -> None:
        payload = copy.deepcopy(self.valid)
        payload.pop("python_score")
        payload.pop("education_level")
        payload.pop("country")

        with self.assertRaises(MissingFieldError) as ctx:
            self.predictor.predict(payload)

        missing = sorted(ctx.exception.missing)
        self.assertEqual(missing, sorted(["python_score", "education_level", "country"]))

    def test_empty_dict_raises(self) -> None:
        with self.assertRaises(MissingFieldError):
            self.predictor.predict({})


# ---------------------------------------------------------------------------
# 3. Invalid category
# ---------------------------------------------------------------------------

class InvalidCategoryTests(PredictorBaseTests):
    def test_unknown_education_raises(self) -> None:
        payload = copy.deepcopy(self.valid)
        payload["education_level"] = "Wizard"

        with self.assertRaises(UnknownCategoryError) as ctx:
            self.predictor.predict(payload)
        self.assertEqual(ctx.exception.field, "education_level")

    def test_unknown_country_raises(self) -> None:
        payload = copy.deepcopy(self.valid)
        payload["country"] = "Atlantis"

        with self.assertRaises(UnknownCategoryError) as ctx:
            self.predictor.predict(payload)
        self.assertEqual(ctx.exception.field, "country")

    def test_non_string_category_raises(self) -> None:
        payload = copy.deepcopy(self.valid)
        payload["remote_preference"] = 42  # numeric where a string is expected

        with self.assertRaises(UnknownCategoryError) as ctx:
            self.predictor.predict(payload)
        self.assertEqual(ctx.exception.field, "remote_preference")

    def test_unknown_category_message_lists_allowed_values(self) -> None:
        payload = copy.deepcopy(self.valid)
        payload["employment_type"] = "Internship"  # not in the valid set

        with self.assertRaises(UnknownCategoryError) as ctx:
            self.predictor.predict(payload)
        message = str(ctx.exception)
        for allowed in CATEGORICAL_VALUES["employment_type"]:
            self.assertIn(allowed, message)


# ---------------------------------------------------------------------------
# 4. Invalid range
# ---------------------------------------------------------------------------

class InvalidRangeTests(PredictorBaseTests):
    def test_skill_score_above_maximum(self) -> None:
        payload = copy.deepcopy(self.valid)
        payload["python_score"] = 11  # skill scores are 0–10

        with self.assertRaises(OutOfRangeError) as ctx:
            self.predictor.predict(payload)
        self.assertEqual(ctx.exception.field, "python_score")
        self.assertIn("above", str(ctx.exception))

    def test_skill_score_below_minimum(self) -> None:
        payload = copy.deepcopy(self.valid)
        payload["data_analysis_score"] = -1

        with self.assertRaises(OutOfRangeError) as ctx:
            self.predictor.predict(payload)
        self.assertEqual(ctx.exception.field, "data_analysis_score")
        self.assertIn("below", str(ctx.exception))

    def test_negative_years_experience(self) -> None:
        payload = copy.deepcopy(self.valid)
        payload["years_experience"] = -2.0

        with self.assertRaises(OutOfRangeError) as ctx:
            self.predictor.predict(payload)
        self.assertEqual(ctx.exception.field, "years_experience")

    def test_non_numeric_value_in_numeric_column(self) -> None:
        payload = copy.deepcopy(self.valid)
        payload["projects_completed"] = "lots"

        with self.assertRaises(InvalidValueError):
            self.predictor.predict(payload)


# ---------------------------------------------------------------------------
# 5. Top-3 ordering
# ---------------------------------------------------------------------------

class TopOrderingTests(PredictorBaseTests):
    def test_top3_is_sorted_descending(self) -> None:
        result = self.predictor.predict(self.valid)
        probs = [entry["probability"] for entry in result["top3"]]
        self.assertEqual(probs, sorted(probs, reverse=True))

    def test_top3_contains_at_most_three_unique_careers(self) -> None:
        result = self.predictor.predict(self.valid)
        careers = [entry["career"] for entry in result["top3"]]
        self.assertLessEqual(len(careers), 3)
        # All entries must have distinct careers (model produces a
        # probability vector – duplicates cannot occur by construction).
        self.assertEqual(len(set(careers)), len(careers))

    def test_top3_always_includes_prediction(self) -> None:
        result = self.predictor.predict(self.valid)
        top_careers = {entry["career"] for entry in result["top3"]}
        self.assertIn(result["prediction"], top_careers)


# ---------------------------------------------------------------------------
# 6. Determinism / reusability
# ---------------------------------------------------------------------------

class ReusabilityTests(PredictorBaseTests):
    def test_predict_is_deterministic_for_same_input(self) -> None:
        first = self.predictor.predict(self.valid)
        second = self.predictor.predict(copy.deepcopy(self.valid))
        self.assertEqual(first, second)

    def test_predict_does_not_mutate_input(self) -> None:
        before = copy.deepcopy(self.valid)
        _ = self.predictor.predict(self.valid)
        self.assertEqual(self.valid, before)


# ---------------------------------------------------------------------------
# 7. Edge case – no input at all
# ---------------------------------------------------------------------------

class NonDictInputTests(unittest.TestCase):
    def test_non_dict_input_raises(self) -> None:
        predictor = Predictor()
        with self.assertRaises(InvalidValueError):
            predictor.predict("not a dict")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Misc – light schema sanity checks (so the test module is self-contained)
# ---------------------------------------------------------------------------

class SchemaSanityTests(unittest.TestCase):
    def test_numeric_ranges_have_lower_bound(self) -> None:
        for column, (lower, _upper) in NUMERIC_RANGES.items():
            self.assertIsNotNone(lower, f"{column} should have a lower bound")

    def test_categorical_values_are_non_empty(self) -> None:
        for column, values in CATEGORICAL_VALUES.items():
            self.assertGreater(len(values), 0)
            for value in values:
                self.assertIsInstance(value, str)


if __name__ == "__main__":
    unittest.main()
