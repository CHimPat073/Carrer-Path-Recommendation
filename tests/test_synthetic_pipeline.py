import unittest

from ml.preprocessing.production_synthetic_pipeline import DatasetValidator, SyntheticProfileGenerator


class SyntheticPipelineTests(unittest.TestCase):
    def test_generator_and_validator_produce_realistic_rows(self) -> None:
        generator = SyntheticProfileGenerator(rows=120, seed=7)
        rows, report = generator.generate()

        self.assertEqual(len(rows), 120)
        self.assertEqual(report["rows_generated"], 120)
        self.assertTrue(all(row.get("career") for row in rows))
        self.assertTrue(all(row.get("education_level") for row in rows))
        self.assertTrue(all(row.get("remote_preference") for row in rows))
        self.assertTrue(all(row.get("country") for row in rows))
        self.assertTrue(all(row.get("industry") for row in rows))
        self.assertTrue(all(row.get("employment_type") for row in rows))
        self.assertTrue(all(1 <= row.get("python_score", 0) <= 10 for row in rows))

        validator = DatasetValidator()
        validation_result = validator.validate_rows(rows)

        self.assertIn("quality_score", validation_result)
        self.assertGreaterEqual(validation_result["quality_score"], 0)
        self.assertLessEqual(validation_result["quality_score"], 100)
        self.assertIn("career_validation", validation_result)


if __name__ == "__main__":
    unittest.main()
    