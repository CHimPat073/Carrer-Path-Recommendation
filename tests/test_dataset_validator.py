import tempfile
import unittest
from pathlib import Path

from dataset_validator import validate_dataset


class DatasetValidatorTests(unittest.TestCase):
    def test_validate_dataset_creates_reports_for_sample_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dataset_path = tmp_path / "sample.csv"
            dataset_path.write_text(
                "career,years_experience,python_score,javascript_score,education_level\n"
                "Software Engineer,3,7,6,Bachelor\n"
                "Data Scientist,5,8,3,Bachelor\n"
                "Software Engineer,3,7,6,Bachelor\n",
                encoding="utf-8",
            )

            report = validate_dataset(dataset_path, output_dir=tmp_path)

            self.assertTrue((tmp_path / "dataset_validation_report.html").exists())
            self.assertTrue((tmp_path / "dataset_validation_report.pdf").exists())
            self.assertIn("missing_values", report)
            self.assertIn("career_balance", report)


if __name__ == "__main__":
    unittest.main()
