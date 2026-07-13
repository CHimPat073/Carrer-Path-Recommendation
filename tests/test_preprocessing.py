import tempfile
import unittest
from pathlib import Path

from ml.preprocessing.data_loader import load_dataset


class DataLoaderTests(unittest.TestCase):
    def test_load_dataset_accepts_latin1_encoded_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "sample.csv"
            file_path.write_text("career,skills\nSoftware Engineer,café\n", encoding="latin-1")

            rows = load_dataset(file_path)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["skills"], "café")


if __name__ == "__main__":
    unittest.main()
