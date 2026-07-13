import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from ml.preprocessing.cleaning import clean_missing_values, merge_rows, save_dataset, standardize_row
from ml.preprocessing.data_loader import iter_raw_datasets, load_dataset


def run_preprocessing(raw_dir: str | Path | None = None, output_path: str | Path | None = None) -> Path:
    raw_dir = Path(raw_dir) if raw_dir else project_root / "datasets" / "raw"
    output_path = Path(output_path) if output_path else project_root / "datasets" / "processed" / "final_dataset.csv"

    dataset_files = iter_raw_datasets(raw_dir)
    if not dataset_files:
        raise FileNotFoundError(f"No supported datasets found in {raw_dir}")

    all_rows: list[dict[str, object]] = []
    for file_path in dataset_files:
        rows = load_dataset(file_path)
        standardized_rows = [standardize_row(row, file_path.name) for row in rows]
        all_rows.extend(standardized_rows)

    cleaned_rows = clean_missing_values(all_rows)
    merged_rows = merge_rows(cleaned_rows)

    save_dataset(merged_rows, output_path)
    return output_path


if __name__ == "__main__":
    output_file = run_preprocessing()
    print(f"Processed dataset saved to {output_file}")
