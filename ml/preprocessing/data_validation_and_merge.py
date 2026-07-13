from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINAL_DATASET_PATH = PROJECT_ROOT / "datasets" / "processed" / "final_dataset.csv"
SYNTHETIC_DATASET_PATH = PROJECT_ROOT / "datasets" / "synthetic" / "synthetic_dataset.csv"
HYBRID_OUTPUT_PATH = PROJECT_ROOT / "datasets" / "processed" / "hybrid_dataset.csv"
MIN_COMMON_COLUMNS_FOR_HYBRID = 5


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the real and synthetic datasets into pandas DataFrames."""
    final_df = pd.read_csv(FINAL_DATASET_PATH, low_memory=False)
    synthetic_df = pd.read_csv(SYNTHETIC_DATASET_PATH)
    return final_df, synthetic_df


def ensure_target_column(df: pd.DataFrame, target_name: str = "Career") -> pd.DataFrame:
    """Ensure the target column exists and rename common aliases to Career."""
    if target_name in df.columns:
        return df

    candidate_columns = [
        col for col in df.columns if col.lower() in {"career", "target", "label", "career_label", "job_title", "role"}
    ]

    if candidate_columns:
        if "career_label" in df.columns:
            source_column = "career_label"
        elif "career" in df.columns:
            source_column = "career"
        else:
            source_column = candidate_columns[0]
        df = df.rename(columns={source_column: target_name})

    if target_name not in df.columns:
        df[target_name] = "Unknown"

    df[target_name] = df[target_name].fillna("Unknown").astype(str).str.strip()
    return df


def align_columns(final_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align both datasets to the same feature schema using the union of columns."""
    final_df = ensure_target_column(final_df)
    synthetic_df = ensure_target_column(synthetic_df)

    union_columns = sorted(set(final_df.columns) | set(synthetic_df.columns))
    ordered_columns = ["Career"] + [col for col in union_columns if col != "Career"]

    final_df = final_df.reindex(columns=ordered_columns, fill_value=None)
    synthetic_df = synthetic_df.reindex(columns=ordered_columns, fill_value=None)

    for column in ordered_columns:
        if column == "Career":
            continue

        final_is_numeric = pd.api.types.is_numeric_dtype(final_df[column])
        synthetic_is_numeric = pd.api.types.is_numeric_dtype(synthetic_df[column])

        if final_is_numeric or synthetic_is_numeric:
            final_df[column] = pd.to_numeric(final_df[column], errors="coerce").fillna(0)
            synthetic_df[column] = pd.to_numeric(synthetic_df[column], errors="coerce").fillna(0)
        else:
            final_df[column] = final_df[column].fillna("missing").astype(str)
            synthetic_df[column] = synthetic_df[column].fillna("missing").astype(str)

    final_df["Career"] = final_df["Career"].fillna("Unknown")
    synthetic_df["Career"] = synthetic_df["Career"].fillna("Unknown")

    return final_df, synthetic_df


def validate_datasets(final_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> dict:
    """Validate schema, missing values, duplicates, and datatype consistency."""
    final_df = ensure_target_column(final_df)
    synthetic_df = ensure_target_column(synthetic_df)

    final_columns = list(final_df.columns)
    synthetic_columns = list(synthetic_df.columns)

    same_columns = set(final_columns) == set(synthetic_columns)
    same_order = final_columns == synthetic_columns
    mismatched_columns = {
        "final_only": [column for column in final_columns if column not in synthetic_columns],
        "synthetic_only": [column for column in synthetic_columns if column not in final_columns],
    }
    common_columns = sorted(set(final_columns) & set(synthetic_columns))

    common_columns = sorted(set(final_columns) & set(synthetic_columns))

    report = {
        "final_shape": final_df.shape,
        "synthetic_shape": synthetic_df.shape,
        "same_columns": same_columns,
        "same_order": same_order,
        "common_columns": common_columns,
        "mismatched_columns": mismatched_columns,
        "final_columns": final_columns,
        "synthetic_columns": synthetic_columns,
        "final_missing": final_df.isna().sum().to_dict(),
        "synthetic_missing": synthetic_df.isna().sum().to_dict(),
        "final_duplicates": int(final_df.duplicated().sum()),
        "synthetic_duplicates": int(synthetic_df.duplicated().sum()),
        "final_dtypes": final_df.dtypes.astype(str).to_dict(),
        "synthetic_dtypes": synthetic_df.dtypes.astype(str).to_dict(),
        "final_career_distribution": final_df["Career"].value_counts(dropna=False).to_dict(),
        "synthetic_career_distribution": synthetic_df["Career"].value_counts(dropna=False).to_dict(),
    }

    return report


def suggest_fixes(report: dict) -> list[str]:
    """Suggest practical fixes when the datasets are not directly compatible."""
    suggestions: list[str] = []

    if not report["same_columns"]:
        if report["mismatched_columns"]["final_only"]:
            suggestions.append(
                f"The final dataset contains unique columns not present in synthetic data: {report['mismatched_columns']['final_only']}"
            )
        if report["mismatched_columns"]["synthetic_only"]:
            suggestions.append(
                f"The synthetic dataset contains unique columns not present in final data: {report['mismatched_columns']['synthetic_only']}"
            )
        if report["common_columns"]:
            suggestions.append(
                f"The datasets share only these columns: {report['common_columns']}. To create a meaningful hybrid dataset, align the final dataset with the synthetic feature schema or engineer shared features from the real data."
            )
        else:
            suggestions.append(
                "No shared feature columns exist between the real and synthetic datasets. Transform one dataset to the other's schema before merging."
            )

    if not report["same_order"]:
        suggestions.append("Reorder columns so both datasets share the same schema for training.")

    if report["final_missing"] or report["synthetic_missing"]:
        suggestions.append("Fill missing values before training or remove rows with unresolved features.")

    if report["final_duplicates"] or report["synthetic_duplicates"]:
        suggestions.append("Remove duplicate rows to prevent learning bias.")

    return suggestions


def merge_datasets(final_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> pd.DataFrame:
    """Create a hybrid dataset by aligning both sources to the same schema and concatenating them."""
    final_df, synthetic_df = align_columns(final_df, synthetic_df)
    final_df = final_df.copy()
    synthetic_df = synthetic_df.copy()

    final_df["data_source"] = "real"
    synthetic_df["data_source"] = "synthetic"

    union_columns = [col for col in final_df.columns if col in synthetic_df.columns]
    final_df = final_df[union_columns + ["data_source"]]
    synthetic_df = synthetic_df[union_columns + ["data_source"]]

    hybrid_df = pd.concat([final_df, synthetic_df], ignore_index=True)
    hybrid_df = hybrid_df.sample(frac=1, random_state=42).reset_index(drop=True)
    return hybrid_df


def save_hybrid_dataset(hybrid_df: pd.DataFrame, output_path: Path = HYBRID_OUTPUT_PATH) -> Path:
    """Save the merged training dataset to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hybrid_df.to_csv(output_path, index=False)
    return output_path


def build_quality_report(hybrid_df: pd.DataFrame) -> dict:
    """Build a concise data quality report for ML readiness."""
    if hybrid_df.empty:
        return {
            "total_rows": 0,
            "total_columns": 0,
            "missing_values": 0,
            "duplicate_rows": 0,
            "class_balance": {},
            "ready_for_ml_training": False,
        }

    missing_values = int(hybrid_df.isna().sum().sum())
    duplicate_rows = int(hybrid_df.duplicated().sum())
    class_balance = hybrid_df["Career"].value_counts(dropna=False).to_dict() if "Career" in hybrid_df.columns else {}

    return {
        "total_rows": int(hybrid_df.shape[0]),
        "total_columns": int(hybrid_df.shape[1]),
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "class_balance": class_balance,
        "ready_for_ml_training": hybrid_df.shape[0] > 0 and missing_values == 0 and duplicate_rows == 0 and len(class_balance) > 1,
    }


def run_pipeline() -> tuple[pd.DataFrame, dict, list[str]]:
    """Run validation, alignment, merging, saving, and reporting steps."""
    final_df, synthetic_df = load_datasets()
    report = validate_datasets(final_df, synthetic_df)
    suggestions = suggest_fixes(report)

    print("final_df.shape")
    print(final_df.shape)
    print("synthetic_df.shape")
    print(synthetic_df.shape)

    print("\nFinal columns:")
    print(report["final_columns"])

    print("\nSynthetic columns:")
    print(report["synthetic_columns"])

    print("\nMissing values - final_df:")
    print(report["final_missing"])

    print("\nMissing values - synthetic_df:")
    print(report["synthetic_missing"])

    print("\nDuplicate rows:")
    print({"final_df": report["final_duplicates"], "synthetic_df": report["synthetic_duplicates"]})

    print("\nCareer distribution - final_df:")
    print(pd.Series(report["final_career_distribution"]))

    print("\nCareer distribution - synthetic_df:")
    print(pd.Series(report["synthetic_career_distribution"]))

    if suggestions:
        print("\nSuggested fixes:")
        for suggestion in suggestions:
            print(f"- {suggestion}")
    else:
        print("\nNo schema fixes required. Datasets are aligned.")

    hybrid_df = merge_datasets(final_df, synthetic_df)
    save_hybrid_dataset(hybrid_df)

    print("\nHybrid dataset shape:")
    print(hybrid_df.shape)

    quality_report = build_quality_report(hybrid_df)
    print("\nData quality report:")
    for key, value in quality_report.items():
        print(f"{key}: {value}")

    return hybrid_df, quality_report, suggestions


if __name__ == "__main__":
    run_pipeline()
