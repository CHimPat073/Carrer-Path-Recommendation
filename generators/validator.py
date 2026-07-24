
"""
CareerPilot AI
Dataset Validator

Validates generated synthetic datasets before ML training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd


@dataclass
class ValidationReport:
    total_rows: int
    total_columns: int
    missing_values: int
    duplicate_rows: int
    class_balance: Dict[str, int]
    ready_for_ml: bool
    quality_score: float


class DatasetValidator:

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe

    def missing_values(self) -> int:
        return int(self.df.isna().sum().sum())

    def duplicate_rows(self) -> int:
        return int(self.df.duplicated().sum())

    def class_balance(self) -> Dict[str, int]:
        if "Career" not in self.df.columns:
            return {}
        return self.df["Career"].value_counts().to_dict()

    def validate_score_ranges(self):
        issues = []

        score_columns = [
            c for c in self.df.columns
            if c.endswith("_score")
        ]

        for col in score_columns:
            invalid = self.df[
                (self.df[col] < 1) |
                (self.df[col] > 10)
            ]

            if not invalid.empty:
                issues.append(
                    f"{col}: {len(invalid)} invalid values"
                )

        return issues

    def quality_score(self) -> float:

        score = 100.0

        score -= self.missing_values() * 0.5
        score -= self.duplicate_rows() * 0.2
        score -= len(self.validate_score_ranges()) * 2

        return max(score, 0)

    def generate_report(self) -> ValidationReport:

        quality = self.quality_score()

        return ValidationReport(
            total_rows=len(self.df),
            total_columns=len(self.df.columns),
            missing_values=self.missing_values(),
            duplicate_rows=self.duplicate_rows(),
            class_balance=self.class_balance(),
            ready_for_ml=quality >= 90,
            quality_score=quality,
        )

    def print_report(self):

        report = self.generate_report()

        print("=" * 60)
        print("CareerPilot AI Dataset Validation Report")
        print("=" * 60)

        print(f"Rows               : {report.total_rows}")
        print(f"Columns            : {report.total_columns}")
        print(f"Missing Values     : {report.missing_values}")
        print(f"Duplicate Rows     : {report.duplicate_rows}")
        print(f"Quality Score      : {report.quality_score:.2f}/100")
        print(f"Ready for ML       : {report.ready_for_ml}")

        print("\nClass Balance")
        for career, count in report.class_balance.items():
            print(f"  {career:<25} {count}")

        print("\nRange Validation")
        issues = self.validate_score_ranges()

        if issues:
            for issue in issues:
                print(" -", issue)
        else:
            print(" All score columns are within [1, 10].")


if __name__ == "__main__":

    # Example usage
    try:
        df = pd.read_csv("datasets/synthetic/careerpilot_v3.csv")

        validator = DatasetValidator(df)

        validator.print_report()

    except FileNotFoundError:
        print("Example dataset not found.")
