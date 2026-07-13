import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.preprocessing.data_loader import load_dataset
from ml.preprocessing.role_normalizer import TARGET_CAREERS


def _normalize_feature_name(name: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")
    aliases = {
        "years_experience": "years_experience",
        "years": "years_experience",
        "projects_completed": "projects_completed",
        "projects": "projects_completed",
        "certifications": "certifications",
        "python": "python_score",
        "python_score": "python_score",
        "java": "java_score",
        "java_score": "java_score",
        "javascript": "javascript_score",
        "javascript_score": "javascript_score",
        "sql": "sql_score",
        "sql_score": "sql_score",
        "machine_learning": "machine_learning_score",
        "machine_learning_score": "machine_learning_score",
        "deep_learning": "deep_learning_score",
        "deep_learning_score": "deep_learning_score",
        "cloud": "cloud_score",
        "cloud_score": "cloud_score",
        "devops": "devops_score",
        "devops_score": "devops_score",
        "cybersecurity": "cybersecurity_score",
        "cybersecurity_score": "cybersecurity_score",
        "data_analysis": "data_analysis_score",
        "data_analysis_score": "data_analysis_score",
        "database": "database_score",
        "database_score": "database_score",
        "networking": "networking_score",
        "networking_score": "networking_score",
        "mobile": "mobile_score",
        "mobile_score": "mobile_score",
        "game_dev": "game_dev_score",
        "game_dev_score": "game_dev_score",
        "testing": "testing_score",
        "testing_score": "testing_score",
        "business_analysis": "business_analysis_score",
        "business_analysis_score": "business_analysis_score",
        "product_management": "product_management_score",
        "product_management_score": "product_management_score",
        "ui_design": "ui_design_score",
        "ui_design_score": "ui_design_score",
        "ux_research": "ux_research_score",
        "ux_research_score": "ux_research_score",
        "communication": "communication_score",
        "communication_score": "communication_score",
        "leadership": "leadership_score",
        "leadership_score": "leadership_score",
        "problem_solving": "problem_solving_score",
        "problem_solving_score": "problem_solving_score",
        "teamwork": "teamwork_score",
        "teamwork_score": "teamwork_score",
        "agile": "agile_score",
        "agile_score": "agile_score",
        "research": "research_score",
        "research_score": "research_score",
    }
    return aliases.get(cleaned, cleaned)


def _load_feature_ranges() -> dict[str, dict[str, Any]]:
    path = PROJECT_ROOT / "knowledge_base" / "feature_ranges.json"
    if not path.exists():
        return {}

    payload = json.loads(path.read_text(encoding="utf-8"))
    ranges: dict[str, dict[str, Any]] = {}
    for entry in payload.get("career_feature_ranges", []):
        career = entry.get("career")
        features = entry.get("features", {})
        if not career:
            continue
        converted: dict[str, Any] = {}
        for feature_name, feature_info in features.items():
            normalized = _normalize_feature_name(feature_name)
            if isinstance(feature_info, dict):
                converted[normalized] = feature_info
        ranges[career] = converted
    return ranges


def _load_career_personas() -> dict[str, dict[str, Any]]:
    path = PROJECT_ROOT / "knowledge_base" / "career_personas.json"
    if not path.exists():
        return {}

    payload = json.loads(path.read_text(encoding="utf-8"))
    personas: dict[str, dict[str, Any]] = {}
    for entry in payload.get("career_personas", []):
        career = entry.get("career")
        if career:
            personas[career] = entry
    return personas


def _education_rank(education_level: Any) -> int:
    if not isinstance(education_level, str):
        return 0
    ranking = {
        "high school": 1,
        "associate": 2,
        "bachelor": 3,
        "master": 4,
        "phd": 5,
    }
    lowered = education_level.strip().lower()
    for key, value in ranking.items():
        if lowered.startswith(key):
            return value
    return 0


def _parse_dataset(dataset_path: Path) -> pd.DataFrame:
    rows = load_dataset(dataset_path)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    for column in frame.columns:
        if column == "career":
            continue
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _check_missing_values(frame: pd.DataFrame) -> dict[str, Any]:
    missing_by_column = frame.isna().sum().to_dict()
    missing_rows = int(frame.isna().any(axis=1).sum())
    return {
        "missing_by_column": {k: int(v) for k, v in missing_by_column.items()},
        "missing_rows": missing_rows,
        "status": "pass" if missing_rows == 0 else "fail",
    }


def _check_duplicate_rows(frame: pd.DataFrame) -> dict[str, Any]:
    duplicate_count = int(frame.duplicated().sum())
    return {
        "duplicate_count": duplicate_count,
        "status": "pass" if duplicate_count == 0 else "fail",
    }


def _check_class_balance(frame: pd.DataFrame) -> dict[str, Any]:
    if "career" not in frame.columns:
        return {"status": "pass", "counts": {}, "message": "No career column available"}

    counts = frame["career"].fillna("Unknown").value_counts().to_dict()
    counts = {k: int(v) for k, v in counts.items()}
    total = max(1, sum(counts.values()))
    balance_ratio = max(counts.values()) / total if counts else 0.0
    return {
        "counts": counts,
        "balance_ratio": round(balance_ratio, 3),
        "status": "pass" if balance_ratio <= 0.75 else "fail",
    }


def _check_career_balance(frame: pd.DataFrame) -> dict[str, Any]:
    if "career" not in frame.columns:
        return {"status": "pass", "counts": {}, "message": "No career column available"}

    counts = frame["career"].fillna("Unknown").value_counts().to_dict()
    expected = max(1, len(frame) // max(1, len(TARGET_CAREERS)))
    deviations = {
        career: int(counts.get(career, 0) - expected)
        for career in TARGET_CAREERS
    }
    return {
        "expected_per_career": expected,
        "counts": {career: int(counts.get(career, 0)) for career in TARGET_CAREERS},
        "deviations": deviations,
        "status": "pass" if all(value >= -max(1, expected // 4) for value in deviations.values()) else "fail",
    }


def _check_feature_ranges(frame: pd.DataFrame) -> dict[str, Any]:
    feature_ranges = _load_feature_ranges()
    issues: list[dict[str, Any]] = []
    for column in frame.columns:
        if column == "career":
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            series = frame[column].dropna()
            if series.empty:
                continue
            expected_limits: dict[str, Any] | None = None
            for career_range in feature_ranges.values():
                if column in career_range:
                    expected_limits = career_range[column]
                    break
            if expected_limits is None:
                continue
            minimum = expected_limits.get("min")
            maximum = expected_limits.get("max")
            if minimum is not None and maximum is not None:
                out_of_range_rows = int(((series < minimum) | (series > maximum)).sum())
                if out_of_range_rows:
                    issues.append({
                        "column": column,
                        "expected_range": [minimum, maximum],
                        "out_of_range_rows": out_of_range_rows,
                    })
    return {
        "issues": issues,
        "status": "pass" if not issues else "fail",
    }


def _check_impossible_combinations(frame: pd.DataFrame) -> dict[str, Any]:
    personas = _load_career_personas()
    issues: list[dict[str, Any]] = []
    for index, row in frame.iterrows():
        career = row.get("career")
        education_level = row.get("education_level")
        if not isinstance(career, str) or not isinstance(education_level, str):
            continue
        persona = personas.get(career, {})
        typical_education = str(persona.get("typical_education", "Bachelor's degree")).lower()
        required_rank = _education_rank(typical_education)
        observed_rank = _education_rank(education_level)
        if required_rank and observed_rank and observed_rank < required_rank:
            issues.append({
                "row_index": int(index),
                "career": career,
                "education_level": education_level,
                "typical_education": persona.get("typical_education"),
            })
    return {
        "issues": issues,
        "status": "pass" if not issues else "fail",
    }


def _check_outliers(frame: pd.DataFrame) -> dict[str, Any]:
    outlier_summary: list[dict[str, Any]] = []
    numeric_columns = [column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]
    for column in numeric_columns:
        series = frame[column].dropna()
        if series.empty:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = series[(series < lower) | (series > upper)]
        if not outliers.empty:
            outlier_summary.append({
                "column": column,
                "count": int(outliers.shape[0]),
                "lower_bound": round(float(lower), 2),
                "upper_bound": round(float(upper), 2),
            })
    return {
        "outliers": outlier_summary,
        "status": "pass" if not outlier_summary else "fail",
    }


def _save_correlations(frame: pd.DataFrame, output_dir: Path) -> str:
    numeric_columns = [column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]
    if len(numeric_columns) < 2:
        return ""

    correlation = frame[numeric_columns].corr(numeric_only=True)
    plt.figure(figsize=(14, 12))
    sns.heatmap(correlation, cmap="coolwarm", annot=False)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    path = output_dir / "correlation_heatmap.png"
    plt.savefig(path, dpi=200)
    plt.close()
    return str(path.name)


def _save_distribution_plot(frame: pd.DataFrame, output_dir: Path) -> str:
    numeric_columns = [column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]
    selected = [column for column in ["years_experience", "python_score", "communication_score", "salary_band"] if column in numeric_columns]
    if not selected:
        selected = numeric_columns[:4]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for axis, column in zip(axes, selected):
        frame[column].dropna().hist(ax=axis, bins=15, color="#4C78A8", edgecolor="black")
        axis.set_title(column)
        axis.set_xlabel("Value")
        axis.set_ylabel("Count")
    for axis in axes[len(selected):]:
        axis.axis("off")
    plt.tight_layout()
    path = output_dir / "distribution_plots.png"
    plt.savefig(path, dpi=200)
    plt.close(fig)
    return str(path.name)


def _write_html_report(report: dict[str, Any], output_path: Path, image_files: list[str]) -> None:
    html_lines = [
        "<!doctype html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='utf-8'>",
        "<title>Dataset Validation Report</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:24px;line-height:1.5;}",
        "h1,h2{color:#1f4e79;}",
        "table{border-collapse:collapse;width:100%;margin-bottom:16px;}",
        "th,td{border:1px solid #ddd;padding:8px;text-align:left;}",
        "th{background:#f2f2f2;}",
        "img{max-width:100%;margin-top:12px;border:1px solid #ddd;}",
        "</style>",
        "</head>",
        "<body>",
        "<h1>Dataset Validation Report</h1>",
        f"<p>Dataset: {report['dataset_path']}</p>",
        "<h2>Summary</h2>",
        "<table>",
        "<tr><th>Check</th><th>Status</th><th>Details</th></tr>",
    ]
    for key, value in report["checks"].items():
        status = value.get("status", "unknown")
        details = json.dumps(value, ensure_ascii=False)
        html_lines.append(f"<tr><td>{key}</td><td>{status}</td><td>{details}</td></tr>")
    html_lines.extend(["</table>", "<h2>Plots</h2>"])
    for image_name in image_files:
        html_lines.append(f"<img src='{image_name}' alt='{image_name}'>")
    html_lines.extend(["</body>", "</html>"])
    output_path.write_text("\n".join(html_lines), encoding="utf-8")


def _write_pdf_report(report: dict[str, Any], output_path: Path, image_files: list[str]) -> None:
    with PdfPages(output_path) as pdf:
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis("off")
        content = ["Dataset Validation Report", f"Dataset: {report['dataset_path']}", ""]
        for key, value in report["checks"].items():
            content.append(f"{key}: {value.get('status', 'unknown')}")
            if isinstance(value, dict):
                for item_key, item_value in value.items():
                    if item_key != "status":
                        content.append(f"  - {item_key}: {item_value}")
            content.append("")
        ax.text(0.02, 0.98, "\n".join(content), va="top", ha="left", fontsize=9, family="monospace")
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        for image_name in image_files:
            image_path = output_path.parent / image_name
            if not image_path.exists():
                continue
            fig, ax = plt.subplots(figsize=(8.27, 11.69))
            ax.axis("off")
            image = plt.imread(image_path)
            ax.imshow(image)
            ax.set_title(image_name)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def validate_dataset(dataset_path: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir) if output_dir is not None else PROJECT_ROOT / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = _parse_dataset(dataset_path)
    checks = {
        "missing_values": _check_missing_values(frame),
        "duplicate_rows": _check_duplicate_rows(frame),
        "class_balance": _check_class_balance(frame),
        "career_balance": _check_career_balance(frame),
        "feature_ranges": _check_feature_ranges(frame),
        "impossible_combinations": _check_impossible_combinations(frame),
        "outliers": _check_outliers(frame),
    }
    report = {
        "dataset_path": str(dataset_path),
        "checks": checks,
        **checks,
    }

    image_files: list[str] = []
    correlation_image = _save_correlations(frame, output_dir)
    if correlation_image:
        image_files.append(correlation_image)
    distribution_image = _save_distribution_plot(frame, output_dir)
    if distribution_image:
        image_files.append(distribution_image)

    html_path = output_dir / "dataset_validation_report.html"
    pdf_path = output_dir / "dataset_validation_report.pdf"
    _write_html_report(report, html_path, image_files)
    _write_pdf_report(report, pdf_path, image_files)
    return report


if __name__ == "__main__":
    parser = None
    try:
        import argparse
        parser = argparse.ArgumentParser(description="Validate a dataset and generate HTML/PDF reports")
        parser.add_argument("--dataset", default=str(PROJECT_ROOT / "datasets" / "synthetic" / "synthetic_dataset_v2.csv"))
        parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "reports"))
        args = parser.parse_args()
        validate_dataset(args.dataset, args.output_dir)
        print(f"Validation report written to {Path(args.output_dir) / 'dataset_validation_report.html'}")
        print(f"Validation PDF written to {Path(args.output_dir) / 'dataset_validation_report.pdf'}")
    except SystemExit:
        raise
