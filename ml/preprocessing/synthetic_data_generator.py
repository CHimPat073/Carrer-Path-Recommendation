import csv
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Final

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.preprocessing.role_normalizer import TARGET_CAREERS

RANDOM_SEED: Final[int] = 42
random.seed(RANDOM_SEED)

KNOWLEDGE_ROOT = PROJECT_ROOT / "knowledge_base"
OUTPUT_DIR = PROJECT_ROOT / "datasets" / "synthetic"
DEFAULT_OUTPUT = OUTPUT_DIR / "synthetic_dataset_v2.csv"
VALIDATION_OUTPUT = OUTPUT_DIR / "validation_report_v2.json"

FEATURE_NAMES: Final[list[str]] = [
    "years_experience",
    "education_level",
    "projects_completed",
    "certifications",
    "python_score",
    "java_score",
    "javascript_score",
    "sql_score",
    "machine_learning_score",
    "deep_learning_score",
    "cloud_score",
    "devops_score",
    "cybersecurity_score",
    "data_analysis_score",
    "database_score",
    "networking_score",
    "mobile_score",
    "game_dev_score",
    "testing_score",
    "business_analysis_score",
    "product_management_score",
    "ui_design_score",
    "ux_research_score",
    "communication_score",
    "leadership_score",
    "problem_solving_score",
    "teamwork_score",
    "agile_score",
    "research_score",
]

EDUCATION_LEVELS: Final[list[str]] = ["High School", "Associate", "Bachelor", "Master", "PhD"]
PERSONA_LEVELS: Final[list[str]] = ["Beginner", "Junior", "Mid-Level", "Senior", "Expert"]
PERSONA_WEIGHTS: Final[list[float]] = [0.15, 0.25, 0.30, 0.20, 0.10]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _bounded_int(value: float, low: int, high: int) -> int:
    return int(round(_clamp(value, low, high)))


def _sample_from_range(range_info: dict[str, Any], *, default_mean: float | None = None, sigma: float | None = None) -> float:
    if not isinstance(range_info, dict):
        return float(default_mean or 5)

    minimum = float(range_info.get("min", 1))
    maximum = float(range_info.get("max", 10))
    average = float(range_info.get("average", (minimum + maximum) / 2))
    spread = sigma if sigma is not None else max(0.6, (maximum - minimum) / 6)
    value = random.gauss(average, spread)
    return _clamp(value, minimum, maximum)


def _parse_salary_band(band_text: str) -> tuple[int, int]:
    numbers = [int(n) for n in re.findall(r"\d+", band_text)]
    if len(numbers) >= 2:
        return numbers[0] * 1000, numbers[1] * 1000
    if numbers:
        return numbers[0] * 1000, numbers[0] * 1000
    return 60000, 120000


def _education_from_text(education_text: str) -> str:
    lowered = education_text.lower()
    if "phd" in lowered:
        return "PhD"
    if "master" in lowered:
        return "Master"
    if "bachelor" in lowered:
        return "Bachelor"
    if "associate" in lowered:
        return "Associate"
    return "Bachelor"


def _career_anchors(career: str) -> list[str]:
    anchor_map: dict[str, list[str]] = {
        "Software Engineer": ["python_score", "java_score", "javascript_score", "sql_score", "testing_score", "problem_solving_score"],
        "Backend Developer": ["python_score", "java_score", "sql_score", "database_score", "problem_solving_score"],
        "Frontend Developer": ["javascript_score", "ui_design_score", "ux_research_score", "communication_score"],
        "Full Stack Developer": ["python_score", "javascript_score", "sql_score", "cloud_score", "problem_solving_score"],
        "Mobile App Developer": ["mobile_score", "javascript_score", "ui_design_score", "testing_score"],
        "Data Scientist": ["python_score", "machine_learning_score", "data_analysis_score", "sql_score", "research_score"],
        "ML Engineer": ["python_score", "machine_learning_score", "deep_learning_score", "data_analysis_score", "research_score"],
        "Data Engineer": ["sql_score", "database_score", "python_score", "cloud_score", "devops_score"],
        "AI Research Engineer": ["python_score", "machine_learning_score", "deep_learning_score", "research_score"],
        "DevOps Engineer": ["cloud_score", "devops_score", "networking_score", "problem_solving_score"],
        "Cloud Engineer": ["cloud_score", "devops_score", "networking_score", "cybersecurity_score"],
        "Cyber Security Analyst": ["cybersecurity_score", "networking_score", "problem_solving_score", "research_score"],
        "Ethical Hacker": ["cybersecurity_score", "networking_score", "problem_solving_score", "testing_score"],
        "Network Engineer": ["networking_score", "cloud_score", "cybersecurity_score", "problem_solving_score"],
        "UI/UX Designer": ["ui_design_score", "ux_research_score", "communication_score", "research_score"],
        "Product Manager": ["product_management_score", "leadership_score", "communication_score", "agile_score", "business_analysis_score"],
        "Business Analyst": ["business_analysis_score", "communication_score", "problem_solving_score", "sql_score"],
        "QA Engineer": ["testing_score", "problem_solving_score", "communication_score", "agile_score"],
        "Database Administrator": ["database_score", "sql_score", "problem_solving_score", "cybersecurity_score"],
        "Game Developer": ["game_dev_score", "javascript_score", "python_score", "teamwork_score", "ui_design_score"],
    }
    return anchor_map.get(career, [])


def _persona_level() -> str:
    return random.choices(PERSONA_LEVELS, weights=PERSONA_WEIGHTS, k=1)[0]


def _persona_profile(persona_name: str) -> dict[str, Any]:
    personas = _load_json(KNOWLEDGE_ROOT / "user_personas.json").get("user_personas", [])
    for persona in personas:
        if persona.get("persona") == persona_name:
            return persona
    return {
        "years_of_experience": 3,
        "projects_completed": 8,
        "certifications": 2,
        "leadership": 5,
        "communication": 6,
        "problem_solving": 6,
        "expected_salary": 90000,
        "career_growth": 8,
    }


def _choose_education_level(career: str, persona_name: str, career_entry: dict[str, Any]) -> str:
    base_level = _education_from_text(career_entry.get("typical_education", "Bachelor's degree"))
    options = [base_level]
    if base_level == "Bachelor":
        options.extend(["Associate", "Master"])
    elif base_level == "Master":
        options.extend(["Bachelor", "PhD"])
    elif base_level == "PhD":
        options.extend(["Master"])
    else:
        options.extend(["Bachelor", "Associate"])

    persona = _persona_profile(persona_name)
    if persona_name == "Beginner":
        options = ["High School", "Associate", "Bachelor"]
    elif persona_name == "Junior":
        options = ["Associate", "Bachelor", "Master"]
    elif persona_name == "Mid-Level":
        options = ["Bachelor", "Master"]
    elif persona_name == "Senior":
        options = ["Bachelor", "Master", "PhD"]
    elif persona_name == "Expert":
        options = ["Master", "PhD"]

    return random.choice(options)


def _generate_base_row(career: str, persona_name: str, feature_ranges: dict[str, Any], career_entry: dict[str, Any]) -> dict[str, Any]:
    persona = _persona_profile(persona_name)
    anchors = _career_anchors(career)
    row: dict[str, Any] = {}

    career_range = feature_ranges.get(career, {}).get("features", {}) if isinstance(feature_ranges.get(career), dict) else {}

    years_floor = int(career_range.get("Years Experience", {}).get("min", 0)) if isinstance(career_range.get("Years Experience"), dict) else 0
    years_ceiling = int(career_range.get("Years Experience", {}).get("max", 20)) if isinstance(career_range.get("Years Experience"), dict) else 20
    projects_floor = int(career_range.get("Projects Completed", {}).get("min", 0)) if isinstance(career_range.get("Projects Completed"), dict) else 0
    projects_ceiling = int(career_range.get("Projects Completed", {}).get("max", 30)) if isinstance(career_range.get("Projects Completed"), dict) else 30
    cert_floor = int(career_range.get("Certifications", {}).get("min", 0)) if isinstance(career_range.get("Certifications"), dict) else 0
    cert_ceiling = int(career_range.get("Certifications", {}).get("max", 6)) if isinstance(career_range.get("Certifications"), dict) else 6

    experience_anchor = max(years_floor, min(years_ceiling, persona.get("years_of_experience", 3) + random.randint(-1, 2)))
    projects_anchor = max(projects_floor, min(projects_ceiling, persona.get("projects_completed", 8) + random.randint(-2, 3)))
    cert_anchor = max(cert_floor, min(cert_ceiling, persona.get("certifications", 2) + random.randint(-1, 1)))

    row["years_experience"] = _bounded_int(random.gauss(experience_anchor, 1.6), years_floor, years_ceiling)
    row["projects_completed"] = _bounded_int(random.gauss(projects_anchor, 2.2), projects_floor, projects_ceiling)
    row["certifications"] = _bounded_int(random.gauss(cert_anchor, 1.0), cert_floor, cert_ceiling)
    row["education_level"] = _choose_education_level(career, persona_name, career_entry)

    for feature in FEATURE_NAMES:
        if feature in {"years_experience", "projects_completed", "certifications", "education_level"}:
            continue

        feature_info = career_range.get(feature.replace("_score", "")) if feature.replace("_score", "") in career_range else None
        feature_name = feature.replace("_score", "")
        if feature_name == "years_experience":
            continue

        if feature_info is None:
            feature_info = {"min": 1, "max": 10, "average": 5}

        mean_value = float(feature_info.get("average", 5))
        if feature in anchors:
            mean_value += 0.5
        if feature in {"communication_score", "leadership_score", "problem_solving_score", "teamwork_score", "research_score"}:
            mean_value += 0.2

        if feature == "communication_score":
            mean_value += max(0, (persona.get("communication", 6) - 6) * 0.25)
        if feature == "leadership_score":
            mean_value += max(0, (persona.get("leadership", 5) - 5) * 0.25)
        if feature == "problem_solving_score":
            mean_value += max(0, (persona.get("problem_solving", 6) - 6) * 0.25)

        sampled = _sample_from_range(feature_info, default_mean=mean_value, sigma=max(0.8, (float(feature_info.get("max", 10)) - float(feature_info.get("min", 1))) / 8))
        row[feature] = _bounded_int(sampled, 1, 10)

    return row


def _apply_correlation_rules(row: dict[str, Any], career: str, correlation_rules: list[dict[str, Any]]) -> None:
    for rule in correlation_rules:
        pair = rule.get("feature_pair", [])
        if len(pair) != 2:
            continue

        first, second = pair[0], pair[1]
        if first == "AI careers" or second == "AI careers":
            continue

        first_key = first.replace(" ", "_").lower()
        second_key = second.replace(" ", "_").lower()
        if first_key == "years_experience":
            first_key = "years_experience"
        if second_key == "years_experience":
            second_key = "years_experience"

        if first_key == "machine_learning" and second_key == "python":
            first_key = "machine_learning_score"
            second_key = "python_score"
        elif first_key == "deep_learning" and second_key == "machine_learning":
            first_key = "deep_learning_score"
            second_key = "machine_learning_score"
        elif first_key == "leadership" and second_key == "years_experience":
            first_key = "leadership_score"
            second_key = "years_experience"
        elif first_key == "projects_completed" and second_key == "years_experience":
            first_key = "projects_completed"
            second_key = "years_experience"
        elif first_key == "communication" and second_key == "years_experience":
            first_key = "communication_score"
            second_key = "years_experience"
        elif first_key == "research" and second_key == "ai_careers":
            first_key = "research_score"
            second_key = "python_score"
        elif first_key == "cloud" and second_key == "devops":
            first_key = "cloud_score"
            second_key = "devops_score"
        elif first_key == "cybersecurity" and second_key == "networking":
            first_key = "cybersecurity_score"
            second_key = "networking_score"
        elif first_key == "ui_design" and second_key == "ux_research":
            first_key = "ui_design_score"
            second_key = "ux_research_score"
        elif first_key == "product_management" and second_key == "business_analysis":
            first_key = "product_management_score"
            second_key = "business_analysis_score"
        elif first_key == "database" and second_key == "sql":
            first_key = "database_score"
            second_key = "sql_score"
        elif first_key == "mobile_development" and second_key == "ui_design":
            first_key = "mobile_score"
            second_key = "ui_design_score"
        elif first_key == "game_development" and second_key == "programming":
            first_key = "game_dev_score"
            second_key = "python_score"
        elif first_key == "problem_solving" and second_key == "years_experience":
            first_key = "problem_solving_score"
            second_key = "years_experience"

        if first_key not in row or second_key not in row:
            continue

        strength = float(rule.get("strength", 0.5))
        if second_key == "years_experience":
            delta = max(0, int(row[first_key]) - 5)
            row[second_key] = _bounded_int(row[second_key] + delta * strength * 0.4, 0, 20)
        else:
            delta = max(0, int(row[first_key]) - 5)
            row[second_key] = _bounded_int(row[second_key] + delta * strength * 0.3, 1, 10)


def _apply_noise(row: dict[str, Any], career: str, noise_rules: dict[str, Any]) -> None:
    rules = noise_rules.get("rules", [])
    if random.random() >= 0.08 or not rules:
        return

    selected = random.choice(rules)
    noise_type = selected.get("type")
    if noise_type == "weak_skill":
        target_skill = random.choice(["python_score", "sql_score", "cloud_score", "communication_score", "testing_score", "database_score"])
        row[target_skill] = max(1, row.get(target_skill, 5) - random.randint(1, 2))
    elif noise_type == "interest_mismatch":
        row["machine_learning_score"] = min(10, row.get("machine_learning_score", 5) + 1)
        row["python_score"] = max(1, row.get("python_score", 5) - 1)
    elif noise_type == "average_soft_skill":
        soft_target = random.choice(["communication_score", "leadership_score", "teamwork_score"])
        row[soft_target] = 5
    elif noise_type == "lower_certification_level":
        row["certifications"] = max(0, row.get("certifications", 2) - random.randint(1, 2))
    elif noise_type == "limited_experience":
        row["years_experience"] = max(0, row.get("years_experience", 3) - random.randint(1, 2))
        row["projects_completed"] = max(0, row.get("projects_completed", 8) - random.randint(1, 3))


def _apply_overlap(row: dict[str, Any], career: str, overlap_map: dict[str, list[dict[str, Any]]]) -> None:
    similar = overlap_map.get(career, [])
    if not similar or random.random() >= 0.25:
        return

    selected = random.choices(similar, weights=[item.get("similarity_percentage", 50) for item in similar], k=1)[0]
    similar_career = selected.get("career")
    if not similar_career:
        return

    for feature in ["python_score", "sql_score", "javascript_score", "cloud_score", "devops_score", "machine_learning_score", "ui_design_score", "communication_score", "leadership_score"]:
        if random.random() < 0.35:
            row[feature] = _bounded_int((row.get(feature, 5) + random.randint(4, 7)) / 2, 1, 10)


def _finalize_row(row: dict[str, Any], career: str, career_entry: dict[str, Any]) -> dict[str, Any]:
    salary_min, salary_max = _parse_salary_band(career_entry.get("salary_band", "$80k-$120k USD"))
    base_salary = random.randint(salary_min, salary_max)
    row["career"] = career
    row["salary_band"] = base_salary
    row["remote_preference"] = random.choice(["Remote", "Hybrid", "On-site"])
    row["career_growth_score"] = _bounded_int(6 + random.random() * 3, 1, 10)
    row["job_satisfaction"] = round(random.uniform(3.2, 4.9), 2)
    row["work_hours_per_week"] = random.randint(35, 60)
    row["country"] = random.choice(["USA", "Canada", "UK", "India", "Australia", "Germany"])
    row["industry"] = random.choice(["Technology", "Finance", "Healthcare", "Education", "Retail", "Consulting", "Gaming"])
    row["employment_type"] = random.choice(["Full-time", "Part-time", "Contract", "Freelance"])
    return row


def generate_synthetic_dataset(rows: int = 20000) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    career_personas = _load_json(KNOWLEDGE_ROOT / "career_personas.json").get("career_personas", [])
    feature_ranges = _load_json(KNOWLEDGE_ROOT / "feature_ranges.json").get("career_feature_ranges", [])
    overlap_data = _load_json(KNOWLEDGE_ROOT / "career_overlap.json").get("career_similarity_matrix", [])
    noise_rules = _load_json(KNOWLEDGE_ROOT / "noise_rules.json").get("noise_generation_rules", {})
    correlation_rules = _load_json(KNOWLEDGE_ROOT / "correlation_rules.json").get("correlation_rules", {}).get("rules", [])

    career_lookup = {item.get("career"): item for item in career_personas}
    feature_lookup = {}
    for item in feature_ranges:
        career_name = item.get("career")
        if career_name:
            feature_lookup[career_name] = item.get("features", {})

    overlap_lookup = {item.get("career"): item.get("similar_careers", []) for item in overlap_data}

    balanced_careers = [career for career in TARGET_CAREERS for _ in range(rows // len(TARGET_CAREERS))]
    if len(balanced_careers) < rows:
        balanced_careers.extend(TARGET_CAREERS[: rows - len(balanced_careers)])
    random.shuffle(balanced_careers)

    generated_rows: list[dict[str, Any]] = []
    for career in balanced_careers[:rows]:
        persona_name = _persona_level()
        row = _generate_base_row(career, persona_name, feature_lookup, career_lookup.get(career, {}))
        _apply_correlation_rules(row, career, correlation_rules)
        _apply_overlap(row, career, overlap_lookup)
        _apply_noise(row, career, noise_rules)
        row = _finalize_row(row, career, career_lookup.get(career, {}))
        generated_rows.append(row)

    report = {
        "rows_generated": len(generated_rows),
        "career_distribution": dict(Counter(row["career"] for row in generated_rows)),
        "persona_distribution": dict(Counter(_persona_level() for _ in generated_rows)),
        "noise_injected": sum(1 for row in generated_rows if row.get("salary_band") and row.get("career")),
        "feature_columns": FEATURE_NAMES + ["career", "salary_band", "remote_preference", "career_growth_score", "job_satisfaction", "work_hours_per_week", "country", "industry", "employment_type"],
    }
    return generated_rows, report


def save_synthetic_dataset(rows: list[dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "career",
        "years_experience",
        "education_level",
        "projects_completed",
        "certifications",
        "python_score",
        "java_score",
        "javascript_score",
        "sql_score",
        "machine_learning_score",
        "deep_learning_score",
        "cloud_score",
        "devops_score",
        "cybersecurity_score",
        "data_analysis_score",
        "database_score",
        "networking_score",
        "mobile_score",
        "game_dev_score",
        "testing_score",
        "business_analysis_score",
        "product_management_score",
        "ui_design_score",
        "ux_research_score",
        "communication_score",
        "leadership_score",
        "problem_solving_score",
        "teamwork_score",
        "agile_score",
        "research_score",
        "salary_band",
        "remote_preference",
        "career_growth_score",
        "job_satisfaction",
        "work_hours_per_week",
        "country",
        "industry",
        "employment_type",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    return output_path


def save_validation_report(report: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return output_path


def generate_and_save(output_path: str | Path | None = None, rows: int = 20000) -> tuple[Path, Path]:
    output_path = Path(output_path) if output_path else DEFAULT_OUTPUT
    rows_data, report = generate_synthetic_dataset(rows=rows)
    csv_path = save_synthetic_dataset(rows_data, output_path)
    report_path = save_validation_report(report, VALIDATION_OUTPUT)
    return csv_path, report_path


if __name__ == "__main__":
    csv_file, report_file = generate_and_save()
    print(f"Synthetic dataset saved to {csv_file}")
    print(f"Validation report saved to {report_file}")