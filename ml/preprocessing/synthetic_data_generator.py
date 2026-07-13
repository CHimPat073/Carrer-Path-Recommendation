import csv
import random
import sys
from pathlib import Path
from typing import Final

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.preprocessing.role_normalizer import TARGET_CAREERS

RANDOM_SEED: Final[int] = 42
random.seed(RANDOM_SEED)

FEATURE_NAMES: Final[list[str]] = [
    "years_experience",
    "education_level",
    "projects_completed",
    "certifications",
    "python_score",
    "sql_score",
    "java_score",
    "javascript_score",
    "machine_learning_score",
    "deep_learning_score",
    "cloud_score",
    "devops_score",
    "cybersecurity_score",
    "data_analysis_score",
    "product_management_score",
    "ui_design_score",
    "ux_research_score",
    "communication_score",
    "leadership_score",
    "problem_solving_score",
    "teamwork_score",
    "agile_score",
    "research_score",
    "database_score",
    "networking_score",
    "mobile_score",
    "game_dev_score",
    "testing_score",
    "business_analysis_score",
    "sales_score",
    "marketing_score",
]

EDUCATION_LEVELS: Final[list[str]] = [
    "High School",
    "Associate",
    "Bachelor",
    "Master",
    "PhD",
]


def _weighted_choice(options: list[tuple[str, int]]) -> str:
    return random.choices([item[0] for item in options], weights=[item[1] for item in options], k=1)[0]


def _bounded_score(mean: float, sigma: float = 1.2) -> int:
    score = int(round(random.gauss(mean, sigma)))
    return min(10, max(1, score))


def _generate_feature_values(career: str) -> dict[str, object]:
    base = {
        "years_experience": int(max(0, min(20, random.gauss(5.5, 2.5)))),
        "education_level": _weighted_choice([
            ("High School", 5),
            ("Associate", 10),
            ("Bachelor", 50),
            ("Master", 30),
            ("PhD", 5),
        ]),
        "projects_completed": int(max(0, min(30, random.gauss(8, 4)))),
        "certifications": int(max(0, min(5, random.gauss(1.5, 1.2)))),
    }

    skill_bias = {
        "Software Engineer": {"python_score": 7, "java_score": 6, "javascript_score": 6, "sql_score": 6, "testing_score": 6},
        "Backend Developer": {"python_score": 7, "java_score": 6, "sql_score": 7, "database_score": 6, "api_score": 6},
        "Frontend Developer": {"javascript_score": 7, "ui_design_score": 7, "ux_research_score": 6, "communication_score": 6},
        "Full Stack Developer": {"python_score": 7, "javascript_score": 7, "sql_score": 6, "cloud_score": 6},
        "Mobile App Developer": {"mobile_score": 7, "javascript_score": 6, "ui_design_score": 6, "testing_score": 6},
        "Data Scientist": {"python_score": 7, "machine_learning_score": 7, "data_analysis_score": 7, "sql_score": 6},
        "ML Engineer": {"python_score": 7, "machine_learning_score": 7, "deep_learning_score": 7, "data_analysis_score": 6},
        "Data Engineer": {"sql_score": 7, "database_score": 7, "python_score": 6, "cloud_score": 5},
        "AI Research Engineer": {"python_score": 7, "machine_learning_score": 7, "deep_learning_score": 7, "research_score": 7},
        "DevOps Engineer": {"cloud_score": 7, "devops_score": 7, "linux_score": 6, "networking_score": 6},
        "Cloud Engineer": {"cloud_score": 7, "devops_score": 6, "networking_score": 6, "security_score": 6},
        "Cyber Security Analyst": {"cybersecurity_score": 7, "networking_score": 6, "python_score": 5, "problem_solving_score": 6},
        "Ethical Hacker": {"cybersecurity_score": 7, "networking_score": 7, "problem_solving_score": 6, "testing_score": 5},
        "Network Engineer": {"networking_score": 7, "cloud_score": 5, "security_score": 6, "problem_solving_score": 6},
        "UI/UX Designer": {"ui_design_score": 7, "ux_research_score": 7, "communication_score": 6, "leadership_score": 5},
        "Product Manager": {"product_management_score": 7, "leadership_score": 7, "communication_score": 7, "agile_score": 6},
        "Business Analyst": {"business_analysis_score": 7, "communication_score": 6, "problem_solving_score": 7, "sql_score": 5},
        "QA Engineer": {"testing_score": 7, "python_score": 5, "problem_solving_score": 6, "communication_score": 5},
        "Database Administrator": {"database_score": 7, "sql_score": 6, "problem_solving_score": 5, "networking_score": 5},
        "Game Developer": {"game_dev_score": 7, "javascript_score": 6, "python_score": 5, "teamwork_score": 6},
    }

    skill_map: dict[str, int] = {}
    for skill_name in FEATURE_NAMES:
        if skill_name.endswith("_score"):
            bias = skill_bias.get(career, {}).get(skill_name, 5)
            skill_map[skill_name] = _bounded_score(bias, sigma=1.1)

    for skill_name, value in skill_bias.get(career, {}).items():
        # Slightly perturb career-specific strengths for realism.
        skill_map[skill_name] = _bounded_score(value + random.choice([-1, 0, 1]), sigma=0.9)

    for skill_name in list(skill_map.keys()):
        if random.random() < 0.05:
            skill_map[skill_name] = _bounded_score(skill_map[skill_name] + random.choice([-1, 1]), sigma=0.8)

    base.update(skill_map)
    return base


def generate_synthetic_dataset(rows: int = 20000) -> list[dict[str, object]]:
    generated_rows: list[dict[str, object]] = []
    for _ in range(rows):
        career = random.choice(TARGET_CAREERS)
        row = _generate_feature_values(career)
        row["career"] = career
        row["country"] = random.choice(["USA", "Canada", "UK", "India", "Australia", "Germany"])
        row["industry"] = random.choice(["Technology", "Finance", "Healthcare", "Education", "Retail", "Consulting", "Gaming"])
        row["employment_type"] = random.choice(["Full-time", "Part-time", "Contract", "Freelance"])
        row["remote_preference"] = random.choice(["Remote", "Hybrid", "On-site"])
        row["salary_band"] = random.randint(40_000, 180_000)
        row["job_satisfaction"] = round(random.uniform(2.5, 5.0), 2)
        row["work_hours_per_week"] = random.randint(35, 60)
        row["career_growth_score"] = round(random.uniform(2.0, 5.0), 2)
        generated_rows.append(row)

    return generated_rows


def save_synthetic_dataset(rows: list[dict[str, object]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["career"] + FEATURE_NAMES + [
        "country",
        "industry",
        "employment_type",
        "remote_preference",
        "salary_band",
        "job_satisfaction",
        "work_hours_per_week",
        "career_growth_score",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    return output_path


def generate_and_save(output_path: str | Path | None = None, rows: int = 20000) -> Path:
    output_path = Path(output_path) if output_path else PROJECT_ROOT / "datasets" / "synthetic" / "synthetic_dataset.csv"
    rows_data = generate_synthetic_dataset(rows=rows)
    return save_synthetic_dataset(rows_data, output_path)


if __name__ == "__main__":
    output_file = generate_and_save()
    print(f"Synthetic dataset saved to {output_file}")