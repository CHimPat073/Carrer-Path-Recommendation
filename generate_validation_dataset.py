"""
Validation Dataset Generator for CareerPilot AI

Generates 500 synthetic candidates balanced across all careers.
Preserves all statistical constraints from knowledge base.
"""

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.preprocessing.role_normalizer import TARGET_CAREERS

RANDOM_SEED: int = 123  # Different seed for validation set

KNOWLEDGE_ROOT = PROJECT_ROOT / "knowledge_base"

FEATURE_NAMES = [
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

EDUCATION_LEVELS = ["High School", "Associate", "Bachelor", "Master", "PhD"]
PERSONA_LEVELS = ["Beginner", "Junior", "Mid-Level", "Senior", "Expert"]
PERSONA_WEIGHTS = [0.15, 0.25, 0.30, 0.20, 0.10]


class ValidationDatasetGenerator:
    """Generate validation dataset with balanced careers and preserved constraints."""

    def __init__(self, rows: int = 500, seed: int = RANDOM_SEED) -> None:
        self.rows = rows
        self.seed = seed
        self.random = random.Random(seed)
        self.knowledge_root = KNOWLEDGE_ROOT
        self.career_personas = self._load_json(self.knowledge_root / "career_personas.json").get("career_personas", [])
        self.feature_ranges = self._load_json(self.knowledge_root / "feature_ranges.json").get("career_feature_ranges", [])
        self.overlap_data = self._load_json(self.knowledge_root / "career_overlap.json").get("career_similarity_matrix", [])
        self.noise_rules = self._load_json(self.knowledge_root / "noise_rules.json").get("noise_generation_rules", {})
        self.correlation_rules = self._load_json(self.knowledge_root / "correlation_rules.json").get("correlation_rules", {}).get("rules", [])
        self.career_lookup = {item.get("career"): item for item in self.career_personas if item.get("career")}
        self.feature_lookup = {
            item.get("career"): item.get("features", {})
            for item in self.feature_ranges
            if item.get("career")
        }
        self.overlap_lookup = {
            item.get("career"): item.get("similar_careers", [])
            for item in self.overlap_data
            if item.get("career")
        }
        self.user_personas = self._load_json(self.knowledge_root / "user_personas.json").get("user_personas", [])

    @staticmethod
    def _load_json(path: Path) -> dict:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    @staticmethod
    def _bounded_int(value: float, low: int, high: int) -> int:
        return int(round(ValidationDatasetGenerator._clamp(value, low, high)))

    def _sample_from_range(
    self,
    range_info: dict[str, any],
    default_mean: float,
    sigma: float | None = None,) -> float:
        """
        Sample a value using a truncated Gaussian distribution.

        Improvements:
        - Lower variance
        - Stay closer to career average
        - Prevent excessive overlap between careers
        """

        if not isinstance(range_info, dict):
            return float(default_mean)

        minimum = float(range_info.get("min", 1))
        maximum = float(range_info.get("max", 10))

        mean = float(default_mean)

        feature_range = maximum - minimum

        # Much tighter sigma
        if sigma is None:
            sigma = max(0.30, feature_range / 8)

        # Generate value
        value = self.random.gauss(mean, sigma)

        # Clamp inside valid range
        value = max(minimum, min(value, maximum))

        return value

    def _persona_level(self) -> str:
        return self.random.choices(PERSONA_LEVELS, weights=PERSONA_WEIGHTS, k=1)[0]

    def _persona_profile(self, persona_name: str) -> dict:
        for persona in self.user_personas:
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

    def _education_from_text(self, education_text: str) -> str:
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

    def _choose_education_level(self, career: str, persona_name: str, career_entry: dict) -> str:
        base_level = self._education_from_text(career_entry.get("typical_education", "Bachelor's degree"))
        options = [base_level]
        if base_level == "Bachelor":
            options.extend(["Associate", "Master"])
        elif base_level == "Master":
            options.extend(["Bachelor", "PhD"])
        elif base_level == "PhD":
            options.extend(["Master"])
        else:
            options.extend(["Bachelor", "Associate"])

        persona = self._persona_profile(persona_name)
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

        return self.random.choice(options)

    def _career_anchor_features(self, career: str) -> list:
        anchor_map = {
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

    def _generate_base_row(self, career: str, persona_name: str, career_entry: dict) -> dict:
        persona = self._persona_profile(persona_name)
        anchors = self._career_anchor_features(career)
        career_range = self.feature_lookup.get(career, {})
        row = {}

        years_floor = int(career_range.get("years_experience", {}).get("min", 0)) if isinstance(career_range.get("years_experience"), dict) else 0
        years_ceiling = int(career_range.get("years_experience", {}).get("max", 20)) if isinstance(career_range.get("years_experience"), dict) else 20
        projects_floor = int(career_range.get("projects_completed", {}).get("min", 0)) if isinstance(career_range.get("projects_completed"), dict) else 0
        projects_ceiling = int(career_range.get("projects_completed", {}).get("max", 30)) if isinstance(career_range.get("projects_completed"), dict) else 30
        cert_floor = int(career_range.get("certifications", {}).get("min", 0)) if isinstance(career_range.get("certifications"), dict) else 0
        cert_ceiling = int(career_range.get("certifications", {}).get("max", 6)) if isinstance(career_range.get("certifications"), dict) else 6

        experience_anchor = max(years_floor, min(years_ceiling, persona.get("years_of_experience", 3) + self.random.randint(-1, 2)))
        projects_anchor = max(projects_floor, min(projects_ceiling, persona.get("projects_completed", 8) + self.random.randint(-2, 3)))
        cert_anchor = max(cert_floor, min(cert_ceiling, persona.get("certifications", 2) + self.random.randint(-1, 1)))

        row["years_experience"] = self._bounded_int(self.random.gauss(experience_anchor, 1.6), years_floor, years_ceiling)
        row["projects_completed"] = self._bounded_int(self.random.gauss(projects_anchor, 2.2), projects_floor, projects_ceiling)
        row["certifications"] = self._bounded_int(self.random.gauss(cert_anchor, 1.0), cert_floor, cert_ceiling)
        row["education_level"] = self._choose_education_level(career, persona_name, career_entry)

        for feature in FEATURE_NAMES:
            if feature in {"years_experience", "projects_completed", "certifications", "education_level"}:
                continue

            feature_name = feature.replace("_score", "")
            title_map = {
                "python":"Python",
                "java":"Java",
                "javascript":"JavaScript",
                "sql":"SQL",
                "machine_learning":"Machine Learning",
                "deep_learning":"Deep Learning",
                "cloud":"Cloud",
                "devops":"DevOps",
                "cybersecurity":"Cybersecurity",
                "data_analysis":"Data Analysis",
                "database":"Database",
                "networking":"Networking",
                "mobile":"Mobile Development",
                "game_dev":"Game Development",
                "testing":"Testing",
                "business_analysis":"Business Analysis",
                "product_management":"Product Management",
                "ui_design":"UI Design",
                "ux_research":"UX Research",
                "communication":"Communication",
                "leadership":"Leadership",
                "problem_solving":"Problem Solving",
                "teamwork":"Teamwork",
                "agile":"Agile",
                "research":"Research",
            }
            feature_name = title_map.get(feature_name, feature_name)
            feature_info = career_range.get(feature_name) if feature_name in career_range else None
            if feature_info is None:
                feature_info = {"min": 1, "max": 10, "average": 5}

            mean_value = float(feature_info.get("average", 5))
            if feature in anchors:
                mean_value += 1.0
            if feature in {"communication_score", "leadership_score", "problem_solving_score", "teamwork_score", "research_score"}:
                mean_value += 0.2
            if feature == "communication_score":
                mean_value += max(0, (persona.get("communication", 6) - 6) * 0.25)
            if feature == "leadership_score":
                mean_value += max(0, (persona.get("leadership", 5) - 5) * 0.25)
            if feature == "problem_solving_score":
                mean_value += max(0, (persona.get("problem_solving", 6) - 6) * 0.25)

            sigma = max(0.5, (float(feature_info.get("max", 10)) - float(feature_info.get("min", 1))) / 4)
            sampled = self._sample_from_range(feature_info, default_mean=mean_value, sigma=sigma)
            row[feature] = self._bounded_int(sampled, int(feature_info.get("min",1)), int(feature_info.get("max",10)))

        return row

    def _apply_correlation_rules(self, row: dict) -> None:
        for rule in self.correlation_rules:
            pair = rule.get("feature_pair", [])
            if len(pair) != 2:
                continue
            first, second = pair[0], pair[1]
            first_key = self._normalize_feature_name(first)
            second_key = self._normalize_feature_name(second)
            if first_key not in row or second_key not in row:
                continue
            strength = float(rule.get("strength", 0.5))
            if second_key == "years_experience":
                delta = max(0, int(row[first_key]) - 5)
                row[second_key] = self._bounded_int(row[second_key] + delta * strength * 0.4, 0, 20)
            else:
                delta = max(0, int(row[first_key]) - 5)
                row[second_key] = self._bounded_int(row[second_key] + delta * strength * 0.3, 1, 10)

    @staticmethod
    def _normalize_feature_name(value: str) -> str:
        cleaned = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
        aliases = {
            "years_experience": "years_experience",
            "projects_completed": "projects_completed",
            "python": "python_score",
            "java": "java_score",
            "javascript": "javascript_score",
            "sql": "sql_score",
            "machine_learning": "machine_learning_score",
            "deep_learning": "deep_learning_score",
            "cloud": "cloud_score",
            "devops": "devops_score",
            "cybersecurity": "cybersecurity_score",
            "data_analysis": "data_analysis_score",
            "database": "database_score",
            "networking": "networking_score",
            "mobile_development": "mobile_score",
            "mobile": "mobile_score",
            "game_development": "game_dev_score",
            "game_dev": "game_dev_score",
            "testing": "testing_score",
            "business_analysis": "business_analysis_score",
            "product_management": "product_management_score",
            "ui_design": "ui_design_score",
            "ux_research": "ux_research_score",
            "communication": "communication_score",
            "leadership": "leadership_score",
            "problem_solving": "problem_solving_score",
            "teamwork": "teamwork_score",
            "agile": "agile_score",
            "research": "research_score",
        }
        return aliases.get(cleaned, cleaned)

    def _apply_noise(self, row: dict) -> None:
        rules = self.noise_rules.get("rules", [])
        if self.random.random() >= 0.08 or not rules:
            return
        selected = self.random.choice(rules)
        noise_type = selected.get("type")
        if noise_type == "weak_skill":
            target_skill = self.random.choice(["python_score", "sql_score", "cloud_score", "communication_score", "testing_score", "database_score"])
            row[target_skill] = max(1, row.get(target_skill, 5) - self.random.randint(1, 2))
        elif noise_type == "interest_mismatch":
            row["machine_learning_score"] = min(10, row.get("machine_learning_score", 5) + 1)
            row["python_score"] = max(1, row.get("python_score", 5) - 1)
        elif noise_type == "average_soft_skill":
            soft_target = self.random.choice(["communication_score", "leadership_score", "teamwork_score"])
            row[soft_target] = 5
        elif noise_type == "lower_certification_level":
            row["certifications"] = max(0, row.get("certifications", 2) - self.random.randint(1, 2))
        elif noise_type == "limited_experience":
            row["years_experience"] = max(0, row.get("years_experience", 3) - self.random.randint(1, 2))
            row["projects_completed"] = max(0, row.get("projects_completed", 8) - self.random.randint(1, 3))

    def _apply_overlap(self, row: dict, career: str) -> None:
        similar = self.overlap_lookup.get(career, [])
        if not similar or self.random.random() >= 0.25:
            return
        selected = self.random.choices(similar, weights=[item.get("similarity_percentage", 50) for item in similar], k=1)[0]
        if not selected.get("career"):
            return
        for feature in ["python_score", "sql_score", "javascript_score", "cloud_score", "devops_score", "machine_learning_score", "ui_design_score", "communication_score", "leadership_score"]:
            if self.random.random() < 0.35:
                row[feature] = self._bounded_int((row.get(feature, 5) + self.random.randint(4, 7)) / 2, 1, 10)

    def _finalize_row(self, row: dict, career: str, career_entry: dict) -> dict:
        salary_min, salary_max = self._parse_salary_band(career_entry.get("salary_band", "$80k-$120k USD"))
        base_salary = self.random.randint(salary_min, salary_max)
        row["career"] = career
        row["salary_band"] = base_salary
        row["remote_preference"] = self.random.choice(["Remote", "Hybrid", "On-site"])
        row["career_growth_score"] = self._bounded_int(6 + self.random.random() * 3, 1, 10)
        row["job_satisfaction"] = round(self.random.uniform(3.2, 4.9), 2)
        row["work_hours_per_week"] = self.random.randint(35, 60)
        row["country"] = self.random.choice(["USA", "Canada", "UK", "India", "Australia", "Germany"])
        row["industry"] = self.random.choice(["Technology", "Finance", "Healthcare", "Education", "Retail", "Consulting", "Gaming"])
        row["employment_type"] = self.random.choice(["Full-time", "Part-time", "Contract", "Freelance"])
        return row

    @staticmethod
    def _parse_salary_band(band_text: str) -> tuple:
        numbers = [int(n) for n in re.findall(r"\d+", band_text)]
        if len(numbers) >= 2:
            return numbers[0] * 1000, numbers[1] * 1000
        if numbers:
            return numbers[0] * 1000, numbers[0] * 1000
        return 60000, 120000

    def generate(self) -> list[dict]:
        """Generate balanced validation dataset across all careers."""
        # Calculate samples per career for balanced distribution
        samples_per_career = self.rows // len(TARGET_CAREERS)

        balanced_careers = []
        for career in TARGET_CAREERS:
            balanced_careers.extend([career] * samples_per_career)

        # Add remaining careers to reach exact count
        remaining = self.rows - len(balanced_careers)
        if remaining > 0:
            balanced_careers.extend(TARGET_CAREERS[:remaining])

        self.random.shuffle(balanced_careers)

        generated_rows = []
        for career in balanced_careers:
            persona_name = self._persona_level()
            row = self._generate_base_row(career, persona_name, self.career_lookup.get(career, {}))
            self._apply_correlation_rules(row)
            self._apply_overlap(row, career)
            self._apply_noise(row)
            row = self._finalize_row(row, career, self.career_lookup.get(career, {}))
            generated_rows.append(row)

        return generated_rows


def analyze_dataset(rows: list[dict]) -> dict:
    """Analyze the generated dataset and return statistics."""
    import pandas as pd

    df = pd.DataFrame(rows)

    # Shape
    shape = df.shape

    # Career distribution
    career_dist = df['career'].value_counts().to_dict()

    # Missing values
    missing = df.isnull().sum().to_dict()
    missing = {k: int(v) for k, v in missing.items() if v > 0}

    # Duplicate rows
    duplicates = df.duplicated().sum()

    # Feature ranges
    feature_ranges = {}
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        if col != 'salary_band':  # Skip salary as it has different scale
            feature_ranges[col] = {
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'mean': float(df[col].mean()),
                'std': float(df[col].std())
            }

    return {
        'shape': {'rows': shape[0], 'columns': shape[1]},
        'career_distribution': career_dist,
        'missing_values': missing,
        'duplicate_rows': int(duplicates),
        'feature_ranges': feature_ranges
    }


def generate_report(analysis: dict, output_path: Path) -> None:
    """Generate HTML report for validation dataset."""
    import pandas as pd

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Validation Dataset Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 40px;
            background-color: #f5f7fa;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 15px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .stat-box {{
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px 30px;
            border-radius: 8px;
            margin: 10px;
            min-width: 150px;
        }}
        .stat-box .label {{
            font-size: 14px;
            opacity: 0.9;
        }}
        .stat-box .value {{
            font-size: 32px;
            font-weight: bold;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .career-dist {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 10px;
            margin: 20px 0;
        }}
        .career-item {{
            background: #ecf0f1;
            padding: 10px 15px;
            border-radius: 5px;
            text-align: center;
        }}
        .career-item .count {{
            font-size: 24px;
            font-weight: bold;
            color: #3498db;
        }}
        .feature-table {{
            max-height: 400px;
            overflow-y: auto;
        }}
        .warning {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Validation Dataset Report</h1>
        <p><strong>Generated:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <h2>Dataset Overview</h2>
        <div class="stat-box">
            <div class="label">Total Rows</div>
            <div class="value">{analysis['shape']['rows']}</div>
        </div>
        <div class="stat-box">
            <div class="label">Total Columns</div>
            <div class="value">{analysis['shape']['columns']}</div>
        </div>
        <div class="stat-box">
            <div class="label">Duplicate Rows</div>
            <div class="value">{analysis['duplicate_rows']}</div>
        </div>

        <h2>Career Distribution</h2>
        <div class="career-dist">
"""

    for career, count in sorted(analysis['career_distribution'].items()):
        html_content += f"""
            <div class="career-item">
                <div>{career}</div>
                <div class="count">{count}</div>
            </div>
"""

    html_content += """
        </div>

        <h2>Missing Values</h2>
"""

    if analysis['missing_values']:
        html_content += "<table><tr><th>Column</th><th>Missing Count</th></tr>"
        for col, count in analysis['missing_values'].items():
            html_content += f"<tr><td>{col}</td><td>{count}</td></tr>"
        html_content += "</table>"
    else:
        html_content += "<p>No missing values found!</p>"

    html_content += """
        <h2>Feature Ranges</h2>
        <div class="feature-table">
            <table>
                <tr>
                    <th>Feature</th>
                    <th>Min</th>
                    <th>Max</th>
                    <th>Mean</th>
                    <th>Std Dev</th>
                </tr>
"""

    for feature, stats in analysis['feature_ranges'].items():
        html_content += f"""
                <tr>
                    <td>{feature}</td>
                    <td>{stats['min']:.2f}</td>
                    <td>{stats['max']:.2f}</td>
                    <td>{stats['mean']:.2f}</td>
                    <td>{stats['std']:.2f}</td>
                </tr>
"""

    html_content += """
            </table>
        </div>
    </div>
</body>
</html>
"""

    with output_path.open('w', encoding='utf-8') as f:
        f.write(html_content)


def main():
    """Main function to generate validation dataset."""
    import pandas as pd

    print("=" * 60)
    print("VALIDATION DATASET GENERATOR")
    print("=" * 60)

    # Create validation directory
    validation_dir = PROJECT_ROOT / "datasets" / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)

    reports_dir = PROJECT_ROOT / "reports"

    # Generate dataset
    print("\n[1/4] Generating 500 synthetic candidates...")
    generator = ValidationDatasetGenerator(rows=500, seed=123)
    rows = generator.generate()
    print(f"      [OK] Generated {len(rows)} candidates")

    # Analyze dataset
    print("\n[2/4] Analyzing dataset...")
    analysis = analyze_dataset(rows)

    # Print results
    print("\n" + "=" * 60)
    print("DATASET ANALYSIS")
    print("=" * 60)

    print(f"\n[SIZE]")
    print(f"   Rows: {analysis['shape']['rows']}")
    print(f"   Columns: {analysis['shape']['columns']}")

    print(f"\n[CAREER DISTRIBUTION]")
    for career, count in sorted(analysis['career_distribution'].items()):
        print(f"   {career}: {count}")

    print(f"\n[MISSING VALUES]")
    if analysis['missing_values']:
        for col, count in analysis['missing_values'].items():
            print(f"   {col}: {count}")
    else:
        print("   None")

    print(f"\n[DUPLICATE ROWS]: {analysis['duplicate_rows']}")

    print(f"\n[FEATURE RANGES]")
    print(f"   {'Feature':<25} {'Min':>8} {'Max':>8} {'Mean':>10} {'Std':>8}")
    print(f"   {'-'*60}")
    for feature, stats in analysis['feature_ranges'].items():
        print(f"   {feature:<25} {stats['min']:>8.2f} {stats['max']:>8.2f} {stats['mean']:>10.2f} {stats['std']:>8.2f}")

    # Save to CSV
    print("\n[3/4] Saving to CSV...")
    df = pd.DataFrame(rows)
    csv_path = validation_dir / "validation_dataset_v2.csv"
    df.to_csv(csv_path, index=False)
    print(f"      [OK] Saved to: {csv_path}")

    # Generate report
    print("\n[4/4] Generating HTML report...")
    report_path = reports_dir / "validation_dataset_report.html"
    generate_report(analysis, report_path)
    print(f"      [OK] Report saved to: {report_path}")

    print("\n" + "=" * 60)
    print("[DONE] VALIDATION DATASET GENERATION COMPLETE!")
    print("=" * 60)

    return analysis


if __name__ == "__main__":
    main()