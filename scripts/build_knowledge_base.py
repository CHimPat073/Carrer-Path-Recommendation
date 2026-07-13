"""Knowledge base builder entry point.

This module imports the individual pipeline modules, composes their outputs,
and prepares the final payloads that can be exported to JSON files.
The implementation is intentionally side-effect free until the caller
explicitly executes the script.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.extract_skills import calculate_skill_frequency, extract_skills_from_dataframe
from scripts.generate_correlations import generate_correlation_rules
from scripts.generate_feature_ranges import generate_feature_ranges
from scripts.generate_noise_rules import generate_noise_rules
from scripts.generate_overlap import compute_career_overlap
from scripts.generate_personas import generate_career_personas
from scripts.generate_user_personas import generate_user_personas
from scripts.utils import configure_logging, ensure_output_directory
from scripts.validate_knowledge_base import validate_knowledge_base

logger = configure_logging("build_knowledge_base")


def build_knowledge_base(
    career_names: list[str],
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Create a full knowledge base payload from modular components.

    Input:
        career_names (list[str]): Careers to include in the knowledge base.
        output_dir (str | Path | None): Optional directory for exporting JSON files.

    Output:
        dict[str, Any]: The composed knowledge base payload.

    Purpose:
        Orchestrate the full pipeline without executing it automatically.
    """
    logger.info("Building knowledge base payload")

    career_personas = generate_career_personas(career_names)
    feature_ranges = generate_feature_ranges(career_names)
    user_personas = generate_user_personas()
    noise_rules = generate_noise_rules()
    correlation_rules = generate_correlation_rules()

    sample_skill_map: dict[str, list[str]] = {}
    for career in career_names:
        profile = career_personas.get(career, {})
        sample_skills = profile.get("required_technical_skills", []) + profile.get("nice_to_have_skills", [])
        sample_skill_map[career] = sample_skills

    overlap = compute_career_overlap(sample_skill_map)
    validation_report = validate_knowledge_base(
        {
            "career_personas": career_personas,
            "feature_ranges": feature_ranges,
            "career_overlap": overlap,
            "user_personas": user_personas,
            "noise_rules": noise_rules,
            "correlation_rules": correlation_rules,
        }
    )

    payload = {
        "career_personas": career_personas,
        "feature_ranges": feature_ranges,
        "career_overlap": overlap,
        "user_personas": user_personas,
        "noise_rules": noise_rules,
        "correlation_rules": correlation_rules,
        "validation_report": validation_report,
    }

    if output_dir is not None:
        output_path = ensure_output_directory(output_dir)
        for name, value in payload.items():
            if name == "validation_report":
                continue
            file_path = output_path / f"{name}.json"
            file_path.write_text(json.dumps(value, indent=2), encoding="utf-8")
        logger.info("Knowledge base files written to %s", output_path)

    return payload


if __name__ == "__main__":
    default_careers = [
        "Software Engineer",
        "Backend Developer",
        "Frontend Developer",
        "Full Stack Developer",
        "Mobile App Developer",
        "Data Scientist",
        "ML Engineer",
        "Data Engineer",
        "AI Research Engineer",
        "DevOps Engineer",
        "Cloud Engineer",
        "Cyber Security Analyst",
        "Ethical Hacker",
        "Network Engineer",
        "UI/UX Designer",
        "Product Manager",
        "Business Analyst",
        "QA Engineer",
        "Database Administrator",
        "Game Developer",
    ]
    build_knowledge_base(default_careers, output_dir="knowledge_base")
