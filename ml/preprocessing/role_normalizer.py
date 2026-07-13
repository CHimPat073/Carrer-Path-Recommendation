import re
from typing import Final

TARGET_CAREERS: Final[list[str]] = [
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

ROLE_MAPPING: Final[dict[str, str]] = {
    "software engineer": "Software Engineer",
    "software developer": "Software Engineer",
    "backend developer": "Backend Developer",
    "backend engineer": "Backend Developer",
    "frontend developer": "Frontend Developer",
    "frontend engineer": "Frontend Developer",
    "full stack developer": "Full Stack Developer",
    "fullstack developer": "Full Stack Developer",
    "full stack engineer": "Full Stack Developer",
    "mobile app developer": "Mobile App Developer",
    "mobile developer": "Mobile App Developer",
    "ios developer": "Mobile App Developer",
    "android developer": "Mobile App Developer",
    "data scientist": "Data Scientist",
    "machine learning engineer": "ML Engineer",
    "ml engineer": "ML Engineer",
    "data engineer": "Data Engineer",
    "ai research engineer": "AI Research Engineer",
    "research engineer": "AI Research Engineer",
    "devops engineer": "DevOps Engineer",
    "devops": "DevOps Engineer",
    "cloud engineer": "Cloud Engineer",
    "cloud architect": "Cloud Engineer",
    "cyber security analyst": "Cyber Security Analyst",
    "security analyst": "Cyber Security Analyst",
    "ethical hacker": "Ethical Hacker",
    "penetration tester": "Ethical Hacker",
    "network engineer": "Network Engineer",
    "ui/ux designer": "UI/UX Designer",
    "ux designer": "UI/UX Designer",
    "product manager": "Product Manager",
    "business analyst": "Business Analyst",
    "qa engineer": "QA Engineer",
    "quality assurance engineer": "QA Engineer",
    "database administrator": "Database Administrator",
    "db administrator": "Database Administrator",
    "game developer": "Game Developer",
    "game engineer": "Game Developer",
}


def normalize_role(role: str | None) -> str:
    """Normalize a role name into one of the target career labels."""
    if not role:
        return "Other"

    normalized = re.sub(r"[^a-z0-9]+", " ", str(role).strip().lower()).strip()
    if not normalized:
        return "Other"

    if normalized in ROLE_MAPPING:
        return ROLE_MAPPING[normalized]

    for career in TARGET_CAREERS:
        if normalized == re.sub(r"[^a-z0-9]+", " ", career.lower()).strip():
            return career

    for alias, career in ROLE_MAPPING.items():
        if alias in normalized or normalized in alias:
            return career

    return "Other"
