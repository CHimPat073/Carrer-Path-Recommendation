"""
CareerPilot AI
Knowledge Base - Career Profiles

Defines reusable career profiles for synthetic dataset generation.

NOTE:
- This file includes 3 fully implemented reference profiles.
- Follow the same pattern to add the remaining careers.
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class SkillDistribution:
    mean: float
    std: float
    min_value: int
    max_value: int


@dataclass(frozen=True)
class MetadataDistribution:
    education: Dict[str, float]
    industries: List[str]
    remote_probability: float


@dataclass(frozen=True)
class CareerProfile:
    name: str
    description: str
    technical_skills: Dict[str, SkillDistribution]
    soft_skills: Dict[str, SkillDistribution]
    metadata: MetadataDistribution
    overlaps: Dict[str, float]


ML_ENGINEER = CareerProfile(
    name="ML Engineer",
    description="Designs, trains and deploys ML systems.",
    technical_skills={
        "python_score": SkillDistribution(9.3,0.5,8,10),
        "machine_learning_score": SkillDistribution(9.6,0.4,8,10),
        "deep_learning_score": SkillDistribution(8.8,0.6,7,10),
        "sql_score": SkillDistribution(7.8,0.8,6,10),
        "cloud_score": SkillDistribution(7.6,0.8,5,10),
        "data_analysis_score": SkillDistribution(8.3,0.7,6,10),
        "research_score": SkillDistribution(8.5,0.6,7,10),
        "devops_score": SkillDistribution(5.8,1.1,2,9),
    },
    soft_skills={
        "communication_score": SkillDistribution(6.5,1.0,4,9),
        "leadership_score": SkillDistribution(5.5,1.1,2,9),
        "problem_solving_score": SkillDistribution(9.5,0.4,8,10),
        "teamwork_score": SkillDistribution(7.2,0.8,5,10),
        "agile_score": SkillDistribution(6.8,0.8,4,9),
    },
    metadata=MetadataDistribution(
        education={"Bachelor":0.45,"Master":0.40,"PhD":0.15},
        industries=["AI","Healthcare","Finance","Cloud"],
        remote_probability=0.75,
    ),
    overlaps={
        "Data Scientist":0.20,
        "Data Engineer":0.15,
        "Backend Developer":0.05,
    }
)


BACKEND_DEVELOPER = CareerProfile(
    name="Backend Developer",
    description="Develops scalable APIs and backend services.",
    technical_skills={
        "java_score": SkillDistribution(9.2,0.5,8,10),
        "python_score": SkillDistribution(6.5,0.9,4,9),
        "sql_score": SkillDistribution(8.8,0.6,7,10),
        "database_score": SkillDistribution(8.7,0.7,7,10),
        "cloud_score": SkillDistribution(7.5,0.8,5,10),
        "devops_score": SkillDistribution(7.2,0.9,5,10),
        "machine_learning_score": SkillDistribution(2.5,1.0,1,5),
    },
    soft_skills={
        "communication_score": SkillDistribution(6.2,1.0,4,9),
        "leadership_score": SkillDistribution(5.5,1.0,2,9),
        "problem_solving_score": SkillDistribution(9.0,0.5,8,10),
        "teamwork_score": SkillDistribution(7.3,0.8,5,10),
        "agile_score": SkillDistribution(8.0,0.7,6,10),
    },
    metadata=MetadataDistribution(
        education={"Diploma":0.10,"Bachelor":0.60,"Master":0.30},
        industries=["Finance","Cloud","Healthcare","E-Commerce"],
        remote_probability=0.60,
    ),
    overlaps={
        "Software Engineer":0.20,
        "Full Stack Developer":0.25,
        "Cloud Engineer":0.10,
    }
)


PRODUCT_MANAGER = CareerProfile(
    name="Product Manager",
    description="Leads product vision and strategy.",
    technical_skills={
        "product_management_score": SkillDistribution(9.5,0.4,8,10),
        "business_analysis_score": SkillDistribution(8.8,0.5,7,10),
        "python_score": SkillDistribution(2.5,1.0,1,5),
        "java_score": SkillDistribution(1.8,0.8,1,4),
    },
    soft_skills={
        "communication_score": SkillDistribution(9.3,0.5,8,10),
        "leadership_score": SkillDistribution(9.0,0.5,8,10),
        "problem_solving_score": SkillDistribution(8.5,0.6,7,10),
        "teamwork_score": SkillDistribution(9.0,0.5,8,10),
        "agile_score": SkillDistribution(8.8,0.5,7,10),
    },
    metadata=MetadataDistribution(
        education={"Bachelor":0.50,"Master":0.35,"MBA":0.15},
        industries=["EdTech","Finance","Healthcare","E-Commerce"],
        remote_probability=0.65,
    ),
    overlaps={
        "Business Analyst":0.25,
        "Software Engineer":0.10,
    }
)


CAREER_PROFILES = {
    ML_ENGINEER.name: ML_ENGINEER,
    BACKEND_DEVELOPER.name: BACKEND_DEVELOPER,
    PRODUCT_MANAGER.name: PRODUCT_MANAGER,
}

ALL_CAREERS = [
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

if __name__ == "__main__":
    print(f"Implemented profiles: {len(CAREER_PROFILES)}")
    print("Remaining profiles to implement:")
    for career in ALL_CAREERS:
        if career not in CAREER_PROFILES:
            print(" -", career)
