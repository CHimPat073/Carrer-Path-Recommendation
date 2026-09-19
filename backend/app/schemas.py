from typing import Literal

from pydantic import BaseModel, Field


class CareerProfile(BaseModel):
    """The candidate fields expected by the trained model."""

    years_experience: float = Field(ge=0, le=60)
    projects_completed: float = Field(ge=0, le=500)
    certifications: float = Field(ge=0, le=50)

    python_score: float = Field(ge=0, le=10)
    java_score: float = Field(ge=0, le=10)
    javascript_score: float = Field(ge=0, le=10)
    sql_score: float = Field(ge=0, le=10)
    machine_learning_score: float = Field(ge=0, le=10)
    deep_learning_score: float = Field(ge=0, le=10)
    cloud_score: float = Field(ge=0, le=10)
    devops_score: float = Field(ge=0, le=10)
    cybersecurity_score: float = Field(ge=0, le=10)
    data_analysis_score: float = Field(ge=0, le=10)
    database_score: float = Field(ge=0, le=10)
    networking_score: float = Field(ge=0, le=10)
    mobile_score: float = Field(ge=0, le=10)
    game_dev_score: float = Field(ge=0, le=10)
    testing_score: float = Field(ge=0, le=10)
    business_analysis_score: float = Field(ge=0, le=10)
    product_management_score: float = Field(ge=0, le=10)
    ui_design_score: float = Field(ge=0, le=10)
    ux_research_score: float = Field(ge=0, le=10)
    communication_score: float = Field(ge=0, le=10)
    leadership_score: float = Field(ge=0, le=10)
    problem_solving_score: float = Field(ge=0, le=10)
    teamwork_score: float = Field(ge=0, le=10)
    agile_score: float = Field(ge=0, le=10)
    research_score: float = Field(ge=0, le=10)

    salary_band: float = Field(ge=0)
    career_growth_score: float = Field(ge=0, le=10)
    job_satisfaction: float = Field(ge=0, le=10)
    work_hours_per_week: float = Field(ge=0, le=80)

    education_level: Literal["High School", "Associate", "Bachelor", "Master", "PhD"]
    remote_preference: Literal["Hybrid", "On-site", "Remote"]
    country: Literal["Australia", "Canada", "Germany", "India", "UK", "USA"]
    industry: Literal[
        "Consulting", "Education", "Finance", "Gaming", "Healthcare", "Retail", "Technology"
    ]
    employment_type: Literal["Contract", "Freelance", "Full-time", "Part-time"]