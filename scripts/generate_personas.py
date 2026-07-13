"""Career persona generation utilities.

This module defines functions that create structured career persona objects
for downstream knowledge base generation. The implementation returns
Python dictionaries and intentionally avoids writing JSON files.
"""

from __future__ import annotations

from typing import Any

from scripts.utils import configure_logging

logger = configure_logging("generate_personas")


def generate_career_personas(career_names: list[str]) -> dict[str, dict[str, Any]]:
    """Generate persona dictionaries for a list of careers.

    Input:
        career_names (list[str]): Careers to describe.

    Output:
        dict[str, dict[str, Any]]: Mapping of career name to persona payload.

    Purpose:
        Create rich, reusable persona metadata for each target profession.
    """
    personas: dict[str, dict[str, Any]] = {}
    for career in career_names:
        personas[career] = _build_persona(career)
    return personas


def _build_persona(career: str) -> dict[str, Any]:
    """Create a single career persona dictionary.

    Input:
        career (str): A normalized career name.

    Output:
        dict[str, Any]: Persona definition for the provided career.

    Purpose:
        Encapsulate the career-specific persona template logic in one place.
    """
    persona_templates: dict[str, dict[str, Any]] = {
        "Software Engineer": {
            "career_description": "Builds reliable software products and services using engineering best practices.",
            "required_technical_skills": ["Python", "Java", "SQL", "Testing"],
            "nice_to_have_skills": ["Cloud", "DevOps", "Machine Learning"],
            "soft_skills": ["Problem Solving", "Communication", "Teamwork"],
            "preferred_education": "Bachelor's degree",
            "experience_range": "2-8 years",
            "salary_band": "$80k-$140k USD",
            "industry": ["Technology", "Finance", "Healthcare"],
            "career_growth": "High",
            "future_demand": "Very High",
        },
        "Backend Developer": {
            "career_description": "Designs backend services, APIs, and data access layers for modern applications.",
            "required_technical_skills": ["Python", "SQL", "Databases", "APIs"],
            "nice_to_have_skills": ["Cloud", "DevOps", "Distributed Systems"],
            "soft_skills": ["Problem Solving", "Communication", "Attention to Detail"],
            "preferred_education": "Bachelor's degree",
            "experience_range": "2-10 years",
            "salary_band": "$85k-$150k USD",
            "industry": ["Technology", "E-commerce", "Fintech"],
            "career_growth": "High",
            "future_demand": "Very High",
        },
        "Frontend Developer": {
            "career_description": "Creates compelling and responsive user interfaces for web and mobile applications.",
            "required_technical_skills": ["JavaScript", "UI Design", "UX Research", "HTML/CSS"],
            "nice_to_have_skills": ["Accessibility", "React", "Testing"],
            "soft_skills": ["Communication", "Creativity", "Collaboration"],
            "preferred_education": "Bachelor's degree",
            "experience_range": "2-8 years",
            "salary_band": "$75k-$130k USD",
            "industry": ["Technology", "Media", "Retail"],
            "career_growth": "Moderate",
            "future_demand": "High",
        },
        "Full Stack Developer": {
            "career_description": "Works across both frontend and backend layers to deliver complete solutions.",
            "required_technical_skills": ["Python", "JavaScript", "SQL", "Cloud"],
            "nice_to_have_skills": ["DevOps", "Testing", "APIs"],
            "soft_skills": ["Problem Solving", "Adaptability", "Teamwork"],
            "preferred_education": "Bachelor's degree",
            "experience_range": "3-10 years",
            "salary_band": "$90k-$155k USD",
            "industry": ["Technology", "Startups", "Consulting"],
            "career_growth": "High",
            "future_demand": "Very High",
        },
        "Mobile App Developer": {
            "career_description": "Builds polished mobile experiences for iOS and Android platforms.",
            "required_technical_skills": ["Mobile Development", "UI Design", "Testing"],
            "nice_to_have_skills": ["Swift", "Kotlin", "Cloud"],
            "soft_skills": ["Creativity", "Communication", "Problem Solving"],
            "preferred_education": "Bachelor's degree",
            "experience_range": "2-8 years",
            "salary_band": "$80k-$140k USD",
            "industry": ["Technology", "Gaming", "E-commerce"],
            "career_growth": "Moderate",
            "future_demand": "High",
        },
        "Data Scientist": {
            "career_description": "Applies statistical and machine learning methods to derive insights from complex data.",
            "required_technical_skills": ["Python", "Machine Learning", "SQL", "Data Analysis"],
            "nice_to_have_skills": ["Deep Learning", "Research", "Cloud"],
            "soft_skills": ["Problem Solving", "Communication", "Curiosity"],
            "preferred_education": "Master's degree",
            "experience_range": "3-10 years",
            "salary_band": "$100k-$180k USD",
            "industry": ["Technology", "Finance", "Healthcare"],
            "career_growth": "Very High",
            "future_demand": "Very High",
        },
        "ML Engineer": {
            "career_description": "Deploys and scales machine learning systems into production environments.",
            "required_technical_skills": ["Python", "Machine Learning", "Deep Learning", "Cloud"],
            "nice_to_have_skills": ["Research", "MLOps", "Data Engineering"],
            "soft_skills": ["Problem Solving", "Collaboration", "Communication"],
            "preferred_education": "Master's degree",
            "experience_range": "3-10 years",
            "salary_band": "$110k-$190k USD",
            "industry": ["Technology", "Finance", "Autonomous Systems"],
            "career_growth": "Very High",
            "future_demand": "Very High",
        },
        "Data Engineer": {
            "career_description": "Designs data pipelines and storage systems that power analytics and products.",
            "required_technical_skills": ["SQL", "Database", "Python", "Cloud"],
            "nice_to_have_skills": ["Spark", "DevOps", "Data Modeling"],
            "soft_skills": ["Problem Solving", "Communication", "Ownership"],
            "preferred_education": "Bachelor's degree",
            "experience_range": "3-10 years",
            "salary_band": "$95k-$170k USD",
            "industry": ["Technology", "Finance", "Retail"],
            "career_growth": "High",
            "future_demand": "Very High",
        },
        "AI Research Engineer": {
            "career_description": "Advances state-of-the-art AI systems through research and experimentation.",
            "required_technical_skills": ["Python", "Machine Learning", "Deep Learning", "Research"],
            "nice_to_have_skills": ["Mathematics", "Statistics", "Cloud"],
            "soft_skills": ["Analytical Thinking", "Curiosity", "Communication"],
            "preferred_education": "PhD",
            "experience_range": "4-12 years",
            "salary_band": "$120k-$220k USD",
            "industry": ["Technology", "Research", "Healthcare"],
            "career_growth": "Very High",
            "future_demand": "High",
        },
        "DevOps Engineer": {
            "career_description": "Improves software delivery reliability through automation and infrastructure tooling.",
            "required_technical_skills": ["Cloud", "DevOps", "Networking", "Automation"],
            "nice_to_have_skills": ["Security", "Kubernetes", "Monitoring"],
            "soft_skills": ["Problem Solving", "Collaboration", "Ownership"],
            "preferred_education": "Bachelor's degree",
            "experience_range": "3-9 years",
            "salary_band": "$95k-$165k USD",
            "industry": ["Technology", "Cloud Services", "Finance"],
            "career_growth": "High",
            "future_demand": "Very High",
        },
        "Cloud Engineer": {
            "career_description": "Designs and operates cloud-native systems and infrastructure services.",
            "required_technical_skills": ["Cloud", "DevOps", "Networking", "Security"],
            "nice_to_have_skills": ["Automation", "Containers", "Monitoring"],
            "soft_skills": ["Problem Solving", "Communication", "Adaptability"],
            "preferred_education": "Bachelor's degree",
            "experience_range": "3-9 years",
            "salary_band": "$100k-$170k USD",
            "industry": ["Technology", "Cloud Services", "Healthcare"],
            "career_growth": "High",
            "future_demand": "Very High",
        },
        "Cyber Security Analyst": {
            "career_description": "Monitors and protects digital systems from security threats and incidents.",
            "required_technical_skills": ["Cybersecurity", "Networking", "Problem Solving"],
            "nice_to_have_skills": ["Cloud", "Threat Analysis", "Automation"],
            "soft_skills": ["Analytical Thinking", "Communication", "Attention to Detail"],
            "preferred_education": "Bachelor's degree",
            "experience_range": "2-8 years",
            "salary_band": "$85k-$150k USD",
            "industry": ["Technology", "Government", "Finance"],
            "career_growth": "High",
            "future_demand": "High",
        },
        "Ethical Hacker": {
            "career_description": "Tests systems for vulnerabilities and strengthens security posture.",
            "required_technical_skills": ["Cybersecurity", "Networking", "Testing"],
            "nice_to_have_skills": ["Cloud", "Automation", "Reverse Engineering"],
            "soft_skills": ["Problem Solving", "Creativity", "Persistence"],
            "preferred_education": "Bachelor's degree",
            "experience_range": "2-8 years",
            "salary_band": "$90k-$160k USD",
            "industry": ["Technology", "Government", "Security"],
            "career_growth": "High",
            "future_demand": "High",
        },
        "Network Engineer": {
            "career_description": "Designs and maintains reliable network infrastructure and connectivity.",
            "required_technical_skills": ["Networking", "Cloud", "Security"],
            "nice_to_have_skills": ["Automation", "Monitoring", "DevOps"],
            "soft_skills": ["Communication", "Problem Solving", "Patience"],
            "preferred_education": "Bachelor's degree",
            "experience_range": "3-9 years",
            "salary_band": "$85k-$150k USD",
            "industry": ["Technology", "Telecommunications", "Government"],
            "career_growth": "Moderate",
            "future_demand": "High",
        },
        "UI/UX Designer": {
            "career_description": "Crafts intuitive, accessible, and user-centered design experiences.",
            "required_technical_skills": ["UI Design", "UX Research", "Communication"],
            "nice_to_have_skills": ["Research", "Design Systems", "Prototyping"],
            "soft_skills": ["Creativity", "Empathy", "Collaboration"],
            "preferred_education": "Bachelor's degree",
            "experience_range": "2-8 years",
            "salary_band": "$75k-$135k USD",
            "industry": ["Technology", "Design", "Retail"],
            "career_growth": "Moderate",
            "future_demand": "High",
        },
        "Product Manager": {
            "career_description": "Aligns business strategy, customer needs, and delivery execution across teams.",
            "required_technical_skills": ["Product Management", "Business Analysis", "Leadership"],
            "nice_to_have_skills": ["Communication", "Agile", "Strategy"],
            "soft_skills": ["Leadership", "Communication", "Negotiation"],
            "preferred_education": "Bachelor's degree",
            "experience_range": "4-10 years",
            "salary_band": "$110k-$180k USD",
            "industry": ["Technology", "Finance", "Healthcare"],
            "career_growth": "Very High",
            "future_demand": "High",
        },
        "Business Analyst": {
            "career_description": "Bridges business goals and technical solutions through analysis and requirements work.",
            "required_technical_skills": ["Business Analysis", "Communication", "SQL"],
            "nice_to_have_skills": ["Product Management", "Data Analysis", "Strategy"],
            "soft_skills": ["Problem Solving", "Communication", "Stakeholder Management"],
            "preferred_education": "Bachelor's degree",
            "experience_range": "2-8 years",
            "salary_band": "$75k-$125k USD",
            "industry": ["Finance", "Healthcare", "Consulting"],
            "career_growth": "Moderate",
            "future_demand": "High",
        },
        "QA Engineer": {
            "career_description": "Ensures software quality through structured testing and automation.",
            "required_technical_skills": ["Testing", "Problem Solving", "Automation"],
            "nice_to_have_skills": ["Python", "DevOps", "CI/CD"],
            "soft_skills": ["Attention to Detail", "Communication", "Teamwork"],
            "preferred_education": "Bachelor's degree",
            "experience_range": "2-7 years",
            "salary_band": "$70k-$120k USD",
            "industry": ["Technology", "Finance", "Healthcare"],
            "career_growth": "Moderate",
            "future_demand": "High",
        },
        "Database Administrator": {
            "career_description": "Maintains databases, performance, and secure data access for enterprise applications.",
            "required_technical_skills": ["Database", "SQL", "Security"],
            "nice_to_have_skills": ["Cloud", "Automation", "Backup Management"],
            "soft_skills": ["Problem Solving", "Communication", "Reliability"],
            "preferred_education": "Bachelor's degree",
            "experience_range": "3-9 years",
            "salary_band": "$85k-$145k USD",
            "industry": ["Technology", "Finance", "Healthcare"],
            "career_growth": "Moderate",
            "future_demand": "High",
        },
        "Game Developer": {
            "career_description": "Builds interactive and entertaining game experiences using software engineering principles.",
            "required_technical_skills": ["Game Development", "Programming", "UI Design"],
            "nice_to_have_skills": ["Graphics", "Physics", "Testing"],
            "soft_skills": ["Creativity", "Teamwork", "Problem Solving"],
            "preferred_education": "Bachelor's degree",
            "experience_range": "2-8 years",
            "salary_band": "$80k-$140k USD",
            "industry": ["Gaming", "Technology", "Entertainment"],
            "career_growth": "Moderate",
            "future_demand": "Moderate",
        },
    }

    template = persona_templates.get(career)
    if template is None:
        logger.warning("No persona template found for %s; using default profile", career)
        return {
            "career_description": f"Profession focused on {career}.",
            "required_technical_skills": [],
            "nice_to_have_skills": [],
            "soft_skills": [],
            "preferred_education": "Bachelor's degree",
            "experience_range": "2-6 years",
            "salary_band": "$70k-$120k USD",
            "industry": ["Technology"],
            "career_growth": "Moderate",
            "future_demand": "High",
        }

    return template
