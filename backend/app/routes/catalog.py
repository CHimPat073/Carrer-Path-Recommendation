from fastapi import APIRouter, HTTPException

from backend.app.services.recommendation_service import (
    get_career,
    get_careers,
    get_roadmap,
)


router = APIRouter(prefix="/api/v1", tags=["catalog"])


@router.get("/careers")
def careers() -> dict:
    """List the careers available in the knowledge base."""
    return {"careers": get_careers()}


@router.get("/careers/{career_name}")
def career(career_name: str) -> dict:
    """Return one career from the knowledge base."""
    result = get_career(career_name)
    if result is None:
        raise HTTPException(status_code=404, detail="Career not found")
    return result


@router.get("/roadmaps/{career_name}")
def roadmap(career_name: str) -> dict:
    """Return the career progression roadmap from the knowledge base."""
    result = get_roadmap(career_name)
    if result is None:
        raise HTTPException(status_code=404, detail="Roadmap not found")
    return result