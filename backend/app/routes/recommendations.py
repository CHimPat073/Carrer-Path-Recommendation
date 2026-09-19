from fastapi import APIRouter, HTTPException

from backend.app.schemas import CareerProfile
from backend.app.services.recommendation_service import build_recommendation
from ml.inference.predictor import PredictorError


router = APIRouter(prefix="/api/v1", tags=["recommendations"])


@router.post("/recommendations")
def recommendations(profile: CareerProfile) -> dict:
    """Return the recommended career with confidence and explanations."""
    try:
        return build_recommendation(profile.model_dump())
    except (PredictorError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error