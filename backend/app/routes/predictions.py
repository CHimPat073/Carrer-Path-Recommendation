from fastapi import APIRouter, HTTPException

from backend.app.schemas import CareerProfile
from backend.app.services.prediction_service import predict_profile
from ml.inference.predictor import PredictorError


router = APIRouter(prefix="/api/v1", tags=["predictions"])


@router.post("/predict")
def predict(profile: CareerProfile) -> dict:
    """Predict the top career matches for one candidate profile."""
    try:
        return predict_profile(profile.model_dump())
    except PredictorError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error