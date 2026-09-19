from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.routes.catalog import router as catalog_router
from backend.app.routes.predictions import router as predictions_router
from backend.app.routes.recommendations import router as recommendations_router
from ml.inference.predictor import (
    BEST_MODEL_PATH,
    PREPROCESSOR_PATH,
    TARGET_ENCODER_PATH,
)


app = FastAPI(title="CareerPilot-AI API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(predictions_router)
app.include_router(recommendations_router)
app.include_router(catalog_router)

@app.get("/")
def read_root():
    return {"message": "CareerPilot-AI backend is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    artifacts = {
        "model": BEST_MODEL_PATH,
        "preprocessor": PREPROCESSOR_PATH,
        "target_encoder": TARGET_ENCODER_PATH,
    }
    missing = [name for name, path in artifacts.items() if not path.is_file()]
    if missing:
        return {"ready": False, "missing": missing}
    return {"ready": True}
