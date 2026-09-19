# CareerPilot-AI Deployment Roadmap

## Goal

Build a production-ready CareerPilot-AI service with a FastAPI backend, Docker deployment, and a frontend interface for career prediction, recommendations, skill gaps, and learning roadmaps.

## Current Constraints

- Model artifacts are stored locally in `models/` and are intentionally excluded from Git.
- The backend currently exposes only a basic root endpoint.
- The prediction module expects these files at runtime:
  - `models/best_model.pkl`
  - `models/preprocessor.joblib`
  - `models/target_encoder.joblib`
- The knowledge base in `knowledge_base/` is required by recommendation and roadmap features.

## Phase 1: Backend Foundation

- [ ] Add backend configuration using environment variables.
- [ ] Add a clear application package structure for routes, schemas, and services.
- [ ] Update `backend/requirements.txt` with the inference dependencies:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `joblib`
- [ ] Configure CORS for the future frontend.
- [ ] Add application-wide error handling and structured logging.

## Phase 2: Model Runtime

- [ ] Create a model service that loads the model bundle once during application startup.
- [ ] Validate that all required model files exist.
- [ ] Keep model loading out of individual request handlers.
- [ ] Add model version information to the readiness response.
- [ ] Document how developers place the model bundle in the local `models/` directory.

## Phase 3: Initial API

Implement and document these endpoints:

- [ ] `GET /health` for process health.
- [ ] `GET /ready` for model and dependency readiness.
- [ ] `POST /api/v1/predict` for top career predictions.
- [ ] `POST /api/v1/recommendations` for confidence, explanations, and skill gaps.
- [ ] `GET /api/v1/careers` for available career profiles.
- [ ] `GET /api/v1/careers/{career_name}` for career details.
- [ ] `GET /api/v1/roadmaps/{career_name}` for learning roadmaps.

Routes should call service-layer functions and should not contain model or knowledge-base logic directly.

## Phase 4: Validation and Testing

- [ ] Define Pydantic request and response schemas.
- [ ] Test valid prediction requests.
- [ ] Test missing fields and invalid categories.
- [ ] Test numeric range validation.
- [ ] Test missing model files.
- [ ] Test readiness before and after model loading.
- [ ] Test recommendation and roadmap responses.
- [ ] Add an API smoke-test command for local development.

## Phase 5: Docker Development Setup

- [ ] Create a production-oriented Dockerfile for the FastAPI service.
- [ ] Install only backend and inference dependencies in the image.
- [ ] Copy `backend/`, `ml/`, and `knowledge_base/` into the image.
- [ ] Do not copy training datasets, reports, plots, or experiment logs.
- [ ] Run the container as a non-root user.
- [ ] Mount local models read-only during development:

```yaml
volumes:
  - ./models:/app/models:ro
```

- [ ] Add a Docker health check using `/health` or `/ready`.
- [ ] Add a local Docker Compose service for the API.

## Phase 6: Model Delivery

For production, keep model artifacts outside Git and deliver a versioned model bundle through one of these options:

1. Object storage such as Azure Blob Storage.
2. A model registry.
3. A deployment volume or managed persistent storage.

The deployment process should download or mount the selected model version into `/app/models/` before the API starts.

Required runtime files:

```text
/app/models/best_model.pkl
/app/models/preprocessor.joblib
/app/models/target_encoder.joblib
```

## Phase 7: Frontend Integration

After the API contract is stable:

- [ ] Create the frontend application structure.
- [ ] Add the career assessment form.
- [ ] Connect the form to `POST /api/v1/predict`.
- [ ] Display top predictions and confidence.
- [ ] Display skill gaps and explanations.
- [ ] Display the learning roadmap and resources.
- [ ] Add loading, validation-error, API-error, and empty states.
- [ ] Configure the frontend API URL through an environment variable.

## Phase 8: Production Deployment

- [ ] Choose a hosting platform such as Azure Container Apps or Azure App Service.
- [ ] Create a private model-storage location.
- [ ] Configure environment variables and secrets through the platform.
- [ ] Build and scan the Docker image in CI.
- [ ] Run automated tests before deployment.
- [ ] Add HTTPS and restrict database access if MongoDB is used.
- [ ] Configure application logs and monitoring.
- [ ] Add a rollback procedure based on image and model versions.
- [ ] Verify `/health`, `/ready`, and prediction behavior after deployment.

## Recommended Implementation Order

1. Backend requirements and configuration.
2. Model loading service.
3. Health and readiness endpoints.
4. Prediction schemas and `POST /api/v1/predict`.
5. API tests.
6. Dockerfile and Docker Compose development setup.
7. Recommendation and roadmap endpoints.
8. Frontend interface.
9. External model storage and production deployment.
10. CI/CD, monitoring, and rollback.

## Definition of Done

The first production milestone is complete when:

- The API starts without errors when the model bundle is mounted.
- `/health` returns a successful response.
- `/ready` confirms that the model bundle is available.
- `/api/v1/predict` returns validated top career predictions.
- The API runs in Docker with models mounted read-only.
- Automated tests pass.
- The frontend can submit an assessment and display the prediction response.
