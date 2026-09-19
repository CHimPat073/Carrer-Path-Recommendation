from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_prediction_rejects_an_empty_profile() -> None:
    response = client.post("/api/v1/predict", json={})

    assert response.status_code == 422