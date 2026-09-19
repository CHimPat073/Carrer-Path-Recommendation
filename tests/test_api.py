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


def test_careers_endpoint_returns_catalog() -> None:
    response = client.get("/api/v1/careers")

    assert response.status_code == 200
    assert len(response.json()["careers"]) == 20


def test_roadmap_endpoint_returns_known_career_roadmap() -> None:
    response = client.get("/api/v1/roadmaps/Software%20Engineer")

    assert response.status_code == 200
    assert response.json()["career"] == "Software Engineer"