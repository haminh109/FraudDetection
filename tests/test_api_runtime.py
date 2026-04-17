from fastapi.testclient import TestClient

from src import api


client = TestClient(api.app)


def test_health_reports_raw_pipeline_status():
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"ok", "degraded"}
    assert "model_ready" in payload
    assert "raw_pipeline_ready" in payload


def test_predict_returns_503_when_model_artifact_is_missing(monkeypatch):
    monkeypatch.setattr(api, "artifact", None)
    monkeypatch.setattr(api, "model", None)
    monkeypatch.setattr(api, "artifact_error", "mock missing model")

    response = client.post(
        "/predict",
        json={"records": [{"TransactionAmt": 100.0}]},
    )

    assert response.status_code == 503
    assert "Model artifact is unavailable" in response.json()["detail"]


def test_predict_raw_returns_503_when_pipeline_artifact_is_missing(monkeypatch):
    monkeypatch.setattr(api, "raw_pipeline", None)
    monkeypatch.setattr(api, "raw_pipeline_error", "mock missing raw pipeline")

    response = client.post(
        "/predict_raw",
        json={"records": [{"TransactionAmt": 100.0}]},
    )

    assert response.status_code == 503
    assert "Raw prediction pipeline is unavailable" in response.json()["detail"]
