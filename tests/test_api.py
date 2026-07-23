import pytest


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert "models_loaded" in data


def test_models(client):
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.get_json()
    assert "classifier" in data
    assert "entity_extractor" in data
    assert "sentiment_analyzer" in data


def test_api_docs(client):
    response = client.get("/api/docs")
    assert response.status_code == 200
    data = response.get_json()
    assert data["openapi"] == "3.0.0"
    assert "/api/analyze" in data["paths"]
    assert "/api/extract" in data["paths"]
    assert "/api/classify" in data["paths"]


def test_analyze_valid(client, sample_well_text):
    response = client.post("/api/analyze", json={"text": sample_well_text})
    assert response.status_code == 200
    data = response.get_json()
    assert "classification" in data
    assert "type" in data["classification"]
    assert "probabilities" in data["classification"]
    assert "entities" in data
    assert "sentiment" in data
    assert "keywords" in data
    assert "text_length" in data
    assert "token_count" in data


def test_analyze_empty_text(client):
    response = client.post("/api/analyze", json={"text": ""})
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data


def test_extract_valid(client, sample_well_text):
    response = client.post("/api/extract", json={"text": sample_well_text})
    assert response.status_code == 200
    data = response.get_json()
    assert "entities" in data
    assert "keywords" in data


def test_extract_empty_text(client):
    response = client.post("/api/extract", json={"text": ""})
    assert response.status_code == 400


def test_classify_valid(client, sample_well_text):
    response = client.post("/api/classify", json={"text": sample_well_text})
    assert response.status_code == 200
    data = response.get_json()
    assert "classification" in data
    assert "probabilities" in data
    assert isinstance(data["classification"], str)


def test_classify_empty_text(client):
    response = client.post("/api/classify", json={"text": ""})
    assert response.status_code == 400


def test_analyze_completion_report(client):
    text = "Completion operations were successful. The well was perforated at 6200 ft in the Target formation. Flow test showed 500 bbl/day oil rate."
    response = client.post("/api/analyze", json={"text": text})
    assert response.status_code == 200
    data = response.get_json()
    assert data["classification"]["type"] in ["drilling", "completion", "workover", "production"]
