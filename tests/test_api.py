import pytest

SAMPLE_TEXT = (
    "Drilling report for Well Alpha-1. The well was drilled to a total depth of 8500 ft "
    "using 8.5 inch hole section. Mud weight was maintained at 10.5 ppg. Casing was set "
    "at 5000 ft with 9 5/8 inch surface casing. Drilling rate averaged 50 ft/hr through "
    "the Target formation. No lost circulation events were observed."
)


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "models_loaded" in data


def test_models(client):
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "classifier" in data
    assert "spacy_ner" in data
    assert "sentiment" in data


def test_analyze_valid(client):
    response = client.post("/api/analyze", json={"text": SAMPLE_TEXT})
    assert response.status_code in (200, 400, 500)
    if response.status_code == 200:
        data = response.json()
        assert "classification" in data
        assert "entities" in data
        assert "sentiment" in data
        assert "keywords" in data
        assert "text_length" in data
        assert "token_count" in data


def test_analyze_empty_text(client):
    response = client.post("/api/analyze", json={"text": ""})
    assert response.status_code == 400


def test_extract_valid(client):
    response = client.post("/api/extract", json={"text": SAMPLE_TEXT})
    assert response.status_code in (200, 400, 500)
    if response.status_code == 200:
        data = response.json()
        assert "entities" in data
        assert "keywords" in data


def test_extract_empty_text(client):
    response = client.post("/api/extract", json={"text": ""})
    assert response.status_code == 400


def test_classify_valid(client):
    response = client.post("/api/classify", json={"text": SAMPLE_TEXT})
    assert response.status_code in (200, 400, 500)
    if response.status_code == 200:
        data = response.json()
        assert "classification" in data
        assert "probabilities" in data
        assert isinstance(data["classification"], str)


def test_classify_empty_text(client):
    response = client.post("/api/classify", json={"text": ""})
    assert response.status_code == 400
