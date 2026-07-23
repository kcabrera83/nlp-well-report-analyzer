import pytest
import os
import json

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "models")


def test_classifier_loads():
    path = os.path.join(OUTPUT_DIR, "classifier.pkl")
    if not os.path.exists(path):
        pytest.skip("classifier.pkl not found - run train.py first")
    assert os.path.getsize(path) > 0


def test_entity_extractor_loads():
    path = os.path.join(OUTPUT_DIR, "entity_extractor.pkl")
    if not os.path.exists(path):
        pytest.skip("entity_extractor.pkl not found - run train.py first")
    assert os.path.getsize(path) > 0


def test_sentiment_loads():
    path = os.path.join(OUTPUT_DIR, "sentiment.pkl")
    if not os.path.exists(path):
        pytest.skip("sentiment.pkl not found - run train.py first")
    assert os.path.getsize(path) > 0


def test_training_data_exists():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "training_data.json")
    if not os.path.exists(path):
        pytest.skip("training_data.json not found")
    with open(path) as f:
        data = json.load(f)
    assert len(data) > 0
    assert "text" in data[0]
    assert "structured" in data[0]


def test_training_data_has_valid_types():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "training_data.json")
    if not os.path.exists(path):
        pytest.skip("training_data.json not found")
    with open(path) as f:
        data = json.load(f)
    valid_types = {"drilling", "completion", "workover", "production"}
    for item in data:
        assert item["structured"]["report_type"] in valid_types
