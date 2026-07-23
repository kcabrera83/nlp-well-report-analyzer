import pytest
import os

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "models")


def test_outputs_directory_exists():
    assert os.path.exists(OUTPUT_DIR)


def test_model_files_exist():
    model_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith((".pkl", ".joblib", ".h5", ".pt"))]
    assert len(model_files) > 0


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
