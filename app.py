"""Flask application for NLP Well Report Analyzer."""

import os
from flask import Flask, request, jsonify, render_template

from nlp_analyzer.models.report_classifier import ReportClassifier
from nlp_analyzer.models.entity_extractor import EntityExtractor
from nlp_analyzer.models.sentiment_analyzer import SentimentAnalyzer
from nlp_analyzer.utils.text_processor import (
    extract_keywords_tfidf, extract_entities, tokenize,
)

app = Flask(__name__)

classifier = ReportClassifier()
entity_extractor = EntityExtractor()
sentiment_analyzer = SentimentAnalyzer()


def _load_models():
    classifier.load()
    entity_extractor.load()
    sentiment_analyzer.load()


_load_models()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "models_loaded": {
            "classifier": classifier.is_trained,
            "entity_extractor": entity_extractor.is_trained,
            "sentiment_analyzer": sentiment_analyzer.is_trained,
        },
    })


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True, silent=True) or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    classification = classifier.predict(text)
    classification_probs = classifier.predict_proba(text)
    entities = entity_extractor.extract(text)
    sentiment = sentiment_analyzer.analyze(text)
    keywords = extract_keywords_tfidf(text, top_n=10)

    return jsonify({
        "classification": {
            "type": classification,
            "probabilities": classification_probs,
        },
        "entities": entities,
        "sentiment": sentiment,
        "keywords": keywords,
        "text_length": len(text),
        "token_count": len(tokenize(text)),
    })


@app.route("/api/extract", methods=["POST"])
def extract():
    data = request.get_json(force=True, silent=True) or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    entities = entity_extractor.extract(text)
    keywords = extract_keywords_tfidf(text, top_n=15)

    return jsonify({
        "entities": entities,
        "keywords": keywords,
    })


@app.route("/api/classify", methods=["POST"])
def classify():
    data = request.get_json(force=True, silent=True) or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    classification = classifier.predict(text)
    probabilities = classifier.predict_proba(text)

    return jsonify({
        "classification": classification,
        "probabilities": probabilities,
    })


@app.route("/api/models", methods=["GET"])
def models():
    return jsonify({
        "classifier": {
            "loaded": classifier.is_trained,
            "types": ["drilling", "completion", "workover", "production"],
        },
        "entity_extractor": {
            "loaded": entity_extractor.is_trained,
            "entity_types": ["well_name", "depths", "formation", "equipment"],
        },
        "sentiment_analyzer": {
            "loaded": sentiment_analyzer.is_trained,
            "labels": ["positive", "negative", "neutral"],
        },
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5017, debug=True)
