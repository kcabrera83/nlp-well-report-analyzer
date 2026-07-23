"""FastAPI application for NLP Well Report Analyzer."""

import os
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from nlp_analyzer.models.report_classifier import ReportClassifier
from nlp_analyzer.models.entity_extractor import EntityExtractor
from nlp_analyzer.models.sentiment_analyzer import SentimentAnalyzer
from nlp_analyzer.utils.text_processor import (
    extract_keywords_tfidf, extract_entities, tokenize,
)

app = FastAPI(
    title="NLP Well Report Analyzer",
    description="NLP analysis of well reports for Oil & Gas",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)

classifier = ReportClassifier()
entity_extractor = EntityExtractor()
sentiment_analyzer = SentimentAnalyzer()


def _load_models():
    classifier.load()
    entity_extractor.load()
    sentiment_analyzer.load()


class TextRequest(BaseModel):
    text: str


class AnalyzeResponse(BaseModel):
    classification: Dict[str, Any]
    entities: Any
    sentiment: Any
    keywords: list
    text_length: int
    token_count: int


class ExtractResponse(BaseModel):
    entities: Any
    keywords: list


class ClassifyResponse(BaseModel):
    classification: str
    probabilities: Dict[str, float]


@app.on_event("startup")
async def startup_event():
    _load_models()


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "models_loaded": {
            "classifier": classifier.is_trained,
            "entity_extractor": entity_extractor.is_trained,
            "sentiment_analyzer": sentiment_analyzer.is_trained,
        },
    }


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(request: TextRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    classification = classifier.predict(text)
    classification_probs = classifier.predict_proba(text)
    entities = entity_extractor.extract(text)
    sentiment = sentiment_analyzer.analyze(text)
    keywords = extract_keywords_tfidf(text, top_n=10)

    return AnalyzeResponse(
        classification={
            "type": classification,
            "probabilities": classification_probs,
        },
        entities=entities,
        sentiment=sentiment,
        keywords=keywords,
        text_length=len(text),
        token_count=len(tokenize(text)),
    )


@app.post("/api/extract", response_model=ExtractResponse)
async def extract(request: TextRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    entities = entity_extractor.extract(text)
    keywords = extract_keywords_tfidf(text, top_n=15)

    return ExtractResponse(entities=entities, keywords=keywords)


@app.post("/api/classify", response_model=ClassifyResponse)
async def classify(request: TextRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    classification = classifier.predict(text)
    probabilities = classifier.predict_proba(text)

    return ClassifyResponse(classification=classification, probabilities=probabilities)


@app.get("/api/models")
async def models_info():
    return {
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
    }


@app.get("/api/docs")
async def api_docs():
    return {
        "openapi": "3.0.0",
        "info": {"title": "NLP Well Report Analyzer", "version": "1.0.0"},
        "paths": {
            "/api/health": {"get": {"summary": "Health check"}},
            "/api/models": {"get": {"summary": "Model info"}},
            "/api/analyze": {"post": {"summary": "Full NLP analysis of well report text"}},
            "/api/extract": {"post": {"summary": "Extract entities and keywords"}},
            "/api/classify": {"post": {"summary": "Classify report type"}},
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5017)

