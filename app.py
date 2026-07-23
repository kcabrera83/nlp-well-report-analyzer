"""FastAPI application for NLP Well Report Analyzer."""

import os
import json
import re
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

import spacy
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim.models import Word2Vec

app = FastAPI(
    title="NLP Well Report Analyzer",
    description="NLP analysis of well reports for Oil & Gas",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "models")

WELL_PATTERN = re.compile(r"[A-Z]-\d+")
FORMATION_WORDS = {"Sandstone", "Limestone", "Shale", "Dolomite"}
MEASUREMENT_PATTERN = re.compile(r"\d+\s*(bbl|psi|ft|md|°F)")

TYPE_KEYWORDS = {
    "drilling": ["drill", "bit", "mud", "rotation", "penetration", "depth", "borehole"],
    "completion": ["casing", "cement", "perforation", "completion", "tubing", "packer"],
    "workover": ["workover", "repair", "stimulation", "acidize", "recompletion"],
    "production": ["production", "pump", "flow", "lift", "wellhead", "separator"],
}

POS_KEYWORDS = {"successful", "excellent", "good", "improved", "completed", "optimized", "achieved"}
NEG_KEYWORDS = {"failed", "failure", "problem", "damage", "loss", "lost", "stuck", "severe", "critical", "incident"}

models = {}


def _load_models():
    spacy_path = os.path.join(OUTPUT_DIR, "spacy_ner")
    w2v_path = os.path.join(OUTPUT_DIR, "word2vec.model")

    if os.path.exists(spacy_path):
        models["nlp"] = spacy.load(spacy_path)
        print(f"  Loaded spaCy NER from {spacy_path}")
    else:
        try:
            models["nlp"] = spacy.load("en_core_web_sm")
        except OSError:
            models["nlp"] = spacy.blank("en")
        print("  Warning: Using fallback spaCy model")

    if os.path.exists(w2v_path):
        models["word2vec"] = Word2Vec.load(w2v_path)
        print(f"  Loaded Word2Vec from {w2v_path}")
    else:
        models["word2vec"] = None
        print("  Warning: Word2Vec model not found")


def extract_entities(text):
    """Extract entities using spaCy NER + regex patterns"""
    nlp = models.get("nlp")
    entities = []
    if nlp:
        doc = nlp(text)
        for ent in doc.ents:
            entities.append({"text": ent.text, "label": ent.label_})
    for match in WELL_PATTERN.finditer(text):
        entities.append({"text": match.group(), "label": "WELL"})
    for match in MEASUREMENT_PATTERN.finditer(text):
        entities.append({"text": match.group(), "label": "MEASUREMENT"})
    return entities


def classify_report(text):
    """Classify report type using keyword features"""
    text_lower = text.lower()
    scores = {}
    for rtype, keywords in TYPE_KEYWORDS.items():
        scores[rtype] = sum(1 for kw in keywords if kw in text_lower)
    best = max(scores, key=scores.get) if max(scores.values()) > 0 else "production"
    total = sum(scores.values()) + 1
    probs = {k: round(v / total, 4) for k, v in scores.items()}
    return best, probs


def analyze_sentiment(text):
    """Sentiment analysis using keyword matching"""
    text_lower = text.lower()
    pos = sum(1 for kw in POS_KEYWORDS if kw in text_lower)
    neg = sum(1 for kw in NEG_KEYWORDS if kw in text_lower)
    total = pos + neg + 1
    if pos > neg:
        return {"label": "positive", "score": round(pos / total, 4)}
    elif neg > pos:
        return {"label": "negative", "score": round(neg / total, 4)}
    return {"label": "neutral", "score": 0.5}


def extract_keywords(text, top_n=10):
    """Extract keywords using tokenization"""
    tokens = word_tokenize(text.lower())
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for", "of", "and", "or", "with", "by", "from"}
    filtered = [t for t in tokens if len(t) > 2 and t.isalpha() and t not in stop_words]
    freq = {}
    for t in filtered:
        freq[t] = freq.get(t, 0) + 1
    sorted_kw = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [kw for kw, _ in sorted_kw[:top_n]]


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
    try:
        _load_models()
    except Exception as e:
        print(f"[WARN] Error loading models: {e}")


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "models_loaded": {
            "spacy_ner": models.get("nlp") is not None,
            "word2vec": models.get("word2vec") is not None,
        },
        "version": "2.0.0",
    }


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(request: TextRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    classification, probabilities = classify_report(text)
    entities = extract_entities(text)
    sentiment = analyze_sentiment(text)
    keywords = extract_keywords(text, top_n=10)

    return AnalyzeResponse(
        classification={"type": classification, "probabilities": probabilities},
        entities=entities,
        sentiment=sentiment,
        keywords=keywords,
        text_length=len(text),
        token_count=len(word_tokenize(text)),
    )


@app.post("/api/extract", response_model=ExtractResponse)
async def extract(request: TextRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    entities = extract_entities(text)
    keywords = extract_keywords(text, top_n=15)

    return ExtractResponse(entities=entities, keywords=keywords)


@app.post("/api/classify", response_model=ClassifyResponse)
async def classify(request: TextRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    classification, probabilities = classify_report(text)
    return ClassifyResponse(classification=classification, probabilities=probabilities)


@app.get("/api/models")
async def models_info():
    info = {
        "spacy_ner": {
            "loaded": models.get("nlp") is not None,
            "entity_types": ["WELL", "FORMATION", "MEASUREMENT"],
        },
        "word2vec": {
            "loaded": models.get("word2vec") is not None,
        },
        "classifier": {
            "type": "Rule-based + NLP features",
            "types": ["drilling", "completion", "workover", "production"],
        },
        "sentiment": {
            "type": "Keyword-based + NLP preprocessing",
            "labels": ["positive", "negative", "neutral"],
        },
    }
    if models.get("word2vec"):
        info["word2vec"]["vocab_size"] = len(models["word2vec"].wv)
    return info


@app.get("/api/docs")
async def api_docs():
    return {
        "openapi": "3.0.0",
        "info": {"title": "NLP Well Report Analyzer", "version": "2.0.0"},
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
