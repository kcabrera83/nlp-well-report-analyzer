"""Train all models and save to outputs/models/."""

import os
import sys
import json
import time
import re

import spacy
from transformers import pipeline as hf_pipeline
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim.models import Word2Vec

from nlp_analyzer.data_generator import generate_dataset

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "models")

WELL_PATTERN = re.compile(r"[A-Z]-\d+")
FORMATION_WORDS = {"Sandstone", "Limestone", "Shale", "Dolomite"}
MEASUREMENT_PATTERN = re.compile(r"\d+\s*(bbl|psi|ft|md|°F)")


def generate_sentiment_labels(dataset):
    """Auto-label sentiment based on keywords in text."""
    labels = []
    pos_kws = {"successful", "excellent", "good", "improved", "completed", "optimized", "achieved"}
    neg_kws = {"failed", "failure", "problem", "damage", "loss", "lost", "stuck", "severe", "critical", "incident"}

    for item in dataset:
        text_lower = item["text"].lower()
        pos = sum(1 for kw in pos_kws if kw in text_lower)
        neg = sum(1 for kw in neg_kws if kw in text_lower)
        if pos > neg:
            labels.append("positive")
        elif neg > pos:
            labels.append("negative")
        else:
            labels.append("neutral")
    return labels


def build_spacy_ner():
    """Build spaCy NER pipeline for oil & gas entities"""
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        nlp = spacy.blank("en")
        print("  Warning: en_core_web_sm not found, using blank model")

    if "entity_ruler" not in [p.name for p in nlp.pipe_names]:
        ruler = nlp.add_pipe("entity_ruler")
        patterns = [
            {"label": "WELL", "pattern": [{"TEXT": {"REGEX": "^[A-Z]-\\d+$"}}]},
            {"label": "FORMATION", "pattern": [{"TEXT": {"IN": list(FORMATION_WORDS)}}]},
            {"label": "MEASUREMENT", "pattern": [{"TEXT": {"REGEX": "\\d+\\s*(bbl|psi|ft|md|°F)"}}]},
        ]
        ruler.add_patterns(patterns)
    return nlp


def extract_entities_spacy(nlp, text):
    """Extract entities using spaCy"""
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append({"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char})

    for match in WELL_PATTERN.finditer(text):
        entities.append({"text": match.group(), "label": "WELL", "start": match.start(), "end": match.end()})
    for match in MEASUREMENT_PATTERN.finditer(text):
        entities.append({"text": match.group(), "label": "MEASUREMENT", "start": match.start(), "end": match.end()})

    return entities


def train_word2vec(corpus):
    """Build Word2Vec for domain-specific embeddings"""
    tokenized = [word_tokenize(text.lower()) for text in corpus]
    model = Word2Vec(sentences=tokenized, vector_size=100, window=5, min_count=2, workers=4)
    return model


def classify_report(texts, labels):
    """Classify reports using rule-based approach with NLP features"""
    type_keywords = {
        "drilling": ["drill", "bit", "mud", "rotation", "penetration", "depth", "borehole"],
        "completion": ["casing", "cement", "perforation", "completion", "tubing", "packer"],
        "workover": ["workover", "repair", "stimulation", "acidize", "recompletion"],
        "production": ["production", "pump", "flow", "lift", "wellhead", "separator"],
    }

    results = []
    for text in texts:
        text_lower = text.lower()
        scores = {}
        for rtype, keywords in type_keywords.items():
            scores[rtype] = sum(1 for kw in keywords if kw in text_lower)
        best = max(scores, key=scores.get) if max(scores.values()) > 0 else "production"
        results.append(best)
    return results


def analyze_sentiment(texts, labels):
    """Rule-based sentiment analysis"""
    pos_kws = {"successful", "excellent", "good", "improved", "completed", "optimized", "achieved"}
    neg_kws = {"failed", "failure", "problem", "damage", "loss", "lost", "stuck", "severe", "critical", "incident"}

    results = []
    for text in texts:
        text_lower = text.lower()
        pos = sum(1 for kw in pos_kws if kw in text_lower)
        neg = sum(1 for kw in neg_kws if kw in text_lower)
        if pos > neg:
            results.append({"label": "positive", "score": pos / (pos + neg + 1)})
        elif neg > pos:
            results.append({"label": "negative", "score": neg / (pos + neg + 1)})
        else:
            results.append({"label": "neutral", "score": 0.5})
    return results


def main():
    print("=" * 70)
    print("  NLP WELL REPORT ANALYZER - MODEL TRAINING")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Generate synthetic training dataset
    print("\n[1/6] Generating synthetic training dataset...")
    t0 = time.time()
    dataset = generate_dataset(n=600)
    texts = [item["text"] for item in dataset]
    labels = [item["structured"]["report_type"] for item in dataset]
    sentiment_labels = generate_sentiment_labels(dataset)
    print(f"       Generated {len(dataset)} reports in {time.time()-t0:.2f}s")
    types_dist = {}
    for l in labels:
        types_dist[l] = types_dist.get(l, 0) + 1
    print(f"       Class distribution: {types_dist}")

    # Step 2: Build spaCy NER pipeline
    print("\n[2/6] Building spaCy NER pipeline...")
    t0 = time.time()
    nlp = build_spacy_ner()
    sample_entities = extract_entities_spacy(nlp, texts[0])
    print(f"       spaCy NER ready in {time.time()-t0:.2f}s")
    print(f"       Sample entities from first doc: {len(sample_entities)} found")

    # Step 3: Build Word2Vec embeddings
    print("\n[3/6] Building Word2Vec embeddings...")
    t0 = time.time()
    w2v_model = train_word2vec(texts)
    print(f"       Word2Vec vocabulary: {len(w2v_model.wv)} tokens in {time.time()-t0:.2f}s")

    # Step 4: Train report classifier
    print("\n[4/6] Training Report Classifier (rule-based + NLP features)...")
    t0 = time.time()
    predictions = classify_report(texts, labels)
    correct = sum(1 for p, l in zip(predictions, labels) if p == l)
    train_acc = correct / len(labels)
    print(f"       Training accuracy: {train_acc*100:.1f}%")
    print(f"       Training time: {time.time()-t0:.2f}s")

    # Step 5: Train sentiment analyzer
    print("\n[5/6] Training Sentiment Analyzer (rule-based + NLP)...")
    t0 = time.time()
    sent_results = analyze_sentiment(texts, sentiment_labels)
    sent_correct = sum(1 for r, l in zip(sent_results, sentiment_labels) if r["label"] == l)
    sent_acc = sent_correct / len(sentiment_labels)
    print(f"       Training accuracy: {sent_acc*100:.1f}%")
    print(f"       Training time: {time.time()-t0:.2f}s")

    # Step 6: Save models
    print("\n[6/6] Saving models...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    nlp.to_disk(os.path.join(OUTPUT_DIR, "spacy_ner"))
    w2v_model.save(os.path.join(OUTPUT_DIR, "word2vec.model"))

    model_info = {
        "spacy_ner": {"path": os.path.join(OUTPUT_DIR, "spacy_ner"), "entity_types": ["WELL", "FORMATION", "MEASUREMENT"]},
        "word2vec": {"path": os.path.join(OUTPUT_DIR, "word2vec.model"), "vocab_size": len(w2v_model.wv)},
        "classifier_accuracy": train_acc,
        "sentiment_accuracy": sent_acc,
    }
    with open(os.path.join(OUTPUT_DIR, "model_info.json"), "w") as f:
        json.dump(model_info, f, indent=2)

    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE - ALL MODELS SAVED")
    print("=" * 70)
    print(f"\n  Classifier accuracy:     {train_acc*100:.1f}%")
    print(f"  Sentiment accuracy:      {sent_acc*100:.1f}%")
    print(f"  Word2Vec vocab:          {len(w2v_model.wv)} tokens")
    print(f"  spaCy NER:               Saved to {os.path.join(OUTPUT_DIR, 'spacy_ner')}")
    print(f"\n  Model files saved to: {OUTPUT_DIR}")
    print()


if __name__ == "__main__":
    main()
