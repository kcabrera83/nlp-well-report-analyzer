"""Train all models and save to outputs/models/."""

import os
import sys
import json
import random
import time

from nlp_analyzer.data_generator import generate_dataset, generate_report
from nlp_analyzer.models.report_classifier import ReportClassifier
from nlp_analyzer.models.entity_extractor import EntityExtractor
from nlp_analyzer.models.sentiment_analyzer import SentimentAnalyzer


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


def main():
    print("=" * 70)
    print("  NLP WELL REPORT ANALYZER - MODEL TRAINING")
    print("=" * 70)

    os.makedirs("outputs/models", exist_ok=True)

    print("\n[1/5] Generating synthetic training dataset...")
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
    sent_dist = {}
    for l in sentiment_labels:
        sent_dist[l] = sent_dist.get(l, 0) + 1
    print(f"       Sentiment distribution: {sent_dist}")

    print("\n[2/5] Training Report Classifier...")
    t0 = time.time()
    classifier = ReportClassifier()
    classifier.train(texts, labels)
    train_correct = 0
    for item in dataset:
        pred = classifier.predict(item["text"])
        if pred == item["structured"]["report_type"]:
            train_correct += 1
    train_acc = train_correct / len(dataset)
    classifier.save()
    print(f"       Training accuracy: {train_acc*100:.1f}%")
    print(f"       Training time: {time.time()-t0:.2f}s")
    print(f"       Model saved to {classifier.model_path}")

    print("\n[3/5] Training Entity Extractor...")
    t0 = time.time()
    entity_extractor = EntityExtractor()
    entity_extractor.train(texts)
    entity_extractor.save()
    print(f"       Vocabulary size: {len(entity_extractor.entity_freq.get('well_names', {}))} well names")
    print(f"                         {len(entity_extractor.entity_freq.get('formations', {}))} formations")
    print(f"                         {len(entity_extractor.entity_freq.get('equipment', {}))} equipment types")
    print(f"       Training time: {time.time()-t0:.2f}s")

    print("\n[4/5] Training Sentiment Analyzer...")
    t0 = time.time()
    sentiment_analyzer = SentimentAnalyzer()
    sentiment_analyzer.train(texts, sentiment_labels)
    train_correct = 0
    for item, sl in zip(dataset, sentiment_labels):
        result = sentiment_analyzer.analyze(item["text"])
        if result["label"] == sl:
            train_correct += 1
    sent_acc = train_correct / len(dataset)
    sentiment_analyzer.save()
    print(f"       Training accuracy: {sent_acc*100:.1f}%")
    print(f"       Training time: {time.time()-t0:.2f}s")

    print("\n[5/5] Saving training data...")
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/training_data.json", "w", encoding="utf-8") as f:
        json.dump(dataset[:20], f, indent=2, ensure_ascii=False)
    print("       Saved 20 sample records to outputs/training_data.json")

    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE - ALL MODELS SAVED")
    print("=" * 70)
    print(f"\n  Classifier accuracy:     {train_acc*100:.1f}%")
    print(f"  Sentiment accuracy:      {sent_acc*100:.1f}%")
    print(f"  Entity extractor:        Trained on {len(texts)} documents")
    print(f"\n  Model files:")
    print(f"    - outputs/models/classifier.pkl")
    print(f"    - outputs/models/entity_extractor.pkl")
    print(f"    - outputs/models/sentiment.pkl")
    print()


if __name__ == "__main__":
    main()
