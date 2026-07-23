"""Sentiment analysis for report tone: positive, negative, neutral."""

import os
import pickle
from collections import Counter

from nlp_analyzer.utils.text_processor import tokenize


POSITIVE_WORDS = {
    "successful", "success", "optimized", "excellent", "good", "improved",
    "positive", "efficient", "completed", "achieved", "exceeded", "production",
    "high", "best", "stable", "normal", "clean", "smooth", "effective",
    "productive", "profitable", "safe", "stable", "optimal", "favorable",
    "increased", "gained", "improved", "beneficial", "satisfactory", "promising",
}

NEGATIVE_WORDS = {
    "failed", "failure", "problem", "damage", "damaged", "loss", "lost",
    "poor", "bad", "stuck", "broken", "defect", "incident", "accident",
    "severe", "critical", "dangerous", "hazard", "issue", "issues",
    "complication", "abandon", "abandoned", "collapse", "corrosion",
    "sand production", "water breakthrough", "gas breakthrough",
    "inadequate", "deterioration", "decline", "depleted", "emergency",
    "kick", "blowout", "unplanned", "unexpected", "unfavorable", "unsuccessful",
    "problematic", "remedial", "shortfall",
}


class SentimentAnalyzer:
    def __init__(self):
        self.is_trained = False
        self.pos_weights = {}
        self.neg_weights = {}
        self.model_path = os.path.join("outputs", "models", "sentiment.pkl")

    def analyze(self, text):
        """Analyze sentiment of text and return label with score."""
        tokens = set(tokenize(text))
        text_lower = text.lower()

        pos_count = 0
        neg_count = 0
        pos_found = []
        neg_found = []

        for word in POSITIVE_WORDS:
            if word in text_lower:
                weight = self.pos_weights.get(word, 1.0)
                pos_count += weight
                pos_found.append(word)

        for word in NEGATIVE_WORDS:
            if word in text_lower:
                weight = self.neg_weights.get(word, 1.0)
                neg_count += weight
                neg_found.append(word)

        total = pos_count + neg_count
        if total == 0:
            label = "neutral"
            score = 0.5
        elif pos_count > neg_count:
            score = 0.5 + 0.5 * (pos_count - neg_count) / total
            label = "positive"
        else:
            score = 0.5 - 0.5 * (neg_count - pos_count) / total
            label = "negative"

        return {
            "label": label,
            "score": round(score, 4),
            "positive_indicators": pos_found,
            "negative_indicators": neg_found,
            "positive_count": int(pos_count),
            "negative_count": int(neg_count),
        }

    def train(self, texts, labels):
        """Train weights based on word frequency in labeled data."""
        pos_texts = [t for t, l in zip(texts, labels) if l == "positive"]
        neg_texts = [t for t, l in zip(texts, labels) if l == "negative"]

        pos_words = Counter()
        neg_words = Counter()

        for t in pos_texts:
            pos_words.update(tokenize(t.lower()))
        for t in neg_texts:
            neg_words.update(tokenize(t.lower()))

        total_pos = sum(pos_words.values()) or 1
        total_neg = sum(neg_words.values()) or 1

        for word in POSITIVE_WORDS:
            self.pos_weights[word] = (pos_words.get(word, 0) / total_pos) * 10 + 1
        for word in NEGATIVE_WORDS:
            self.neg_weights[word] = (neg_words.get(word, 0) / total_neg) * 10 + 1

        self.is_trained = True

    def save(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump({
                "pos_weights": self.pos_weights,
                "neg_weights": self.neg_weights,
                "is_trained": self.is_trained,
            }, f)

    def load(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                data = pickle.load(f)
            self.pos_weights = data["pos_weights"]
            self.neg_weights = data["neg_weights"]
            self.is_trained = data["is_trained"]
            return True
        return False
