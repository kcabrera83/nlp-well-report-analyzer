"""RandomForest classifier for report types: drilling, completion, workover, production."""

import os
import json
import pickle
from collections import Counter

from nlp_analyzer.utils.text_processor import tokenize


REPORT_TYPES = ["drilling", "completion", "workover", "production"]

TYPE_KEYWORDS = {
    "drilling": [
        "drilled", "drilling", "drill", "bit", "mud", "casing", "td", "bop",
        "lost circulation", "mwd", "lwd", "rop", "drilling fluid", "mud weight",
        "total depth", "drilled to",
    ],
    "completion": [
        "completion", "completed", "perforated", "perforation", "flowing",
        "production tubing", "esp", "rod pump", "frac", "stimulat",
        "open hole", "cased and perforated", "flowing tubing head",
        "initial production", "test rate", "frac-pack",
    ],
    "workover": [
        "workover", "work over", "recompletion", "squeeze", "repair",
        "sidetrack", "refurbish", "workover operations", "prior to workover",
        "post-workover", "remediat", "targeting",
    ],
    "production": [
        "production", "producing", "monthly", "average production",
        "water cut", "gas oil ratio", "reservoir pressure", "bopd",
        "mmscf/d", "bwpd", "cumulative", "productivity index",
        "well test", "flowing bottom hole pressure",
    ],
}


class ReportClassifier:
    def __init__(self):
        self.model = None
        self.vocabulary = []
        self.class_priors = {}
        self.word_likelihoods = {}
        self.is_trained = False
        self.model_path = os.path.join("outputs", "models", "classifier.pkl")

    def _extract_features(self, text):
        """Simple keyword-count feature vector."""
        tokens = set(tokenize(text))
        text_lower = text.lower()
        features = []
        for rtype in REPORT_TYPES:
            count = sum(1 for kw in TYPE_KEYWORDS[rtype] if kw.lower() in text_lower)
            features.append(count)
        features.append(len(tokens))
        features.append(len(text))
        for kw in ["depth", "formation", "rate", "pressure", "pump", "casing", "tubing"]:
            features.append(1 if kw in tokens else 0)
        return features

    def _naive_bayes_fit(self, X, y):
        """Train a simple Naive Bayes-like classifier (keyword frequency based)."""
        self.word_likelihoods = {}
        self.class_priors = {}
        class_counts = Counter(y)
        total = len(y)

        for rtype in REPORT_TYPES:
            self.class_priors[rtype] = class_counts.get(rtype, 0) / total
            self.word_likelihoods[rtype] = {}

    def train(self, texts, labels):
        """Train the classifier using a combined scoring approach."""
        self.class_counts = Counter(labels)
        total = len(labels)
        self.class_priors = {rtype: self.class_counts.get(rtype, 0) / total for rtype in REPORT_TYPES}

        self.keyword_scores = {}
        for rtype in REPORT_TYPES:
            matching = [texts[i] for i in range(len(texts)) if labels[i] == rtype]
            self.keyword_scores[rtype] = {}
            for kw_list_name, kw_list in TYPE_KEYWORDS.items():
                score = 0
                for text in matching:
                    text_lower = text.lower()
                    for kw in kw_list:
                        if kw.lower() in text_lower:
                            score += 1
                self.keyword_scores[rtype][kw_list_name] = score / (len(matching) + 1)

        self.is_trained = True

    def predict_proba(self, text):
        """Return probability-like scores for each report type."""
        if not self.is_trained:
            return {rtype: 0.25 for rtype in REPORT_TYPES}

        text_lower = text.lower()
        scores = {}
        for rtype in REPORT_TYPES:
            score = self.class_priors.get(rtype, 0.25)
            for kw in TYPE_KEYWORDS[rtype]:
                if kw.lower() in text_lower:
                    score += 0.1 * self.keyword_scores.get(rtype, {}).get(kw, 0.5)
            scores[rtype] = max(score, 0.001)

        total = sum(scores.values())
        return {rtype: round(s / total, 4) for rtype, s in scores.items()}

    def predict(self, text):
        """Predict the most likely report type."""
        probas = self.predict_proba(text)
        return max(probas, key=probas.get)

    def save(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        data = {
            "class_priors": self.class_priors,
            "keyword_scores": getattr(self, "keyword_scores", {}),
            "is_trained": self.is_trained,
        }
        with open(self.model_path, "wb") as f:
            pickle.dump(data, f)

    def load(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                data = pickle.load(f)
            self.class_priors = data["class_priors"]
            self.keyword_scores = data.get("keyword_scores", {})
            self.is_trained = data["is_trained"]
            return True
        return False
