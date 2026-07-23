"""Rule-based + ML entity extraction for well reports."""

import os
import re
import pickle
from collections import Counter

from nlp_analyzer.utils.text_processor import (
    extract_entities, extract_keywords_tfidf, tokenize,
)


class EntityExtractor:
    """Extracts entities from well report text using rules and frequency-based scoring."""

    def __init__(self):
        self.is_trained = False
        self.entity_freq = {}
        self.model_path = os.path.join("outputs", "models", "entity_extractor.pkl")

    def extract(self, text):
        """Extract all entities from text using rule-based methods."""
        rule_entities = extract_entities(text)
        keywords = extract_keywords_tfidf(text, top_n=10)

        structured = {
            "well_name": self._best_match_well(rule_entities["well_names"]),
            "depths": self._format_depths(rule_entities["depths"]),
            "formation": self._best_match_formation(rule_entities["formations"]),
            "equipment": rule_entities["equipment"],
            "keywords": keywords,
            "all_well_names": rule_entities["well_names"],
            "all_formations": rule_entities["formations"],
        }
        return structured

    def _best_match_well(self, well_names):
        if not well_names:
            return None
        return well_names[0]

    def _best_match_formation(self, formations):
        if not formations:
            return None
        return formations[0]

    def _format_depths(self, depths):
        if not depths:
            return None
        return [
            {"value": d["value"], "unit": d["unit"]}
            for d in depths
        ]

    def train(self, texts):
        """Build frequency stats from corpus."""
        all_entities = {"well_names": Counter(), "formations": Counter(), "equipment": Counter()}
        for text in texts:
            ents = extract_entities(text)
            for key in all_entities:
                for item in ents[key]:
                    all_entities[key][item] += 1
        self.entity_freq = {k: dict(v.most_common(50)) for k, v in all_entities.items()}
        self.is_trained = True

    def save(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump({"entity_freq": self.entity_freq, "is_trained": self.is_trained}, f)

    def load(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                data = pickle.load(f)
            self.entity_freq = data["entity_freq"]
            self.is_trained = data["is_trained"]
            return True
        return False
