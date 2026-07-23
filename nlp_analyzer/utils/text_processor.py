"""Tokenization, TF-IDF, keyword extraction, and simple rule-based NER."""

import re
import math
from collections import Counter


def tokenize(text):
    """Basic whitespace + punctuation tokenizer."""
    return re.findall(r"[A-Za-z0-9/\-]+(?:\.[0-9]+)?", text.lower())


def tokenize_sentences(text):
    """Split text into sentences."""
    return [s.strip() for s in re.split(r'[.!?\n]+', text) if s.strip()]


def compute_tfidf(documents):
    """Compute TF-IDF scores for a list of tokenized documents."""
    vocabulary = set()
    for doc in documents:
        vocabulary.update(doc)
    vocab_list = sorted(vocabulary)
    vocab_index = {w: i for i, w in enumerate(vocab_list)}

    df = Counter()
    for doc in documents:
        unique_tokens = set(doc)
        for token in unique_tokens:
            df[token] += 1

    n_docs = len(documents)
    idf = {}
    for word in vocab_list:
        idf[word] = math.log((n_docs + 1) / (df.get(word, 0) + 1)) + 1

    tfidf_matrix = []
    for doc in documents:
        tf = Counter(doc)
        total = len(doc) if doc else 1
        vector = []
        for word in vocab_list:
            score = (tf.get(word, 0) / total) * idf.get(word, 1)
            vector.append(round(score, 6))
        tfidf_matrix.append(vector)

    return tfidf_matrix, vocab_list, idf


def extract_keywords_tfidf(text, top_n=10):
    """Extract top keywords from a single text using TF-IDF."""
    tokens = tokenize(text)
    if not tokens:
        return []
    matrix, vocab, idf = compute_tfidf([tokens])
    scores = list(zip(vocab, matrix[0]))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [word for word, score in scores[:top_n] if score > 0]


def extract_keywords_textstat(text, top_n=10):
    """Extract keywords based on frequency analysis (fallback without textstat)."""
    tokens = tokenize(text)
    if not tokens:
        return []
    tf = Counter(tokens)
    stop_words = {
        "the", "a", "an", "is", "was", "were", "are", "been", "be", "have",
        "has", "had", "do", "does", "did", "will", "would", "could", "should",
        "may", "might", "shall", "can", "need", "dare", "used", "to", "of",
        "in", "for", "on", "with", "at", "by", "from", "as", "into", "during",
        "through", "before", "after", "above", "below", "between", "and", "but",
        "or", "nor", "not", "no", "if", "then", "that", "this", "these", "those",
        "it", "its", "at", "all", "each", "every", "both", "few", "more", "most",
        "other", "some", "such", "than", "too", "very", "just", "also",
    }
    filtered = [(word, count) for word, count in tf.items() if word not in stop_words and len(word) > 2]
    filtered.sort(key=lambda x: x[1], reverse=True)
    return [word for word, count in filtered[:top_n]]


# ---------- Rule-based NER ----------

DEPTH_PATTERN = re.compile(r'\b(\d[\d,]*\.?\d*)\s*(feet|ft|meter|m)\b', re.IGNORECASE)
WELL_PATTERN = re.compile(r'\b(Pemex|Cantarell|Ku-Maloob|Zama|Ixachi|Litoral|Altamira|Burgos|Chicontepec|Tampico|Veracruz|Macuspana|Musaceo|Cordon|Sierra|La-Cruz|Jujo|El-Angel)[\-\s]?([A-Z0-9\-]+)\b', re.IGNORECASE)
FORMATION_PATTERN = re.compile(
    r'\b(Ku-Nu|Nito|Arenoso|Lime Creek|Edwards|Eocene|Cretaceous|Jurassic|Turonian|Cenomanian|'
    r'Cantarell Formation|Ku-Maloob Formation|Shila Formation|Tuxpan Platform|Golden Lane|'
    r'Sureste Basin|Burgos Basin|Veracruz Basin|Chiapas Fold Belt|Mesozoic Carbonate|'
    r'Tertiary Sandstone|Volcanic Sequence|La Luna Formation|Misoa Formation|Lagunillas Formation|'
    r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+Formation)\b'
)
EQUIPMENT_KEYWORDS = [
    "drill bit", "PDC", "tricone", "casing", "cementing unit", "mud pump",
    "BOP", "blowout preventer", "production tubing", "ESP", "rod pump",
    "hydraulic fracturing", "coiled tubing", "wireline", "MWD", "LWD",
    "packer", "liner", "perforating gun", "acidizing", "sand control",
    "gravel pack", "wellhead", "Christmas tree", "choke manifold",
    "separator", "flowback",
]


def extract_depths(text):
    """Extract depth mentions with units."""
    results = []
    for match in DEPTH_PATTERN.finditer(text):
        value = float(match.group(1).replace(",", ""))
        unit = match.group(2)
        results.append({"value": value, "unit": unit, "raw": match.group(0)})
    return results


def extract_well_names(text):
    """Extract well name mentions."""
    results = []
    for match in WELL_PATTERN.finditer(text):
        results.append(match.group(0).strip())
    return list(set(results))


def extract_formations(text):
    """Extract formation names."""
    results = []
    for match in FORMATION_PATTERN.finditer(text):
        results.append(match.group(0).strip())
    return list(set(results))


def extract_equipment(text):
    """Extract equipment mentions using keyword matching."""
    found = []
    text_lower = text.lower()
    for eq in EQUIPMENT_KEYWORDS:
        if eq.lower() in text_lower:
            found.append(eq)
    return found


def extract_entities(text):
    """Run all rule-based entity extractors and return consolidated results."""
    return {
        "depths": extract_depths(text),
        "well_names": extract_well_names(text),
        "formations": extract_formations(text),
        "equipment": extract_equipment(text),
    }
