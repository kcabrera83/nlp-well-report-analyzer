# API Documentation - NLP Well Report Analyzer

## Base URL

```
http://localhost:5017
```

## Endpoints

### GET /

Serves the web interface (HTML dashboard).

**Response:** HTML page

---

### GET /api/health

Health check endpoint with model status.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": {
    "classifier": true,
    "entity_extractor": true,
    "sentiment_analyzer": true
  }
}
```

---

### GET /api/models

Returns model information and supported types.

**Response:**
```json
{
  "classifier": {
    "loaded": true,
    "types": ["drilling", "completion", "workover", "production"]
  },
  "entity_extractor": {
    "loaded": true,
    "entity_types": ["well_name", "depths", "formation", "equipment"]
  },
  "sentiment_analyzer": {
    "loaded": true,
    "labels": ["positive", "negative", "neutral"]
  }
}
```

---

### POST /api/analyze

Full NLP analysis: classification + entity extraction + sentiment + keywords.

**Request:**
```json
{
  "text": "Drilling report for well Cantarell-X1. Total depth reached 12500 feet in the Cretaceous formation. Excellent drilling performance achieved with minimal issues."
}
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| text | string | Yes | Report text to analyze |

**Response:**
```json
{
  "classification": {
    "type": "drilling",
    "probabilities": {
      "drilling": 0.82,
      "completion": 0.10,
      "workover": 0.05,
      "production": 0.03
    }
  },
  "entities": {
    "well_names": ["Cantarell-X1"],
    "depths": ["12500 feet"],
    "formations": ["Cretaceous"],
    "equipment": []
  },
  "sentiment": {
    "label": "positive",
    "score": 0.65,
    "positive_words": ["excellent", "minimal"],
    "negative_words": []
  },
  "keywords": ["drilling", "cantarell", "depth", "cretaceous", "performance"],
  "text_length": 152,
  "token_count": 24
}
```

**Error Response:**
```json
{"error": "No text provided"}
```
Status: `400 Bad Request`

---

### POST /api/classify

Classify report type only.

**Request:**
```json
{
  "text": "Workover operations performed on well Santa Barbara-3. Pump replaced and tubing cleaned."
}
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| text | string | Yes | Report text to classify |

**Response:**
```json
{
  "classification": "workover",
  "probabilities": {
    "drilling": 0.08,
    "completion": 0.12,
    "workover": 0.72,
    "production": 0.08
  }
}
```

**Error Response:**
```json
{"error": "No text provided"}
```
Status: `400 Bad Request`

---

### POST /api/extract

Extract entities and keywords from report text.

**Request:**
```json
{
  "text": "Completion report for well Ku-Maloob-Zaap-5. Perforated interval 8200-8500 feet in breccia formation. ESP installed with 400 bbl/day capacity."
}
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| text | string | Yes | Report text to extract from |

**Response:**
```json
{
  "entities": {
    "well_names": ["Ku-Maloob-Zaap-5"],
    "depths": ["8200-8500 feet"],
    "formations": ["breccia"],
    "equipment": ["ESP"]
  },
  "keywords": ["completion", "perforated", "interval", "esp", "capacity", "breccia", "formation"]
}
```

**Error Response:**
```json
{"error": "No text provided"}
```
Status: `400 Bad Request`

---

### GET /api/docs

Returns OpenAPI 3.0.0 specification.

**Response:**
```json
{
  "openapi": "3.0.0",
  "info": {"title": "NLP Well Report Analyzer", "version": "1.0.0"},
  "paths": {
    "/api/health": {"get": {"summary": "Health check"}},
    "/api/models": {"get": {"summary": "Model info"}},
    "/api/analyze": {"post": {"summary": "Full NLP analysis of well report text"}},
    "/api/extract": {"post": {"summary": "Extract entities and keywords"}},
    "/api/classify": {"post": {"summary": "Classify report type"}}
  }
}
```

## Entity Types

| Type | Description | Examples |
|------|-------------|----------|
| well_name | Oil well identifiers | Cantarell-X1, Santa Barbara-3 |
| depths | Depth measurements | 12500 feet, 8200-8500 ft |
| formation | Geological formations | Cretaceous, breccia, limestone |
| equipment | Oil & gas equipment | ESP, rod pump, BOP |

## Error Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad request - missing or invalid parameters |
| 500 | Internal server error |
