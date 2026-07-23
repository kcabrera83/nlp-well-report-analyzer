# User Guide - NLP Well Report Analyzer

## Overview

The NLP Well Report Analyzer is a natural language processing system for automatically classifying, analyzing, and extracting information from oil and gas well reports. It processes drilling, completion, workover, and production reports using lightweight custom NLP implementations.

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
cd nlp-well-report-analyzer
pip install -r requirements.txt
```

### Train Models

```bash
python train.py
```

Generates 600 synthetic well reports and trains all three models. Models are saved to `outputs/models/`.

### Start the API Server

```bash
python app.py
```

The server starts on `http://localhost:5017`.

### Open the Dashboard

Navigate to `http://localhost:5017` in your browser.

### Run Tests

```bash
python test_api.py
```

## Dashboard Features

- **Full Analysis**: Paste a well report to get classification, entities, sentiment, and keywords
- **Report Classification**: Categorize reports into drilling, completion, workover, or production
- **Entity Extraction**: Extract well names, depths, formations, and equipment
- **Sentiment Analysis**: Determine report tone (positive, negative, neutral)
- **Keyword Extraction**: Identify key terms using TF-IDF scoring

## Report Types

| Type | Description | Key Indicators |
|------|-------------|----------------|
| drilling | Drilling operations reports | drill, bit, mud, casing, depth, wellbore |
| completion | Well completion reports | perforated, completion, liner, cemented |
| workover | Workover/intervention reports | workover, repairs, replaced, pulled |
| production | Production operations reports | production, barrels, oil rate, water cut |

## API Usage

### Using curl

**Full analysis:**
```bash
curl -X POST http://localhost:5017/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Drilling report for well Cantarell-X1. Total depth reached 12500 feet in the Cretaceous formation."}'
```

**Classify report type:**
```bash
curl -X POST http://localhost:5017/api/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Workover operations performed on well Santa Barbara-3."}'
```

**Extract entities:**
```bash
curl -X POST http://localhost:5017/api/extract \
  -H "Content-Type: application/json" \
  -d '{"text": "ESP installed on well Ku-Maloob-Zaap-5 at 8200 feet depth."}'
```

### Using Python

```python
import requests

text = "Drilling report for well Cantarell-X1. Total depth reached 12500 feet in the Cretaceous formation."

# Full analysis
response = requests.post("http://localhost:5017/api/analyze", json={"text": text})
result = response.json()
print(f"Type: {result['classification']['type']}")
print(f"Sentiment: {result['sentiment']['label']}")
print(f"Keywords: {result['keywords'][:5]}")

# Extract entities
response = requests.post("http://localhost:5017/api/extract", json={"text": text})
entities = response.json()["entities"]
for well in entities["well_names"]:
    print(f"Well: {well}")
for depth in entities["depths"]:
    print(f"Depth: {depth}")
```

### Using JavaScript

```javascript
const response = await fetch("http://localhost:5017/api/analyze", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ text: "Your well report text here..." })
});
const data = await response.json();
console.log(`Type: ${data.classification.type}`);
console.log(`Sentiment: ${data.sentiment.label}`);
```

## Dependencies

No heavy NLP libraries required - the system uses lightweight custom implementations built with Python standard library (math, re, json, collections, pickle).
