# NLP Well Report Analyzer

An NLP system for analyzing well completion reports and extracting structured information using modern NLP frameworks.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| NLP Framework | **spaCy** - industrial-strength NLP |
| Text Processing | **NLTK** - tokenization, stemming, stopwords |
| Topic Modeling | **Gensim** - topic modeling and embeddings |
| Transformers | **HuggingFace** - pre-trained language models |
| Data Processing | pandas, numpy, joblib |
| Web Server | **FastAPI** + uvicorn |
| Monitoring | prometheus-fastapi-instrumentator |
| Validation | pydantic v2 |

### Key Libraries
- spaCy - Industrial-strength NLP for entity extraction
- NLTK - Natural language toolkit for text processing
- Gensim - Topic modeling and document similarity
- HuggingFace Transformers - Pre-trained language models
- FastAPI - Modern async web framework

## Overview

This project uses NLP and machine learning to automatically classify, analyze, and extract entities from oil and gas well reports. The system can process drilling reports, completion reports, workover reports, and production reports.

## Features

- **Report Classification**: Automatically categorize reports into drilling, completion, workover, or production types
- **Entity Extraction**: Extract well names, depths, formations, and equipment from report text using spaCy NER
- **Sentiment Analysis**: Determine the overall tone of a report using HuggingFace models
- **Topic Modeling**: Identify key topics using Gensim LDA
- **Keyword Extraction**: Identify key terms using TF-IDF scoring
- **REST API**: FastAPI-based API with multiple endpoints for programmatic access
- **Web Interface**: Dark-themed dashboard with Chart.js visualizations

## Project Structure

```
nlp-well-report-analyzer/
  nlp_analyzer/
    __init__.py
    data_generator.py
    models/
      __init__.py
      report_classifier.py
      entity_extractor.py
      sentiment_analyzer.py
    utils/
      __init__.py
      text_processor.py
  outputs/
    models/
  templates/
    index.html
  .github/
    workflows/
      ci.yml
  train.py
  app.py
  test_api.py
  requirements.txt
  setup.py
  .gitignore
  README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Training

Generate synthetic data and train all models:

```bash
python train.py
```

## Running the Application

Start the FastAPI server on port 5017:

```bash
python app.py
```

Then open http://127.0.0.1:5017 in your browser.

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Web interface |
| `/api/health` | GET | Health check and model status |
| `/api/models` | GET | Model information |
| `/api/analyze` | POST | Full analysis (classify + extract + sentiment + keywords) |
| `/api/classify` | POST | Classify report type only |
| `/api/extract` | POST | Extract entities and keywords only |

### Example Request

```bash
curl -X POST http://127.0.0.1:5017/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Drilling report for well Cantarell-X1. Total depth reached 12500 feet in the Cretaceous formation."}'
```

## Testing

```bash
python test_api.py
```

## How It Works

1. **Data Generation**: Synthetic well reports are generated with realistic oil and gas industry terminology
2. **Text Processing**: Tokenization using spaCy/NLTK, TF-IDF computation with Gensim
3. **Classification**: HuggingFace transformer-based classification
4. **Entity Extraction**: spaCy NER pipeline for well names, depths, formations, and equipment
5. **Sentiment Analysis**: HuggingFace sentiment models fine-tuned for industrial text
6. **Topic Modeling**: Gensim LDA for document topic discovery

---

Elaborado por Ing. Kelvin Cabrera
