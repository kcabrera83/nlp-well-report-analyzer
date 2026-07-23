# NLP Well Report Analyzer

An NLP system for analyzing well completion reports and extracting structured information from unstructured text.

## Overview

This project uses natural language processing and machine learning to automatically classify, analyze, and extract entities from oil and gas well reports. The system can process drilling reports, completion reports, workover reports, and production reports.

## Features

- **Report Classification**: Automatically categorize reports into drilling, completion, workover, or production types
- **Entity Extraction**: Extract well names, depths, formations, and equipment from report text
- **Sentiment Analysis**: Determine the overall tone of a report (positive, negative, neutral)
- **Keyword Extraction**: Identify key terms using TF-IDF scoring
- **REST API**: Flask-based API with multiple endpoints for programmatic access
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

Start the Flask server on port 5017:

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

1. **Data Generation**: Synthetic well reports are generated with realistic oil and gas industry terminology, well names, formations, and equipment references
2. **Text Processing**: Tokenization, TF-IDF computation, and keyword extraction using custom implementations without heavy NLP dependencies
3. **Classification**: Keyword-frequency-based classification that matches report text against domain-specific term patterns for each report type
4. **Entity Extraction**: Rule-based extraction using regex patterns for well names, depth measurements, formation names, and equipment mentions
5. **Sentiment Analysis**: Lexicon-based sentiment scoring using industry-specific positive and negative indicator words

## Dependencies

- Flask (web framework)
- Python standard library (math, re, json, collections, pickle)

No heavy NLP libraries required - the system is designed to work with lightweight custom implementations.

## License

MIT License

## Author

Elaborado por Ing. Kelvin Cabrera
