# Architecture - NLP Well Report Analyzer

## System Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Web Dashboard (HTML)                   │
│                    Port 5017 /                           │
├─────────────────────────────────────────────────────────┤
│                    Flask API Layer                       │
│      /api/analyze  /api/classify  /api/extract          │
├──────────┬──────────────────┬───────────────────────────┤
│ Report   │ Entity           │ Sentiment                 │
│ Classif. │ Extractor        │ Analyzer                  │
│ (Keyword)│ (Regex/Rules)    │ (Lexicon)                 │
├──────────┴──────────────────┴───────────────────────────┤
│         Text Processor (TF-IDF, Tokenizer)               │
│           Synthetic Data Generator (600 reports)          │
└─────────────────────────────────────────────────────────┘
```

## Components

### Data Layer

- **Data Generator**: Creates synthetic well reports with realistic oil & gas terminology
- **Report Types**: drilling, completion, workover, production (150 each, 600 total)
- **Content**: Includes well names, depths, formations, equipment references
- **Sentiment Labels**: Auto-labeled using keyword-based heuristics

### Model Layer

#### Report Classifier (`ReportClassifier`)
- **Algorithm**: Keyword-frequency-based classification
- **Approach**: Matches report text against domain-specific term patterns for each report type
- **Scoring**: Counts keyword occurrences per category, normalizes to probabilities
- **Categories**: drilling, completion, workover, production
- **Persistence**: Pickle (.pkl) at `outputs/models/classifier.pkl`

#### Entity Extractor (`EntityExtractor`)
- **Algorithm**: Rule-based extraction using regex patterns
- **Entity Types**:
  - `well_name`: Matches patterns like `Cantarell-X1`, `Santa Barbara-3`
  - `depths`: Matches depth measurements (e.g., `12500 feet`, `8200-8500 ft`)
  - `formation`: Matches geological formation names
  - `equipment`: Matches oil & gas equipment references (ESP, BOP, etc.)
- **Training**: Builds frequency dictionaries from training corpus
- **Persistence**: Pickle (.pkl) at `outputs/models/entity_extractor.pkl`

#### Sentiment Analyzer (`SentimentAnalyzer`)
- **Algorithm**: Lexicon-based sentiment scoring
- **Approach**: Uses industry-specific positive and negative indicator words
- **Positive Keywords**: successful, excellent, good, improved, completed, optimized, achieved
- **Negative Keywords**: failed, failure, problem, damage, loss, lost, stuck, severe, critical, incident
- **Scoring**: Weighted sum of positive/negative word counts
- **Labels**: positive, negative, neutral
- **Persistence**: Pickle (.pkl) at `outputs/models/sentiment.pkl`

#### Text Processor Utilities
- **Tokenization**: Custom word tokenizer with lowercase normalization
- **TF-IDF**: Term frequency-inverse document frequency computation
- **Keyword Extraction**: TF-IDF weighted term extraction

### API Layer

- **Framework**: Flask (Python)
- **Serialization**: JSON request/response
- **Model Loading**: Pickle deserialization at startup
- **Port**: 5017

### Dashboard Layer

- **Frontend**: HTML/CSS/JavaScript (single page)
- **Visualization**: Chart.js for classification probabilities and sentiment
- **Style**: Dark-themed responsive UI

## Data Flow

### Full Analysis Pipeline

```
1. Input Report Text
   ↓
2. Text Processor (tokenize, normalize)
   ↓
3. Parallel Analysis
   ├── Report Classifier → report type + probabilities
   ├── Entity Extractor → well names, depths, formations, equipment
   ├── Sentiment Analyzer → label + score
   └── Keyword Extractor → TF-IDF keywords
   ↓
4. Combined JSON Response
   ↓
5. Dashboard Display
```

### Classification Pipeline

```
1. Input Text
   ↓
2. Tokenization + Normalization
   ↓
3. Keyword Pattern Matching (per category)
   ↓
4. Score Normalization
   ↓
5. Report Type + Probabilities
```

### Entity Extraction Pipeline

```
1. Input Text
   ↓
2. Regex Pattern Matching
   ├── Well Name Patterns
   ├── Depth Measurement Patterns
   ├── Formation Name Patterns
   └── Equipment Name Patterns
   ↓
3. Extracted Entities
```

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.8+ |
| Web Framework | Flask |
| NLP | Custom (no heavy NLP libraries) |
| Text Processing | regex, collections, math |
| Model Persistence | Pickle |
| Frontend | HTML/CSS/JS + Chart.js |

## Model Artifacts

| File | Description |
|------|-------------|
| `outputs/models/classifier.pkl` | Keyword-based report classifier |
| `outputs/models/entity_extractor.pkl` | Rule-based entity extractor |
| `outputs/models/sentiment.pkl` | Lexicon-based sentiment analyzer |
| `outputs/training_data.json` | Sample training data (20 records) |
