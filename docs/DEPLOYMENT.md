# Deployment Guide - NLP Well Report Analyzer

## Docker Deployment

### Build the Image

```bash
cd nlp-well-report-analyzer
docker build -t nlp-well-report-analyzer .
```

### Run the Container

```bash
docker run -p 5017:5017 nlp-well-report-analyzer
```

### With Model Training

```bash
docker run -p 5017:5017 nlp-well-report-analyzer bash -c "python train.py && python app.py"
```

## Docker Compose

```yaml
version: '3.8'
services:
  nlp-analyzer:
    build: .
    ports:
      - "5017:5017"
    volumes:
      - ./outputs:/app/outputs
    environment:
      - FLASK_ENV=production
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| FLASK_ENV | Flask environment mode | development |
| PYTHONUNBUFFERED | Disable Python output buffering | 1 |
| PORT | Server port (hardcoded in app.py) | 5017 |

## Manual Deployment

### Install Dependencies

```bash
pip install -r requirements.txt
```

No heavy NLP libraries required - uses only Flask and Python standard library.

### Train Models

```bash
python train.py
```

### Run with Gunicorn (Production)

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5017 app:app
```

### Run with Flask Development Server

```bash
python app.py
```

## Production Considerations

- Use **gunicorn** with multiple workers for production deployments
- Set `debug=True` to `debug=False` in `app.py` for production
- Configure proper logging for request/error tracking
- Place behind a reverse proxy (nginx/Apache) for SSL termination
- Models auto-load from `outputs/models/` at startup
- Pre-train models with `train.py` before starting the server
- Lightweight dependencies - no heavy NLP libraries needed

## Health Check

```bash
curl http://localhost:5017/api/health
```

Expected response:
```json
{"status": "healthy", "models_loaded": {"classifier": true, "entity_extractor": true, "sentiment_analyzer": true}}
```

## Ports

| Service | Port |
|---------|------|
| NLP Well Report Analyzer | 5017 |
