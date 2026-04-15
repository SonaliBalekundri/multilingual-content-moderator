# 🛡️ Multilingual Content Moderator API

A FastAPI-powered content moderation service that detects toxicity, hate speech, spam, and harmful content across **English, Hindi, and Arabic** using Hugging Face transformer models (XLM-RoBERTa).

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)

## Features

- **Multilingual toxicity detection** — English, Hindi, Arabic
- **Multi-label classification** — toxic, severe toxic, obscene, threat, insult, hate speech
- **Language auto-detection** — automatically identifies input language
- **Batch processing** — moderate hundreds of texts in one request
- **Configurable thresholds** — adjust sensitivity per category
- **Interactive dashboard** — Streamlit UI with visual breakdowns
- **RESTful API** — FastAPI with auto-generated Swagger docs
- **Dockerised** — one command to run everything

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│   Client     │────▶│   FastAPI    │────▶│  XLM-RoBERTa     │
│  (Streamlit/ │     │   Backend    │     │  Toxicity Model   │
│   any app)   │◀────│              │◀────│                   │
└──────────────┘     └──────┬───────┘     └──────────────────┘
                            │
                     ┌──────▼───────┐
                     │  Language    │
                     │  Detection   │
                     └──────────────┘
```

## Quick Start

### Prerequisites
- Python 3.10+
- pip
- (Optional) Docker & Docker Compose

### Local Setup

```bash
git clone https://github.com/SonaliBalekundri/multilingual-content-moderator.git
cd multilingual-content-moderator

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Run API
uvicorn app.main:app --reload --port 8000

# Run Streamlit dashboard (separate terminal)
streamlit run streamlit_app.py
```

### Docker Setup

```bash
docker-compose up --build
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/moderate` | Moderate a single text |
| POST | `/moderate/batch` | Moderate multiple texts |
| GET | `/languages` | List supported languages |
| GET | `/categories` | List moderation categories |
| GET | `/health` | Health check |

## Multilingual Performance Benchmarks

| Language | Accuracy | F1 Score | Avg Latency |
|----------|----------|----------|-------------|
| English  | TBD      | TBD      | TBD         |
| Hindi    | TBD      | TBD      | TBD         |
| Arabic   | TBD      | TBD      | TBD         |

## Tech Stack

- **ML/AI**: PyTorch, Hugging Face Transformers (XLM-RoBERTa)
- **Backend**: FastAPI, Pydantic, Uvicorn
- **Frontend**: Streamlit
- **Infrastructure**: Docker, Docker Compose
- **Utilities**: langdetect, pandas, plotly

## What I Learned

- **Week 1**: PyTorch fundamentals, multilingual NLP, XLM-RoBERTa cross-lingual transfer
- **Week 2**: FastAPI design patterns, batch processing, input validation
- **Week 3**: Docker multi-stage builds, Streamlit dashboards, deployment

## License

MIT
