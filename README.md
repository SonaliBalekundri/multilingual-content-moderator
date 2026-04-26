# 🛡️ Multilingual Content Moderator API

A FastAPI-powered content moderation service that detects toxicity and harmful content across **English, Hindi, and Arabic** using Hugging Face transformer models. Features language-aware threshold calibration to handle cross-lingual bias in multilingual NLP models.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![Tests](https://img.shields.io/badge/Tests-31%20passing-brightgreen)
![Accuracy](https://img.shields.io/badge/Accuracy-91.7%25-orange)

## Demo

https://github.com/SonaliBalekundri/multilingual-content-moderator/raw/main/docs/Multilingual_Content_Moderator_Demo.mp4

## Features

- **Multilingual toxicity detection** — English, Hindi, Arabic with per-language threshold calibration
- **Interactive dashboard** — Streamlit UI with real-time moderation, batch upload, and Plotly charts
- **Language auto-detection** — identifies input language and applies calibrated thresholds
- **Batch processing** — moderate up to 100 texts in one request (14.6 texts/sec throughput)
- **Input sanitisation** — handles short text, whitespace, repeated characters, edge cases
- **Configurable thresholds** — override defaults via API or use calibrated per-language defaults
- **RESTful API** — FastAPI with auto-generated Swagger docs at `/docs`
- **31 tests passing** — language detection, model, and API endpoint coverage
- **Dockerised** — one command to run everything: `docker-compose up`

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌───────────────────────┐
│   Client     │────▶│   FastAPI    │────▶│  DistilBERT           │
│  (Streamlit/ │     │   Backend    │     │  Multilingual         │
│   curl/app)  │◀────│   :8000     │◀────│  Toxicity Model       │
└──────────────┘     └──────┬───────┘     └───────────────────────┘
                            │
                     ┌──────▼───────┐
                     │  Language    │
                     │  Detection   │
                     │  (langdetect)│
                     └──────────────┘
```

**Request flow:** Text input → Language detection → Text sanitisation → Tokenization → Model inference → Sigmoid → Language-aware thresholding → Structured JSON response

## Quick Start

### Option 1: Docker (Recommended)

```bash
git clone https://github.com/SonaliBalekundri/multilingual-content-moderator.git
cd multilingual-content-moderator

docker-compose up --build
```

Then open:
- **Dashboard:** http://localhost:8501
- **API Docs:** http://localhost:8000/docs

### Option 2: Local Setup

**Prerequisites:** Python 3.12, pip

```bash
git clone https://github.com/SonaliBalekundri/multilingual-content-moderator.git
cd multilingual-content-moderator

python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\Activate

pip install -r requirements.txt

# Terminal 1: Run API
uvicorn app.main:app --reload --port 8000

# Terminal 2: Run Dashboard
streamlit run streamlit_app.py
```

First request will take ~30 seconds (model download + loading). Subsequent requests: 16–40ms.

### Run Tests

```bash
pytest tests/test_moderator.py -v

# Performance benchmarks (API must be running)
python tests/test_performance.py
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/moderate` | Moderate a single text |
| POST | `/api/v1/moderate/batch` | Moderate up to 100 texts |
| GET | `/api/v1/languages` | List supported languages |
| GET | `/api/v1/categories` | List moderation categories |
| GET | `/api/v1/health` | Health check with model status |

**Example request:**
```bash
curl -X POST http://localhost:8000/api/v1/moderate \
  -H "Content-Type: application/json" \
  -d '{"text": "You are a disgusting person"}'
```

**Example response:**
```json
{
  "text": "You are a disgusting person",
  "language": "en",
  "verdict": "toxic",
  "categories": {
    "toxic": {"score": 0.9301, "flagged": true},
    "not_toxic": {"score": 0.1017, "flagged": false}
  },
  "confidence": 0.9301,
  "processing_time_ms": 40.03,
  "threshold_used": 0.5,
  "warnings": []
}
```

See [API_DOCS.md](API_DOCS.md) for full documentation with all endpoints, examples, and error handling.

## Streamlit Dashboard

The interactive dashboard provides:
- **Single text moderation** — type or paste text, click Analyse, see verdict with colour-coded results
- **Example buttons** — one-click testing in English, Hindi, and Arabic
- **Batch CSV upload** — upload a CSV file to moderate hundreds of texts at once
- **Plotly charts** — verdict distribution pie chart, language distribution bar chart
- **Moderation history** — tracks recent results within the session
- **Threshold override** — slider to override language-aware defaults
- **API health status** — sidebar indicator showing model and connection status

## Multilingual Performance Benchmarks

**Model:** `citizenlab/distilbert-base-multilingual-cased-toxicity`

| Metric | English | Hindi | Arabic | Overall |
|--------|---------|-------|--------|---------|
| Threshold | 0.50 | 0.15 | 0.20 | — |
| Benchmark Accuracy | — | — | — | 91.7% (11/12) |

**Performance (CPU):**

| Metric | Value |
|--------|-------|
| Single request latency | 16–40ms |
| Batch (50 texts) throughput | 14.6 texts/sec |
| Model load time | ~2.3 seconds |

### Why Language-Aware Thresholds?

The same toxic content scores differently across languages due to training data bias:

```
"You are stupid"         → EN score: 0.93 (easily caught at 0.50)
"तुम बेवकूफ हो"           → HI score: 0.42 (missed at 0.50, caught at 0.15)
"أنت غبي"                → AR score: 0.34 (missed at 0.50, caught at 0.20)
```

Without per-language thresholds, Hindi and Arabic toxic content goes undetected. This is the key insight from our model evaluation.

### Known Limitations

- **Indirect threats missed:** "I will make sure you regret being alive" scores 0.034 — the model relies on surface-level toxic keywords, not semantic understanding. Unfixable by threshold alone.
- **Cross-lingual bias:** Non-English toxic content scores lower than equivalent English content. Mitigated with per-language thresholds but not fully eliminated.
- **Short text unreliable:** Texts under 3 words produce unreliable scores. API returns a warning for these cases.
- **False positive on "You are a wonderful person!":** Model pattern-matches "You are a ___ person" as toxic regardless of the adjective (scores 0.81).

### Model Selection

Two models were evaluated during Week 1:

| Model | Accuracy | Key Issue |
|-------|----------|-----------|
| `unitary/multilingual-toxic-xlm-roberta` | 50% | Inverted scores — clean text scored higher than toxic |
| `citizenlab/distilbert-base-multilingual-cased-toxicity` | 91.7% | Selected — correct score separation with calibrated thresholds |

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| ML/AI | PyTorch (CPU), Hugging Face Transformers (DistilBERT multilingual), langdetect |
| Backend | FastAPI, Pydantic, Uvicorn |
| Frontend | Streamlit, Plotly |
| Infrastructure | Docker, Docker Compose, WSL 2 |
| Testing | pytest (31 tests), FastAPI TestClient |
| Utilities | pandas, matplotlib, requests |

## Project Structure

```
multilingual-content-moderator/
├── app/
│   ├── main.py              # FastAPI application entry point
│   ├── config.py             # Settings, thresholds, language aliases
│   ├── api/
│   │   └── routes.py         # API endpoint handlers
│   ├── models/
│   │   └── moderator.py      # ML model wrapper (ContentModerator)
│   ├── schemas/
│   │   └── moderation.py     # Pydantic request/response models
│   └── utils/
│       └── language.py       # Language detection + alias mapping
├── tests/
│   ├── test_moderator.py     # 31 tests (language, model, API)
│   └── test_performance.py   # Performance benchmarking script
├── streamlit_app.py          # Interactive dashboard
├── docs/
│   └── demo.mp4              # Demo recording
├── API_DOCS.md               # Full API documentation
├── Dockerfile                # Multi-stage container build (CPU-only PyTorch)
├── docker-compose.yml        # API + Streamlit orchestration
├── .dockerignore             # Exclude venv, .git, __pycache__ from image
└── requirements.txt          # Python dependencies
```

## What I Learned

### Week 1 — PyTorch + Multilingual NLP
- PyTorch fundamentals: tensors, autograd, nn.Module, device management
- Hugging Face Transformers: tokenizers, model loading, inference pipelines
- Evaluated two multilingual toxicity models — discovered severe cross-lingual bias
- Implemented per-language threshold calibration to compensate for model bias
- Built a 30-text benchmark dataset across 3 languages with error analysis

### Week 2 — FastAPI Service + Testing
- Built REST API with FastAPI: single and batch moderation endpoints
- Input sanitisation: short text detection, whitespace handling, character normalisation
- Language-aware threshold routing: API auto-detects language and applies correct threshold
- 31 pytest tests across 3 layers: language detection, model wrapper, API endpoints
- Performance benchmarking: measured latency, throughput, and accuracy through the API

### Week 3 — Docker + Streamlit Dashboard
- Docker containerisation with CPU-only PyTorch (reduced image from 2.5GB CUDA to ~200MB CPU)
- Multi-stage Dockerfile build for lean production images
- docker-compose orchestration: API + Streamlit running with one command
- Interactive Streamlit dashboard: single text moderation, batch CSV upload, Plotly charts
- Language-aware threshold visualisation in the UI
- Session-based moderation history tracking
- WSL 2 + Docker Engine setup on Windows 11

## License

MIT
