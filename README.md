# 🛡️ Multilingual Content Moderator API

A FastAPI-powered content moderation service that detects toxicity and harmful content across **English, Hindi, and Arabic** using Hugging Face transformer models. Features language-aware threshold calibration to handle cross-lingual bias in multilingual NLP models.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![Tests](https://img.shields.io/badge/Tests-31%20passing-brightgreen)

## Features

- **Multilingual toxicity detection** — English, Hindi, Arabic with per-language threshold calibration
- **Language auto-detection** — identifies input language and applies calibrated thresholds
- **Batch processing** — moderate up to 100 texts in one request (14.6 texts/sec throughput)
- **Input sanitisation** — handles short text, whitespace, repeated characters, edge cases
- **Configurable thresholds** — override defaults via API or use calibrated per-language defaults
- **RESTful API** — FastAPI with auto-generated Swagger docs at `/docs`
- **31 tests passing** — language detection, model, and API endpoint coverage
- **Dockerised** — one command to run everything (Week 3)

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌───────────────────────┐
│   Client     │────▶│   FastAPI    │────▶│  DistilBERT           │
│  (Streamlit/ │     │   Backend    │     │  Multilingual         │
│   curl/app)  │◀────│              │◀────│  Toxicity Model       │
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

### Prerequisites
- Python 3.12 (PyTorch does not support 3.14 yet)
- pip

### Local Setup

```bash
git clone https://github.com/SonaliBalekundri/multilingual-content-moderator.git
cd multilingual-content-moderator

python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\Activate

pip install -r requirements.txt

# Run API
uvicorn app.main:app --reload --port 8000

# Open Swagger docs
# http://localhost:8000/docs
```

First request will take ~30 seconds (model download + loading). Subsequent requests: 16-40ms.

### Run Tests

```bash
pytest tests/test_moderator.py -v
```

### Docker Setup (Week 3)

```bash
docker-compose up --build
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/moderate` | Moderate a single text |
| POST | `/api/v1/moderate/batch` | Moderate up to 100 texts |
| GET | `/api/v1/languages` | List supported languages |
| GET | `/api/v1/categories` | List moderation categories |
| GET | `/api/v1/health` | Health check with model status |

See [API_DOCS.md](API_DOCS.md) for full documentation with example requests and responses.

## Multilingual Performance Benchmarks

**Model:** `citizenlab/distilbert-base-multilingual-cased-toxicity`

| Metric | English | Hindi | Arabic | Overall |
|--------|---------|-------|--------|---------|
| Threshold | 0.50 | 0.15 | 0.20 | — |
| Benchmark Accuracy | — | — | — | 91.7% (11/12) |

**Performance (CPU — Intel, 16GB RAM):**

| Metric | Value |
|--------|-------|
| Single request latency | 16–40ms |
| Batch (50 texts) throughput | 14.6 texts/sec |
| Model load time | ~2.3 seconds |

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
| ML/AI | PyTorch, Hugging Face Transformers (DistilBERT multilingual), langdetect |
| Backend | FastAPI, Pydantic, Uvicorn |
| Frontend | Streamlit (Week 3) |
| Infrastructure | Docker, Docker Compose (Week 3) |
| Testing | pytest, FastAPI TestClient |
| Utilities | pandas, matplotlib |

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
├── notebooks/                # Week 1 learning notebooks
├── docs/                     # Benchmark charts
├── API_DOCS.md               # Full API documentation
├── Dockerfile                # Container setup
├── docker-compose.yml        # Multi-service orchestration
└── requirements.txt          # Python dependencies
```

## What I Learned

### Week 1 — PyTorch + Multilingual NLP
- PyTorch fundamentals: tensors, autograd, nn.Module, device management
- Hugging Face Transformers: tokenizers, model loading, inference pipelines
- Evaluated two multilingual toxicity models — discovered severe cross-lingual bias in XLM-RoBERTa
- Implemented per-language threshold calibration to compensate for model bias
- Built a 30-text benchmark dataset across 3 languages with error analysis

### Week 2 — FastAPI Service + Testing
- Built REST API with FastAPI: single and batch moderation endpoints
- Input sanitisation: short text detection, whitespace handling, character normalisation
- Language-aware threshold routing: API auto-detects language and applies correct threshold
- 31 pytest tests across 3 layers: language detection, model wrapper, API endpoints
- Performance benchmarking: measured latency, throughput, and accuracy through the API

### Week 3 — Docker + Dashboard (upcoming)
- Docker multi-stage builds, Streamlit dashboard, deployment

## License

MIT
