# 📡 API Documentation — Multilingual Content Moderator

**Base URL:** `http://localhost:8000/api/v1`
**Interactive Docs:** `http://localhost:8000/docs` (Swagger UI)

---

## Authentication

No authentication required (development mode). Add API key middleware for production.

---

## Endpoints

### POST /moderate

Moderate a single text for toxicity. Auto-detects language and applies calibrated thresholds.

**Request Body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `text` | string | Yes | — | Text to moderate (1–5000 characters) |
| `threshold` | float | No | Per-language | Override threshold (0.0–1.0). If not set, uses language-aware defaults: EN=0.5, HI=0.15, AR=0.20 |

**Example Request:**

```json
{
  "text": "You are a disgusting person"
}
```

**Example Response (toxic):**

```json
{
  "text": "You are a disgusting person",
  "language": "en",
  "verdict": "toxic",
  "categories": {
    "toxic": {
      "score": 0.9301,
      "flagged": true
    },
    "not_toxic": {
      "score": 0.1017,
      "flagged": false
    }
  },
  "confidence": 0.9301,
  "processing_time_ms": 40.03,
  "threshold_used": 0.5,
  "warnings": []
}
```

**Example Response (clean):**

```json
{
  "text": "Have a wonderful day!",
  "language": "en",
  "verdict": "clean",
  "categories": {
    "toxic": {
      "score": 0.0687,
      "flagged": false
    },
    "not_toxic": {
      "score": 0.9061,
      "flagged": true
    }
  },
  "confidence": 0.9061,
  "processing_time_ms": 16.59,
  "threshold_used": 0.5,
  "warnings": []
}
```

**Example — Hindi with language-aware threshold:**

```json
// Request
{"text": "तुम बेवकूफ हो"}

// Response
{
  "text": "तुम बेवकूफ हो",
  "language": "hi",
  "verdict": "toxic",
  "categories": {
    "toxic": {"score": 0.4212, "flagged": true},
    "not_toxic": {"score": 0.5547, "flagged": true}
  },
  "confidence": 0.5547,
  "processing_time_ms": 41.1,
  "threshold_used": 0.15,
  "warnings": []
}
```

Note: The toxic score (0.4212) would be missed with a default 0.5 threshold. The Hindi-calibrated threshold of 0.15 correctly catches it.

**Example — Short text warning:**

```json
// Request
{"text": "hi"}

// Response
{
  "text": "hi",
  "language": "en",
  "verdict": "clean",
  "categories": {
    "toxic": {"score": 0.206, "flagged": false},
    "not_toxic": {"score": 0.7573, "flagged": true}
  },
  "confidence": 0.7573,
  "processing_time_ms": 19.17,
  "threshold_used": 0.5,
  "warnings": [
    "Short text (1 words) — classification may be unreliable"
  ]
}
```

**Example — Whitespace-only text:**

```json
// Request
{"text": "   "}

// Response
{
  "text": "   ",
  "language": "unknown",
  "verdict": "clean",
  "categories": {},
  "confidence": 0.0,
  "processing_time_ms": 0.0,
  "threshold_used": null,
  "warnings": [
    "Text is empty or whitespace-only — skipped classification"
  ]
}
```

**Example — Custom threshold override:**

```json
// Request — force high threshold
{"text": "You are a terrible person", "threshold": 0.95}

// Response — not flagged because score (0.87) < 0.95
{
  "text": "You are a terrible person",
  "language": "en",
  "verdict": "clean",
  "threshold_used": 0.95,
  ...
}
```

**Error Responses:**

| Status | Cause | Example |
|--------|-------|---------|
| 422 | Invalid input (empty text, threshold out of range) | `{"text": "", "threshold": 0.5}` |
| 500 | Model inference failure | Internal server error |

---

### POST /moderate/batch

Moderate multiple texts in one request. More efficient than calling `/moderate` individually.

**Request Body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `texts` | string[] | Yes | — | List of texts (1–100 items) |
| `threshold` | float | No | Per-language | Override threshold for all texts |

**Example Request:**

```json
{
  "texts": [
    "You are a disgusting person",
    "Have a wonderful day!",
    "तुम बेवकूफ हो",
    "آپکا دن شبھ ہو",
    "أنت غبي ولا تستحق الاحترام",
    "شكراً لك على مساعدتك"
  ]
}
```

**Example Response:**

```json
{
  "results": [
    {
      "text": "You are a disgusting person",
      "language": "en",
      "verdict": "toxic",
      "categories": {
        "toxic": {"score": 0.9301, "flagged": true},
        "not_toxic": {"score": 0.1017, "flagged": false}
      },
      "confidence": 0.9301,
      "processing_time_ms": 390.48,
      "threshold_used": 0.5,
      "warnings": []
    },
    {
      "text": "Have a wonderful day!",
      "language": "en",
      "verdict": "clean",
      "categories": {
        "toxic": {"score": 0.0687, "flagged": false},
        "not_toxic": {"score": 0.9061, "flagged": true}
      },
      "confidence": 0.9061,
      "processing_time_ms": 16.64,
      "threshold_used": 0.5,
      "warnings": []
    }
  ],
  "total_texts": 6,
  "flagged_count": 3,
  "clean_count": 3,
  "total_processing_time_ms": 539.86
}
```

**Batch Performance:**

| Batch Size | Total Time | Per Text | Throughput |
|-----------|-----------|---------|------------|
| 5 | 148ms | 30ms | 33 texts/sec |
| 10 | 253ms | 25ms | 40 texts/sec |
| 20 | 548ms | 27ms | 36 texts/sec |
| 50 | 1420ms | 28ms | 35 texts/sec |

Note: Throughput numbers above reflect model inference time only. End-to-end HTTP latency adds ~2ms overhead per request.

---

### GET /languages

Return list of supported languages.

**Example Response:**

```json
{
  "languages": [
    {"code": "en", "name": "English"},
    {"code": "hi", "name": "Hindi"},
    {"code": "ar", "name": "Arabic"},
    {"code": "es", "name": "Spanish"}
  ]
}
```

---

### GET /categories

Return list of model output categories.

**Example Response:**

```json
{
  "categories": ["toxic", "non-toxic"]
}
```

---

### GET /health

Health check endpoint. Confirms model is loaded and ready.

**Example Response:**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "citizenlab/distilbert-base-multilingual-cased-toxicity",
  "device": "cpu"
}
```

---

## Language-Aware Thresholds

The model exhibits cross-lingual bias — identical toxic content scores differently across languages. To compensate, language-specific thresholds are applied automatically:

| Language | Threshold | Why |
|----------|-----------|-----|
| English | 0.50 | Standard — model trained primarily on English data |
| Spanish | 0.50 | Similar performance to English |
| Arabic | 0.20 | Arabic toxic content scores lower (0.10–0.40 range) |
| Hindi | 0.15 | Hindi toxic content scores lowest (0.15–0.45 range) |

When no threshold is provided in the request, the API auto-detects the language and applies the calibrated threshold. You can override this by explicitly passing a `threshold` value.

### Language Alias Mapping

Some languages share scripts with supported languages. `langdetect` sometimes confuses them, so aliases map to the closest supported language:

| Detected | Mapped To | Reason |
|----------|-----------|--------|
| Marathi (mr) | Hindi (hi) | Both use Devanagari script |
| Nepali (ne) | Hindi (hi) | Both use Devanagari script |
| Urdu (ur) | Arabic (ar) | Similar script |
| Farsi (fa) | Arabic (ar) | Similar script |
| Catalan (ca) | Spanish (es) | langdetect frequently confuses them |
| Portuguese (pt) | Spanish (es) | Similar threshold needs |
| Galician (gl) | Spanish (es) | Similar language family |

---

## Input Sanitisation

The API applies these guardrails before classification:

| Guard | What It Does | Example |
|-------|-------------|---------|
| Whitespace check | Skips classification for empty/whitespace-only text | `"   "` → clean with warning |
| Short text warning | Warns when text has fewer than 3 words | `"hi"` → classified but with warning |
| Character normalisation | Reduces repeated characters | `"stuuuuupid"` → `"stuupid"` |

---

## Error Handling

| Status Code | Meaning | When |
|------------|---------|------|
| 200 | Success | Request processed successfully |
| 422 | Validation Error | Invalid input (empty text, threshold out of range, missing required field) |
| 500 | Server Error | Model inference failure or unexpected error |

**422 Example:**

```json
// Request with empty text
{"text": ""}

// Response
{
  "detail": [
    {
      "type": "string_too_short",
      "loc": ["body", "text"],
      "msg": "String should have at least 1 character",
      "input": ""
    }
  ]
}
```

---

## Testing

```bash
# Run all 31 tests
pytest tests/test_moderator.py -v

# Run performance benchmarks (server must be running)
python tests/test_performance.py
```

### Test Coverage

| Group | Tests | What It Covers |
|-------|-------|---------------|
| Language Detection | 6 | English/Hindi/Arabic detection, short text fallback, supported languages list |
| Content Moderator | 10 | Toxic/clean classification, threshold override, field validation, score ranges, batch processing |
| API Endpoints | 15 | All 5 endpoints, input validation, edge cases, batch counts |
