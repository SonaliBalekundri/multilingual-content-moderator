"""
Tests for the Multilingual Content Moderator.
Run with: pytest tests/test_moderator.py -v

Structure:
    TestLanguageDetection — language.py utility tests (fast, no model)
    TestContentModerator  — model wrapper tests (needs model, slower)
    TestAPI               — FastAPI endpoint tests (full integration)
"""

import pytest
import time
from fastapi.testclient import TestClient

from app.main import app
from app.utils.language import detect_language, get_supported_languages
from app.models.moderator import ContentModerator


# ============================================
# Fixtures — shared setup across tests
# ============================================

@pytest.fixture(scope="module")
def moderator():
    """
    Load the ContentModerator once for all tests in this module.
    scope="module" means it loads once, not once per test.
    Without this, each test would spend 30 seconds loading the model.
    """
    return ContentModerator()


@pytest.fixture(scope="module")
def client():
    """Create a TestClient for API testing."""
    return TestClient(app)


# ============================================
# 1. Language Detection Tests (fast, no model)
# ============================================

class TestLanguageDetection:
    """Tests for the language detection utility."""

    def test_detect_english(self):
        """English text should return 'en'."""
        result = detect_language("This is a normal English sentence")
        assert result == "en"

    def test_detect_hindi(self):
        """Hindi text should return 'hi'."""
        result = detect_language("यह एक हिंदी वाक्य है")
        assert result == "hi"

    def test_detect_arabic(self):
        """Arabic text should return 'ar'."""
        result = detect_language("هذه جملة باللغة العربية")
        assert result == "ar"

    def test_short_text_fallback(self):
        """Very short text should not crash — falls back to 'en'."""
        result = detect_language("hi")
        assert isinstance(result, str)
        # Don't assert specific language — short text detection is unreliable
        # Just verify it returns a valid string without crashing

    def test_supported_languages_list(self):
        """get_supported_languages() should return a list of dicts with code and name."""
        languages = get_supported_languages()
        assert isinstance(languages, list)
        assert len(languages) >= 3  # At least en, hi, ar

        # Each entry should have 'code' and 'name'
        for lang in languages:
            assert "code" in lang
            assert "name" in lang

    def test_supported_languages_contains_target_languages(self):
        """Our three target languages must be in the supported list."""
        languages = get_supported_languages()
        codes = [lang["code"] for lang in languages]
        assert "en" in codes
        assert "hi" in codes
        assert "ar" in codes


# ============================================
# 2. Content Moderator Tests (needs model)
# ============================================

class TestContentModerator:
    """Tests for the ML model wrapper."""

    def test_model_loads(self, moderator):
        """Model should load successfully."""
        assert moderator.model is not None
        assert moderator.tokenizer is not None

    def test_moderate_english_toxic(self, moderator):
        """Clearly toxic English text should be flagged."""
        result = moderator.moderate("You are a disgusting person")
        assert result["verdict"] == "toxic"
        assert result["language"] == "en"
        assert result["categories"]["toxic"]["score"] > 0.5
        assert result["categories"]["toxic"]["flagged"] is True

    def test_moderate_english_clean(self, moderator):
        """Clearly clean English text should not be flagged."""
        result = moderator.moderate("Have a wonderful day!")
        assert result["verdict"] == "clean"
        assert result["language"] == "en"
        assert result["categories"]["toxic"]["flagged"] is False

    def test_moderate_hindi_toxic(self, moderator):
        """Hindi toxic text should be caught with language-aware threshold."""
        result = moderator.moderate("तुम बेवकूफ और बेकार हो")
        assert result["verdict"] == "toxic"
        assert result["language"] == "hi"
        assert result["threshold_used"] == 0.15  # Hindi threshold

    def test_moderate_arabic_clean(self, moderator):
        """Arabic clean text should not be flagged."""
        result = moderator.moderate("شكراً لك على مساعدتك")
        assert result["verdict"] == "clean"
        assert result["language"] == "ar"
        assert result["threshold_used"] == 0.10  # Arabic threshold

    def test_threshold_override(self, moderator):
        """Passing a custom threshold should override language-aware defaults."""
        # With default Hindi threshold (0.15), this would be toxic
        # With high threshold (0.99), nothing gets flagged
        result = moderator.moderate("तुम बेवकूफ हो", threshold=0.99)
        assert result["verdict"] == "clean"
        assert result["threshold_used"] == 0.99

    def test_result_has_required_fields(self, moderator):
        """Moderate result should contain all expected fields."""
        result = moderator.moderate("Test text for field validation")
        required_fields = [
            "language", "language_raw", "categories",
            "verdict", "confidence", "threshold_used",
            "processing_time_ms",
        ]
        for field in required_fields:
            assert field in result, f"Missing field: {field}"

    def test_scores_are_valid_range(self, moderator):
        """All scores should be between 0.0 and 1.0."""
        result = moderator.moderate("Testing score ranges")
        for cat_name, cat_data in result["categories"].items():
            assert 0.0 <= cat_data["score"] <= 1.0, (
                f"{cat_name} score {cat_data['score']} out of range"
            )

    def test_processing_time_is_reasonable(self, moderator):
        """Single text should process in under 5 seconds on CPU."""
        result = moderator.moderate("Performance test")
        assert result["processing_time_ms"] < 5000

    def test_moderate_batch(self, moderator):
        """Batch moderation should return one result per text."""
        texts = ["Hello!", "You are stupid", "Thank you"]
        results = moderator.moderate_batch(texts)
        assert len(results) == 3
        # Each result should have a verdict
        for r in results:
            assert r["verdict"] in ["toxic", "clean"]


# ============================================
# 3. API Endpoint Tests (full integration)
# ============================================

class TestAPI:
    """Tests for FastAPI endpoints."""

    def test_root_endpoint(self, client):
        """Root endpoint should return API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data

    def test_health_endpoint(self, client):
        """Health check should confirm model is loaded."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

    def test_languages_endpoint(self, client):
        """Languages endpoint should return supported languages."""
        response = client.get("/api/v1/languages")
        assert response.status_code == 200
        data = response.json()
        assert "languages" in data
        assert len(data["languages"]) >= 3

    def test_categories_endpoint(self, client):
        """Categories endpoint should return model categories."""
        response = client.get("/api/v1/categories")
        assert response.status_code == 200
        data = response.json()
        assert "categories" in data
        assert "toxic" in data["categories"]

    def test_moderate_endpoint_toxic(self, client):
        """POST /moderate with toxic text should return toxic verdict."""
        response = client.post(
            "/api/v1/moderate",
            json={"text": "You are a disgusting and terrible person"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["verdict"] == "toxic"
        assert data["language"] == "en"
        assert "categories" in data
        assert "toxic" in data["categories"]

    def test_moderate_endpoint_clean(self, client):
        """POST /moderate with clean text should return clean verdict."""
        response = client.post(
            "/api/v1/moderate",
            json={"text": "Have a wonderful day, thank you!"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["verdict"] == "clean"

    def test_moderate_endpoint_returns_threshold(self, client):
        """Response should include the threshold that was used."""
        response = client.post(
            "/api/v1/moderate",
            json={"text": "This is a test sentence"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "threshold_used" in data

    def test_moderate_endpoint_with_custom_threshold(self, client):
        """Custom threshold should override language-aware default."""
        response = client.post(
            "/api/v1/moderate",
            json={"text": "This is a test", "threshold": 0.3},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["threshold_used"] == 0.3

    def test_moderate_endpoint_empty_text(self, client):
        """Empty text should be rejected by Pydantic validation."""
        response = client.post(
            "/api/v1/moderate",
            json={"text": ""},
        )
        # Pydantic rejects empty string (min_length=1) → 422
        assert response.status_code == 422

    def test_moderate_endpoint_whitespace_only(self, client):
        """Whitespace-only text should return clean with warning."""
        response = client.post(
            "/api/v1/moderate",
            json={"text": "   "},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["verdict"] == "clean"
        assert len(data["warnings"]) > 0

    def test_moderate_endpoint_short_text_warning(self, client):
        """Short text should include a warning about unreliability."""
        response = client.post(
            "/api/v1/moderate",
            json={"text": "hi"},
        )
        assert response.status_code == 200
        data = response.json()
        assert any("Short text" in w for w in data["warnings"])

    def test_batch_endpoint(self, client):
        """Batch endpoint should process multiple texts."""
        response = client.post(
            "/api/v1/moderate/batch",
            json={
                "texts": [
                    "You are terrible",
                    "Have a great day!",
                    "Thank you for your help",
                ]
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total_texts"] == 3
        assert data["flagged_count"] + data["clean_count"] == 3
        assert len(data["results"]) == 3

    def test_batch_endpoint_counts(self, client):
        """Batch counts should match individual verdicts."""
        response = client.post(
            "/api/v1/moderate/batch",
            json={
                "texts": [
                    "You are a disgusting person",
                    "Have a wonderful day!",
                ]
            },
        )
        assert response.status_code == 200
        data = response.json()

        # Count verdicts manually
        toxic_count = sum(1 for r in data["results"] if r["verdict"] == "toxic")
        clean_count = sum(1 for r in data["results"] if r["verdict"] == "clean")
        assert data["flagged_count"] == toxic_count
        assert data["clean_count"] == clean_count

    def test_invalid_threshold_rejected(self, client):
        """Threshold outside 0.0-1.0 should be rejected."""
        response = client.post(
            "/api/v1/moderate",
            json={"text": "Test text", "threshold": 1.5},
        )
        assert response.status_code == 422  # Pydantic validation error

    def test_missing_text_rejected(self, client):
        """Request without text field should be rejected."""
        response = client.post(
            "/api/v1/moderate",
            json={},
        )
        assert response.status_code == 422