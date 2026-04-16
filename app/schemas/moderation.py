"""
Pydantic schemas for API request/response models.
These define the contract for the API — what clients send and receive.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# ============================================
# Request Models
# ============================================

class ModerationRequest(BaseModel):
    """Request body for single text moderation."""
    text: str = Field(..., min_length=1, max_length=5000, description="Text to moderate")
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Override threshold (uses language-aware defaults if not set)")
    thresholds: Optional[Dict[str, float]] = Field(None, description="Per-category thresholds (overrides global)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "You are a wonderful person!",
                    "threshold": 0.5,
                }
            ]
        }
    }


class BatchModerationRequest(BaseModel):
    """Request body for batch text moderation."""
    texts: List[str] = Field(..., min_length=1, max_length=100, description="List of texts to moderate")
    threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Global threshold")
    thresholds: Optional[Dict[str, float]] = Field(None, description="Per-category thresholds")


# ============================================
# Response Models
# ============================================

class CategoryResult(BaseModel):
    """Result for a single toxicity category."""
    score: float = Field(..., description="Toxicity score (0.0 to 1.0)")
    flagged: bool = Field(..., description="Whether this category exceeds the threshold")


class ModerationResult(BaseModel):
    """Response for a single text moderation."""
    text: str
    language: str = Field(..., description="Detected language code (en, hi, ar)")
    verdict: str = Field(..., description="'toxic' or 'clean'")
    categories: Dict[str, CategoryResult]
    confidence: float
    processing_time_ms: float


class BatchModerationResult(BaseModel):
    """Response for batch text moderation."""
    results: List[ModerationResult]
    total_texts: int
    flagged_count: int
    clean_count: int
    total_processing_time_ms: float


class LanguagesResponse(BaseModel):
    """Response for supported languages endpoint."""
    languages: List[Dict[str, str]]


class CategoriesResponse(BaseModel):
    """Response for moderation categories endpoint."""
    categories: List[str]


class HealthResponse(BaseModel):
    """Response for health check endpoint."""
    status: str
    model_loaded: bool
    model_name: str
    device: str
