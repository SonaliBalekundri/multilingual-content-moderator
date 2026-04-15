"""
API route handlers for the Content Moderator.
You'll implement these in Week 2.
"""

from fastapi import APIRouter, HTTPException
from loguru import logger

from app.schemas.moderation import (
    ModerationRequest,
    ModerationResult,
    BatchModerationRequest,
    BatchModerationResult,
    LanguagesResponse,
    CategoriesResponse,
    HealthResponse,
)
from app.utils.language import detect_language, get_supported_languages
from app.config import settings

router = APIRouter()

# Model will be loaded once at startup and shared across requests
moderator = None


def get_moderator():
    """Get the loaded moderator model (lazy initialization)."""
    global moderator
    if moderator is None:
        from app.models.moderator import ContentModerator
        moderator = ContentModerator()
    return moderator


@router.post("/moderate", response_model=ModerationResult)
async def moderate_text(request: ModerationRequest):
    """
    Moderate a single text for toxicity.

    Detects the language, runs the toxicity model, and returns
    per-category scores with a verdict.
    """
    # TODO: Week 2 - Implement this endpoint
    # Steps:
    # 1. Detect language using detect_language()
    # 2. Build thresholds dict from request
    # 3. Call moderator.moderate(text, thresholds)
    # 4. Combine language + moderation result into ModerationResult
    raise HTTPException(status_code=501, detail="Coming in Week 2!")


@router.post("/moderate/batch", response_model=BatchModerationResult)
async def moderate_batch(request: BatchModerationRequest):
    """
    Moderate multiple texts in one request.

    Processes up to 100 texts and returns individual results
    plus aggregate statistics.
    """
    # TODO: Week 2 - Implement batch endpoint
    raise HTTPException(status_code=501, detail="Coming in Week 2!")


@router.get("/languages", response_model=LanguagesResponse)
async def list_languages():
    """Return list of supported languages."""
    return LanguagesResponse(languages=get_supported_languages())


@router.get("/categories", response_model=CategoriesResponse)
async def list_categories():
    """Return list of moderation categories."""
    return CategoriesResponse(categories=settings.categories)


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    mod = get_moderator()
    return HealthResponse(
        status="healthy",
        model_loaded=mod.model is not None,
        model_name=mod.model_name,
        device=mod.device,
    )
