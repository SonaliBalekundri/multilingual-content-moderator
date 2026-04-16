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
    try:
        mod = get_moderator()

        # Call the model — handles language detection, tokenization,
        # inference, and threshold application internally
        result = mod.moderate(
            text=request.text,
            threshold=request.threshold,
        )

        # Build the response — add the original text (moderate() doesn't return it)
        return ModerationResult(
            text=request.text,
            language=result["language"],
            verdict=result["verdict"],
            categories=result["categories"],   # Pydantic auto-converts dicts → CategoryResult
            confidence=result["confidence"],
            processing_time_ms=result["processing_time_ms"],
        )

    except Exception as e:
        logger.error(f"Moderation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Moderation failed: {str(e)}")

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
