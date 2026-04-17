"""
API route handlers for the Content Moderator.
You'll implement these in Week 2.
"""

from fastapi import APIRouter, HTTPException
from loguru import logger
import time
import re

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


def sanitise_text(text: str) -> dict:
    """
    Clean and validate input text before sending to the model.
    Returns dict with cleaned text and any warnings.

    Guardrails from Week 1 testing:
    - Very short text (<3 words) → model unreliably flags "hi", "ok" as toxic
    - Whitespace-only → reject
    - Repeated characters → normalise ("stuuuuupid" → "stuupid")
    """
    warnings = []

    # Strip leading/trailing whitespace
    cleaned = text.strip()

    # Check for empty or whitespace-only text
    if not cleaned:
        return {"text": cleaned, "skip": True, "reason": "empty_text", "warnings": warnings}

    # Check minimum word count
    word_count = len(cleaned.split())
    if word_count < 3:
        warnings.append(f"Short text ({word_count} words) — classification may be unreliable")

    # Normalise repeated characters: "stuuuuupid" → "stuupid"
    cleaned = re.sub(r'(.)\1{2,}', r'\1\1', cleaned)

    return {"text": cleaned, "skip": False, "reason": None, "warnings": warnings}


@router.post("/moderate", response_model=ModerationResult)
async def moderate_text(request: ModerationRequest):
    """
    Moderate a single text for toxicity.

    Detects the language, runs the toxicity model, and returns
    per-category scores with a verdict.
    """
    try:
        # Step 1: Sanitise input
        sanitised = sanitise_text(request.text)

        # Handle empty text
        if sanitised["skip"]:
            return ModerationResult(
                text=request.text,
                language="unknown",
                verdict="clean",
                categories={},
                confidence=0.0,
                processing_time_ms=0.0,
                threshold_used=None,
                warnings=["Text is empty or whitespace-only — skipped classification"],
            )

        # Step 2: Run moderation on cleaned text
        mod = get_moderator()
        result = mod.moderate(
            text=sanitised["text"],
            threshold=request.threshold,
        )

        return ModerationResult(
            text=request.text,              # Return ORIGINAL text, not cleaned
            language=result["language"],
            verdict=result["verdict"],
            categories=result["categories"],
            confidence=result["confidence"],
            processing_time_ms=result["processing_time_ms"],
            threshold_used=result["threshold_used"],
            warnings=sanitised["warnings"],
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
    try:
        mod = get_moderator()
        start = time.time()

        results = []
        for text in request.texts:
            # Sanitise each text
            sanitised = sanitise_text(text)

            if sanitised["skip"]:
                results.append(ModerationResult(
                    text=text,
                    language="unknown",
                    verdict="clean",
                    categories={},
                    confidence=0.0,
                    processing_time_ms=0.0,
                    threshold_used=None,
                    warnings=["Text is empty or whitespace-only — skipped classification"],
                ))
                continue

            result = mod.moderate(
                text=sanitised["text"],
                threshold=request.threshold,
            )

            results.append(ModerationResult(
                text=text,
                language=result["language"],
                verdict=result["verdict"],
                categories=result["categories"],
                confidence=result["confidence"],
                processing_time_ms=result["processing_time_ms"],
                threshold_used=result["threshold_used"],
                warnings=sanitised["warnings"],
            ))

        total_ms = (time.time() - start) * 1000
        flagged = sum(1 for r in results if r.verdict == "toxic")

        return BatchModerationResult(
            results=results,
            total_texts=len(results),
            flagged_count=flagged,
            clean_count=len(results) - flagged,
            total_processing_time_ms=round(total_ms, 2),
        )

    except Exception as e:
        logger.error(f"Batch moderation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch moderation failed: {str(e)}")

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
