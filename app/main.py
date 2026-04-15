"""
Multilingual Content Moderator - FastAPI Application.

Run with: uvicorn app.main:app --reload --port 8000
Docs at:  http://localhost:8000/docs
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api.routes import router
from app.config import settings

app = FastAPI(
    title="Multilingual Content Moderator",
    description="Detect toxicity, hate speech, and harmful content across English, Hindi, and Arabic",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware (needed for Streamlit / React frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1", tags=["moderation"])


@app.on_event("startup")
async def startup_event():
    """Pre-load the model on startup so first request isn't slow."""
    logger.info("Starting Multilingual Content Moderator...")
    logger.info(f"Model: {settings.model_name}")
    logger.info(f"Device: {settings.device}")
    # Model loads lazily on first request - uncomment below to pre-load:
    # from app.api.routes import get_moderator
    # get_moderator()


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Multilingual Content Moderator",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
