"""
Content Moderation Model Wrapper.

Uses CitizenLab's DistilBERT multilingual toxicity model with
per-language threshold calibration to handle cross-lingual bias.

Key findings from model evaluation:
    - unitary/multilingual-toxic-xlm-roberta: 57% accuracy (poor Hindi/Arabic)
    - citizenlab/distilbert-base-multilingual-cased-toxicity: 100% accuracy
      with per-language thresholds (English 0.5, Hindi 0.15, Arabic 0.10)

Architecture:
    1. Detect input language (langdetect + alias mapping)
    2. Load DistilBERT multilingual toxicity model from Hugging Face
    3. Run inference and get toxic/non-toxic probabilities
    4. Apply language-specific threshold to determine verdict
"""

import time
from typing import Dict, List, Optional

import torch
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.config import settings
from app.utils.language import detect_language, detect_language_raw


class ContentModerator:
    """
    Multilingual content moderation using DistilBERT.

    Uses per-language thresholds to compensate for cross-lingual bias.
    English/Spanish toxic content scores ~0.5-0.99, but Hindi/Arabic
    score ~0.15-0.40 for equivalent insults. Per-language thresholds fix this.

    Usage:
        moderator = ContentModerator()
        result = moderator.moderate("some text to check")
    """

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the moderator with a pre-trained model.

        Args:
            model_name: Hugging Face model ID. Defaults to config value.
            device: "cpu" or "cuda". Defaults to config value.
        """
        self.model_name = model_name or settings.model_name
        self.device = device or settings.device
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the tokenizer and model from Hugging Face."""
        logger.info(f"Loading model: {self.model_name} on {self.device}")
        start = time.time()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        elapsed = time.time() - start
        logger.info(f"Model loaded in {elapsed:.2f}s")

        # Log model info
        num_labels = self.model.config.num_labels
        label_names = self.model.config.id2label
        logger.info(f"Model has {num_labels} output labels: {label_names}")

    def moderate(
        self,
        text: str,
        language: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> Dict:
        """
        Moderate a single text and return toxicity scores.

        Args:
            text: The text to moderate.
            language: Language code (auto-detected if not provided).
            threshold: Override threshold (uses per-language if not provided).

        Returns:
            Dictionary with:
                - language: detected/provided language code
                - language_raw: raw langdetect output (before alias mapping)
                - categories: per-category scores and flagged status
                - verdict: "toxic" or "clean"
                - confidence: overall confidence score
                - threshold_used: threshold applied for this language
                - processing_time_ms: inference time in milliseconds
        """
        start = time.time()

        # Step 1: Detect language if not provided
        language_raw = detect_language_raw(text)
        if language is None:
            language = detect_language(text)

        # Step 2: Get language-specific threshold
        if threshold is None:
            threshold = settings.get_threshold_for_language(language)

        # Step 3: Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)

        # Step 4: Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.sigmoid(outputs.logits).squeeze().cpu().numpy()

        # Step 5: Map probabilities to category names
        categories = {}
        any_flagged = False
        max_score = 0.0

        for idx, (label_id, label_name) in enumerate(self.model.config.id2label.items()):
            score = float(probs[idx]) if probs.ndim > 0 else float(probs)
            normalised_name = label_name.lower().replace(" ", "_")
            flagged = score >= threshold

            if flagged and normalised_name == "toxic":
                any_flagged = True

            max_score = max(max_score, score)

            categories[normalised_name] = {
                "score": round(score, 4),
                "flagged": flagged,
            }

        elapsed_ms = (time.time() - start) * 1000

        return {
            "language": language,
            "language_raw": language_raw,
            "categories": categories,
            "verdict": "toxic" if any_flagged else "clean",
            "confidence": round(max_score, 4),
            "threshold_used": threshold,
            "processing_time_ms": round(elapsed_ms, 2),
        }

    def moderate_batch(
        self,
        texts: List[str],
        threshold: Optional[float] = None,
    ) -> List[Dict]:
        """
        Moderate multiple texts with per-language threshold calibration.

        Args:
            texts: List of texts to moderate.
            threshold: Global override threshold (per-language if None).

        Returns:
            List of moderation results (same format as moderate()).
        """
        results = []
        for text in texts:
            result = self.moderate(text, threshold=threshold)
            results.append(result)
        return results
