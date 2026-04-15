"""
Language detection utility.
Detects the language of input text and handles script-sharing aliases.

Key discovery during testing:
- langdetect confuses Marathi with Hindi (both Devanagari script)
- langdetect confuses Catalan with Spanish (similar Romance languages)
- Language aliases map these to the correct supported language threshold
"""

from langdetect import detect, detect_langs, LangDetectException
from loguru import logger
from typing import Optional, List, Dict

# Supported languages and their display names
SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "ar": "Arabic",
    "es": "Spanish",
}

# Languages that share scripts with supported languages or are commonly
# confused by langdetect — map them to the correct supported language.
LANGUAGE_ALIASES = {
    "mr": "hi",   # Marathi → Hindi (both use Devanagari script)
    "ne": "hi",   # Nepali → Hindi (both use Devanagari script)
    "ur": "ar",   # Urdu → Arabic (similar script)
    "fa": "ar",   # Farsi → Arabic (similar script)
    "ca": "es",   # Catalan → Spanish (langdetect often confuses them)
    "pt": "es",   # Portuguese → Spanish (similar threshold needs)
    "gl": "es",   # Galician → Spanish (similar language)
}


def detect_language(text: str) -> str:
    """
    Detect the language of the input text.

    Handles script-sharing aliases (e.g., Marathi → Hindi, Catalan → Spanish).

    Args:
        text: Input text to detect language for.

    Returns:
        ISO 639-1 language code (e.g., "en", "hi", "ar", "es").
        Returns "en" as fallback if detection fails.
    """
    try:
        detected = detect(text)

        # Map aliased languages to supported ones
        mapped = LANGUAGE_ALIASES.get(detected, detected)

        if mapped in SUPPORTED_LANGUAGES:
            return mapped
        else:
            logger.warning(
                f"Detected unsupported language: {detected} "
                f"(mapped: {mapped}), falling back to 'en'"
            )
            return "en"
    except LangDetectException as e:
        logger.warning(f"Language detection failed: {e}, falling back to 'en'")
        return "en"


def detect_language_raw(text: str) -> str:
    """
    Detect language without alias mapping.
    Useful for logging and debugging.

    Returns:
        Raw detected language code before alias mapping.
    """
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def detect_language_with_confidence(text: str) -> Dict:
    """
    Detect language with confidence scores for all candidates.

    Returns both raw and mapped language codes, plus confidence breakdown.
    Useful for debugging language detection issues.

    Args:
        text: Input text.

    Returns:
        Dictionary with detected language, mapped language, and confidence.
    """
    try:
        results = detect_langs(text)
        raw = str(results[0].lang)
        mapped = LANGUAGE_ALIASES.get(raw, raw)

        return {
            "detected_raw": raw,
            "detected_mapped": mapped if mapped in SUPPORTED_LANGUAGES else "en",
            "confidence": round(results[0].prob, 4),
            "all_candidates": [
                {
                    "lang": str(r.lang),
                    "mapped_to": LANGUAGE_ALIASES.get(str(r.lang), str(r.lang)),
                    "confidence": round(r.prob, 4),
                }
                for r in results
            ],
        }
    except LangDetectException:
        return {
            "detected_raw": "unknown",
            "detected_mapped": "en",
            "confidence": 0.0,
            "all_candidates": [],
        }


def get_supported_languages() -> List[Dict[str, str]]:
    """Return list of supported languages with their details."""
    return [
        {"code": code, "name": name}
        for code, name in SUPPORTED_LANGUAGES.items()
    ]


def get_language_aliases() -> Dict[str, str]:
    """Return the alias mapping for reference."""
    return {
        alias: {
            "maps_to": target,
            "target_name": SUPPORTED_LANGUAGES.get(target, "Unknown"),
        }
        for alias, target in LANGUAGE_ALIASES.items()
    }
