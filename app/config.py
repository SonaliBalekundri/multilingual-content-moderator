"""
Configuration management for the Content Moderator.
Uses pydantic-settings to load from .env file or environment variables.
"""

from pydantic_settings import BaseSettings
from typing import Dict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Model — CitizenLab chosen over unitary/multilingual-toxic-xlm-roberta
    # because it shows much better cross-lingual performance for Hindi and Arabic.
    # See notebooks/02_huggingface_intro.py for the comparison analysis.
    model_name: str = "citizenlab/distilbert-base-multilingual-cased-toxicity"
    device: str = "cpu"  # "cpu" or "cuda"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "info"

    # Per-language thresholds (calibrated based on benchmark analysis)
    # English/Spanish score high (~0.5+ for toxic), Hindi/Arabic score lower (~0.15-0.40)
    # These thresholds were calibrated through benchmark testing.
    threshold_en: float = 0.5
    threshold_hi: float = 0.15
    threshold_ar: float = 0.10
    threshold_es: float = 0.5
    threshold_default: float = 0.5

    # Supported languages
    supported_languages: list = ["en", "hi", "ar", "es"]

    # Model output categories (CitizenLab model has 2 labels)
    categories: list = ["toxic", "non-toxic"]

    # Language aliases — languages that share scripts or are commonly
    # confused by langdetect, mapped to a supported language's threshold.
    # Discovered through testing: langdetect confuses Catalan/Spanish,
    # Marathi/Hindi due to shared scripts.
    language_aliases: dict = {
        "mr": "hi",   # Marathi → Hindi (both Devanagari script)
        "ne": "hi",   # Nepali → Hindi (both Devanagari script)
        "ur": "ar",   # Urdu → Arabic (similar script)
        "fa": "ar",   # Farsi → Arabic (similar script)
        "ca": "es",   # Catalan → Spanish (langdetect often confuses them)
        "pt": "es",   # Portuguese → Spanish (similar threshold needs)
        "gl": "es",   # Galician → Spanish (similar language)
    }

    @property
    def language_thresholds(self) -> Dict[str, float]:
        """Return per-language thresholds as a dictionary."""
        return {
            "en": self.threshold_en,
            "hi": self.threshold_hi,
            "ar": self.threshold_ar,
            "es": self.threshold_es,
        }

    def get_threshold_for_language(self, lang_code: str) -> float:
        """Get the calibrated threshold for a language, handling aliases."""
        mapped = self.language_aliases.get(lang_code, lang_code)
        return self.language_thresholds.get(mapped, self.threshold_default)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Singleton instance
settings = Settings()
