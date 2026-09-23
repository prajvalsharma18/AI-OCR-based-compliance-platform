"""Prompt builder service for vision label extraction.

Loads, formats, and manages system prompts and repair retry prompts.
"""

from pathlib import Path
from typing import Optional

PROMPT_FILE_PATH = Path(__file__).resolve().parent.parent / "prompts" / "product_label_extraction.txt"


class PromptBuilder:
    """Manages prompts used for visual label extraction and repair retries."""

    _cached_prompt: Optional[str] = None

    @classmethod
    def get_extraction_prompt(cls) -> str:
        """Loads and caches the product label extraction prompt text."""
        if cls._cached_prompt is None:
            if not PROMPT_FILE_PATH.exists():
                raise FileNotFoundError(
                    f"Extraction prompt template not found at {PROMPT_FILE_PATH}"
                )
            cls._cached_prompt = PROMPT_FILE_PATH.read_text(encoding="utf-8").strip()
        return cls._cached_prompt

    @classmethod
    def build_repair_prompt(cls, invalid_output: str, error_details: str) -> str:
        """Constructs a targeted repair prompt when initial extraction fails validation.

        Args:
            invalid_output: The raw text returned by the model that failed validation.
            error_details: Specific Pydantic validation or JSON syntax errors encountered.

        Returns:
            Formatted repair prompt instructing the model to rectify errors while preserving visible facts.
        """
        base_prompt = cls.get_extraction_prompt()
        return (
            f"{base_prompt}\n\n"
            f"================================================================================\n"
            f"REPAIR ATTEMPT — CRITICAL INSTRUCTION:\n"
            f"Your previous extraction attempt was invalid and failed strict JSON schema validation.\n"
            f"Validation errors:\n"
            f"{error_details}\n\n"
            f"Previous invalid output:\n"
            f"{invalid_output}\n\n"
            f"Please carefully fix the schema errors above. Ensure:\n"
            f"1. Confidence values are strictly between 0.0 and 1.0.\n"
            f"2. Every status is strictly 'present', 'missing', or 'uncertain'.\n"
            f"3. Bounding boxes are either null or a list of exactly 4 numeric values [x_min, y_min, x_max, y_max].\n"
            f"4. Output MUST be valid, parseable JSON with no markdown backticks, explanations, or commentary.\n"
            f"================================================================================\n"
        )
