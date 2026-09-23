"""Configuration settings for SIH PS 26034 Module 1.

Centralized configuration using Pydantic Settings.
Loads values from environment variables or .env file.
"""

import json
from typing import List, Union
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings schema."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    OPENAI_API_KEY: str = Field(
        default="",
        description="OpenAI API authentication key.",
    )
    LLM_MODEL: str = Field(
        default="gpt-5.6-luna",
        description="OpenAI Vision model identifier.",
    )
    MAX_IMAGE_SIZE_MB: float = Field(
        default=15.0,
        description="Maximum upload image file size in megabytes.",
    )
    ALLOWED_IMAGE_TYPES: List[str] = Field(
        default_factory=lambda: ["image/jpeg", "image/png", "image/webp"],
        description="Allowed MIME types for uploaded packaging images.",
    )
    PORT: int = Field(
        default=8001,
        description="Application server port.",
    )
    HOST: str = Field(
        default="127.0.0.1",
        description="Application server host.",
    )
    REPORTS_DIR: str = Field(
        default="reports",
        description="Root directory for inspection image, JSON, and PDF artifacts.",
    )

    # MongoDB Settings
    MONGODB_URI: str = Field(
        default="mongodb://localhost:27017",
        description="MongoDB connection URI string (standalone, replica set, or Atlas).",
    )
    MONGODB_DATABASE: str = Field(
        default="compliance_checker",
        description="MongoDB database name for storing inspections.",
    )
    MONGODB_ENABLED: bool = Field(
        default=True,
        description="Flag to enable or disable MongoDB persistence layer.",
    )
    MONGODB_TIMEOUT_MS: int = Field(
        default=2000,
        description="Server selection and connection timeout in milliseconds.",
    )

    @field_validator("ALLOWED_IMAGE_TYPES", mode="before")
    @classmethod
    def parse_allowed_image_types(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            v_str = v.strip()
            if v_str.startswith("[") and v_str.endswith("]"):
                try:
                    parsed = json.loads(v_str)
                    if isinstance(parsed, list):
                        return [str(item).strip() for item in parsed]
                except Exception:
                    pass
            return [item.strip() for item in v_str.split(",") if item.strip()]
        return v

    @property
    def max_image_size_bytes(self) -> int:
        """Returns max image file size converted to bytes."""
        return int(self.MAX_IMAGE_SIZE_MB * 1024 * 1024)


settings = Settings()
