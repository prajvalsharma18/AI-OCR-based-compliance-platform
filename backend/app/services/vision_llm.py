"""Vision LLM Client service.

Isolates all interactions with the OpenAI Python SDK and vision models using the Responses API.
"""

import asyncio
import base64
import logging
from typing import Any, Optional, Type
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)
from pydantic import BaseModel

from app.config import settings
from app.schemas.extraction import LabelExtractionResult

logger = logging.getLogger(__name__)


class VisionLLMError(Exception):
    """Base exception for vision LLM communication issues."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class VisionLLMConfigurationError(VisionLLMError):
    """Raised when API key or required model settings are missing or invalid."""
    pass


class VisionLLMAPIError(VisionLLMError):
    """Raised when the vision API returns an error or fails to respond."""
    pass


def clean_json_markdown(text: str) -> str:
    """Removes markdown code block formatting (```json ... ```) if present."""
    content = text.strip()
    if "```json" in content:
        content = content.split("```json", 1)[1]
        if "```" in content:
            content = content.split("```", 1)[0]
    elif "```" in content:
        content = content.split("```", 1)[1]
        if "```" in content:
            content = content.split("```", 1)[0]
    return content.strip()


class VisionLLMClient:
    """Wraps OpenAI Python SDK using the Responses API for visual semantic extraction."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.model_name = model_name or settings.LLM_MODEL
        self._client: Optional[OpenAI] = None

    def _get_client(self) -> OpenAI:
        if self._client is not None:
            return self._client

        if not self.api_key or not self.api_key.strip():
            raise VisionLLMConfigurationError(
                code="MISSING_API_KEY",
                message=(
                    "OPENAI_API_KEY is not configured. Please provide a valid OpenAI API key "
                    "in the environment or .env file."
                ),
            )

        try:
            self._client = OpenAI(api_key=self.api_key.strip())
            return self._client
        except Exception as exc:
            raise VisionLLMConfigurationError(
                code="CLIENT_INIT_FAILED",
                message=f"Failed to initialize OpenAI client: {str(exc)}",
            ) from exc

    def _execute_responses_call(
        self,
        image_data_url: str,
        prompt: str,
        schema: Type[BaseModel],
    ) -> str:
        """Synchronously executes the OpenAI Responses API call."""
        client = self._get_client()

        input_payload = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt,
                    },
                    {
                        "type": "input_image",
                        "image_url": image_data_url,
                    },
                ],
            }
        ]

        try:
            response = client.responses.parse(
                model=self.model_name,
                input=input_payload,
                text_format=schema,
            )

            # Prefer raw text if present, fallback to parsed model JSON
            output_text = getattr(response, "output_text", None)
            if output_text and output_text.strip():
                return clean_json_markdown(output_text)

            output_parsed = getattr(response, "output_parsed", None)
            if output_parsed is not None:
                if hasattr(output_parsed, "model_dump_json"):
                    return output_parsed.model_dump_json()
                elif isinstance(output_parsed, dict):
                    import json
                    return json.dumps(output_parsed)

            raise VisionLLMAPIError(
                code="EMPTY_MODEL_RESPONSE",
                message="The OpenAI vision model returned an empty response.",
            )

        except AuthenticationError as exc:
            raise VisionLLMConfigurationError(
                code="OPENAI_AUTH_ERROR",
                message="OpenAI authentication failed. Please check that OPENAI_API_KEY is valid.",
            ) from exc
        except RateLimitError as exc:
            raise VisionLLMAPIError(
                code="OPENAI_RATE_LIMIT",
                message="OpenAI rate limit reached or quota exceeded. Please retry shortly.",
            ) from exc
        except BadRequestError as exc:
            raise VisionLLMAPIError(
                code="OPENAI_BAD_REQUEST",
                message=f"OpenAI bad request: {exc.message}",
            ) from exc
        except APITimeoutError as exc:
            raise VisionLLMAPIError(
                code="OPENAI_TIMEOUT",
                message="OpenAI request timed out.",
            ) from exc
        except APIConnectionError as exc:
            raise VisionLLMAPIError(
                code="OPENAI_CONNECTION_ERROR",
                message="Could not connect to OpenAI API servers.",
            ) from exc
        except APIError as exc:
            raise VisionLLMAPIError(
                code="OPENAI_API_ERROR",
                message=f"OpenAI API error: {exc.message}",
            ) from exc
        except VisionLLMError:
            raise
        except Exception as exc:
            raise VisionLLMAPIError(
                code="LLM_REQUEST_FAILED",
                message=f"Vision model request failed ({self.model_name}): {str(exc)}",
            ) from exc

    async def extract(
        self,
        image_bytes: bytes,
        mime_type: str,
        prompt: str,
        schema: Optional[Type[BaseModel]] = None,
    ) -> str:
        """Sends a product image and extraction prompt to OpenAI Vision via Responses API.

        Args:
            image_bytes: Raw bytes of the packaging photo.
            mime_type: Supported MIME format (e.g. image/jpeg, image/png, image/webp).
            prompt: Semantic extraction instructions.
            schema: Target Pydantic model for structured output enforcement (defaults to LabelExtractionResult).

        Returns:
            JSON-formatted string conforming to the extraction schema.

        Raises:
            VisionLLMConfigurationError: For missing or invalid API keys.
            VisionLLMAPIError: For remote API errors, rate limits, or network timeouts.
        """
        target_schema = schema or LabelExtractionResult
        base64_encoded = base64.b64encode(image_bytes).decode("utf-8")
        image_data_url = f"data:{mime_type};base64,{base64_encoded}"

        return await asyncio.to_thread(
            self._execute_responses_call,
            image_data_url=image_data_url,
            prompt=prompt,
            schema=target_schema,
        )

    async def extract_label_data(
        self,
        image_bytes: bytes,
        mime_type: str,
        prompt: str,
        schema: Optional[Type[BaseModel]] = None,
    ) -> str:
        """Backward-compatible alias for extract()."""
        return await self.extract(
            image_bytes=image_bytes,
            mime_type=mime_type,
            prompt=prompt,
            schema=schema,
        )
