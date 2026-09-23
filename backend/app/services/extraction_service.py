"""Extraction service orchestrator.

Coordinates image validation, prompt construction, OpenAI vision model execution,
JSON parsing, strict Pydantic validation, and single-attempt repair retries.
"""

import json
import logging
from typing import Any, Optional, Tuple
from fastapi import UploadFile
from pydantic import ValidationError

from app.schemas.extraction import LabelExtractionResult
from app.services.prompt_builder import PromptBuilder
from app.services.vision_llm import (
    VisionLLMClient,
    VisionLLMConfigurationError,
    VisionLLMError,
    clean_json_markdown,
)
from app.utils.image import (
    ImageValidationError,
    validate_and_load_image,
)

logger = logging.getLogger(__name__)


class ExtractionServiceError(Exception):
    """Base exception for extraction service workflows."""

    def __init__(self, code: str, message: str, status_code: int = 422):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class ExtractionParsingError(ExtractionServiceError):
    """Raised when model output is not valid JSON or fails schema validation."""
    pass


class ExtractionService:
    """Coordinates semantic label extraction pipeline."""

    def __init__(self, vision_client: Optional[VisionLLMClient] = None):
        self.vision_client = vision_client or VisionLLMClient()

    def _parse_and_validate(self, raw_input: Any) -> Tuple[Optional[LabelExtractionResult], Optional[str]]:
        """Attempts to parse JSON and validate with LabelExtractionResult schema.

        Returns:
            Tuple of (validated_model, None) on success, or (None, error_description) on failure.
        """
        if isinstance(raw_input, LabelExtractionResult):
            return raw_input, None

        if isinstance(raw_input, dict):
            parsed_data = raw_input
        elif isinstance(raw_input, str):
            cleaned = clean_json_markdown(raw_input)
            try:
                parsed_data = json.loads(cleaned)
            except json.JSONDecodeError as exc:
                return None, f"Malformed JSON syntax: {exc.msg} at line {exc.lineno}, column {exc.colno}"
        else:
            return None, f"Expected a JSON string or dict, got {type(raw_input).__name__}"

        if not isinstance(parsed_data, dict):
            return None, f"Expected a JSON object at root, got {type(parsed_data).__name__}"

        try:
            validated = LabelExtractionResult.model_validate(parsed_data)
            return validated, None
        except ValidationError as exc:
            errors = []
            for err in exc.errors():
                loc = " -> ".join(str(p) for p in err.get("loc", []))
                msg = err.get("msg", "")
                errors.append(f"[{loc}]: {msg}")
            return None, "Schema validation errors:\n" + "\n".join(errors)

    async def extract_from_upload(self, file: UploadFile) -> LabelExtractionResult:
        """Processes an uploaded product image and extracts structured semantic label data.

        Workflow:
        1. Validate uploaded image file format, size, and decoding.
        2. Build extraction prompt.
        3. Call OpenAI Vision LLM client with structured schema.
        4. Validate output with Pydantic.
        5. If invalid, retry once with targeted repair prompt.
        6. Return validated result or raise typed exception.

        Args:
            file: FastAPI UploadFile object.

        Returns:
            Validated LabelExtractionResult model instance.

        Raises:
            ExtractionServiceError: For image, configuration, or parsing errors.
        """
        # Step 1: In-memory image validation
        try:
            img_result = await validate_and_load_image(file)
        except ImageValidationError as exc:
            status_code = 400
            if exc.code == "UNSUPPORTED_MEDIA_TYPE":
                status_code = 415
            elif exc.code == "FILE_TOO_LARGE":
                status_code = 413
            raise ExtractionServiceError(
                code=exc.code,
                message=exc.message,
                status_code=status_code,
            ) from exc

        # Step 2: Prompt preparation
        initial_prompt = PromptBuilder.get_extraction_prompt()

        # Step 3: First extraction attempt
        try:
            raw_output = await self.vision_client.extract(
                image_bytes=img_result.content,
                mime_type=img_result.mime_type,
                prompt=initial_prompt,
                schema=LabelExtractionResult,
            )
        except VisionLLMConfigurationError as exc:
            raise ExtractionServiceError(
                code=exc.code,
                message=exc.message,
                status_code=500,
            ) from exc
        except VisionLLMError as exc:
            raise ExtractionServiceError(
                code=exc.code,
                message=exc.message,
                status_code=502,
            ) from exc

        # Step 4: Parse & validate
        validated_result, validation_error = self._parse_and_validate(raw_output)
        if validated_result is not None:
            return validated_result

        # Step 5: Retry once with repair prompt
        logger.warning(
            "Initial model output failed validation: %s. Attempting repair retry...",
            validation_error,
        )
        repair_prompt = PromptBuilder.build_repair_prompt(
            invalid_output=str(raw_output),
            error_details=validation_error or "Unknown validation failure",
        )

        try:
            repair_output = await self.vision_client.extract(
                image_bytes=img_result.content,
                mime_type=img_result.mime_type,
                prompt=repair_prompt,
                schema=LabelExtractionResult,
            )
        except VisionLLMConfigurationError as exc:
            raise ExtractionServiceError(
                code=exc.code,
                message=exc.message,
                status_code=500,
            ) from exc
        except VisionLLMError as exc:
            raise ExtractionServiceError(
                code="REPAIR_CALL_FAILED",
                message=f"Model call failed during repair attempt: {exc.message}",
                status_code=502,
            ) from exc

        # Final validation check
        repaired_result, second_error = self._parse_and_validate(repair_output)
        if repaired_result is not None:
            logger.info("Extraction successfully repaired on second attempt.")
            return repaired_result

        # If still invalid after retry, return structured 422 error
        logger.error("Extraction failed validation after repair attempt: %s", second_error)
        raise ExtractionParsingError(
            code="EXTRACTION_SCHEMA_VALIDATION_FAILED",
            message=(
                f"Model output failed strict schema validation after repair attempt. "
                f"Details: {second_error}"
            ),
            status_code=422,
        )
