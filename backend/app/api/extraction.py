"""API route for label extraction endpoint."""

import logging
from typing import Union
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from app.schemas.extraction import (
    ErrorResponse,
    ExtractionResponse,
    LabelExtractionResult,
)
from app.services.extraction_service import (
    ExtractionService,
    ExtractionServiceError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Semantic Extraction"])


def get_extraction_service() -> ExtractionService:
    """Dependency provider for ExtractionService."""
    return ExtractionService()


@router.post(
    "/extract",
    response_model=ExtractionResponse,
    responses={
        200: {"model": ExtractionResponse, "description": "Structured semantic label extracted successfully"},
        400: {"model": ErrorResponse, "description": "Invalid file upload or corrupted image"},
        415: {"model": ErrorResponse, "description": "Unsupported media format"},
        413: {"model": ErrorResponse, "description": "File exceeds maximum size limit"},
        422: {"model": ErrorResponse, "description": "Model output failed schema validation"},
        500: {"model": ErrorResponse, "description": "Internal server or configuration error"},
        502: {"model": ErrorResponse, "description": "Upstream vision model error"},
    },
    summary="Extract visible label declarations from packaged commodity image",
    description=(
        "Accepts a product image (JPEG, PNG, WEBP), performs visual semantic extraction "
        "via a vision-capable LLM, strictly validates against the SIH PS 26034 Pydantic contract, "
        "and returns structured label information. Does NOT perform legal compliance checks."
    ),
)
async def extract_label(
    image: UploadFile = File(..., description="Product packaging photo (JPEG, PNG, or WEBP)"),
    service: ExtractionService = Depends(get_extraction_service),
) -> Union[ExtractionResponse, JSONResponse]:
    """Receives image upload, delegates extraction to ExtractionService, and returns standardized envelope."""
    try:
        extracted_data: LabelExtractionResult = await service.extract_from_upload(image)
        return ExtractionResponse(success=True, data=extracted_data)

    except ExtractionServiceError as exc:
        logger.warning("Extraction service error [%s]: %s", exc.code, exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                },
            },
        )
    except Exception as exc:
        logger.exception("Unexpected error occurred during extraction: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": f"An unexpected error occurred: {str(exc)}",
                },
            },
        )
