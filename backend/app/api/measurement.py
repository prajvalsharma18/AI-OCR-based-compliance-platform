"""API route for Module 2A Calibrated Numeral Size Measurement."""

import logging
from typing import Optional, Union
from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from fastapi.responses import JSONResponse

from app.schemas.measurement import (
    MeasurementErrorResponse,
    MeasurementResponse,
    MeasurementResponseData,
)
from app.services.measurement_service import (
    MeasurementService,
    MeasurementServiceError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Numeral Size Measurement"])


def get_measurement_service() -> MeasurementService:
    """Dependency provider for MeasurementService."""
    return MeasurementService()


@router.post(
    "/measure",
    response_model=MeasurementResponse,
    responses={
        200: {"model": MeasurementResponse, "description": "Numerals successfully isolated and measured"},
        400: {"model": MeasurementErrorResponse, "description": "Invalid package dimensions, malformed JSON, or invalid image"},
        413: {"model": MeasurementErrorResponse, "description": "File exceeds maximum size limit"},
        415: {"model": MeasurementErrorResponse, "description": "Unsupported media format"},
        422: {"model": MeasurementErrorResponse, "description": "Invalid extraction schema, missing numeral_region, or invalid package_bbox_px"},
        500: {"model": MeasurementErrorResponse, "description": "Internal server error"},
    },
    summary="Calibrated physical measurement of statutory package numerals",
    description=(
        "Consumes a product packaging image, Module 1 semantic extraction JSON, "
        "and user-supplied real-world package dimensions (in mm). "
        "Automatically detects package boundary for pixel-to-mm calibration (or uses optional manual package_bbox_px override), "
        "isolates numeral glyphs within semantic regions, and calculates "
        "physical numeral height in millimetres. Does NOT make legal compliance decisions."
    ),
)
async def measure_numerals(
    image: UploadFile = File(..., description="Product packaging photo (JPEG, PNG, or WEBP)"),
    extraction_json: str = Form(..., description="Structured JSON extraction produced by Module 1"),
    package_width_mm: float = Form(..., description="Physical package width in millimetres (> 0)"),
    package_height_mm: float = Form(..., description="Physical package height in millimetres (> 0)"),
    package_bbox_px: Optional[str] = Form(
        default=None,
        description="Optional development/debug pixel bounding box override: [x_min, y_min, x_max, y_max]. If omitted, the package boundary is automatically detected from the image.",
        examples=["[170, 30, 830, 980]"],
    ),
    include_debug_image: bool = Form(
        default=False,
        description="Optional flag: if true, returns in-memory base64 JPEG with debug visualization overlays (Green=Package, Blue=Semantic, Red=Numeral).",
    ),
    service: MeasurementService = Depends(get_measurement_service),
) -> Union[MeasurementResponse, JSONResponse]:
    """Receives image, Module 1 JSON, package dimensions, and package bbox to calculate physical numeral sizes."""
    try:
        data: MeasurementResponseData = await service.measure_packaging_numerals(
            image_file=image,
            extraction_json=extraction_json,
            package_width_mm=package_width_mm,
            package_height_mm=package_height_mm,
            package_bbox_px=package_bbox_px,
            include_debug_image=include_debug_image,
        )
        return MeasurementResponse(success=True, data=data)

    except MeasurementServiceError as exc:
        logger.warning("Measurement service error [%s]: %s", exc.code, exc.message)
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
        logger.exception("Unexpected error during numeral measurement: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": f"An unexpected error occurred during measurement: {str(exc)}",
                },
            },
        )
