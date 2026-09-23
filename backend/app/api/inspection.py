"""End-to-end inspection orchestration endpoint for SIH 2026 PS 26034.

Automated Compliance Checker for Packaged Commodities:
Single entrypoint API that orchestrates the complete compliance pipeline:
1. Exact original image receipt and storage via ImageStorageService.
2. Cryptographic SHA-256 provenance calculation.
3. Module 1: Semantic extraction via OpenAI Vision.
4. Module 2A: Calibrated numeral height measurement via OpenCV.
5. Module 3C: Rule 6 declaration visibility evaluation.
6. Module 3A: Rule 7 numeral-height statutory evaluation.
7. Module 3B: Rule 8 spatial clearance analysis.
8. Deterministic Unified Inspection JSON aggregation.
9. Persistent snapshot storage under reports/inspections/<inspection_id>/inspection.json.
10. Complete InspectionResponse envelope return.
"""

import io
import json
import logging
import math
from typing import Any, Dict, List, Optional, Union
import cv2
from fastapi import APIRouter, Depends, File, Form, Request, UploadFile, status
from fastapi.responses import JSONResponse
import numpy as np
from app.config import settings
from app.repositories.inspection_repository import InspectionRepository, InspectionRepositoryError
from app.schemas.history import PersistenceInfo

from app.cv.package_detector import detect_package_boundary
from app.schemas.compliance import RuleSourceMode
from app.schemas.inspection import (
    InspectionErrorResponse,
    InspectionResponse,
    InspectionResponseData,
)
from app.services.extraction_service import (
    ExtractionParsingError,
    ExtractionService,
    ExtractionServiceError,
)
from app.services.image_storage_service import (
    ImageStorageError,
    ImageStorageService,
)
from app.services.inspection_service import (
    InspectionService,
    InspectionServiceError,
)
from app.services.measurement_service import (
    MeasurementService,
    MeasurementServiceError,
)
from app.services.rule6_service import (
    Rule6Service,
    Rule6ServiceError,
)
from app.services.rule7_service import (
    Rule7Service,
    Rule7ServiceError,
)
from app.services.rule8_service import (
    Rule8Service,
    Rule8ServiceError,
)
from app.services.report_service import ReportService, ReportServiceError
from app.utils.image import (
    ImageValidationError,
    get_supported_mime_types,
    validate_and_load_image,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Inspection"])

ALLOWED_RULE_SOURCE_MODES = {"sih_ps_26034", "doca_statutory_2011"}


def get_image_storage_service() -> ImageStorageService:
    """Dependency provider for ImageStorageService."""
    return ImageStorageService()


def get_extraction_service() -> ExtractionService:
    """Dependency provider for ExtractionService."""
    return ExtractionService()


def get_measurement_service() -> MeasurementService:
    """Dependency provider for MeasurementService."""
    return MeasurementService()


def get_rule6_service() -> Rule6Service:
    """Dependency provider for Rule6Service."""
    return Rule6Service()


def get_rule7_service() -> Rule7Service:
    """Dependency provider for Rule7Service."""
    return Rule7Service()


def get_rule8_service() -> Rule8Service:
    """Dependency provider for Rule8Service."""
    return Rule8Service()


def get_inspection_service() -> InspectionService:
    """Dependency provider for InspectionService."""
    return InspectionService()


@router.post(
    "/inspection",
    response_model=InspectionResponse,
    responses={
        200: {
            "model": InspectionResponse,
            "description": "Full end-to-end compliance inspection completed successfully.",
        },
        400: {
            "model": InspectionErrorResponse,
            "description": "Invalid image file, empty bytes, or malformed parameters.",
        },
        415: {
            "model": InspectionErrorResponse,
            "description": "Unsupported image media type.",
        },
        422: {
            "model": InspectionErrorResponse,
            "description": "Validation failure, invalid rule source mode, or unprocessable image content.",
        },
        500: {
            "model": InspectionErrorResponse,
            "description": "Internal server error during inspection pipeline execution.",
        },
        502: {
            "model": InspectionErrorResponse,
            "description": "External Vision LLM service communication error.",
        },
    },
    summary="End-to-End Packaged Commodity Compliance Inspection",
    description=(
        "Production-style single entrypoint for SIH 2026 PS 26034 compliance checking. "
        "Accepts a packaged commodity label image and optional packaging metadata. "
        "Stores the exact original uploaded bytes without re-encoding, calculates SHA-256 "
        "cryptographic provenance, executes semantic extraction (Module 1), calibrated numeral "
        "measurement (Module 2A), Rule 6 visibility evaluation (Module 3C), Rule 7 numeral-height "
        "evaluation (Module 3A), Rule 8 spatial clearance analysis (Module 3B), aggregates the "
        "Unified Inspection JSON, and persists inspection snapshot for downstream reporting."
    ),
)
async def perform_end_to_end_inspection(
    image: UploadFile = File(..., description="Original packaged commodity label image file (JPEG, PNG, WebP)"),
    brand_name: Optional[str] = Form(default=None, description="Optional brand name override"),
    generic_name: Optional[str] = Form(default=None, description="Optional generic/commodity name override"),
    package_type: Optional[str] = Form(default="retail", description="Package type: retail, wholesale, combination_pack"),
    package_width_mm: Optional[float] = Form(default=None, description="Physical package width in millimetres (> 0)"),
    package_height_mm: Optional[float] = Form(default=None, description="Physical package height in millimetres (> 0)"),
    pdp_area_cm2: Optional[float] = Form(default=None, description="Principal Display Panel area in cm² (> 0)"),
    rule_source_mode: str = Form(default="sih_ps_26034", description="Statutory rule mode: 'sih_ps_26034' or 'doca_statutory_2011'"),
    storage_service: ImageStorageService = Depends(get_image_storage_service),
    extraction_service: ExtractionService = Depends(get_extraction_service),
    measurement_service: MeasurementService = Depends(get_measurement_service),
    rule6_service: Rule6Service = Depends(get_rule6_service),
    rule7_service: Rule7Service = Depends(get_rule7_service),
    rule8_service: Rule8Service = Depends(get_rule8_service),
    inspection_service: InspectionService = Depends(get_inspection_service),
) -> Union[InspectionResponse, JSONResponse]:
    """Executes the full automated compliance inspection pipeline."""
    # ------------------------------------------------------------------
    # Step 0: Parameter validation
    # ------------------------------------------------------------------
    clean_rule_mode = rule_source_mode.strip() if rule_source_mode else "sih_ps_26034"
    if clean_rule_mode not in ALLOWED_RULE_SOURCE_MODES:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT if hasattr(status, "HTTP_422_UNPROCESSABLE_CONTENT") else 422,
            content={
                "success": False,
                "error": {
                    "code": "INVALID_RULE_SOURCE_MODE",
                    "message": (
                        f"rule_source_mode '{clean_rule_mode}' is invalid. "
                        f"Allowed modes: {sorted(list(ALLOWED_RULE_SOURCE_MODES))}."
                    ),
                },
            },
        )

    if package_width_mm is not None and package_width_mm <= 0:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "error": {
                    "code": "INVALID_PACKAGE_WIDTH",
                    "message": f"package_width_mm must be strictly positive (> 0 mm), got {package_width_mm}.",
                },
            },
        )

    if package_height_mm is not None and package_height_mm <= 0:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "error": {
                    "code": "INVALID_PACKAGE_HEIGHT",
                    "message": f"package_height_mm must be strictly positive (> 0 mm), got {package_height_mm}.",
                },
            },
        )

    if pdp_area_cm2 is not None and pdp_area_cm2 <= 0:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "error": {
                    "code": "INVALID_PDP_AREA",
                    "message": f"pdp_area_cm2 must be strictly positive (> 0 cm²), got {pdp_area_cm2}.",
                },
            },
        )

    # ------------------------------------------------------------------
    # Step 1: Read raw original uploaded bytes (source of truth)
    # ------------------------------------------------------------------
    if not image or not image.filename:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "error": {
                    "code": "MISSING_IMAGE_FILE",
                    "message": "No image file was provided in the upload.",
                },
            },
        )

    try:
        image_bytes = await image.read()
    except Exception as exc:
        logger.exception("Failed to read uploaded image bytes: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "error": {
                    "code": "IMAGE_READ_ERROR",
                    "message": f"Could not read uploaded image bytes: {str(exc)}",
                },
            },
        )

    if not image_bytes or len(image_bytes) == 0:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "error": {
                    "code": "EMPTY_IMAGE_BYTES",
                    "message": "Uploaded image file is empty (0 bytes).",
                },
            },
        )

    # ------------------------------------------------------------------
    # Step 2: Validate image format and decoding in-memory
    # ------------------------------------------------------------------
    validation_upload = UploadFile(
        file=io.BytesIO(image_bytes),
        filename=image.filename,
        headers=image.headers,
    )

    try:
        img_validation_result = await validate_and_load_image(validation_upload)
    except ImageValidationError as exc:
        status_code = status.HTTP_400_BAD_REQUEST
        if exc.code == "UNSUPPORTED_MEDIA_TYPE":
            status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
        elif exc.code == "FILE_TOO_LARGE":
            status_code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
        return JSONResponse(
            status_code=status_code,
            content={
                "success": False,
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                },
            },
        )

    # ------------------------------------------------------------------
    # Step 3: Generate inspection_id and calculate SHA-256 provenance
    # ------------------------------------------------------------------
    inspection_id = storage_service.generate_inspection_id()
    sha256_hash = storage_service.calculate_sha256(image_bytes)
    original_filename = image.filename

    # ------------------------------------------------------------------
    # Step 4: Store exact original uploaded bytes without re-encoding
    # ------------------------------------------------------------------
    try:
        stored_path = storage_service.save_original(
            image_bytes=image_bytes,
            filename=original_filename,
            inspection_id=inspection_id,
        )
        logical_storage_path = storage_service.get_logical_source_path(
            inspection_id=inspection_id,
            filename=original_filename,
        )
    except ImageStorageError as exc:
        logger.error("Failed to store source image [%s]: %s", exc.code, exc.message)
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
        logger.exception("Unexpected error during source image persistence: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": {
                    "code": "IMAGE_STORAGE_FAILED",
                    "message": f"Failed to persist original source image: {str(exc)}",
                },
            },
        )

    # ------------------------------------------------------------------
    # Step 5: Resolve Package Dimensions (user-supplied vs automatic CV)
    # ------------------------------------------------------------------
    nparr = np.frombuffer(image_bytes, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image_bgr is None or image_bgr.size == 0:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "error": {
                    "code": "CORRUPTED_IMAGE",
                    "message": "OpenCV could not decode uploaded image bytes.",
                },
            },
        )

    img_h, img_w = image_bgr.shape[:2]

    # Detect package boundary for aspect-ratio estimation if dimensions omitted
    boundary_result = detect_package_boundary(image_bgr)
    if boundary_result.quality != "failed" and boundary_result.height_px > 0:
        pkg_w_px = float(boundary_result.width_px)
        pkg_h_px = float(boundary_result.height_px)
    else:
        pkg_w_px = float(img_w)
        pkg_h_px = float(img_h)

    aspect_ratio = pkg_w_px / max(1.0, pkg_h_px)

    resolved_w_mm: float
    resolved_h_mm: float
    resolved_pdp_area: Optional[float] = pdp_area_cm2

    if package_width_mm is not None and package_height_mm is not None:
        resolved_w_mm = float(package_width_mm)
        resolved_h_mm = float(package_height_mm)
        if resolved_pdp_area is None:
            resolved_pdp_area = round((resolved_w_mm * resolved_h_mm) / 100.0, 2)
    elif package_width_mm is not None and package_height_mm is None:
        resolved_w_mm = float(package_width_mm)
        resolved_h_mm = round(resolved_w_mm / aspect_ratio, 2)
        if resolved_pdp_area is None:
            resolved_pdp_area = round((resolved_w_mm * resolved_h_mm) / 100.0, 2)
    elif package_height_mm is not None and package_width_mm is None:
        resolved_h_mm = float(package_height_mm)
        resolved_w_mm = round(resolved_h_mm * aspect_ratio, 2)
        if resolved_pdp_area is None:
            resolved_pdp_area = round((resolved_w_mm * resolved_h_mm) / 100.0, 2)
    else:
        # User did not provide dimensions; use automatic aspect ratio with standardized nominal baseline
        if resolved_pdp_area is not None and resolved_pdp_area > 0:
            area_mm2 = resolved_pdp_area * 100.0
            resolved_h_mm = round(math.sqrt(area_mm2 / aspect_ratio), 2)
            resolved_w_mm = round(resolved_h_mm * aspect_ratio, 2)
        else:
            # Standard retail packaged commodity baseline (200.0 mm height, width derived from detected AR)
            resolved_h_mm = 200.0
            resolved_w_mm = round(resolved_h_mm * aspect_ratio, 2)
            resolved_pdp_area = round((resolved_w_mm * resolved_h_mm) / 100.0, 2)

    # ------------------------------------------------------------------
    # Step 6: Run Module 1 Semantic Extraction
    # ------------------------------------------------------------------
    extraction_upload = UploadFile(
        file=io.BytesIO(image_bytes),
        filename=original_filename,
        headers=image.headers,
    )

    try:
        extraction_result = await extraction_service.extract_from_upload(extraction_upload)
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
        logger.exception("Unexpected error during semantic extraction: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": {
                    "code": "EXTRACTION_FAILED",
                    "message": f"Unexpected error during semantic extraction: {str(exc)}",
                },
            },
        )

    # Apply optional client overrides if provided
    if brand_name and extraction_result.semantic_extraction.package:
        extraction_result.semantic_extraction.package.brand_name = brand_name.strip()
    if generic_name and extraction_result.semantic_extraction.package:
        extraction_result.semantic_extraction.package.generic_name = generic_name.strip()
    if package_type and extraction_result.semantic_extraction.package:
        extraction_result.semantic_extraction.package.package_type = package_type.strip()  # type: ignore

    # ------------------------------------------------------------------
    # Step 7: Run Module 2A Calibrated Numeral Measurement
    # ------------------------------------------------------------------
    measurement_upload = UploadFile(
        file=io.BytesIO(image_bytes),
        filename=original_filename,
        headers=image.headers,
    )

    try:
        measurement_result = await measurement_service.measure_packaging_numerals(
            image_file=measurement_upload,
            extraction_json=extraction_result.model_dump_json(),
            package_width_mm=resolved_w_mm,
            package_height_mm=resolved_h_mm,
            package_bbox_px=None,  # Triggers automatic package boundary detection
            include_debug_image=False,
            allow_no_numeral_regions=True,
        )
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
                    "code": "MEASUREMENT_FAILED",
                    "message": f"Unexpected error during numeral measurement: {str(exc)}",
                },
            },
        )

    # ------------------------------------------------------------------
    # Step 8: Run Module 3C Rule 6 Declaration Visibility Evaluation
    # ------------------------------------------------------------------
    try:
        rule6_result = rule6_service.evaluate(
            extraction=extraction_result,
            package_type=package_type or extraction_result.semantic_extraction.package.package_type,
        )
    except Rule6ServiceError as exc:
        logger.warning("Rule 6 service error [%s]: %s", exc.code, exc.message)
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
        logger.exception("Unexpected error during Rule 6 evaluation: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": {
                    "code": "RULE6_EVALUATION_FAILED",
                    "message": f"Unexpected error during Rule 6 evaluation: {str(exc)}",
                },
            },
        )

    # ------------------------------------------------------------------
    # Step 9: Run Module 3A Rule 7 Numeral-Height Evaluation
    # ------------------------------------------------------------------
    try:
        rule7_result = rule7_service.evaluate(
            extraction=extraction_result,
            measurements=measurement_result,
            font_category="normal",
            pdp_area_cm2=resolved_pdp_area,
            rule_source_mode=clean_rule_mode,  # type: ignore
        )
    except Rule7ServiceError as exc:
        logger.warning("Rule 7 service error [%s]: %s", exc.code, exc.message)
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
        logger.exception("Unexpected error during Rule 7 evaluation: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": {
                    "code": "RULE7_EVALUATION_FAILED",
                    "message": f"Unexpected error during Rule 7 evaluation: {str(exc)}",
                },
            },
        )

    # ------------------------------------------------------------------
    # Step 10: Run Module 3B Rule 8 Spatial Clearance Analysis
    # ------------------------------------------------------------------
    try:
        rule8_result = rule8_service.evaluate(
            extraction=extraction_result,
            measurements=measurement_result,
            image_input=image_bgr,
            include_debug_image=False,
        )
    except Rule8ServiceError as exc:
        logger.warning("Rule 8 service error [%s]: %s", exc.code, exc.message)
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
        logger.exception("Unexpected error during Rule 8 evaluation: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": {
                    "code": "RULE8_EVALUATION_FAILED",
                    "message": f"Unexpected error during Rule 8 evaluation: {str(exc)}",
                },
            },
        )

    # ------------------------------------------------------------------
    # Step 11: Aggregate Unified Inspection Findings
    # ------------------------------------------------------------------
    try:
        inspection_data: InspectionResponseData = inspection_service.aggregate(
            extraction=extraction_result,
            measurements=measurement_result,
            rule6=rule6_result,
            rule7=rule7_result,
            rule8=rule8_result,
            inspection_id=inspection_id,
            source_image_sha256=sha256_hash,
            source_image_filename=original_filename,
            source_image_path=logical_storage_path,
            source_image_storage_path=logical_storage_path,
        )
    except InspectionServiceError as exc:
        logger.warning("Inspection aggregation error [%s]: %s", exc.code, exc.message)
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
        logger.exception("Unexpected error during unified inspection aggregation: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": {
                    "code": "AGGREGATION_FAILED",
                    "message": f"Unexpected error during Unified Inspection aggregation: {str(exc)}",
                },
            },
        )

    # ------------------------------------------------------------------
    # Step 12: Persist inspection snapshot to disk, then MongoDB
    # ------------------------------------------------------------------
    inspection_json_path = None
    try:
        inspection_json_path = storage_service.save_inspection_json(
            inspection_id=inspection_id,
            inspection_data=inspection_data,
        )
        logger.info(
            "Persisted unified inspection record %s to reports/inspections/%s/inspection.json",
            inspection_id,
            inspection_id,
        )
    except Exception as exc:
        logger.exception("Failed to persist inspection JSON snapshot: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": {
                    "code": "PERSISTENCE_FAILED",
                    "message": f"Failed to persist inspection.json snapshot: {str(exc)}",
                },
            },
        )

    # Generate the canonical stored report before recording artifact references in MongoDB.
    try:
        ReportService(storage_service=storage_service).generate_report(inspection_data)
    except ReportServiceError as exc:
        logger.warning("Report generation failed for %s [%s]: %s", inspection_id, exc.code, exc.message)
    except Exception as exc:
        logger.exception("Unexpected report generation failure for %s: %s", inspection_id, exc)

    # Refresh the JSON reference only after the report path is established.
    try:
        storage_service.save_inspection_json(inspection_id=inspection_id, inspection_data=inspection_data)
    except Exception as exc:
        logger.warning("Could not refresh inspection JSON artifact references for %s: %s", inspection_id, exc)

    persistence_status = "disabled"
    persistence_reason = None
    if settings.MONGODB_ENABLED:
        mongo_document = inspection_data.model_dump(mode="json")
        mongo_document["inspection_persistence"] = {"status": "persisted"}
        mongo_document["artifacts"] = {
            "inspection_json_path": inspection_json_path,
            "report_pdf_path": str(storage_service.get_report_path(inspection_id)),
        }
        try:
            InspectionRepository().create_inspection(mongo_document)
            persistence_status = "persisted"
            logger.info("Persisted inspection %s to MongoDB", inspection_id)
        except InspectionRepositoryError as exc:
            persistence_status = "not_persisted"
            persistence_reason = exc.message
            logger.warning("MongoDB persistence unavailable for %s [%s]: %s", inspection_id, exc.code, exc.message)
        except Exception as exc:
            persistence_status = "not_persisted"
            persistence_reason = "MongoDB persistence failed."
            logger.exception("Unexpected MongoDB persistence failure for %s: %s", inspection_id, exc)

    inspection_data = inspection_data.model_copy(
        update={
            "inspection_persistence": PersistenceInfo(
                status=persistence_status,
                reason=persistence_reason,
            )
        }
    )
    try:
        storage_service.save_inspection_json(inspection_id=inspection_id, inspection_data=inspection_data)
    except Exception as exc:
        logger.warning("Could not finalize inspection JSON persistence metadata for %s: %s", inspection_id, exc)

    # ------------------------------------------------------------------
    # Step 13: Return structured Unified Inspection response
    # ------------------------------------------------------------------
    return InspectionResponse(success=True, data=inspection_data)
