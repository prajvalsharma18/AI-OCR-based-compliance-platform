"""API routes for Module 3: Rule 6 Visibility, Rule 7 Numeral-Height, and Rule 8 Spatial Clearance."""

import logging
from typing import Union
from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse

from app.schemas.compliance import (
    Rule7ErrorResponse,
    Rule7EvaluationRequest,
    Rule7Response,
    Rule7ResponseData,
)
from app.schemas.rule6 import (
    Rule6ErrorResponse,
    Rule6EvaluationRequest,
    Rule6Response,
    Rule6ResponseData,
)
from app.schemas.inspection import (
    InspectionErrorResponse,
    InspectionEvaluationRequest,
    InspectionResponse,
    InspectionResponseData,
)
from app.schemas.rule8 import (
    Rule8ErrorResponse,
    Rule8EvaluationRequest,
    Rule8Response,
    Rule8ResponseData,
)
from app.services.inspection_service import InspectionService, InspectionServiceError
from app.services.rule6_service import Rule6Service, Rule6ServiceError
from app.services.rule7_service import Rule7Service, Rule7ServiceError
from app.services.rule8_service import Rule8Service, Rule8ServiceError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Statutory Compliance (Rule 6, Rule 7 & Rule 8)"])


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
    "/compliance/rule6",
    response_model=Rule6Response,
    responses={
        200: {"model": Rule6Response, "description": "Mandatory declarations successfully evaluated for visibility"},
        400: {"model": Rule6ErrorResponse, "description": "Malformed input JSON"},
        422: {"model": Rule6ErrorResponse, "description": "Schema validation failure in extraction payload"},
        500: {"model": Rule6ErrorResponse, "description": "Internal server error during Rule 6 evaluation"},
    },
    summary="Deterministic Rule 6 declaration visibility evaluation",
    description=(
        "Consumes structured output from Module 1 (semantic extraction) to evaluate visibility "
        "of all six mandatory declarations required under Rule 6 of the Legal Metrology (Packaged Commodities) "
        "Rules, 2011: (1) Manufacturer/Packer/Importer, (2) Generic Name, (3) MRP, (4) Manufacturing/Packing Date, "
        "(5) Net Quantity, and (6) Consumer Care Details. "
        "Strictly distinguishes 'Not visible in supplied image' from legal non-compliance, preserving neutrality "
        "for unsubmitted package faces."
    ),
)
async def evaluate_rule6_compliance(
    request: Rule6EvaluationRequest,
    service: Rule6Service = Depends(get_rule6_service),
) -> Union[Rule6Response, JSONResponse]:
    """Evaluates mandatory declaration visibility against Rule 6."""
    try:
        data: Rule6ResponseData = service.evaluate(
            extraction=request.extraction,
            package_type=request.package_type,
        )
        return Rule6Response(success=True, data=data)

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
        logger.exception("Unexpected error during Rule 6 compliance evaluation: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": f"An unexpected error occurred during Rule 6 evaluation: {str(exc)}",
                },
            },
        )


@router.post(
    "/compliance/rule7",
    response_model=Rule7Response,
    responses={
        200: {"model": Rule7Response, "description": "Statutory numeral declarations successfully evaluated against Rule 7"},
        400: {"model": Rule7ErrorResponse, "description": "Malformed input JSON"},
        422: {"model": Rule7ErrorResponse, "description": "Schema validation failure in extraction or measurement payloads"},
        500: {"model": Rule7ErrorResponse, "description": "Internal server error during rule evaluation"},
    },
    summary="Deterministic Rule 7 numeral-height compliance evaluation",
    description=(
        "Consumes structured outputs from Module 1 (semantic extraction) and Module 2A "
        "(calibrated numeral measurements) to determine applicable Rule 7 numeral-height "
        "statutory thresholds (Second Schedule Table 1) and compare measured physical sizes. "
        "Strictly deterministic: uses no LLMs and preserves measurement uncertainty. "
        "Does not constitute a final legal enforcement order."
    ),
)
async def evaluate_rule7_compliance(
    request: Rule7EvaluationRequest,
    service: Rule7Service = Depends(get_rule7_service),
) -> Union[Rule7Response, JSONResponse]:
    """Evaluates statutory declaration numeral heights against Rule 7."""
    try:
        data: Rule7ResponseData = service.evaluate(
            extraction=request.extraction,
            measurements=request.measurements,
            font_category=request.font_category,
            pdp_area_cm2=request.pdp_area_cm2,
            rule_source_mode=request.rule_source_mode,
        )
        return Rule7Response(success=True, data=data)

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
        logger.exception("Unexpected error during Rule 7 compliance evaluation: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": f"An unexpected error occurred during Rule 7 evaluation: {str(exc)}",
                },
            },
        )


@router.post(
    "/compliance/rule8",
    response_model=Rule8Response,
    responses={
        200: {"model": Rule8Response, "description": "Net quantity spatial clearance successfully evaluated against Rule 8"},
        400: {"model": Rule8ErrorResponse, "description": "Malformed input JSON"},
        422: {"model": Rule8ErrorResponse, "description": "Schema validation failure in extraction or measurement payloads"},
        500: {"model": Rule8ErrorResponse, "description": "Internal server error during Rule 8 evaluation"},
    },
    summary="Deterministic Rule 8 spatial clearance compliance evaluation",
    description=(
        "Consumes structured outputs from Module 1 (semantic extraction) and Module 2A "
        "(calibrated numeral measurements) to evaluate statutory clear space surrounding the "
        "Net Quantity declaration per Rule 8(1) proviso of the Legal Metrology (Packaged Commodities) Rules, 2011. "
        "Required clear space: Top=H, Bottom=H, Left=2H, Right=2H, where H is the numeral height in mm. "
        "Strictly deterministic: uses no LLMs and preserves measurement uncertainty. "
        "Does not constitute a final legal enforcement order."
    ),
)
async def evaluate_rule8_compliance(
    request: Rule8EvaluationRequest,
    service: Rule8Service = Depends(get_rule8_service),
) -> Union[Rule8Response, JSONResponse]:
    """Evaluates Net Quantity spatial clearance against Rule 8(1) proviso."""
    try:
        data: Rule8ResponseData = service.evaluate(
            extraction=request.extraction,
            measurements=request.measurements,
            image_input=request.image_base64,
            include_debug_image=request.include_debug_image,
        )
        return Rule8Response(success=True, data=data)

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
        logger.exception("Unexpected error during Rule 8 compliance evaluation: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": f"An unexpected error occurred during Rule 8 evaluation: {str(exc)}",
                },
            },
        )


@router.post(
    "/compliance/inspection",
    response_model=InspectionResponse,
    responses={
        200: {"model": InspectionResponse, "description": "Unified inspection findings successfully aggregated"},
        400: {"model": InspectionErrorResponse, "description": "Malformed input JSON"},
        422: {"model": InspectionErrorResponse, "description": "Schema validation failure in supplied module outputs"},
        500: {"model": InspectionErrorResponse, "description": "Internal server error during inspection aggregation"},
    },
    summary="Deterministic Unified Inspection aggregation",
    description=(
        "Consumes structured outputs from Module 1 (extraction), Module 2A (measurements), "
        "Module 3C (Rule 6), Module 3A (Rule 7), and Module 3B (Rule 8) to produce a single normalized "
        "Unified Inspection JSON. Pure aggregation and normalization: does not re-evaluate compliance rules, "
        "run additional OCR/CV passes, or execute LLMs. Preserves evidence, measurement values, and individual "
        "rule outcomes without issuing an overall pass/fail legal verdict."
    ),
)
async def evaluate_unified_inspection(
    request: InspectionEvaluationRequest,
    service: InspectionService = Depends(get_inspection_service),
) -> Union[InspectionResponse, JSONResponse]:
    """Aggregates upstream module outputs into the Unified Inspection JSON."""
    try:
        data: InspectionResponseData = service.aggregate(
            extraction=request.extraction,
            measurements=request.measurements,
            rule6=request.rule6,
            rule7=request.rule7,
            rule8=request.rule8,
            inspection_id=request.inspection_id,
            source_image_sha256=request.source_image_sha256,
            source_image_filename=request.source_image_filename,
            source_image_path=request.source_image_path,
            source_image_storage_path=request.source_image_storage_path,
        )
        return InspectionResponse(success=True, data=data)

    except InspectionServiceError as exc:
        logger.warning("Inspection service error [%s]: %s", exc.code, exc.message)
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
        logger.exception("Unexpected error during Unified Inspection aggregation: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": f"An unexpected error occurred during Unified Inspection aggregation: {str(exc)}",
                },
            },
        )


