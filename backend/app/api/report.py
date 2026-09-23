"""API routes for PDF Inspection Report Generator."""

import logging
from pathlib import Path
from typing import Any, Optional, Union
from fastapi import APIRouter, Depends, status
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from app.services.report_service import ReportService, ReportServiceError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Inspection Report Generation"])


class ReportGenerationRequest(BaseModel):
    """Request payload for PDF Inspection Report generation."""
    model_config = ConfigDict(extra="allow")

    inspection: Any = Field(
        ...,
        description="Unified Inspection JSON object, dictionary, envelope, or JSON string."
    )
    image_base64: Optional[str] = Field(
        default=None,
        description="Optional base64 encoded original packaging photo (PNG or JPEG)."
    )


def get_report_service() -> ReportService:
    """Dependency provider for ReportService."""
    return ReportService()


@router.post(
    "/report/pdf",
    response_model=None,
    summary="Generate statutory PDF inspection report",
    description=(
        "Consumes the Unified Inspection JSON and renders a structured multi-page PDF inspection report "
        "using ReportLab. The PDF is a pure presentation layer: zero rule recalculation, zero LLM calls, "
        "and zero image manipulation. Returns the generated PDF document."
    ),
    responses={
        200: {
            "content": {"application/pdf": {}},
            "description": "Generated PDF inspection report file",
        },
        422: {
            "description": "Validation error in inspection payload",
        },
        500: {
            "description": "Internal server error during PDF generation",
        },
    },
)
async def generate_pdf_report(
    request: ReportGenerationRequest,
    service: ReportService = Depends(get_report_service),
) -> Union[FileResponse, JSONResponse]:
    """Generates and streams the statutory PDF inspection report."""
    try:
        pdf_path_str = service.generate_report(
            inspection=request.inspection,
            image_base64=request.image_base64,
        )
        pdf_path = Path(pdf_path_str)

        if not pdf_path.exists():
            raise FileNotFoundError(f"Generated PDF file not found at {pdf_path}")

        insp_data = service._normalize_inspection(request.inspection)
        download_filename = f"inspection_{insp_data.inspection_id}.pdf"
        return FileResponse(
            path=pdf_path,
            media_type="application/pdf",
            filename=download_filename,
            headers={
                "Content-Disposition": f'inline; filename="{download_filename}"',
            },
        )

    except ReportServiceError as exc:
        logger.warning("Report service error [%s]: %s", exc.code, exc.message)
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
        logger.exception("Unexpected error during PDF report generation: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": f"An unexpected error occurred during PDF report generation: {str(exc)}",
                },
            },
        )
