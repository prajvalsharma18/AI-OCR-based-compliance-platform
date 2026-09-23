"""Inspection history and manual review APIs backed by MongoDB."""

from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Query, status as http_status
from fastapi.responses import FileResponse, JSONResponse

from app.config import settings
from app.repositories.inspection_repository import InspectionRepository, InspectionRepositoryError
from app.schemas.history import (
    InspectionHistoryListResponse,
    ReviewUpdateRequest,
    SingleInspectionResponse,
)
from app.services.image_storage_service import ImageStorageService

router = APIRouter(prefix="/api/v1", tags=["Inspection History"])


def _database_error(exc: InspectionRepositoryError) -> JSONResponse:
    return JSONResponse(
        status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "success": False,
            "error": {"code": exc.code, "message": exc.message},
        },
    )


@router.get(
    "/inspections",
    response_model=InspectionHistoryListResponse,
    summary="List inspection history",
)
def list_inspections(
    limit: int = Query(default=20, ge=1, le=100),
    skip: int = Query(default=0, ge=0),
    date_from: Optional[datetime] = Query(default=None),
    date_to: Optional[datetime] = Query(default=None),
    status: Optional[str] = Query(default=None),
    brand_name: Optional[str] = Query(default=None),
    package_type: Optional[str] = Query(default=None),
    inspection_id: Optional[str] = Query(default=None),
):
    """Returns compact, newest-first inspection summaries."""
    if not settings.MONGODB_ENABLED:
        return JSONResponse(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"success": False, "error": {"code": "MONGODB_DISABLED", "message": "Inspection history is disabled."}},
        )

    try:
        items, total = InspectionRepository().list_inspections(
            limit=limit,
            skip=skip,
            date_from=date_from,
            date_to=date_to,
            status=status,
            brand_name=brand_name,
            package_type=package_type,
            inspection_id=inspection_id,
        )
        return InspectionHistoryListResponse(
            success=True,
            total=total,
            limit=limit,
            skip=skip,
            items=items,
        )
    except InspectionRepositoryError as exc:
        return _database_error(exc)


@router.get(
    "/inspections/{inspection_id}/image",
    response_model=None,
    summary="Retrieve a stored inspection image",
)
def get_inspection_image(inspection_id: str):
    """Streams the original image through FastAPI without exposing filesystem paths."""
    try:
        image_path = ImageStorageService().get_original(inspection_id)
    except Exception:
        image_path = None

    if not image_path or not Path(image_path).is_file():
        return JSONResponse(
            status_code=http_status.HTTP_404_NOT_FOUND,
            content={"success": False, "error": {"code": "IMAGE_NOT_FOUND", "message": "Stored inspection image was not found."}},
        )

    if settings.MONGODB_ENABLED:
        try:
            document = InspectionRepository().get_inspection_by_id(inspection_id)
            expected_hash = (document or {}).get("metadata", {}).get("source_image_sha256")
            if expected_hash:
                actual_hash = ImageStorageService().calculate_sha256(Path(image_path).read_bytes())
                if actual_hash.lower() != expected_hash.lower():
                    return JSONResponse(
                        status_code=http_status.HTTP_409_CONFLICT,
                        content={"success": False, "error": {"code": "IMAGE_HASH_MISMATCH", "message": "Stored image provenance verification failed."}},
                    )
        except InspectionRepositoryError:
            return _database_error(InspectionRepositoryError("Could not verify stored image provenance.", "MONGO_READ_FAILED"))

    return FileResponse(path=image_path, filename=Path(image_path).name)


@router.get(
    "/inspections/{inspection_id}/report",
    response_model=None,
    summary="Retrieve the stored inspection PDF",
    responses={404: {"description": "Stored report was not found."}},
)
def get_inspection_report(inspection_id: str):
    """Returns the existing report.pdf bytes without regenerating the report."""
    try:
        report_path = ImageStorageService().get_report_path(inspection_id)
    except Exception:
        report_path = None

    if not report_path or not report_path.is_file():
        return JSONResponse(
            status_code=http_status.HTTP_404_NOT_FOUND,
            content={"success": False, "error": {"code": "REPORT_NOT_FOUND", "message": "Stored inspection report was not found."}},
        )

    return FileResponse(
        path=report_path,
        media_type="application/pdf",
        filename=f"inspection_{inspection_id}.pdf",
        headers={"Content-Disposition": f'attachment; filename="inspection_{inspection_id}.pdf"'},
    )


@router.get(
    "/inspections/{inspection_id}",
    response_model=SingleInspectionResponse,
    summary="Retrieve a complete persisted inspection",
)
def get_inspection(inspection_id: str):
    if not settings.MONGODB_ENABLED:
        return JSONResponse(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"success": False, "error": {"code": "MONGODB_DISABLED", "message": "Inspection history is disabled."}},
        )

    try:
        document = InspectionRepository().get_inspection_by_id(inspection_id)
        if document is None:
            return JSONResponse(
                status_code=http_status.HTTP_404_NOT_FOUND,
                content={"success": False, "error": {"code": "INSPECTION_NOT_FOUND", "message": "Inspection was not found."}},
            )
        return SingleInspectionResponse(success=True, data=document)
    except InspectionRepositoryError as exc:
        return _database_error(exc)


@router.patch(
    "/inspections/{inspection_id}/review",
    response_model=SingleInspectionResponse,
    summary="Update manual LMO review",
)
def update_inspection_review(inspection_id: str, request: ReviewUpdateRequest):
    if not settings.MONGODB_ENABLED:
        return JSONResponse(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"success": False, "error": {"code": "MONGODB_DISABLED", "message": "Inspection review persistence is disabled."}},
        )

    try:
        document = InspectionRepository().update_review(
            inspection_id=inspection_id,
            review_status=request.status,
            notes=request.notes,
        )
        if document is None:
            return JSONResponse(
                status_code=http_status.HTTP_404_NOT_FOUND,
                content={"success": False, "error": {"code": "INSPECTION_NOT_FOUND", "message": "Inspection was not found."}},
            )
        return SingleInspectionResponse(success=True, data=document)
    except InspectionRepositoryError as exc:
        return _database_error(exc)
