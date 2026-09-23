"""Pydantic schemas for MongoDB inspection history and manual LMO review.

Strictly preserves existing Unified Inspection data models and introduces
compact history listings and review update contracts.
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field

ReviewStatus = Literal["Pending", "Verified", "Issue Raised", "N/A"]
PersistenceStatusType = Literal["persisted", "not_persisted", "disabled"]


class ReviewData(BaseModel):
    """Manual Legal Metrology Officer (LMO) review metadata."""
    model_config = ConfigDict(extra="forbid")

    status: ReviewStatus = Field(
        default="Pending",
        description="Manual review decision by inspecting officer: 'Pending', 'Verified', 'Issue Raised', or 'N/A'.",
    )
    notes: str = Field(
        default="",
        description="Officer inspection notes, physical caliper verification comments, or seizure memo notes.",
    )
    updated_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp when officer review was last updated in UTC.",
    )


class ReviewUpdateRequest(BaseModel):
    """Payload for PATCH /api/v1/inspections/{inspection_id}/review."""
    model_config = ConfigDict(extra="forbid")

    status: ReviewStatus = Field(
        ...,
        description="Updated review status: 'Pending', 'Verified', 'Issue Raised', or 'N/A'.",
    )
    notes: Optional[str] = Field(
        default="",
        description="Updated officer review notes.",
    )


class PersistenceInfo(BaseModel):
    """Application-level persistence status reported alongside inspection results."""
    model_config = ConfigDict(extra="allow")

    status: PersistenceStatusType = Field(
        ...,
        description="Persistence state: 'persisted', 'not_persisted', or 'disabled'.",
    )
    reason: Optional[str] = Field(
        default=None,
        description="Diagnostic explanation if persistence failed or was bypassed.",
    )


class InspectionHistoryItem(BaseModel):
    """Compact summary item returned by GET /api/v1/inspections for fast list rendering."""
    model_config = ConfigDict(extra="allow")

    inspection_id: str = Field(..., description="Unique inspection identifier.")
    created_at: datetime = Field(..., description="Inspection completion timestamp in UTC.")
    brand_name: Optional[str] = Field(default=None, description="Packaging brand name.")
    generic_name: Optional[str] = Field(default=None, description="Generic commodity name.")
    package_type: Optional[str] = Field(default="retail", description="Packaging classification.")
    pdp_area_cm2: Optional[float] = Field(default=None, description="Principal Display Panel area in cm².")
    rule_source_mode: Optional[str] = Field(default="sih_ps_26034", description="Applied statutory standard.")
    total_declarations: int = Field(default=0, ge=0, description="Total declarations evaluated.")
    pass_count: int = Field(default=0, ge=0, description="Total passed statutory checks.")
    non_compliant_count: int = Field(default=0, ge=0, description="Total non-compliant statutory checks.")
    not_visible_count: int = Field(default=0, ge=0, description="Total declarations not visible on panel.")
    review_status: str = Field(default="Pending", description="Current manual LMO review status.")


class InspectionHistoryListResponse(BaseModel):
    """Response envelope for GET /api/v1/inspections."""
    model_config = ConfigDict(extra="allow")

    success: bool = Field(default=True, description="API success indicator.")
    total: int = Field(..., ge=0, description="Total count of inspections matching query filters.")
    limit: int = Field(..., ge=1, description="Page limit applied.")
    skip: int = Field(..., ge=0, description="Offset count skipped.")
    items: List[InspectionHistoryItem] = Field(default_factory=list, description="List of compact inspection summaries.")


class SingleInspectionResponse(BaseModel):
    """Response envelope for GET /api/v1/inspections/{inspection_id}."""
    model_config = ConfigDict(extra="allow")

    success: bool = Field(default=True, description="API success indicator.")
    data: Dict[str, Any] = Field(..., description="Complete Unified Inspection document from MongoDB.")
