"""Opt-in live Atlas verification; skipped during normal local test runs."""

import os
import time

import pytest

from app.db.indexes import ensure_indexes
from app.db.mongo import get_database
from app.repositories.inspection_repository import InspectionRepository


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_MONGODB_TESTS") != "1",
    reason="Set RUN_LIVE_MONGODB_TESTS=1 to run against configured MongoDB Atlas.",
)


def test_live_atlas_round_trip():
    database = get_database()
    assert database is not None
    ensure_indexes(database)

    inspection_id = f"INSP-LIVE-TEST-{int(time.time() * 1000)}"
    document = {
        "inspection_id": inspection_id,
        "metadata": {
            "brand_name": "Live Test",
            "source_image_sha256": "b" * 64,
            "source_image_path": f"inspections/{inspection_id}/original_image.jpg",
        },
        "summary": {
            "total_declarations_evaluated": 0,
            "pass_findings_count": 0,
            "non_compliant_findings_count": 0,
            "not_visible_declarations_count": 0,
        },
        "findings": [],
        "artifacts": {
            "inspection_json_path": f"reports/inspections/{inspection_id}/inspection.json",
            "report_pdf_path": f"reports/inspections/{inspection_id}/report.pdf",
        },
    }

    repository = InspectionRepository(db=database)
    try:
        assert repository.create_inspection(document) is True
        stored = repository.get_inspection_by_id(inspection_id)
        assert stored is not None
        assert stored["inspection_id"] == inspection_id
        assert stored["metadata"]["source_image_sha256"] == "b" * 64
    finally:
        database["inspections"].delete_one({"inspection_id": inspection_id})
