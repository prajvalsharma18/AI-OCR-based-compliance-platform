"""MongoDB persistence and history API tests without a live MongoDB server."""

from datetime import datetime, timedelta, timezone

import mongomock
import pytest
from fastapi.testclient import TestClient

from app.db.indexes import ensure_indexes
from app.db.mongo import set_mongo_client
from app.config import settings
from app.main import app
from app.repositories.inspection_repository import InspectionRepository


@pytest.fixture
def mongo_repository():
    client = mongomock.MongoClient()
    set_mongo_client(client)
    db = client[settings.MONGODB_DATABASE]
    ensure_indexes(db)
    try:
        yield InspectionRepository(db=db), db
    finally:
        set_mongo_client(None)


def sample_document(inspection_id: str, created_at: datetime) -> dict:
    return {
        "inspection_id": inspection_id,
        "created_at": created_at,
        "metadata": {
            "brand_name": "Lay's",
            "generic_name": "Potato Chips",
            "package_type": "retail",
            "pdp_area_cm2": 195.0,
            "rule_source_mode": "sih_ps_26034",
            "source_image_sha256": "a" * 64,
            "source_image_path": f"inspections/{inspection_id}/original_image.jpg",
        },
        "summary": {
            "total_declarations_evaluated": 7,
            "pass_findings_count": 3,
            "non_compliant_findings_count": 2,
            "not_visible_declarations_count": 1,
        },
        "findings": [{"field": "mrp", "status": "PASS"}],
        "artifacts": {"inspection_json_path": f"reports/inspections/{inspection_id}/inspection.json"},
    }


def test_repository_create_get_and_idempotency(mongo_repository):
    repository, db = mongo_repository
    document = sample_document("INSP-1", datetime.now(timezone.utc))

    assert repository.create_inspection(document) is True
    document["summary"]["pass_findings_count"] = 4
    assert repository.create_inspection(document) is True

    stored = repository.get_inspection_by_id("INSP-1")
    assert stored["summary"]["pass_findings_count"] == 4
    assert db["inspections"].count_documents({"inspection_id": "INSP-1"}) == 1


def test_repository_list_filters_and_review_update(mongo_repository):
    repository, _ = mongo_repository
    now = datetime.now(timezone.utc)
    repository.create_inspection(sample_document("INSP-NEW", now))
    older = sample_document("INSP-OLD", now - timedelta(days=2))
    older["metadata"]["brand_name"] = "Other Brand"
    older["metadata"]["package_type"] = "wholesale"
    repository.create_inspection(older)

    items, total = repository.list_inspections(
        limit=20,
        date_from=now - timedelta(days=1),
        brand_name="lay",
        package_type="retail",
    )
    assert total == 1
    assert items[0]["inspection_id"] == "INSP-NEW"

    updated = repository.update_review("INSP-NEW", "Verified", "Checked against physical pack")
    assert updated["review"]["status"] == "Verified"
    assert updated["review"]["notes"] == "Checked against physical pack"
    assert updated["summary"]["non_compliant_findings_count"] == 2


def test_history_and_review_api(mongo_repository, monkeypatch):
    repository, _ = mongo_repository
    repository.create_inspection(sample_document("INSP-API", datetime.now(timezone.utc)))
    monkeypatch.setattr("app.config.settings.MONGODB_ENABLED", True)

    client = TestClient(app)
    history = client.get("/api/v1/inspections", params={"brand_name": "lay"})
    assert history.status_code == 200
    assert history.json()["items"][0]["inspection_id"] == "INSP-API"

    detail = client.get("/api/v1/inspections/INSP-API")
    assert detail.status_code == 200
    assert detail.json()["data"]["findings"][0]["status"] == "PASS"

    review = client.patch(
        "/api/v1/inspections/INSP-API/review",
        json={"status": "Issue Raised", "notes": "Manual follow-up required"},
    )
    assert review.status_code == 200
    assert review.json()["data"]["review"]["status"] == "Issue Raised"


def test_history_returns_service_unavailable_when_disabled(mongo_repository, monkeypatch):
    monkeypatch.setattr("app.config.settings.MONGODB_ENABLED", False)
    response = TestClient(app).get("/api/v1/inspections")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "MONGODB_DISABLED"
