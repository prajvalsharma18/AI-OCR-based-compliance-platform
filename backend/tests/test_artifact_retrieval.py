"""Regression tests for byte-preserving historical artifact retrieval."""

import io
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.services.image_storage_service import ImageStorageService


def test_historical_report_returns_exact_stored_bytes(monkeypatch, tmp_path):
    storage_root = tmp_path / "reports" / "inspections"
    monkeypatch.setattr("app.services.image_storage_service.DEFAULT_STORAGE_ROOT", storage_root)
    inspection_id = "INSP-REPORT-TEST"
    report_path = storage_root / inspection_id / "report.pdf"
    expected = b"%PDF-1.7\npre-generated-report-bytes\n%%EOF"
    report_path.parent.mkdir(parents=True)
    report_path.write_bytes(expected)

    response = TestClient(app).get(f"/api/v1/inspections/{inspection_id}/report")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content == expected
    assert f'filename="inspection_{inspection_id}.pdf"' in response.headers["content-disposition"]


def test_historical_report_returns_controlled_404(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "app.services.image_storage_service.DEFAULT_STORAGE_ROOT",
        tmp_path / "reports" / "inspections",
    )

    response = TestClient(app).get("/api/v1/inspections/INSP-MISSING/report")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "REPORT_NOT_FOUND"


def test_historical_image_returns_exact_stored_bytes(monkeypatch, tmp_path):
    storage_root = tmp_path / "reports" / "inspections"
    monkeypatch.setattr("app.services.image_storage_service.DEFAULT_STORAGE_ROOT", storage_root)
    inspection_id = "INSP-IMAGE-TEST"
    image = Image.new("RGB", (8, 8), color=(12, 34, 56))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    expected = buffer.getvalue()
    image_path = storage_root / inspection_id / "original_image.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(expected)

    response = TestClient(app).get(f"/api/v1/inspections/{inspection_id}/image")

    assert response.status_code == 200
    assert response.content == expected


def test_stored_report_path_uses_safe_inspection_id(monkeypatch, tmp_path):
    monkeypatch.setattr("app.services.image_storage_service.DEFAULT_STORAGE_ROOT", tmp_path)
    path = ImageStorageService().get_report_path("INSP-SAFE")
    assert path.name == "report.pdf"
    assert path.parent.name == "INSP-SAFE"
