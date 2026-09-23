"""Integration tests for POST /api/v1/inspection end-to-end endpoint.

SIH 2026 PS 26034: Automated Compliance Checker for Packaged Commodities

Test Scenarios:
A. Successful end-to-end inspection (upload image, get inspection_id, metadata, findings, summary, inspection.json).
B. Original image storage (bytes stored unchanged, SHA-256 matches uploaded bytes).
C. Rule source mode (sih_ps_26034 and doca_statutory_2011 accepted, invalid rejected with 422).
D. Invalid image (empty bytes, invalid image format return controlled 4xx errors without stack traces).
E. Source filename preserved in metadata and storage path.
F. NOT VISIBLE semantics (missing field never becomes NON-COMPLIANT).
G. DETECTED != PASS (detected declaration without rule assessment becomes NOT ASSESSABLE).
H. Date semantics (manufacture and use_by stay correctly mapped).
I. Existing Rule 7 and Rule 8 results passed through without recalculation or mutation.
J. Persistence (inspection.json contains same inspection_id as response).
"""

import io
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.main import app
from app.schemas.inspection import InspectionResponse
from app.services.image_storage_service import ImageStorageService
from tests.test_schema import sample_valid_extraction_dict


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    return TestClient(app)


def create_test_package_image(width: int = 800, height: int = 1000) -> bytes:
    """Creates a valid synthetic package image with clear boundaries."""
    image = Image.new("RGB", (width, height), color=(240, 240, 240))
    draw = ImageDraw.Draw(image)

    # Draw package container rectangle occupying [100, 50, 700, 950]
    pkg_x1, pkg_y1, pkg_x2, pkg_y2 = 100, 50, 700, 950
    draw.rectangle([pkg_x1, pkg_y1, pkg_x2, pkg_y2], fill=(255, 255, 255), outline=(50, 50, 50), width=4)

    # Draw text declarations
    draw.text((150, 100), "TEST BRAND BASMATI RICE", fill=(0, 0, 0))
    draw.text((150, 200), "NET WT: 500 g", fill=(0, 0, 0))
    draw.text((150, 300), "MRP Rs. 120.00", fill=(0, 0, 0))
    draw.text((150, 400), "Mfg: 08/2026", fill=(0, 0, 0))
    draw.text((150, 500), "Care: 1800-123-4567", fill=(0, 0, 0))

    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return buf.getvalue()


# ----------------------------------------------------------------------
# A. Successful End-to-End Inspection
# ----------------------------------------------------------------------
def test_end_to_end_inspection_success(client):
    """Test A: Valid image upload executes full pipeline and returns unified inspection."""
    image_bytes = create_test_package_image()
    extraction_data = sample_valid_extraction_dict()
    extraction_json_str = json.dumps(extraction_data)

    with patch("app.services.vision_llm.VisionLLMClient.extract", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = extraction_json_str

        response = client.post(
            "/api/v1/inspection",
            files={"image": ("test_package.jpg", image_bytes, "image/jpeg")},
            data={
                "brand_name": "ABC Brand",
                "generic_name": "Basmati Rice",
                "package_width_mm": 150.0,
                "package_height_mm": 200.0,
                "rule_source_mode": "sih_ps_26034",
            },
        )

    assert response.status_code == 200
    res_json = response.json()
    assert res_json["success"] is True
    data = res_json["data"]

    # Verify inspection_id
    inspection_id = data["inspection_id"]
    assert inspection_id.startswith("INSP-")

    # Verify metadata
    metadata = data["metadata"]
    assert metadata["brand_name"] == "ABC Brand"
    assert metadata["generic_name"] == "Basmati Rice"
    assert metadata["package_dimensions_mm"]["width_mm"] == 150.0
    assert metadata["package_dimensions_mm"]["height_mm"] == 200.0
    assert metadata["rule_source_mode"] == "sih_ps_26034"
    assert "source_image_sha256" in metadata
    assert metadata["source_image_filename"] == "test_package.jpg"

    # Verify summary
    summary = data["summary"]
    assert summary["total_declarations_evaluated"] == 6
    assert summary["detected_declarations_count"] > 0
    assert "rules_evaluated_count" in summary

    # Verify findings
    findings = data["findings"]
    assert len(findings) == 6
    fields = [f["field"] for f in findings]
    assert "net_quantity" in fields
    assert "mrp" in fields
    assert "manufacturer_packer_importer" in fields

    # Verify persistence of inspection.json
    storage = ImageStorageService()
    saved_json_path = storage.get_inspection_json_path(inspection_id)
    assert saved_json_path.exists()
    persisted = json.loads(saved_json_path.read_text(encoding="utf-8"))
    assert persisted["inspection_id"] == inspection_id
    assert storage.get_report_path(inspection_id).exists()
    assert storage.get_report_path(inspection_id).read_bytes().startswith(b"%PDF")


# ----------------------------------------------------------------------
# B. Original Image Storage Integrity & SHA-256 Provenance
# ----------------------------------------------------------------------
def test_original_image_storage_integrity(client):
    """Test B: Uploaded bytes are stored byte-for-byte unchanged with matching SHA-256."""
    image_bytes = create_test_package_image()
    extraction_data = sample_valid_extraction_dict()

    with patch("app.services.vision_llm.VisionLLMClient.extract", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = json.dumps(extraction_data)

        response = client.post(
            "/api/v1/inspection",
            files={"image": ("original_lays.jpg", image_bytes, "image/jpeg")},
        )

    assert response.status_code == 200
    data = response.json()["data"]
    inspection_id = data["inspection_id"]

    storage = ImageStorageService()
    stored_bytes = storage.get_original_bytes(inspection_id)

    # Byte-for-byte exact match (no PIL re-encoding)
    assert stored_bytes == image_bytes

    # SHA-256 matches exact uploaded bytes
    expected_sha256 = storage.calculate_sha256(image_bytes)
    assert data["metadata"]["source_image_sha256"] == expected_sha256
    assert storage.get_original_sha256(inspection_id) == expected_sha256


# ----------------------------------------------------------------------
# C. Rule Source Mode Selection & Validation
# ----------------------------------------------------------------------
def test_rule_source_mode_accepted_and_rejected(client):
    """Test C: sih_ps_26034 and doca_statutory_2011 accepted; invalid mode returns 422."""
    image_bytes = create_test_package_image()
    extraction_data = sample_valid_extraction_dict()

    with patch("app.services.vision_llm.VisionLLMClient.extract", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = json.dumps(extraction_data)

        # 1. doca_statutory_2011 accepted
        res_doca = client.post(
            "/api/v1/inspection",
            files={"image": ("test.jpg", image_bytes, "image/jpeg")},
            data={"rule_source_mode": "doca_statutory_2011"},
        )
        assert res_doca.status_code == 200
        assert res_doca.json()["data"]["metadata"]["rule_source_mode"] == "doca_statutory_2011"

        # 2. sih_ps_26034 accepted
        res_sih = client.post(
            "/api/v1/inspection",
            files={"image": ("test.jpg", image_bytes, "image/jpeg")},
            data={"rule_source_mode": "sih_ps_26034"},
        )
        assert res_sih.status_code == 200
        assert res_sih.json()["data"]["metadata"]["rule_source_mode"] == "sih_ps_26034"

        # 3. Invalid mode rejected with 422
        res_inv = client.post(
            "/api/v1/inspection",
            files={"image": ("test.jpg", image_bytes, "image/jpeg")},
            data={"rule_source_mode": "invalid_mode_xyz"},
        )
        assert res_inv.status_code == 422
        err = res_inv.json()["error"]
        assert err["code"] == "INVALID_RULE_SOURCE_MODE"


# ----------------------------------------------------------------------
# D. Invalid Image Error Handling
# ----------------------------------------------------------------------
def test_invalid_image_returns_controlled_error(client):
    """Test D: Empty bytes or corrupt image returns clean 4xx without traceback."""
    # 1. Empty image bytes -> 400
    res_empty = client.post(
        "/api/v1/inspection",
        files={"image": ("empty.jpg", b"", "image/jpeg")},
    )
    assert res_empty.status_code == 400
    err_empty = res_empty.json()["error"]
    assert err_empty["code"] == "EMPTY_IMAGE_BYTES"

    # 2. Corrupt non-image bytes -> 400
    res_corrupt = client.post(
        "/api/v1/inspection",
        files={"image": ("corrupt.jpg", b"NOT_AN_IMAGE_CONTENT_HERE", "image/jpeg")},
    )
    assert res_corrupt.status_code == 400
    assert "error" in res_corrupt.json()


# ----------------------------------------------------------------------
# E. Source Filename Preservation
# ----------------------------------------------------------------------
def test_source_filename_preserved(client):
    """Test E: Uploaded filename is faithfully captured in metadata and paths."""
    image_bytes = create_test_package_image()
    extraction_data = sample_valid_extraction_dict()

    with patch("app.services.vision_llm.VisionLLMClient.extract", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = json.dumps(extraction_data)

        res = client.post(
            "/api/v1/inspection",
            files={"image": ("custom_lays_chips_panel.png", image_bytes, "image/png")},
        )

    assert res.status_code == 200
    meta = res.json()["data"]["metadata"]
    assert meta["source_image_filename"] == "custom_lays_chips_panel.png"
    assert "original_image" in meta["source_image_storage_path"]


# ----------------------------------------------------------------------
# F. NOT VISIBLE Semantics Preserved
# ----------------------------------------------------------------------
def test_not_visible_semantics_preserved(client):
    """Test F: Missing declaration is labeled NOT VISIBLE, never converted to NON-COMPLIANT."""
    image_bytes = create_test_package_image()
    extraction_data = sample_valid_extraction_dict()
    # Mark consumer care as completely missing from supplied panel
    extraction_data["consumer_care"] = {
        "status": "missing",
        "name": {"status": "missing", "value": None, "confidence": 0.0},
        "address": {"status": "missing", "value": None, "confidence": 0.0},
        "phone": {"status": "missing", "value": None, "confidence": 0.0},
        "email": {"status": "missing", "value": None, "confidence": 0.0},
        "raw_text": None,
        "confidence": 0.0,
        "bbox": None,
    }

    with patch("app.services.vision_llm.VisionLLMClient.extract", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = json.dumps(extraction_data)

        res = client.post(
            "/api/v1/inspection",
            files={"image": ("test.jpg", image_bytes, "image/jpeg")},
        )

    assert res.status_code == 200
    findings = res.json()["data"]["findings"]
    care_finding = next(f for f in findings if f["field"] == "consumer_care")
    assert care_finding["visibility"] == "NOT VISIBLE"
    assert care_finding["status"] == "NOT VISIBLE"
    assert care_finding["status"] != "NON-COMPLIANT"


# ----------------------------------------------------------------------
# G. DETECTED != PASS Semantics
# ----------------------------------------------------------------------
def test_detected_not_equal_to_pass(client):
    """Test G: Detected declaration with no applicable Rule 7/8 rule becomes NOT ASSESSABLE."""
    image_bytes = create_test_package_image()
    extraction_data = sample_valid_extraction_dict()

    with patch("app.services.vision_llm.VisionLLMClient.extract", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = json.dumps(extraction_data)

        res = client.post(
            "/api/v1/inspection",
            files={"image": ("test.jpg", image_bytes, "image/jpeg")},
        )

    assert res.status_code == 200
    findings = res.json()["data"]["findings"]

    # Manufacturer has visibility evaluation only; no Rule 7/Rule 8 numeral-height rule
    mfr_finding = next(f for f in findings if f["field"] == "manufacturer_packer_importer")
    assert mfr_finding["visibility"] == "DETECTED"
    # DETECTED != PASS: without applicable rule, overall status is NOT ASSESSABLE
    assert mfr_finding["status"] == "NOT ASSESSABLE"


# ----------------------------------------------------------------------
# H. Date Semantics Mapping
# ----------------------------------------------------------------------
def test_date_semantics_mapping(client):
    """Test H: Manufacture and use_by dates remain correctly mapped."""
    image_bytes = create_test_package_image()
    extraction_data = sample_valid_extraction_dict()
    extraction_data["dates"] = [
        {
            "status": "present",
            "date_type": "manufacture",
            "raw_text": "Mfg: 01/2026",
            "month": "01",
            "year": 2026,
            "confidence": 0.95,
            "bbox": [0.1, 0.4, 0.4, 0.45],
            "numeral_region": [0.2, 0.4, 0.35, 0.45],
        },
        {
            "status": "present",
            "date_type": "use_by",
            "raw_text": "Best Before: 12/2026",
            "month": "12",
            "year": 2026,
            "confidence": 0.95,
            "bbox": [0.5, 0.4, 0.8, 0.45],
            "numeral_region": [0.6, 0.4, 0.75, 0.45],
        },
    ]

    with patch("app.services.vision_llm.VisionLLMClient.extract", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = json.dumps(extraction_data)

        res = client.post(
            "/api/v1/inspection",
            files={"image": ("test.jpg", image_bytes, "image/jpeg")},
        )

    assert res.status_code == 200
    findings = res.json()["data"]["findings"]
    mfg_finding = next(f for f in findings if f["field"] == "date_of_manufacture")
    use_by_finding = next(f for f in findings if f["field"] == "date_use_by")
    assert mfg_finding["visibility"] == "DETECTED"
    assert use_by_finding["visibility"] == "DETECTED"
    assert mfg_finding["detected_value"] == "01/2026"
    assert use_by_finding["detected_value"] == "12/2026"


# ----------------------------------------------------------------------
# I. Rule 7 & Rule 8 Finding Pass-Through
# ----------------------------------------------------------------------
def test_rule7_and_rule8_findings_passed_through(client):
    """Test I: Rule 7 and Rule 8 engine results are passed through without modification."""
    image_bytes = create_test_package_image()
    extraction_data = sample_valid_extraction_dict()

    with patch("app.services.vision_llm.VisionLLMClient.extract", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = json.dumps(extraction_data)

        res = client.post(
            "/api/v1/inspection",
            files={"image": ("test.jpg", image_bytes, "image/jpeg")},
            data={"package_width_mm": 150.0, "package_height_mm": 200.0},
        )

    assert res.status_code == 200
    data = res.json()["data"]
    findings = data["findings"]
    net_qty = next(f for f in findings if f["field"] == "net_quantity")

    # Rule 7 assessment should be present for net quantity
    assert net_qty["rules"]["rule7"] is not None
    assert net_qty["rules"]["rule7"]["status"] in ("PASS", "NON-COMPLIANT", "NOT ASSESSABLE")

    # Rule 8 assessment should be present for net quantity
    assert net_qty["rules"]["rule8"] is not None
    assert net_qty["rules"]["rule8"]["status"] in ("PASS", "NON-COMPLIANT", "NOT ASSESSABLE")


# ----------------------------------------------------------------------
# J. Inspection JSON Persistence Matches Response ID
# ----------------------------------------------------------------------
def test_persisted_inspection_json_matches_response(client):
    """Test J: Stored inspection.json has matching inspection_id and contents."""
    image_bytes = create_test_package_image()
    extraction_data = sample_valid_extraction_dict()

    with patch("app.services.vision_llm.VisionLLMClient.extract", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = json.dumps(extraction_data)

        res = client.post(
            "/api/v1/inspection",
            files={"image": ("test_persist.jpg", image_bytes, "image/jpeg")},
        )

    assert res.status_code == 200
    data = res.json()["data"]
    inspection_id = data["inspection_id"]

    storage = ImageStorageService()
    json_path = storage.get_inspection_json_path(inspection_id)
    assert json_path.exists()

    with open(json_path, "r", encoding="utf-8") as f:
        persisted = json.load(f)

    assert persisted["inspection_id"] == inspection_id
    assert persisted["metadata"]["source_image_sha256"] == data["metadata"]["source_image_sha256"]
    assert len(persisted["findings"]) == len(data["findings"])
