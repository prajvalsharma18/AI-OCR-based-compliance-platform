"""Integration tests for POST /api/v1/measure and MeasurementService with user-supplied package_bbox_px."""

import io
import json
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.main import app
from tests.test_schema import sample_valid_extraction_dict


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    return TestClient(app)


def create_sample_package_image(width: int = 1000, height: int = 1000) -> bytes:
    """Creates a synthetic package image with clear printed declarations and numerals."""
    image = Image.new("RGB", (width, height), color=(240, 240, 240))
    draw = ImageDraw.Draw(image)

    # Draw package container rectangle (occupying [170, 30, 830, 980])
    pkg_x1, pkg_y1, pkg_x2, pkg_y2 = 170, 30, 830, 980
    draw.rectangle([pkg_x1, pkg_y1, pkg_x2, pkg_y2], fill=(255, 255, 255), outline=(50, 50, 50), width=4)

    # Draw Brand & Title
    draw.text((250, 100), "GOOD LIFE SONA MASOORI RICE", fill=(0, 0, 0))

    # Draw Net Quantity declaration with numeral "1" around (680, 240)
    # Corresponding normalized region roughly around x=[0.67, 0.23, 0.76, 0.28]
    draw.text((680, 240), "NET WT: 1 kg", fill=(0, 0, 0))

    # Draw MRP declaration
    draw.text((250, 520), "MRP Rs. 120.00", fill=(0, 0, 0))

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_measure_endpoint_success_with_package_bbox(client):
    """Test 1: Valid measurement request with user-supplied package_bbox_px returns 200."""
    image_bytes = create_sample_package_image(1000, 1000)

    # Prepare extraction JSON matching image layout
    extraction_data = sample_valid_extraction_dict()
    extraction_data["net_quantity"]["numeral_region"] = [0.67, 0.23, 0.76, 0.28]
    extraction_data["mrp"]["numeral_region"] = [0.24, 0.51, 0.45, 0.55]
    extraction_json_str = json.dumps(extraction_data)

    package_bbox_px = [170, 30, 830, 980]

    response = client.post(
        "/api/v1/measure",
        files={"image": ("test_package.jpg", image_bytes, "image/jpeg")},
        data={
            "extraction_json": extraction_json_str,
            "package_width_mm": 180.0,
            "package_height_mm": 280.0,
            "package_bbox_px": json.dumps(package_bbox_px),
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    data = body["data"]

    # Verify calibration metadata
    calib = data["calibration"]
    assert calib["package_width_mm"] == 180.0
    assert calib["package_height_mm"] == 280.0
    assert calib["package_bbox_px"] == package_bbox_px
    assert calib["package_width_px"] == 660.0   # 830 - 170
    assert calib["package_height_px"] == 950.0  # 980 - 30
    assert calib["width_mm_per_px"] == round(180.0 / 660.0, 6)
    assert calib["height_mm_per_px"] == round(280.0 / 950.0, 6)

    # Verify measurements
    measurements = data["measurements"]
    assert len(measurements) >= 1

    net_qty_meas = next((m for m in measurements if m["field"] == "net_quantity"), None)
    assert net_qty_meas is not None
    assert net_qty_meas["numeral_height_px"] > 0
    assert net_qty_meas["numeral_height_mm"] > 0
    assert net_qty_meas["measurement_quality"] in ("good", "moderate", "low")
    assert 0.0 <= net_qty_meas["confidence"] <= 1.0

    # Verify absence of legal verdicts
    assert "rule_7_pass" not in body
    assert "compliant" not in body


def test_measure_accepts_comma_separated_bbox_string(client):
    """Test that package_bbox_px accepts comma-separated string format '170, 30, 830, 980' as well as JSON array."""
    image_bytes = create_sample_package_image(1000, 1000)
    extraction_data = sample_valid_extraction_dict()
    extraction_data["net_quantity"]["numeral_region"] = [0.67, 0.23, 0.76, 0.28]

    response = client.post(
        "/api/v1/measure",
        files={"image": ("test_package.jpg", image_bytes, "image/jpeg")},
        data={
            "extraction_json": json.dumps(extraction_data),
            "package_width_mm": 180.0,
            "package_height_mm": 280.0,
            "package_bbox_px": "170, 30, 830, 980",  # Comma separated string without brackets
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["data"]["calibration"]["package_bbox_px"] == [170, 30, 830, 980]


def test_measure_full_image_bbox_warning(client):
    """Test 2: Full-image bounding box sets measurement_quality='low' and gives warning note."""
    image_bytes = create_sample_package_image(1000, 1000)
    extraction_data = sample_valid_extraction_dict()
    extraction_data["net_quantity"]["numeral_region"] = [0.67, 0.23, 0.76, 0.28]

    # Full image as package bbox
    full_image_bbox = [0, 0, 1000, 1000]

    response = client.post(
        "/api/v1/measure",
        files={"image": ("test_package.jpg", image_bytes, "image/jpeg")},
        data={
            "extraction_json": json.dumps(extraction_data),
            "package_width_mm": 180.0,
            "package_height_mm": 280.0,
            "package_bbox_px": json.dumps(full_image_bbox),
        },
    )

    assert response.status_code == 200
    data = response.json()["data"]
    for meas in data["measurements"]:
        assert meas["measurement_quality"] == "low"
        assert "Calibration bbox covers the full image" in meas["notes"]


def test_measure_rejects_inverted_x_coords(client):
    """Test 3: Reject package_bbox_px where x_min >= x_max with 422."""
    image_bytes = create_sample_package_image(1000, 1000)
    response = client.post(
        "/api/v1/measure",
        files={"image": ("pkg.jpg", image_bytes, "image/jpeg")},
        data={
            "extraction_json": json.dumps(sample_valid_extraction_dict()),
            "package_width_mm": 180.0,
            "package_height_mm": 280.0,
            "package_bbox_px": "[830, 30, 170, 980]",  # x_min (830) > x_max (170)
        },
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_PACKAGE_BBOX"
    assert "x_min" in response.json()["error"]["message"]


def test_measure_rejects_inverted_y_coords(client):
    """Test 4: Reject package_bbox_px where y_min >= y_max with 422."""
    image_bytes = create_sample_package_image(1000, 1000)
    response = client.post(
        "/api/v1/measure",
        files={"image": ("pkg.jpg", image_bytes, "image/jpeg")},
        data={
            "extraction_json": json.dumps(sample_valid_extraction_dict()),
            "package_width_mm": 180.0,
            "package_height_mm": 280.0,
            "package_bbox_px": "[170, 980, 830, 30]",  # y_min (980) > y_max (30)
        },
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_PACKAGE_BBOX"
    assert "y_min" in response.json()["error"]["message"]


def test_measure_rejects_negative_coords(client):
    """Test 5: Reject package_bbox_px with negative coordinates."""
    image_bytes = create_sample_package_image(1000, 1000)
    response = client.post(
        "/api/v1/measure",
        files={"image": ("pkg.jpg", image_bytes, "image/jpeg")},
        data={
            "extraction_json": json.dumps(sample_valid_extraction_dict()),
            "package_width_mm": 180.0,
            "package_height_mm": 280.0,
            "package_bbox_px": "[-10, 30, 830, 980]",
        },
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_PACKAGE_BBOX"


def test_measure_rejects_out_of_bounds_coords(client):
    """Test 6: Reject package_bbox_px exceeding image dimensions."""
    image_bytes = create_sample_package_image(1000, 1000)
    response = client.post(
        "/api/v1/measure",
        files={"image": ("pkg.jpg", image_bytes, "image/jpeg")},
        data={
            "extraction_json": json.dumps(sample_valid_extraction_dict()),
            "package_width_mm": 180.0,
            "package_height_mm": 280.0,
            "package_bbox_px": "[170, 30, 1050, 980]",  # x_max 1050 > 1000
        },
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_PACKAGE_BBOX"


def test_measure_debug_visualization_generation(client):
    """Test 7: Verification of optional debug overlay rendering."""
    image_bytes = create_sample_package_image(1000, 1000)
    extraction_data = sample_valid_extraction_dict()
    extraction_data["net_quantity"]["numeral_region"] = [0.67, 0.23, 0.76, 0.28]

    response = client.post(
        "/api/v1/measure",
        files={"image": ("test_package.jpg", image_bytes, "image/jpeg")},
        data={
            "extraction_json": json.dumps(extraction_data),
            "package_width_mm": 180.0,
            "package_height_mm": 280.0,
            "package_bbox_px": "[170, 30, 830, 980]",
            "include_debug_image": "true",
        },
    )

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["debug_image_base64"] is not None
    assert data["debug_image_base64"].startswith("data:image/jpeg;base64,")


def test_measure_rejects_non_positive_dimensions(client):
    """Test 8: Rejection of zero or negative package dimensions."""
    image_bytes = create_sample_package_image()
    extraction_json_str = json.dumps(sample_valid_extraction_dict())

    # Zero width
    resp_zero_w = client.post(
        "/api/v1/measure",
        files={"image": ("pkg.jpg", image_bytes, "image/jpeg")},
        data={
            "extraction_json": extraction_json_str,
            "package_width_mm": 0.0,
            "package_height_mm": 180.0,
            "package_bbox_px": "[100, 75, 700, 925]",
        },
    )
    assert resp_zero_w.status_code == 400
    assert resp_zero_w.json()["error"]["code"] == "INVALID_PACKAGE_WIDTH"

    # Negative height
    resp_neg_h = client.post(
        "/api/v1/measure",
        files={"image": ("pkg.jpg", image_bytes, "image/jpeg")},
        data={
            "extraction_json": extraction_json_str,
            "package_width_mm": 120.0,
            "package_height_mm": -10.0,
            "package_bbox_px": "[100, 75, 700, 925]",
        },
    )
    assert resp_neg_h.status_code == 400
    assert resp_neg_h.json()["error"]["code"] == "INVALID_PACKAGE_HEIGHT"


def test_measure_rejects_missing_numeral_regions(client):
    """Test 9: Structured 422 error when extraction_json has no numeral_region fields."""
    image_bytes = create_sample_package_image()
    data = sample_valid_extraction_dict()
    data["net_quantity"]["numeral_region"] = None
    data["mrp"]["numeral_region"] = None
    data["dates"] = []

    response = client.post(
        "/api/v1/measure",
        files={"image": ("pkg.jpg", image_bytes, "image/jpeg")},
        data={
            "extraction_json": json.dumps(data),
            "package_width_mm": 120.0,
            "package_height_mm": 180.0,
            "package_bbox_px": "[100, 75, 700, 925]",
        },
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "NO_NUMERAL_REGIONS_FOUND"


# ==============================================================================
# MULTI-NUMERAL MEASUREMENT CANDIDATE EXTRACTION TESTS (A THROUGH E)
# ==============================================================================

def test_measure_candidate_extraction_only_net_quantity(client):
    """Test A: Extraction with ONLY net_quantity.numeral_region produces exactly 1 measurement."""
    image_bytes = create_sample_package_image(1000, 1000)
    data = sample_valid_extraction_dict()
    data["net_quantity"]["numeral_region"] = [0.67, 0.23, 0.76, 0.28]
    data["mrp"]["numeral_region"] = None
    data["dates"] = []

    response = client.post(
        "/api/v1/measure",
        files={"image": ("pkg.jpg", image_bytes, "image/jpeg")},
        data={
            "extraction_json": json.dumps(data),
            "package_width_mm": 180.0,
            "package_height_mm": 280.0,
            "package_bbox_px": "[170, 30, 830, 980]",
        },
    )

    assert response.status_code == 200
    measurements = response.json()["data"]["measurements"]
    assert len(measurements) == 1
    assert measurements[0]["field"] == "net_quantity"
    assert measurements[0]["numeral"] == "500"


def test_measure_candidate_extraction_net_qty_and_mrp(client):
    """Test B: Extraction with net_quantity + MRP produces 2 measurements (net_quantity, mrp)."""
    image_bytes = create_sample_package_image(1000, 1000)
    data = sample_valid_extraction_dict()
    data["net_quantity"]["numeral_region"] = [0.67, 0.23, 0.76, 0.28]
    data["mrp"]["numeral_region"] = [0.24, 0.51, 0.45, 0.55]
    data["dates"] = []

    response = client.post(
        "/api/v1/measure",
        files={"image": ("pkg.jpg", image_bytes, "image/jpeg")},
        data={
            "extraction_json": json.dumps(data),
            "package_width_mm": 180.0,
            "package_height_mm": 280.0,
            "package_bbox_px": "[170, 30, 830, 980]",
        },
    )

    assert response.status_code == 200
    measurements = response.json()["data"]["measurements"]
    assert len(measurements) == 2
    fields = [m["field"] for m in measurements]
    assert "net_quantity" in fields
    assert "mrp" in fields

    mrp_meas = next(m for m in measurements if m["field"] == "mrp")
    assert mrp_meas["numeral"] == "120.00"


def test_measure_candidate_extraction_net_qty_mrp_and_one_date(client):
    """Test C: Extraction with net_quantity + MRP + 1 numeric date produces 3 measurements."""
    image_bytes = create_sample_package_image(1000, 1000)
    data = sample_valid_extraction_dict()
    data["net_quantity"]["numeral_region"] = [0.67, 0.23, 0.76, 0.28]
    data["mrp"]["numeral_region"] = [0.24, 0.51, 0.45, 0.55]
    data["dates"] = [
        {
            "status": "present",
            "date_type": "packing",
            "raw_text": "PKD: 08/2026",
            "month": "08",
            "year": 2026,
            "confidence": 0.95,
            "numeral_region": [0.55, 0.50, 0.85, 0.55]
        }
    ]

    response = client.post(
        "/api/v1/measure",
        files={"image": ("pkg.jpg", image_bytes, "image/jpeg")},
        data={
            "extraction_json": json.dumps(data),
            "package_width_mm": 180.0,
            "package_height_mm": 280.0,
            "package_bbox_px": "[170, 30, 830, 980]",
        },
    )

    assert response.status_code == 200
    measurements = response.json()["data"]["measurements"]
    assert len(measurements) == 3
    fields = [m["field"] for m in measurements]
    assert "net_quantity" in fields
    assert "mrp" in fields
    assert "dates[0]" in fields


def test_measure_candidate_extraction_multiple_numeric_dates(client):
    """Test D: Extraction with multiple numeric dates produces indexed fields 'dates[0]', 'dates[1]'."""
    image_bytes = create_sample_package_image(1000, 1000)
    data = sample_valid_extraction_dict()
    data["net_quantity"]["numeral_region"] = None
    data["mrp"]["numeral_region"] = None
    data["dates"] = [
        {
            "status": "present",
            "date_type": "manufacture",
            "raw_text": "MFG: 01/2026",
            "month": "01",
            "year": 2026,
            "confidence": 0.95,
            "numeral_region": [0.55, 0.40, 0.85, 0.45]
        },
        {
            "status": "present",
            "date_type": "packing",
            "raw_text": "PKD: 08/2026",
            "month": "08",
            "year": 2026,
            "confidence": 0.95,
            "numeral_region": [0.55, 0.50, 0.85, 0.55]
        }
    ]

    response = client.post(
        "/api/v1/measure",
        files={"image": ("pkg.jpg", image_bytes, "image/jpeg")},
        data={
            "extraction_json": json.dumps(data),
            "package_width_mm": 180.0,
            "package_height_mm": 280.0,
            "package_bbox_px": "[170, 30, 830, 980]",
        },
    )

    assert response.status_code == 200
    measurements = response.json()["data"]["measurements"]
    assert len(measurements) == 2
    assert measurements[0]["field"] == "dates[0]"
    assert measurements[1]["field"] == "dates[1]"


def test_measure_candidate_extraction_ignores_additional_numeric_info(client):
    """Test E: Non-Rule-7 numbers in additional_numeric_information (phone, PIN, license, barcode) MUST NOT generate measurements."""
    image_bytes = create_sample_package_image(1000, 1000)
    data = sample_valid_extraction_dict()
    data["net_quantity"]["numeral_region"] = [0.67, 0.23, 0.76, 0.28]
    data["mrp"]["numeral_region"] = None
    data["dates"] = []
    data["additional_numeric_information"] = [
        {
            "semantic_role": "phone",
            "raw_text": "Customer Care: 022-67276727",
            "value": "022-67276727",
            "bbox": [0.1, 0.85, 0.6, 0.9],
            "confidence": 0.95
        },
        {
            "semantic_role": "pin_code",
            "raw_text": "Mumbai 400002",
            "value": "400002",
            "bbox": [0.1, 0.75, 0.4, 0.8],
            "confidence": 0.98
        },
        {
            "semantic_role": "other",
            "raw_text": "Lic. No. 11517018000800",
            "value": "11517018000800",
            "bbox": [0.1, 0.9, 0.7, 0.95],
            "confidence": 0.96
        }
    ]

    response = client.post(
        "/api/v1/measure",
        files={"image": ("pkg.jpg", image_bytes, "image/jpeg")},
        data={
            "extraction_json": json.dumps(data),
            "package_width_mm": 180.0,
            "package_height_mm": 280.0,
            "package_bbox_px": "[170, 30, 830, 980]",
        },
    )

    assert response.status_code == 200
    measurements = response.json()["data"]["measurements"]
    # Only net_quantity should be measured; none of the non-Rule 7 numbers should become measurements!
    assert len(measurements) == 1
    assert measurements[0]["field"] == "net_quantity"

