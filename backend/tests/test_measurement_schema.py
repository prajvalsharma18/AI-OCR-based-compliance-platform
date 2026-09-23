"""Unit tests for Module 2A measurement schemas and calibration calculation math."""

import pytest
from pydantic import ValidationError

from app.cv.measurement import (
    calculate_numeral_height_mm,
    compute_calibration_metadata,
)
from app.cv.preprocessing import convert_normalized_bbox_to_pixels
from app.schemas.measurement import (
    CalibrationMetadata,
    MeasurementImageMetadata,
    MeasurementResponse,
    MeasurementResponseData,
    NumeralMeasurement,
)


def sample_valid_numeral_measurement():
    """Returns a valid NumeralMeasurement dictionary."""
    return {
        "field": "net_quantity",
        "raw_text": "NET WT: 1 kg",
        "numeral": "1",
        "semantic_region_normalized": [0.683, 0.244, 0.742, 0.271],
        "numeral_bbox_px": [695, 246, 708, 270],
        "numeral_height_px": 24,
        "numeral_height_mm": 3.0,
        "confidence": 0.91,
        "measurement_quality": "good",
        "notes": "Clear numeral glyph isolated."
    }


def sample_valid_calibration_dict():
    """Returns a valid CalibrationMetadata dictionary."""
    return {
        "method": "package_dimensions",
        "package_width_mm": 180.0,
        "package_height_mm": 280.0,
        "package_bbox_px": [170, 30, 830, 980],
        "package_width_px": 660.0,
        "package_height_px": 950.0,
        "width_mm_per_px": 0.272727,
        "height_mm_per_px": 0.294737
    }


def test_valid_measurement_schema():
    """Test 1: Valid measurement response schema validation."""
    data = {
        "image": {"width": 1000, "height": 1000},
        "calibration": sample_valid_calibration_dict(),
        "measurements": [sample_valid_numeral_measurement()],
        "debug_image_base64": None,
    }

    model = MeasurementResponseData.model_validate(data)
    assert model.image.width == 1000
    assert model.calibration.package_bbox_px == [170, 30, 830, 980]
    assert model.calibration.package_width_px == 660.0
    assert model.calibration.package_height_px == 950.0
    assert model.calibration.width_mm_per_px == 0.272727
    assert len(model.measurements) == 1
    assert model.measurements[0].numeral_height_mm == 3.0
    assert model.measurements[0].measurement_quality == "good"

    # Wrap in API response envelope
    resp = MeasurementResponse(success=True, data=model)
    assert resp.success is True


def test_calibration_schema_requires_package_bbox_px():
    """Test 2: CalibrationMetadata requires package_bbox_px (must not be missing or optional)."""
    calib = sample_valid_calibration_dict()
    del calib["package_bbox_px"]

    with pytest.raises(ValidationError):
        CalibrationMetadata.model_validate(calib)


def test_calibration_math_using_package_bbox():
    """Test 3: Calibration math derived from package_bbox_px dimensions."""
    bbox = [170, 30, 830, 980]
    width_px = 830 - 170   # 660 px
    height_px = 980 - 30   # 950 px

    calib = compute_calibration_metadata(
        package_width_mm=180.0,
        package_height_mm=280.0,
        package_bbox_px=bbox,
    )
    assert calib.package_bbox_px == bbox
    assert calib.package_width_px == width_px
    assert calib.package_height_px == height_px
    assert calib.width_mm_per_px == round(180.0 / 660.0, 6)
    assert calib.height_mm_per_px == round(280.0 / 950.0, 6)


def test_exact_specification_numeral_height_calculation():
    """Test 4: Specific test from SIH PS 26034 specification:
    package_height_mm = 280, package_height_px = 950 -> height_mm_per_px = 280 / 950.
    numeral_height_px = 19 -> numeral_height_mm = 19 * (280 / 950) = 5.60 mm.
    """
    height_mm_per_px = 280.0 / 950.0
    numeral_height_px = 19
    numeral_height_mm = calculate_numeral_height_mm(
        numeral_height_px=numeral_height_px,
        height_mm_per_px=height_mm_per_px,
    )
    assert numeral_height_mm == 5.60


def test_invalid_package_dimensions():
    """Test 5: Calibration rejected when package dimensions are non-positive."""
    with pytest.raises(ValueError, match="Package physical dimensions must be strictly positive"):
        compute_calibration_metadata(
            package_width_mm=-180.0,
            package_height_mm=280.0,
            package_bbox_px=[170, 30, 830, 980],
        )

    with pytest.raises(ValueError, match="Package physical dimensions must be strictly positive"):
        compute_calibration_metadata(
            package_width_mm=180.0,
            package_height_mm=0.0,
            package_bbox_px=[170, 30, 830, 980],
        )


def test_normalized_bbox_to_pixels_conversion():
    """Test 6: Safe conversion of normalized bounding boxes to clamped pixels."""
    norm_bbox = [0.68, 0.241, 0.756, 0.269]
    px_coords = convert_normalized_bbox_to_pixels(norm_bbox, image_width=1000, image_height=1000)

    assert px_coords == (680, 241, 756, 269)
    assert px_coords[2] - px_coords[0] == 76
    assert px_coords[3] - px_coords[1] == 28


def test_measurement_confidence_range():
    """Test 7: Confidence must be bounded between 0.0 and 1.0."""
    data = sample_valid_numeral_measurement()
    data["confidence"] = 1.2

    with pytest.raises(ValidationError):
        NumeralMeasurement.model_validate(data)


def test_explicit_failed_measurement_representation():
    """Test 8: Failed measurement must be represented with quality='failed' without legal verdicts."""
    data = sample_valid_numeral_measurement()
    data["measurement_quality"] = "failed"
    data["notes"] = "Excessive glare obscured glyph boundary."

    model = NumeralMeasurement.model_validate(data)
    assert model.measurement_quality == "failed"
    # Verify no legal verdict fields are present
    assert not hasattr(model, "pass_fail")
    assert not hasattr(model, "rule_7_compliant")
