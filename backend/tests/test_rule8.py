"""Unit and integration tests for Module 3B: Rule 8 Spatial Clearance Analysis.

Tests statutory compliance with Rule 8(1) proviso of the Legal Metrology (Packaged Commodities)
Rules, 2011:
- Top clearance >= H
- Bottom clearance >= H
- Left clearance >= 2H
- Right clearance >= 2H
Where H is numeral_height_mm.

Covers tests 1 through 18, API endpoints, calibration differentiation, and Lay's integration.
"""

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.measurement import (
    CalibrationMetadata,
    MeasurementQuality,
    NumeralMeasurement,
)
from app.services.rule8_service import Rule8Service


def make_sample_calibration(scale_v: float = 0.1, scale_h: float = 0.1, pkg_bbox: list[int] = None) -> CalibrationMetadata:
    """Helper to create valid calibration metadata."""
    if pkg_bbox is None:
        pkg_bbox = [0, 0, 1000, 1000]
    return CalibrationMetadata(
        method="package_dimensions",
        package_width_mm=100.0,
        package_height_mm=100.0,
        package_bbox_px=pkg_bbox,
        package_width_px=1000.0,
        package_height_px=1000.0,
        width_mm_per_px=scale_h,
        height_mm_per_px=scale_v,
        confidence=0.98,
        quality="good",
    )


def make_sample_net_qty_measurement(
    numeral_bbox: list[int] = None,
    height_mm: float = 2.0,
    height_px: int = 20,
    quality: MeasurementQuality = "good",
    confidence: float = 0.95,
) -> NumeralMeasurement:
    """Helper to create valid net quantity numeral measurement."""
    if numeral_bbox is None:
        numeral_bbox = [400, 400, 450, 420]  # height=20, width=50
    return NumeralMeasurement(
        field="net_quantity",
        raw_text="NET QTY: 500 g",
        numeral="500",
        semantic_region_normalized=[0.3, 0.39, 0.55, 0.43],
        numeral_bbox_px=numeral_bbox,
        numeral_height_px=height_px,
        numeral_height_mm=height_mm,
        confidence=confidence,
        measurement_quality=quality,
        notes="Clean isolated glyph",
    )


def make_extraction_payload(raw_blocks: list[dict] = None) -> dict:
    """Helper to create minimal valid extraction payload with raw_text_blocks."""
    return {
        "image": {"width": 1000, "height": 1000, "coordinate_system": "normalized_0_1"},
        "raw_text_blocks": raw_blocks or [],
        "semantic_extraction": {
            "image_quality": {"overall": "good", "notes": "Clear lighting"},
            "package": {
                "package_type": "retail",
                "brand_name": "Test Brand",
                "generic_name": "Namkeen",
                "food_status": "food",
                "retail_unit_count": None,
                "evidence": ["Front label"],
            },
            "manufacturer": {"status": "missing", "entities": []},
            "generic_name": {"status": "present", "raw_text": "Namkeen", "value": "Namkeen", "bbox": None},
            "mrp": {"status": "missing"},
            "dates": [],
            "net_quantity": {
                "status": "present",
                "raw_text": "NET QTY: 500 g",
                "value": 500.0,
                "unit": "g",
                "quantity_type": "weight",
                "when_packed": False,
                "confidence": 0.98,
                "bbox": [0.30, 0.39, 0.55, 0.43],  # [300, 390, 550, 430] in 1000x1000
                "numeral_region": [0.40, 0.40, 0.45, 0.42],
            },
            "consumer_care": {"status": "missing"},
            "stickers": [],
            "additional_numeric_information": [],
            "other_visible_text": [],
        },
    }


# ==============================================================================
# TESTS 1 - 8: DIRECTIONAL THRESHOLDS
# ==============================================================================

def test_1_sufficient_top_spacing():
    """TEST 1: Quantity numeral with sufficient top spacing (actual >= H)."""
    # H = 2.0 mm, scale_v = 0.1 mm/px -> req top = 20 px
    # Declaration bbox: [300, 390, 550, 430]. Declaration top = 390 px.
    # Nearest block at y = [0.34, 0.36] -> by2 = 360 px. Distance = 390 - 360 = 30 px = 3.0 mm >= 2.0 mm
    service = Rule8Service()
    calib = make_sample_calibration(scale_v=0.1, scale_h=0.1)
    meas = make_sample_net_qty_measurement(numeral_bbox=[400, 400, 450, 420], height_mm=2.0, height_px=20)
    extraction = make_extraction_payload([
        {"text": "BRAND NAME", "bbox": [0.38, 0.34, 0.48, 0.36]}  # by2 = 360 px
    ])

    finding = service.evaluate_rule8_finding(
        field="net_quantity",
        numeral_measurement=meas,
        extraction=service._parse_extraction_payload(extraction),
        calibration=calib,
        image_bgr=None,
        img_w=1000,
        img_h=1000,
    )
    assert finding.direction_results.top == "above_threshold"
    assert finding.actual_clearance_mm.top_mm == pytest.approx(3.0, 0.1)
    assert finding.required_clearance_mm.top_mm == 2.0


def test_2_insufficient_top_spacing():
    """TEST 2: Quantity numeral with insufficient top spacing (actual < H)."""
    # Declaration top = 390 px, req top = 20 px (2.0 mm)
    # Nearest block at y = [0.37, 0.385] -> by2 = 385 px. Distance = 390 - 385 = 5 px = 0.5 mm < 2.0 mm
    service = Rule8Service()
    calib = make_sample_calibration(scale_v=0.1, scale_h=0.1)
    meas = make_sample_net_qty_measurement(numeral_bbox=[400, 400, 450, 420], height_mm=2.0, height_px=20)
    extraction = make_extraction_payload([
        {"text": "INTRUDING LINE", "bbox": [0.38, 0.37, 0.48, 0.385]}  # by2 = 385 px
    ])

    finding = service.evaluate_rule8_finding(
        field="net_quantity",
        numeral_measurement=meas,
        extraction=service._parse_extraction_payload(extraction),
        calibration=calib,
        image_bgr=None,
        img_w=1000,
        img_h=1000,
    )
    assert finding.direction_results.top == "below_threshold"
    assert finding.actual_clearance_mm.top_mm < 2.0


def test_3_sufficient_bottom_spacing():
    """TEST 3: Quantity numeral with sufficient bottom spacing (actual >= H)."""
    # Declaration bottom = 430 px. Req bottom = 20 px (2.0 mm).
    # Nearest block at y = [0.47, 0.49] -> by1 = 470 px. Distance = 470 - 430 = 40 px = 4.0 mm >= 2.0 mm
    service = Rule8Service()
    calib = make_sample_calibration(scale_v=0.1, scale_h=0.1)
    meas = make_sample_net_qty_measurement(numeral_bbox=[400, 400, 450, 420], height_mm=2.0, height_px=20)
    extraction = make_extraction_payload([
        {"text": "BOTTOM TEXT", "bbox": [0.38, 0.47, 0.48, 0.49]}  # by1 = 470 px
    ])

    finding = service.evaluate_rule8_finding(
        field="net_quantity",
        numeral_measurement=meas,
        extraction=service._parse_extraction_payload(extraction),
        calibration=calib,
        image_bgr=None,
        img_w=1000,
        img_h=1000,
    )
    assert finding.direction_results.bottom == "above_threshold"
    assert finding.actual_clearance_mm.bottom_mm == pytest.approx(4.0, 0.1)


def test_4_insufficient_bottom_spacing():
    """TEST 4: Quantity numeral with insufficient bottom spacing (actual < H)."""
    # Declaration bottom = 430 px. Req bottom = 20 px (2.0 mm).
    # Nearest block at y = [0.438, 0.45] -> by1 = 438 px. Distance = 438 - 430 = 8 px = 0.8 mm < 2.0 mm
    service = Rule8Service()
    calib = make_sample_calibration(scale_v=0.1, scale_h=0.1)
    meas = make_sample_net_qty_measurement(numeral_bbox=[400, 400, 450, 420], height_mm=2.0, height_px=20)
    extraction = make_extraction_payload([
        {"text": "INTRUDING FOOTER", "bbox": [0.38, 0.438, 0.48, 0.45]}  # by1 = 438 px
    ])

    finding = service.evaluate_rule8_finding(
        field="net_quantity",
        numeral_measurement=meas,
        extraction=service._parse_extraction_payload(extraction),
        calibration=calib,
        image_bgr=None,
        img_w=1000,
        img_h=1000,
    )
    assert finding.direction_results.bottom == "below_threshold"
    assert finding.actual_clearance_mm.bottom_mm < 2.0


def test_5_sufficient_left_spacing():
    """TEST 5: Quantity numeral with sufficient left spacing (actual >= 2H)."""
    # H = 2.0 mm -> required left = 2H = 4.0 mm = 40 px.
    # Numeral left = 400 px. Semantic declaration bbox = [300, 390, 550, 430] (sx1 = 300).
    # External block at x = [0.20, 0.25] -> bx2 = 250 px.
    # Distance from numeral left = 400 - 250 = 150 px = 15.0 mm >= 4.0 mm.
    service = Rule8Service()
    calib = make_sample_calibration(scale_v=0.1, scale_h=0.1)
    meas = make_sample_net_qty_measurement(numeral_bbox=[400, 400, 450, 420], height_mm=2.0, height_px=20)
    extraction = make_extraction_payload([
        {"text": "SIDE LOGO", "bbox": [0.20, 0.39, 0.25, 0.42]}  # bx2 = 250 px
    ])

    finding = service.evaluate_rule8_finding(
        field="net_quantity",
        numeral_measurement=meas,
        extraction=service._parse_extraction_payload(extraction),
        calibration=calib,
        image_bgr=None,
        img_w=1000,
        img_h=1000,
    )
    assert finding.direction_results.left == "above_threshold"
    assert finding.actual_clearance_mm.left_mm >= finding.required_clearance_mm.left_mm
    assert finding.required_clearance_mm.left_mm == 4.0


def test_6_insufficient_left_spacing():
    """TEST 6: Quantity numeral with insufficient left spacing (actual < 2H)."""
    # H = 2.0 mm -> required left = 4.0 mm = 40 px.
    # Numeral left = 400 px. Semantic bbox sx1 = 380 px.
    # External block intruding at x = [0.35, 0.375] -> bx2 = 375 px.
    # Distance = 400 - 375 = 25 px = 2.5 mm < 4.0 mm.
    service = Rule8Service()
    calib = make_sample_calibration(scale_v=0.1, scale_h=0.1)
    meas = make_sample_net_qty_measurement(numeral_bbox=[400, 400, 450, 420], height_mm=2.0, height_px=20)
    extraction = make_extraction_payload([
        {"text": "INTRUDING GRAPHIC", "bbox": [0.35, 0.39, 0.375, 0.42]}  # bx2 = 375 px
    ])
    # Set semantic bbox sx1 to 380
    extraction["semantic_extraction"]["net_quantity"]["bbox"] = [0.38, 0.39, 0.55, 0.43]

    finding = service.evaluate_rule8_finding(
        field="net_quantity",
        numeral_measurement=meas,
        extraction=service._parse_extraction_payload(extraction),
        calibration=calib,
        image_bgr=None,
        img_w=1000,
        img_h=1000,
    )
    assert finding.direction_results.left == "below_threshold"
    assert finding.actual_clearance_mm.left_mm < 4.0


def test_7_sufficient_right_spacing():
    """TEST 7: Quantity numeral with sufficient right spacing (actual >= 2H)."""
    # H = 2.0 mm -> required right = 2H = 4.0 mm = 40 px.
    # Numeral right = 450 px. Semantic declaration bbox sx2 = 500 px.
    # External block at x = [0.60, 0.70] -> bx1 = 600 px.
    # Distance = 600 - 450 = 150 px = 15.0 mm >= 4.0 mm.
    service = Rule8Service()
    calib = make_sample_calibration(scale_v=0.1, scale_h=0.1)
    meas = make_sample_net_qty_measurement(numeral_bbox=[400, 400, 450, 420], height_mm=2.0, height_px=20)
    extraction = make_extraction_payload([
        {"text": "BARCODE BOX", "bbox": [0.60, 0.39, 0.70, 0.43]}  # bx1 = 600 px
    ])

    finding = service.evaluate_rule8_finding(
        field="net_quantity",
        numeral_measurement=meas,
        extraction=service._parse_extraction_payload(extraction),
        calibration=calib,
        image_bgr=None,
        img_w=1000,
        img_h=1000,
    )
    assert finding.direction_results.right == "above_threshold"
    assert finding.actual_clearance_mm.right_mm >= finding.required_clearance_mm.right_mm
    assert finding.required_clearance_mm.right_mm == 4.0


def test_8_insufficient_right_spacing():
    """TEST 8: Quantity numeral with insufficient right spacing (actual < 2H)."""
    # H = 2.0 mm -> required right = 4.0 mm = 40 px.
    # Numeral right = 450 px. Semantic declaration bbox sx2 = 460 px.
    # External block intruding at x = [0.47, 0.55] -> bx1 = 470 px.
    # Distance = 470 - 450 = 20 px = 2.0 mm < 4.0 mm.
    service = Rule8Service()
    calib = make_sample_calibration(scale_v=0.1, scale_h=0.1)
    meas = make_sample_net_qty_measurement(numeral_bbox=[400, 400, 450, 420], height_mm=2.0, height_px=20)
    extraction = make_extraction_payload([
        {"text": "INTRUDING RIGHT TEXT", "bbox": [0.47, 0.39, 0.55, 0.43]}  # bx1 = 470 px
    ])
    extraction["semantic_extraction"]["net_quantity"]["bbox"] = [0.35, 0.39, 0.46, 0.43]

    finding = service.evaluate_rule8_finding(
        field="net_quantity",
        numeral_measurement=meas,
        extraction=service._parse_extraction_payload(extraction),
        calibration=calib,
        image_bgr=None,
        img_w=1000,
        img_h=1000,
    )
    assert finding.direction_results.right == "below_threshold"
    assert finding.actual_clearance_mm.right_mm < 4.0


# ==============================================================================
# TESTS 9 - 10: OVERALL VERDICTS
# ==============================================================================

def test_9_all_four_directions_sufficient_compliant():
    """TEST 9: All 4 directions sufficient -> overall compliant."""
    service = Rule8Service()
    calib = make_sample_calibration(scale_v=0.1, scale_h=0.1)
    meas = make_sample_net_qty_measurement(numeral_bbox=[400, 400, 450, 420], height_mm=2.0, height_px=20)
    extraction = make_extraction_payload([
        {"text": "HEADER", "bbox": [0.38, 0.20, 0.48, 0.25]},  # top distance = 150 px = 15mm >= 2mm
        {"text": "FOOTER", "bbox": [0.38, 0.60, 0.48, 0.65]},  # bottom distance = 180 px = 18mm >= 2mm
        {"text": "LEFT ICON", "bbox": [0.10, 0.39, 0.20, 0.43]},  # left distance = 200 px = 20mm >= 4mm
        {"text": "RIGHT ICON", "bbox": [0.70, 0.39, 0.80, 0.43]},  # right distance = 250 px = 25mm >= 4mm
    ])

    finding = service.evaluate_rule8_finding(
        field="net_quantity",
        numeral_measurement=meas,
        extraction=service._parse_extraction_payload(extraction),
        calibration=calib,
        image_bgr=None,
        img_w=1000,
        img_h=1000,
    )
    assert finding.direction_results.top == "above_threshold"
    assert finding.direction_results.bottom == "above_threshold"
    assert finding.direction_results.left == "above_threshold"
    assert finding.direction_results.right == "above_threshold"
    assert finding.inspection_status == "compliant"


def test_10_one_direction_insufficient_non_compliant():
    """TEST 10: One direction insufficient -> overall non_compliant."""
    service = Rule8Service()
    calib = make_sample_calibration(scale_v=0.1, scale_h=0.1)
    meas = make_sample_net_qty_measurement(numeral_bbox=[400, 400, 450, 420], height_mm=2.0, height_px=20)
    extraction = make_extraction_payload([
        {"text": "HEADER", "bbox": [0.38, 0.20, 0.48, 0.25]},  # top OK
        {"text": "FOOTER", "bbox": [0.38, 0.60, 0.48, 0.65]},  # bottom OK
        {"text": "LEFT ICON", "bbox": [0.10, 0.39, 0.20, 0.43]},  # left OK
        {"text": "CLOSE RIGHT", "bbox": [0.47, 0.39, 0.55, 0.43]},  # right: 470-450 = 20px = 2mm < 4mm (FAIL)
    ])
    extraction["semantic_extraction"]["net_quantity"]["bbox"] = [0.35, 0.39, 0.46, 0.43]

    finding = service.evaluate_rule8_finding(
        field="net_quantity",
        numeral_measurement=meas,
        extraction=service._parse_extraction_payload(extraction),
        calibration=calib,
        image_bgr=None,
        img_w=1000,
        img_h=1000,
    )
    assert finding.direction_results.right == "below_threshold"
    assert finding.inspection_status == "non_compliant"


# ==============================================================================
# TESTS 11 - 12: UNCERTAINTY & MISSING MEASUREMENT
# ==============================================================================

def test_11_low_measurement_quality_uncertainty():
    """TEST 11: Low measurement quality -> indeterminate_low_confidence."""
    service = Rule8Service()
    calib = make_sample_calibration()
    meas = make_sample_net_qty_measurement(quality="low", confidence=0.45)
    extraction = make_extraction_payload([
        {"text": "HEADER", "bbox": [0.38, 0.20, 0.48, 0.25]}
    ])

    finding = service.evaluate_rule8_finding(
        field="net_quantity",
        numeral_measurement=meas,
        extraction=service._parse_extraction_payload(extraction),
        calibration=calib,
        image_bgr=None,
        img_w=1000,
        img_h=1000,
    )
    assert finding.inspection_status == "indeterminate_low_confidence"
    assert "quality is 'low'" in finding.notes


def test_12_missing_numeral_bbox():
    """TEST 12: Missing numeral measurement or bounding box -> indeterminate_missing_measurement."""
    service = Rule8Service()
    finding = service.evaluate_rule8_finding(
        field="net_quantity",
        numeral_measurement=None,
        extraction=None,
        calibration=None,
        image_bgr=None,
        img_w=1000,
        img_h=1000,
    )
    assert finding.inspection_status == "indeterminate_missing_measurement"
    assert finding.direction_results.top == "indeterminate"


# ==============================================================================
# TESTS 13 - 15: NON-RULE 8 FIELD EXCLUSIONS
# ==============================================================================

def test_13_mrp_measurement_supplied():
    """TEST 13: MRP measurement supplied -> not_applicable."""
    service = Rule8Service()
    mrp_meas = NumeralMeasurement(
        field="mrp",
        raw_text="MRP Rs. 20.00",
        numeral="20.00",
        semantic_region_normalized=[0.1, 0.1, 0.3, 0.2],
        numeral_bbox_px=[100, 100, 150, 130],
        numeral_height_px=30,
        numeral_height_mm=3.0,
        confidence=0.98,
        measurement_quality="good",
    )
    finding = service.evaluate_rule8_finding(
        field="mrp",
        numeral_measurement=mrp_meas,
        extraction=None,
        calibration=None,
        image_bgr=None,
        img_w=1000,
        img_h=1000,
    )
    assert finding.inspection_status == "not_applicable"
    assert finding.direction_results.top == "not_applicable"


def test_14_date_measurement_supplied():
    """TEST 14: Date measurement supplied -> not_applicable."""
    service = Rule8Service()
    date_meas = NumeralMeasurement(
        field="dates[0]",
        raw_text="Pkd: 08/2026",
        numeral="08/2026",
        semantic_region_normalized=[0.6, 0.1, 0.8, 0.2],
        numeral_bbox_px=[600, 100, 700, 130],
        numeral_height_px=30,
        numeral_height_mm=3.0,
        confidence=0.98,
        measurement_quality="good",
    )
    finding = service.evaluate_rule8_finding(
        field="dates[0]",
        numeral_measurement=date_meas,
        extraction=None,
        calibration=None,
        image_bgr=None,
        img_w=1000,
        img_h=1000,
    )
    assert finding.inspection_status == "not_applicable"
    assert finding.direction_results.bottom == "not_applicable"


def test_15_phone_pin_barcode_exclusion():
    """TEST 15: Phone, PIN code, and barcode numbers -> not_applicable."""
    service = Rule8Service()
    for field_name in ["additional_numeric_information[0]_phone", "pin_code", "barcode"]:
        finding = service.evaluate_rule8_finding(
            field=field_name,
            numeral_measurement=None,
            extraction=None,
            calibration=None,
            image_bgr=None,
            img_w=1000,
            img_h=1000,
        )
        assert finding.inspection_status == "not_applicable"


# ==============================================================================
# TEST 16: HORIZONTAL VS VERTICAL CALIBRATION SCALE DIFFERENTIATION
# ==============================================================================

def test_16_horizontal_vs_vertical_scale_differentiation():
    """TEST 16: Top/Bottom use vertical mm/px, Left/Right use horizontal mm/px."""
    # Let vertical scale = 0.05 mm/px, horizontal scale = 0.20 mm/px (differ by 4x!)
    # Top distance = 100 px -> top_mm = 100 * 0.05 = 5.0 mm
    # Left distance = 100 px -> left_mm = 100 * 0.20 = 20.0 mm
    service = Rule8Service()
    calib = make_sample_calibration(scale_v=0.05, scale_h=0.20)
    meas = make_sample_net_qty_measurement(numeral_bbox=[400, 400, 450, 420], height_mm=2.0, height_px=40)
    extraction = make_extraction_payload([
        {"text": "TOP BLOCK", "bbox": [0.38, 0.29, 0.48, 0.29]},  # top dist from dy1 (390) = 390 - 290 = 100 px
        {"text": "LEFT BLOCK", "bbox": [0.25, 0.39, 0.25, 0.43]},  # left dist from dx1 (350) = 350 - 250 = 100 px
    ])
    extraction["semantic_extraction"]["net_quantity"]["bbox"] = [0.35, 0.39, 0.55, 0.43]

    finding = service.evaluate_rule8_finding(
        field="net_quantity",
        numeral_measurement=meas,
        extraction=service._parse_extraction_payload(extraction),
        calibration=calib,
        image_bgr=None,
        img_w=1000,
        img_h=1000,
    )
    # top_mm = 100 * 0.05 = 5.0 mm
    assert finding.actual_clearance_mm.top_mm == pytest.approx(5.0, 0.1)
    # left_mm = 100 * 0.20 = 20.0 mm
    assert finding.actual_clearance_mm.left_mm == pytest.approx(20.0, 0.1)


# ==============================================================================
# TEST 17: LAY'S REAL-WORLD INTEGRATION
# ==============================================================================

def test_17_lays_real_world_integration():
    """TEST 17: Lay's product package test.

    Net quantity = 28.5 g
    numeral_height_mm = 1.14 mm
    numeral_bbox_px = [440, 1451, 460, 1462]
    Declaration bbox: [420, 1440, 540, 1464] (encloses 'Net Qty: 28.5 g')
    """
    service = Rule8Service()
    calib = CalibrationMetadata(
        method="package_dimensions_auto_bbox",
        package_width_mm=150.0,
        package_height_mm=200.0,
        package_bbox_px=[50, 100, 1150, 1500],
        package_width_px=1100.0,
        package_height_px=1400.0,
        width_mm_per_px=150.0 / 1100.0,  # ~0.1364 mm/px
        height_mm_per_px=200.0 / 1400.0,  # ~0.1429 mm/px
        confidence=0.95,
        quality="moderate",
    )
    meas = NumeralMeasurement(
        field="net_quantity",
        raw_text="Net Qty: 28.5 g",
        numeral="28.5",
        semantic_region_normalized=[0.35, 0.90, 0.45, 0.915],
        numeral_bbox_px=[440, 1451, 460, 1462],  # height = 11 px -> ~1.14 mm
        numeral_height_px=11,
        numeral_height_mm=1.14,
        confidence=0.92,
        measurement_quality="good",
        notes="Lay's net quantity measurement",
    )
    # Synthetic image for spatial edge analysis (1600x1200)
    fake_img = np.ones((1600, 1200, 3), dtype=np.uint8) * 240
    # Draw visible text 'CHIPS' 70 px above declaration (1440 - 1370 = 70 px -> ~10.0 mm >= 1.14 mm)
    cv2.putText(fake_img, "CHIPS", (400, 1370), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    result = service.evaluate(
        extraction=None,
        measurements={"calibration": calib.model_dump(), "measurements": [meas.model_dump()]},
        image_input=fake_img,
        include_debug_image=True,
    )
    assert result.summary.total_evaluated == 1
    finding = result.findings[0]
    assert finding.field == "net_quantity"
    assert finding.rule_reference == "Legal Metrology (Packaged Commodities) Rules, 2011 — Rule 8(1) proviso"
    assert finding.clearance_reference == "quantity_declaration_bbox"
    assert finding.declaration_bbox_px == [420, 1440, 540, 1464]

    # Required clearances: H = 1.14 mm top/bot, 2H = 2.28 mm left/right
    assert finding.required_clearance_mm.top_mm == pytest.approx(1.14, 0.05)
    assert finding.required_clearance_mm.bottom_mm == pytest.approx(1.14, 0.05)
    assert finding.required_clearance_mm.left_mm == pytest.approx(2.28, 0.05)
    assert finding.required_clearance_mm.right_mm == pytest.approx(2.28, 0.05)

    # Actual clearances must NOT be artificially inflated to 93 mm / 53 mm / 94 mm:
    # Top distance corresponds to 'CHIPS' text ~70 px -> ~10.0 mm (not 93 mm / 193 mm)
    assert finding.actual_clearance_mm.top_mm == pytest.approx(10.0, 1.0)
    # Bottom clearance to package border (1500 - 1464 = 36 px -> ~5.14 mm)
    assert finding.actual_clearance_mm.bottom_mm == pytest.approx(5.14, 0.5)
    # Left and right are bounded within local scanned window (~16.36 mm, not 53 mm or 94 mm)
    assert finding.actual_clearance_mm.left_mm < 25.0
    assert finding.actual_clearance_mm.right_mm < 25.0

    # Overall verdict
    assert finding.direction_results.top == "above_threshold"
    assert finding.direction_results.bottom == "above_threshold"
    assert finding.direction_results.left == "above_threshold"
    assert finding.direction_results.right == "above_threshold"
    assert finding.inspection_status == "compliant"

    # Debug image verification
    assert result.debug_image_base64 is not None
    assert result.debug_image_base64.startswith("data:image/jpeg;base64,")


# ==============================================================================
# TEST 18: NO NEARBY OCCUPIED REGION DETECTED UNCERTAINTY
# ==============================================================================

def test_18_no_nearby_occupied_region_detected_indeterminate():
    """TEST 18: No nearby occupied region confidently detected -> indeterminate."""
    service = Rule8Service()
    calib = make_sample_calibration()
    meas = make_sample_net_qty_measurement()
    # Empty extraction without raw text blocks and no image supplied
    empty_extraction = {
        "image": {"width": 1000, "height": 1000, "coordinate_system": "normalized_0_1"},
        "raw_text_blocks": [],
        "semantic_extraction": {
            "image_quality": {"overall": "good", "notes": "Clear"},
            "package": {"package_type": "retail", "food_status": "food", "retail_unit_count": None, "evidence": []},
            "manufacturer": {"status": "missing", "entities": []},
            "generic_name": {"status": "missing"},
            "mrp": {"status": "missing"},
            "dates": [],
            "net_quantity": {"status": "present", "raw_text": "500 g", "value": 500.0, "unit": "g", "bbox": None},
            "consumer_care": {"status": "missing"},
            "stickers": [],
            "additional_numeric_information": [],
            "other_visible_text": [],
        },
    }
    finding = service.evaluate_rule8_finding(
        field="net_quantity",
        numeral_measurement=meas,
        extraction=service._parse_extraction_payload(empty_extraction),
        calibration=calib,
        image_bgr=None,
        img_w=1000,
        img_h=1000,
    )
    assert finding.inspection_status == "indeterminate_low_confidence"
    assert "No surrounding layout elements" in finding.notes


# ==============================================================================
# TEST 19 - 20: FASTAPI ENDPOINT & MULTI-MEASUREMENT AGGREGATION
# ==============================================================================

def test_19_api_rule8_endpoint_success():
    """TEST 19: Test POST /api/v1/compliance/rule8 endpoint."""
    client = TestClient(app)
    calib = make_sample_calibration()
    meas = make_sample_net_qty_measurement()
    extraction = make_extraction_payload([
        {"text": "HEADER", "bbox": [0.38, 0.20, 0.48, 0.25]},
        {"text": "FOOTER", "bbox": [0.38, 0.60, 0.48, 0.65]},
    ])

    response = client.post(
        "/api/v1/compliance/rule8",
        json={
            "extraction": extraction,
            "measurements": {
                "calibration": calib.model_dump(),
                "measurements": [meas.model_dump()],
            },
            "include_debug_image": False,
        },
    )
    assert response.status_code == 200
    res_json = response.json()
    assert res_json["success"] is True
    assert "data" in res_json
    assert res_json["data"]["summary"]["total_evaluated"] >= 1


def test_20_api_rule8_with_mrp_and_dates_excluded():
    """TEST 20: Test that MRP and Date measurements are marked not_applicable in Rule 8."""
    client = TestClient(app)
    calib = make_sample_calibration()
    net_qty = make_sample_net_qty_measurement()
    mrp_meas = NumeralMeasurement(
        field="mrp",
        raw_text="MRP 10",
        numeral="10",
        semantic_region_normalized=[0.1, 0.1, 0.2, 0.2],
        numeral_bbox_px=[100, 100, 150, 130],
        numeral_height_px=30,
        numeral_height_mm=3.0,
        confidence=0.95,
        measurement_quality="good",
    )
    date_meas = NumeralMeasurement(
        field="dates[0]",
        raw_text="07/2026",
        numeral="07/2026",
        semantic_region_normalized=[0.6, 0.1, 0.7, 0.2],
        numeral_bbox_px=[600, 100, 700, 130],
        numeral_height_px=30,
        numeral_height_mm=3.0,
        confidence=0.95,
        measurement_quality="good",
    )

    response = client.post(
        "/api/v1/compliance/rule8",
        json={
            "measurements": {
                "calibration": calib.model_dump(),
                "measurements": [net_qty.model_dump(), mrp_meas.model_dump(), date_meas.model_dump()],
            },
        },
    )
    assert response.status_code == 200
    findings = response.json()["data"]["findings"]
    f_map = {f["field"]: f for f in findings}

    assert f_map["net_quantity"]["inspection_status"] in ("compliant", "indeterminate_low_confidence")
    assert f_map["mrp"]["inspection_status"] == "not_applicable"
    assert f_map["dates[0]"]["inspection_status"] == "not_applicable"
    assert response.json()["data"]["summary"]["not_applicable_count"] == 2


# ==============================================================================
# TASK 1: RULE 8 STATUTORY CITATION VERIFICATION
# ==============================================================================

def test_task1_statutory_reference_is_rule_8_1_proviso():
    """TASK 1: Verify Rule 8 citation points strictly to Rule 8(1) proviso (not Rule 8(2))."""
    service = Rule8Service()
    calib = make_sample_calibration()
    meas = make_sample_net_qty_measurement()
    finding = service.evaluate_rule8_finding(
        field="net_quantity",
        numeral_measurement=meas,
        extraction=None,
        calibration=calib,
        image_bgr=None,
        img_w=1000,
        img_h=1000,
    )
    assert finding.rule_reference == "Legal Metrology (Packaged Commodities) Rules, 2011 — Rule 8(1) proviso"
    assert "Rule 8(2)" not in finding.rule_reference
    assert finding.clearance_reference == "quantity_declaration_bbox"


# ==============================================================================
# TASK 3: FIXED SYNTHETIC GEOMETRY TEST FIXTURE (7 DETERMINISTIC CASES)
# ==============================================================================

def make_synthetic_geometry_fixture():
    """Helper creating deterministic synthetic geometry:

    Image: 1000 x 1000 px.
    Calibration: 0.10 mm/px (both H and V).
    Declaration 'NET QTY: 28.5 g': bbox = [400, 500, 600, 530] (dx1=400, dy1=500, dx2=600, dy2=530).
    Numeral '28.5': bbox = [480, 505, 540, 525] (H_px = 20 px -> H = 2.0 mm).
    Prefix 'NET QTY:': [400, 500, 480, 530].
    Unit 'g': [540, 500, 600, 530].

    Statutory Requirements:
    - Top: H = 2.0 mm (20 px)
    - Bottom: H = 2.0 mm (20 px)
    - Left: 2H = 4.0 mm (40 px)
    - Right: 2H = 4.0 mm (40 px)
    """
    calib = CalibrationMetadata(
        method="package_dimensions",
        package_width_mm=100.0,
        package_height_mm=100.0,
        package_bbox_px=[0, 0, 1000, 1000],
        package_width_px=1000.0,
        package_height_px=1000.0,
        width_mm_per_px=0.10,
        height_mm_per_px=0.10,
        confidence=1.0,
        quality="good",
    )
    meas = NumeralMeasurement(
        field="net_quantity",
        raw_text="NET QTY: 28.5 g",
        numeral="28.5",
        semantic_region_normalized=[0.40, 0.50, 0.60, 0.53],
        numeral_bbox_px=[480, 505, 540, 525],
        numeral_height_px=20,
        numeral_height_mm=2.0,
        confidence=0.98,
        measurement_quality="good",
        notes="Synthetic test numeral",
    )
    return calib, meas


def test_task3_case1_text_immediately_above():
    """TASK 3 - Case 1: Text immediately above declaration -> non_compliant top."""
    service = Rule8Service()
    calib, meas = make_synthetic_geometry_fixture()
    # Obstacle at y = [495, 498], by2 = 498 px.
    # Expected clearance from dy1 (500): 500 - 498 = 2 px = 0.2 mm.
    # Required: 2.0 mm -> below_threshold.
    extraction = make_extraction_payload([
        {"text": "OVERHEAD TEXT", "bbox": [0.45, 0.495, 0.55, 0.498]}
    ])
    extraction["semantic_extraction"]["net_quantity"]["bbox"] = [0.40, 0.50, 0.60, 0.53]

    finding = service.evaluate_rule8_finding(
        field="net_quantity",
        numeral_measurement=meas,
        extraction=service._parse_extraction_payload(extraction),
        calibration=calib,
        image_bgr=None,
        img_w=1000,
        img_h=1000,
    )
    assert finding.declaration_bbox_px == [400, 500, 600, 530]
    assert finding.actual_clearance_px.top_px == 2.0
    assert finding.actual_clearance_mm.top_mm == pytest.approx(0.2, 0.01)
    assert finding.direction_results.top == "below_threshold"
    assert finding.inspection_status == "non_compliant"


def test_task3_case2_text_immediately_below():
    """TASK 3 - Case 2: Text immediately below declaration -> non_compliant bottom."""
    service = Rule8Service()
    calib, meas = make_synthetic_geometry_fixture()
    # Obstacle at y = [535, 545], by1 = 535 px.
    # Expected clearance from dy2 (530): 535 - 530 = 5 px = 0.5 mm.
    # Required: 2.0 mm -> below_threshold.
    extraction = make_extraction_payload([
        {"text": "UNDERFOOT TEXT", "bbox": [0.45, 0.535, 0.55, 0.545]}
    ])
    extraction["semantic_extraction"]["net_quantity"]["bbox"] = [0.40, 0.50, 0.60, 0.53]

    finding = service.evaluate_rule8_finding(
        field="net_quantity",
        numeral_measurement=meas,
        extraction=service._parse_extraction_payload(extraction),
        calibration=calib,
        image_bgr=None,
        img_w=1000,
        img_h=1000,
    )
    assert finding.declaration_bbox_px == [400, 500, 600, 530]
    assert finding.actual_clearance_px.bottom_px == 5.0
    assert finding.actual_clearance_mm.bottom_mm == pytest.approx(0.5, 0.01)
    assert finding.direction_results.bottom == "below_threshold"
    assert finding.inspection_status == "non_compliant"


def test_task3_case3_text_immediately_left():
    """TASK 3 - Case 3: Text immediately left of declaration -> non_compliant left.

    CRITICAL VALIDATION:
    If measured from numeral nx1 (480), distance would be 480 - 395 = 85 px = 8.5 mm (falsely compliant).
    Measured from declaration dx1 (400), distance is 400 - 395 = 5 px = 0.5 mm < 4.0 mm (correctly non_compliant).
    """
    service = Rule8Service()
    calib, meas = make_synthetic_geometry_fixture()
    # Obstacle at x = [390, 395], bx2 = 395 px.
    extraction = make_extraction_payload([
        {"text": "LEFT BADGE", "bbox": [0.390, 0.505, 0.395, 0.525]}
    ])
    extraction["semantic_extraction"]["net_quantity"]["bbox"] = [0.40, 0.50, 0.60, 0.53]

    finding = service.evaluate_rule8_finding(
        field="net_quantity",
        numeral_measurement=meas,
        extraction=service._parse_extraction_payload(extraction),
        calibration=calib,
        image_bgr=None,
        img_w=1000,
        img_h=1000,
    )
    assert finding.declaration_bbox_px == [400, 500, 600, 530]
    assert finding.actual_clearance_px.left_px == 5.0
    assert finding.actual_clearance_mm.left_mm == pytest.approx(0.5, 0.01)
    assert finding.direction_results.left == "below_threshold"
    assert finding.inspection_status == "non_compliant"


def test_task3_case4_text_immediately_right():
    """TASK 3 - Case 4: Text immediately right of declaration -> non_compliant right.

    CRITICAL VALIDATION:
    If measured from numeral nx2 (540), distance would be 610 - 540 = 70 px = 7.0 mm (falsely compliant).
    Measured from declaration dx2 (600), distance is 610 - 600 = 10 px = 1.0 mm < 4.0 mm (correctly non_compliant).
    """
    service = Rule8Service()
    calib, meas = make_synthetic_geometry_fixture()
    # Obstacle at x = [610, 650], bx1 = 610 px.
    extraction = make_extraction_payload([
        {"text": "RIGHT ICON", "bbox": [0.610, 0.505, 0.650, 0.525]}
    ])
    extraction["semantic_extraction"]["net_quantity"]["bbox"] = [0.40, 0.50, 0.60, 0.53]

    finding = service.evaluate_rule8_finding(
        field="net_quantity",
        numeral_measurement=meas,
        extraction=service._parse_extraction_payload(extraction),
        calibration=calib,
        image_bgr=None,
        img_w=1000,
        img_h=1000,
    )
    assert finding.declaration_bbox_px == [400, 500, 600, 530]
    assert finding.actual_clearance_px.right_px == 10.0
    assert finding.actual_clearance_mm.right_mm == pytest.approx(1.0, 0.01)
    assert finding.direction_results.right == "below_threshold"
    assert finding.inspection_status == "non_compliant"


def test_task3_case5_all_four_sides_clear():
    """TASK 3 - Case 5: Obstacles outside required clearance area on all 4 sides -> compliant."""
    service = Rule8Service()
    calib, meas = make_synthetic_geometry_fixture()
    # Top obstacle at by2 = 450 (dist = 500 - 450 = 50 px = 5.0 mm >= 2.0 mm)
    # Bottom obstacle at by1 = 570 (dist = 570 - 530 = 40 px = 4.0 mm >= 2.0 mm)
    # Left obstacle at bx2 = 340 (dist = 400 - 340 = 60 px = 6.0 mm >= 4.0 mm)
    # Right obstacle at bx1 = 660 (dist = 660 - 600 = 60 px = 6.0 mm >= 4.0 mm)
    extraction = make_extraction_payload([
        {"text": "FAR TOP", "bbox": [0.45, 0.40, 0.55, 0.45]},
        {"text": "FAR BOTTOM", "bbox": [0.45, 0.57, 0.55, 0.62]},
        {"text": "FAR LEFT", "bbox": [0.28, 0.505, 0.34, 0.525]},
        {"text": "FAR RIGHT", "bbox": [0.66, 0.505, 0.72, 0.525]},
    ])
    extraction["semantic_extraction"]["net_quantity"]["bbox"] = [0.40, 0.50, 0.60, 0.53]

    finding = service.evaluate_rule8_finding(
        field="net_quantity",
        numeral_measurement=meas,
        extraction=service._parse_extraction_payload(extraction),
        calibration=calib,
        image_bgr=None,
        img_w=1000,
        img_h=1000,
    )
    assert finding.actual_clearance_mm.top_mm == pytest.approx(5.0, 0.01)
    assert finding.actual_clearance_mm.bottom_mm == pytest.approx(4.0, 0.01)
    assert finding.actual_clearance_mm.left_mm == pytest.approx(6.0, 0.01)
    assert finding.actual_clearance_mm.right_mm == pytest.approx(6.0, 0.01)
    assert finding.direction_results.top == "above_threshold"
    assert finding.direction_results.bottom == "above_threshold"
    assert finding.direction_results.left == "above_threshold"
    assert finding.direction_results.right == "above_threshold"
    assert finding.inspection_status == "compliant"


def test_task3_case6_prefix_and_unit_remain_inside_declaration():
    """TASK 3 - Case 6: Internal prefix 'NET QTY:' and unit 'g' are inside declaration box.

    Neither prefix nor unit is counted towards external clear space or treated as an obstacle.
    """
    service = Rule8Service()
    calib, meas = make_synthetic_geometry_fixture()
    # Declaration box is [400, 500, 600, 530]
    # Add OCR blocks representing 'NET QTY:' [400, 500, 480, 530] and 'g' [540, 500, 600, 530]
    extraction = make_extraction_payload([
        {"text": "NET QTY:", "bbox": [0.40, 0.50, 0.48, 0.53]},
        {"text": "g", "bbox": [0.54, 0.50, 0.60, 0.53]},
        # External obstacle 50 px above declaration
        {"text": "TOP TEXT", "bbox": [0.45, 0.40, 0.55, 0.45]},
    ])
    extraction["semantic_extraction"]["net_quantity"]["bbox"] = [0.40, 0.50, 0.60, 0.53]

    layout_res = service.detect_occupied_regions_from_layout(
        declaration_bbox_px=[400, 500, 600, 530],
        extraction=service._parse_extraction_payload(extraction),
        image_width=1000,
        image_height=1000,
    )
    # Own text 'NET QTY:' and 'g' must be skipped (not treated as left/right obstacles)
    assert layout_res["left"] is None
    assert layout_res["right"] is None
    assert layout_res["top"] == 50.0  # 500 - 450 = 50 px


def test_task3_case7_unrelated_nearby_text_outside_clearance_envelope():
    """TASK 3 - Case 7: Unrelated nearby text outside the corridor does not encroach on clear space."""
    service = Rule8Service()
    calib, meas = make_synthetic_geometry_fixture()
    # Text diagonally offset at x=[0.70, 0.80], y=[0.40, 0.45]
    extraction = make_extraction_payload([
        {"text": "DIAGONAL PROMO", "bbox": [0.70, 0.40, 0.80, 0.45]}
    ])
    extraction["semantic_extraction"]["net_quantity"]["bbox"] = [0.40, 0.50, 0.60, 0.53]

    layout_res = service.detect_occupied_regions_from_layout(
        declaration_bbox_px=[400, 500, 600, 530],
        extraction=service._parse_extraction_payload(extraction),
        image_width=1000,
        image_height=1000,
    )
    # Off-corridor text does not register as a cardinal obstacle
    assert layout_res["top"] is None
    assert layout_res["bottom"] is None
    assert layout_res["left"] is None
    assert layout_res["right"] is None


# ==============================================================================
# TASK 4: LAY'S DEBUG IMAGE VALIDATION
# ==============================================================================

def test_task4_lays_debug_image_visual_verification():
    """TASK 4: Verify debug image generation, ray rendering, and visual bounding."""
    service = Rule8Service()
    calib, meas = make_synthetic_geometry_fixture()
    synthetic_img = np.ones((1000, 1000, 3), dtype=np.uint8) * 255

    # Draw an obstacle above at y = 460 (40 px above declaration top 500)
    cv2.rectangle(synthetic_img, (450, 440), (550, 460), (0, 0, 0), -1)

    result = service.evaluate(
        extraction=None,
        measurements={"calibration": calib.model_dump(), "measurements": [meas.model_dump()]},
        image_input=synthetic_img,
        include_debug_image=True,
    )
    assert result.debug_image_base64 is not None
    assert result.debug_image_base64.startswith("data:image/jpeg;base64,")

    # Decode and verify dimensions of returned debug image
    b64_str = result.debug_image_base64.split(",", 1)[1]
    import base64
    img_bytes = base64.b64decode(b64_str)
    decoded = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape == (1000, 1000, 3)

    finding = result.findings[0]
    # Measured top clearance must hit the rectangle at 460 px (dist = 500 - 460 = 40 px = 4.0 mm)
    assert finding.actual_clearance_px.top_px == pytest.approx(40.0, 1.0)
    assert finding.actual_clearance_mm.top_mm == pytest.approx(4.0, 0.1)


# ==============================================================================
# TASK 5: CONFIDENCE & FOIL / TEXTURE / SHADOW UNCERTAINTY
# ==============================================================================

def test_task5_uncertainty_reflective_foil_glare_indeterminate():
    """TASK 5: Specular glare or high foil reflection returns indeterminate_low_confidence."""
    service = Rule8Service()
    calib, meas = make_synthetic_geometry_fixture()

    # Image with specular glare patches (brightness 255 across > 25% of background)
    glare_img = np.ones((1000, 1000, 3), dtype=np.uint8) * 255  # fully saturated white glare
    # Dark declaration text in center
    cv2.putText(glare_img, "NET QTY: 28.5 g", (420, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    result = service.evaluate(
        extraction=None,
        measurements={"calibration": calib.model_dump(), "measurements": [meas.model_dump()]},
        image_input=glare_img,
        include_debug_image=False,
    )
    finding = result.findings[0]
    assert finding.inspection_status == "indeterminate_low_confidence"
    assert "reflective foil glare" in finding.notes.lower()

