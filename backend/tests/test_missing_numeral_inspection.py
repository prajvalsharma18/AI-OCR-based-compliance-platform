"""Regression coverage for inspections with no measurable statutory numerals."""

import io
import json

import pytest
from fastapi import UploadFile
from starlette.datastructures import Headers
from PIL import Image

from app.services.measurement_service import MeasurementService
from tests.test_schema import sample_valid_extraction_dict


@pytest.mark.asyncio
async def test_inspection_measurement_mode_returns_empty_result_without_fabricating_regions():
    extraction = sample_valid_extraction_dict()
    extraction["net_quantity"]["numeral_region"] = None
    extraction["net_quantity"]["value"] = None
    extraction["net_quantity"]["raw_text"] = "NET WT:"
    extraction["mrp"]["numeral_region"] = None
    extraction["mrp"]["value"] = None
    extraction["mrp"]["raw_text"] = "MRP:"
    for date in extraction.get("dates", []):
        date["numeral_region"] = None
        date["month"] = None
        date["year"] = None
        date["raw_text"] = "Packed On:"

    image_buffer = io.BytesIO()
    Image.new("RGB", (100, 120), color="white").save(image_buffer, format="PNG")
    image_buffer.seek(0)
    upload = UploadFile(
        file=image_buffer,
        filename="blank-declarations.png",
        headers=Headers({"content-type": "image/png"}),
    )

    result = await MeasurementService().measure_packaging_numerals(
        image_file=upload,
        extraction_json=json.dumps(extraction),
        package_width_mm=100.0,
        package_height_mm=120.0,
        allow_no_numeral_regions=True,
    )

    assert result.measurements == []
    assert result.calibration.package_width_mm == 100.0


@pytest.mark.asyncio
async def test_standalone_measurement_contract_still_rejects_missing_regions():
    extraction = sample_valid_extraction_dict()
    extraction["net_quantity"]["numeral_region"] = None
    extraction["net_quantity"]["value"] = None
    extraction["net_quantity"]["raw_text"] = "NET WT:"
    extraction["mrp"]["numeral_region"] = None
    extraction["mrp"]["value"] = None
    extraction["mrp"]["raw_text"] = "MRP:"
    for date in extraction.get("dates", []):
        date["numeral_region"] = None
        date["month"] = None
        date["year"] = None
        date["raw_text"] = "Packed On:"

    image_buffer = io.BytesIO()
    Image.new("RGB", (100, 120), color="white").save(image_buffer, format="PNG")
    image_buffer.seek(0)
    upload = UploadFile(
        file=image_buffer,
        filename="blank-declarations.png",
        headers=Headers({"content-type": "image/png"}),
    )

    with pytest.raises(Exception) as exc_info:
        await MeasurementService().measure_packaging_numerals(
            image_file=upload,
            extraction_json=json.dumps(extraction),
            package_width_mm=100.0,
            package_height_mm=120.0,
        )

    assert getattr(exc_info.value, "code", None) == "NO_NUMERAL_REGIONS_FOUND"


def test_measurement_recovers_exact_declared_numeral_span_only():
    extraction = sample_valid_extraction_dict()
    extraction["net_quantity"]["numeral_region"] = None
    extraction["net_quantity"]["raw_text"] = "NET WT: 500 g"
    extraction["net_quantity"]["bbox"] = [0.2, 0.3, 0.8, 0.4]

    service = MeasurementService()
    parsed = service.parse_extraction_json(json.dumps(extraction))
    candidates = service._extract_measurable_candidates(parsed, allow_region_recovery=True)

    net_candidate = next(item for item in candidates if item["field"] == "net_quantity")
    assert net_candidate["region"] != extraction["net_quantity"]["bbox"]
    assert net_candidate["region"][0] > extraction["net_quantity"]["bbox"][0]
