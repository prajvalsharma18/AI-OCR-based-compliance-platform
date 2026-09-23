"""Measurement service orchestrator for Module 2A.

Coordinates image loading, Module 1 JSON validation, user-supplied package bounding box
calibration, semantic region cropping, and numeral glyph height measurement.
Does NOT call automatic package boundary detection.
"""

import json
import logging
from typing import Any, List, Optional, Union
import cv2
from fastapi import UploadFile
import numpy as np
from pydantic import ValidationError

from app.cv.measurement import calculate_numeral_height_mm, compute_calibration_metadata
from app.cv.numeral_detector import detect_numeral_glyphs
from app.cv.package_detector import detect_package_boundary
from app.cv.preprocessing import crop_semantic_region
from app.cv.visualization import render_measurement_debug_overlay
from app.schemas.extraction import LabelExtractionResult
from app.schemas.measurement import (
    CalibrationMetadata,
    MeasurementImageMetadata,
    MeasurementResponseData,
    NumeralMeasurement,
)
from app.utils.image import ImageValidationError, validate_and_load_image

logger = logging.getLogger(__name__)


class MeasurementServiceError(Exception):
    """Base exception for measurement service failures."""

    def __init__(self, code: str, message: str, status_code: int = 422):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class MeasurementService:
    """Orchestrates Module 2A calibrated numeral size measurement using user-supplied package_bbox_px."""

    def validate_package_dimensions(
        self,
        package_width_mm: float,
        package_height_mm: float,
    ) -> None:
        """Validates real-world package dimension inputs."""
        if package_width_mm <= 0:
            raise MeasurementServiceError(
                code="INVALID_PACKAGE_WIDTH",
                message=f"package_width_mm must be strictly positive (> 0 mm), got {package_width_mm}.",
                status_code=400,
            )
        if package_height_mm <= 0:
            raise MeasurementServiceError(
                code="INVALID_PACKAGE_HEIGHT",
                message=f"package_height_mm must be strictly positive (> 0 mm), got {package_height_mm}.",
                status_code=400,
            )

    def validate_package_bbox_px(
        self,
        raw_bbox_input: Union[str, List[Any]],
        image_width: int,
        image_height: int,
    ) -> List[int]:
        """Validates and parses user-supplied package_bbox_px against image dimensions.

        Ensures:
        - Format: [x_min, y_min, x_max, y_max]
        - x_min < x_max and y_min < y_max
        - 0 <= x_min < image_width and 0 <= x_max <= image_width
        - 0 <= y_min < image_height and 0 <= y_max <= image_height

        Returns:
            List[int] containing [x_min, y_min, x_max, y_max].

        Raises:
            MeasurementServiceError: If parsing or coordinate bounds check fails (HTTP 422).
        """
        if raw_bbox_input is None:
            raise MeasurementServiceError(
                code="INVALID_PACKAGE_BBOX",
                message="package_bbox_px is required for Module 2A calibration.",
                status_code=422,
            )

        # Parse from string if passed as JSON string or comma-separated string
        if isinstance(raw_bbox_input, str):
            trimmed = raw_bbox_input.strip()
            if not trimmed:
                raise MeasurementServiceError(
                    code="INVALID_PACKAGE_BBOX",
                    message="package_bbox_px cannot be empty.",
                    status_code=422,
                )
            if not (trimmed.startswith("[") and trimmed.endswith("]")):
                if "," in trimmed:
                    coords = [p.strip() for p in trimmed.split(",") if p.strip()]
                else:
                    coords = [trimmed]
            else:
                try:
                    coords = json.loads(trimmed)
                except json.JSONDecodeError as exc:
                    raise MeasurementServiceError(
                        code="INVALID_PACKAGE_BBOX",
                        message=f"package_bbox_px must be a valid JSON array of 4 numbers, got: '{trimmed}'. Error: {exc.msg}",
                        status_code=422,
                    ) from exc
        else:
            coords = raw_bbox_input

        if not isinstance(coords, (list, tuple)) or len(coords) != 4:
            raise MeasurementServiceError(
                code="INVALID_PACKAGE_BBOX",
                message=f"package_bbox_px must contain exactly 4 numbers [x_min, y_min, x_max, y_max], got {coords}.",
                status_code=422,
            )

        try:
            x_min = int(round(float(coords[0])))
            y_min = int(round(float(coords[1])))
            x_max = int(round(float(coords[2])))
            y_max = int(round(float(coords[3])))
        except (ValueError, TypeError) as exc:
            raise MeasurementServiceError(
                code="INVALID_PACKAGE_BBOX",
                message=f"All coordinates in package_bbox_px must be numeric, got {coords}.",
                status_code=422,
            ) from exc

        # 1. Ordering validation
        if x_min >= x_max:
            raise MeasurementServiceError(
                code="INVALID_PACKAGE_BBOX",
                message=f"x_min ({x_min}) must be strictly less than x_max ({x_max}).",
                status_code=422,
            )
        if y_min >= y_max:
            raise MeasurementServiceError(
                code="INVALID_PACKAGE_BBOX",
                message=f"y_min ({y_min}) must be strictly less than y_max ({y_max}).",
                status_code=422,
            )

        # 2. Coordinate range validation against image dimensions
        if x_min < 0 or x_min >= image_width:
            raise MeasurementServiceError(
                code="INVALID_PACKAGE_BBOX",
                message=f"x_min ({x_min}) is out of bounds for image width {image_width} (must satisfy 0 <= x_min < {image_width}).",
                status_code=422,
            )
        if x_max <= 0 or x_max > image_width:
            raise MeasurementServiceError(
                code="INVALID_PACKAGE_BBOX",
                message=f"x_max ({x_max}) is out of bounds for image width {image_width} (must satisfy 0 < x_max <= {image_width}).",
                status_code=422,
            )
        if y_min < 0 or y_min >= image_height:
            raise MeasurementServiceError(
                code="INVALID_PACKAGE_BBOX",
                message=f"y_min ({y_min}) is out of bounds for image height {image_height} (must satisfy 0 <= y_min < {image_height}).",
                status_code=422,
            )
        if y_max <= 0 or y_max > image_height:
            raise MeasurementServiceError(
                code="INVALID_PACKAGE_BBOX",
                message=f"y_max ({y_max}) is out of bounds for image height {image_height} (must satisfy 0 < y_max <= {image_height}).",
                status_code=422,
            )

        return [x_min, y_min, x_max, y_max]

    def parse_extraction_json(self, raw_json_str: str) -> LabelExtractionResult:
        """Parses and strictly validates Module 1 extraction JSON."""
        if not raw_json_str or not raw_json_str.strip():
            raise MeasurementServiceError(
                code="EMPTY_EXTRACTION_JSON",
                message="extraction_json payload is required and cannot be empty.",
                status_code=400,
            )

        try:
            parsed_dict = json.loads(raw_json_str)
        except json.JSONDecodeError as exc:
            raise MeasurementServiceError(
                code="MALFORMED_EXTRACTION_JSON",
                message=f"extraction_json contains invalid JSON syntax: {exc.msg}",
                status_code=400,
            ) from exc

        if not isinstance(parsed_dict, dict):
            raise MeasurementServiceError(
                code="INVALID_EXTRACTION_JSON_TYPE",
                message=f"extraction_json must be a JSON object at root, got {type(parsed_dict).__name__}",
                status_code=422,
            )

        # Handle envelope if client passed {"success": true, "data": {...}}
        if "data" in parsed_dict and isinstance(parsed_dict["data"], dict):
            parsed_dict = parsed_dict["data"]

        try:
            return LabelExtractionResult.model_validate(parsed_dict)
        except ValidationError as exc:
            errors = [f"[{' -> '.join(str(p) for p in err.get('loc', []))}]: {err.get('msg', '')}" for err in exc.errors()]
            raise MeasurementServiceError(
                code="INVALID_EXTRACTION_SCHEMA",
                message=f"extraction_json failed Module 1 schema validation: {'; '.join(errors)}",
                status_code=422,
            ) from exc

    def _extract_measurable_candidates(
        self,
        extraction: LabelExtractionResult,
    ) -> List[dict]:
        """Collects relevant statutory declarations that contain a numeral_region."""
        candidates = []

        # 1. Net Quantity numeral region
        net_qty = extraction.net_quantity
        if net_qty and net_qty.numeral_region and len(net_qty.numeral_region) == 4:
            val_str = str(int(net_qty.value)) if (net_qty.value is not None and net_qty.value.is_integer()) else str(net_qty.value or "")
            candidates.append({
                "field": "net_quantity",
                "raw_text": net_qty.raw_text or f"{val_str} {net_qty.unit or ''}".strip(),
                "numeral": val_str,
                "region": net_qty.numeral_region,
            })

        # 2. MRP numeral region
        mrp = extraction.mrp
        if mrp and mrp.numeral_region and len(mrp.numeral_region) == 4:
            mrp_val_str = f"{mrp.value:.2f}" if (mrp.value is not None and mrp.value > 0) else str(mrp.value or "")
            candidates.append({
                "field": "mrp",
                "raw_text": mrp.raw_text or f"MRP {mrp.currency or '₹'}{mrp_val_str}".strip(),
                "numeral": mrp_val_str,
                "region": mrp.numeral_region,
            })

        # 3. Date numeral regions (manufacture / packing / import)
        for idx, dt in enumerate(extraction.dates):
            if dt and dt.numeral_region and len(dt.numeral_region) == 4:
                date_numeral = f"{dt.month or ''}/{dt.year or ''}".strip("/")
                candidates.append({
                    "field": f"dates[{idx}]",
                    "raw_text": dt.raw_text,
                    "numeral": date_numeral or dt.raw_text,
                    "region": dt.numeral_region,
                })

        return candidates

    async def measure_packaging_numerals(
        self,
        image_file: UploadFile,
        extraction_json: str,
        package_width_mm: float,
        package_height_mm: float,
        package_bbox_px: Optional[Union[str, List[Any]]] = None,
        include_debug_image: bool = False,
    ) -> MeasurementResponseData:
        """Executes the complete numeral measurement workflow.

        Workflow:
        1. Validate user-supplied package dimensions.
        2. Validate image file and decode to BGR numpy array.
        3. Parse and validate extraction JSON from Module 1.
        4. Determine package bounding box:
           - If package_bbox_px is provided: validate as manual override (method='package_dimensions_manual_bbox').
           - If omitted: automatically detect package boundary with OpenCV (method='package_dimensions_auto_bbox').
        5. Calculate calibration from package_bbox_px and physical dimensions.
        6. Convert Module 1 normalized numeral_region to pixel coordinates.
        7. Crop numeral semantic region and refine numeral glyph region with CV.
        8. Measure numeral height in pixels and compute calibrated height in mm.
        9. Synthesize measurement quality and notes.
        10. Return structured response (with optional debug visualization).
        """
        # 1. Dimension validation
        self.validate_package_dimensions(package_width_mm, package_height_mm)

        # 2. Image validation & decoding
        try:
            img_result = await validate_and_load_image(image_file)
        except ImageValidationError as exc:
            status_code = 400
            if exc.code == "UNSUPPORTED_MEDIA_TYPE":
                status_code = 415
            elif exc.code == "FILE_TOO_LARGE":
                status_code = 413
            raise MeasurementServiceError(
                code=exc.code,
                message=exc.message,
                status_code=status_code,
            ) from exc

        # Decode image bytes to OpenCV BGR
        nparr = np.frombuffer(img_result.content, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image_bgr is None or image_bgr.size == 0:
            raise MeasurementServiceError(
                code="CORRUPTED_IMAGE",
                message="OpenCV could not decode uploaded image bytes.",
                status_code=400,
            )

        img_h, img_w = image_bgr.shape[:2]

        # 3. Parse Module 1 JSON
        extraction = self.parse_extraction_json(extraction_json)

        # 4. Package bounding box: manual override vs. automatic detection
        has_manual_bbox = False
        if package_bbox_px is not None:
            if isinstance(package_bbox_px, str):
                trimmed = package_bbox_px.strip()
                if trimmed and trimmed.lower() not in ("null", "none"):
                    has_manual_bbox = True
            elif isinstance(package_bbox_px, (list, tuple)) and len(package_bbox_px) > 0:
                has_manual_bbox = True

        if has_manual_bbox:
            parsed_bbox = self.validate_package_bbox_px(package_bbox_px, img_w, img_h)  # type: ignore
            calib_method = "package_dimensions_manual_bbox"
            calib_confidence = 1.0
            calib_quality = "good"
            is_full_image = (parsed_bbox[0] == 0 and parsed_bbox[1] == 0 and parsed_bbox[2] == img_w and parsed_bbox[3] == img_h)
            if is_full_image:
                calib_quality = "low"
            calib_note = "Measurement calibrated using user-supplied package bounding box."
        else:
            det_result = detect_package_boundary(image_bgr)
            if det_result.quality == "failed":
                raise MeasurementServiceError(
                    code="PACKAGE_DETECTION_FAILED",
                    message=f"Automatic package boundary detection failed: {det_result.notes or 'No valid package candidate detected.'}",
                    status_code=422,
                )
            parsed_bbox = [det_result.x_min, det_result.y_min, det_result.x_max, det_result.y_max]
            calib_method = "package_dimensions_auto_bbox"
            calib_confidence = det_result.confidence
            calib_quality = det_result.quality
            is_full_image = (parsed_bbox[0] == 0 and parsed_bbox[1] == 0 and parsed_bbox[2] == img_w and parsed_bbox[3] == img_h)
            calib_note = det_result.notes or "Measurement calibrated using automatic package boundary detection."

        px_x1, px_y1, px_x2, px_y2 = parsed_bbox
        pkg_w_px = float(px_x2 - px_x1)
        pkg_h_px = float(px_y2 - px_y1)

        # 5. Calculate calibration from package_bbox_px + user dimensions
        calibration = compute_calibration_metadata(
            package_width_mm=package_width_mm,
            package_height_mm=package_height_mm,
            package_bbox_px=parsed_bbox,
            package_width_px=pkg_w_px,
            package_height_px=pkg_h_px,
            method=calib_method,
            confidence=calib_confidence,
            quality=calib_quality,
        )

        # 6. Check measurable candidates
        candidates = self._extract_measurable_candidates(extraction)
        if not candidates:
            raise MeasurementServiceError(
                code="NO_NUMERAL_REGIONS_FOUND",
                message=(
                    "No statutory declaration in extraction_json (net_quantity, mrp, dates) "
                    "contains a valid numeral_region for size measurement."
                ),
                status_code=422,
            )

        # 7. Measure each candidate numeral
        measurements: List[NumeralMeasurement] = []

        for cand in candidates:
            # Crop semantic region
            crop, crop_coords = crop_semantic_region(image_bgr, cand["region"])
            crop_origin = (crop_coords[0], crop_coords[1])

            # Detect actual numeral glyphs
            det_result = detect_numeral_glyphs(
                crop_bgr=crop,
                crop_origin_px=crop_origin,
                target_numeral_text=cand["numeral"],
            )

            # Compute physical mm using vertical height_mm_per_px
            height_mm = calculate_numeral_height_mm(
                numeral_height_px=det_result.numeral_height_px,
                height_mm_per_px=calibration.height_mm_per_px,
            )

            # Quality and diagnostic notes synthesis
            final_quality = det_result.quality
            diagnostic_notes = det_result.notes or ""

            if is_full_image:
                final_quality = "low"
                warning_note = "Calibration bbox covers the full image; physical package boundary may not have been isolated."
                diagnostic_notes = f"{warning_note} {diagnostic_notes}".strip()
            elif calib_quality == "low":
                final_quality = "low"
                diagnostic_notes = f"{calib_note} {diagnostic_notes}".strip()
            else:
                diagnostic_notes = f"{calib_note} {diagnostic_notes}".strip()

            measurements.append(
                NumeralMeasurement(
                    field=cand["field"],
                    raw_text=cand["raw_text"],
                    numeral=cand["numeral"],
                    semantic_region_normalized=cand["region"],
                    numeral_bbox_px=det_result.numeral_bbox_px,
                    numeral_height_px=det_result.numeral_height_px,
                    numeral_height_mm=height_mm,
                    confidence=det_result.confidence,
                    measurement_quality=final_quality,  # type: ignore
                    notes=diagnostic_notes,
                )
            )

        # Optional debug visualization overlay
        debug_b64 = None
        if include_debug_image:
            debug_b64 = render_measurement_debug_overlay(
                image_bgr=image_bgr,
                package_bbox_px=parsed_bbox,
                measurements=measurements,
                package_bbox_label="Manual" if has_manual_bbox else "Auto",
            )

        return MeasurementResponseData(
            image=MeasurementImageMetadata(width=img_w, height=img_h),
            calibration=calibration,
            measurements=measurements,
            debug_image_base64=debug_b64,
        )
