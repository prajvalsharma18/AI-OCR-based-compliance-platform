"""Deterministic Rule 8 Spatial Clearance Analysis Service.

Evaluates statutory spatial clear-space compliance for Net Quantity declarations
under the Legal Metrology (Packaged Commodities) Rules, 2011, Rule 8(1) proviso and SIH 2026 PS 26034.

Rule 8 Requirement:
For Net Quantity declarations, the area surrounding the quantity declaration must be free
from printed information:
- Top: equal to numeral height (H)
- Bottom: equal to numeral height (H)
- Left: equal to 2 x numeral height (2H)
- Right: equal to 2 x numeral height (2H)

Where H = numeral_height_mm.

Clearance Reference:
Clear space is measured from the perimeter of the complete quantity declaration bounding box
(including prefix e.g. 'NET QTY:' and unit e.g. 'g'), NOT from the isolated numeral bbox.

Exclusions:
MRP, Manufacture Date, Use-by Date, Phone numbers, PIN codes, Barcodes, and License numbers
are explicitly excluded from Rule 8 and evaluated as 'not_applicable'.
"""

import base64
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from app.schemas.extraction import LabelExtractionResult
from app.schemas.measurement import (
    CalibrationMetadata,
    MeasurementQuality,
    MeasurementResponseData,
    NumeralMeasurement,
)
from app.schemas.rule8 import (
    DirectionStatus,
    DirectionalClearanceMm,
    DirectionalClearancePx,
    DirectionalRequirement,
    DirectionalResults,
    Rule8EvaluationRequest,
    Rule8Finding,
    Rule8InspectionStatus,
    Rule8MeasurementSummary,
    Rule8ResponseData,
    Rule8Summary,
)

logger = logging.getLogger(__name__)

NON_RULE8_FIELDS = {
    "mrp",
    "dates",
    "dates[0]",
    "dates[1]",
    "dates[2]",
    "date",
    "date[0]",
    "date[1]",
    "phone",
    "pin_code",
    "barcode",
    "license",
    "survey_number",
    "other",
}


class Rule8ServiceError(Exception):
    """Exception raised for Rule 8 service failures."""

    def __init__(self, code: str, message: str, status_code: int = 422):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class Rule8Service:
    """Deterministic Rule 8 spatial clearance evaluation engine."""

    def _parse_extraction_payload(self, raw_input: Any) -> Optional[LabelExtractionResult]:
        """Parses and validates Module 1 extraction payload."""
        if raw_input is None:
            return None
        if isinstance(raw_input, LabelExtractionResult):
            return raw_input
        if isinstance(raw_input, dict):
            if "data" in raw_input and isinstance(raw_input["data"], dict):
                return LabelExtractionResult.model_validate(raw_input["data"])
            return LabelExtractionResult.model_validate(raw_input)
        if isinstance(raw_input, str):
            try:
                parsed = json.loads(raw_input)
                return self._parse_extraction_payload(parsed)
            except Exception as exc:
                raise Rule8ServiceError(
                    code="INVALID_EXTRACTION_JSON",
                    message=f"Failed to parse extraction JSON string: {exc}",
                ) from exc
        return None

    def _parse_measurement_payload(self, raw_input: Any) -> Tuple[Optional[CalibrationMetadata], List[NumeralMeasurement], Tuple[int, int]]:
        """Parses Module 2A measurement payload into calibration, measurement list, and (img_w, img_h)."""
        if raw_input is None:
            return None, [], (0, 0)

        data_dict = raw_input
        if isinstance(raw_input, str):
            try:
                data_dict = json.loads(raw_input)
            except Exception as exc:
                raise Rule8ServiceError(
                    code="INVALID_MEASUREMENT_JSON",
                    message=f"Failed to parse measurement JSON string: {exc}",
                ) from exc

        if isinstance(data_dict, dict) and "data" in data_dict and isinstance(data_dict["data"], dict):
            data_dict = data_dict["data"]

        if isinstance(data_dict, MeasurementResponseData):
            img_dims = (data_dict.image.width, data_dict.image.height)
            return data_dict.calibration, data_dict.measurements, img_dims

        if isinstance(data_dict, dict):
            img_dims = (0, 0)
            if "image" in data_dict and isinstance(data_dict["image"], dict):
                img_dims = (data_dict["image"].get("width", 0), data_dict["image"].get("height", 0))

            calib = None
            if "calibration" in data_dict and isinstance(data_dict["calibration"], dict):
                calib = CalibrationMetadata.model_validate(data_dict["calibration"])

            measurements = []
            raw_meas_list = data_dict.get("measurements", [])
            if isinstance(raw_meas_list, list):
                for m in raw_meas_list:
                    if isinstance(m, NumeralMeasurement):
                        measurements.append(m)
                    elif isinstance(m, dict):
                        measurements.append(NumeralMeasurement.model_validate(m))
            return calib, measurements, img_dims

        return None, [], (0, 0)

    def _decode_image(self, image_input: Optional[Union[str, np.ndarray, bytes]]) -> Optional[np.ndarray]:
        """Decodes base64 string, bytes, or returns existing numpy image."""
        if image_input is None:
            return None
        if isinstance(image_input, np.ndarray):
            return image_input
        if isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if isinstance(image_input, str):
            data_str = image_input.strip()
            if "," in data_str and "base64" in data_str[:50]:
                data_str = data_str.split(",", 1)[1]
            try:
                img_bytes = base64.b64decode(data_str)
                nparr = np.frombuffer(img_bytes, np.uint8)
                return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception as exc:
                logger.warning("Failed to decode base64 image: %s", exc)
                return None
        return None

    def _convert_normalized_bbox_to_pixels(self, norm_bbox: List[float], img_w: int, img_h: int) -> List[int]:
        """Converts normalized [x_min, y_min, x_max, y_max] box to pixel coordinates."""
        if not norm_bbox or len(norm_bbox) != 4 or img_w <= 0 or img_h <= 0:
            return [0, 0, 0, 0]
        x1 = max(0, min(img_w, int(round(norm_bbox[0] * img_w))))
        y1 = max(0, min(img_h, int(round(norm_bbox[1] * img_h))))
        x2 = max(0, min(img_w, int(round(norm_bbox[2] * img_w))))
        y2 = max(0, min(img_h, int(round(norm_bbox[3] * img_h))))
        return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

    def detect_occupied_regions_from_layout(
        self,
        declaration_bbox_px: List[int],
        extraction: Optional[LabelExtractionResult],
        image_width: int,
        image_height: int,
    ) -> Dict[str, Optional[float]]:
        """Determines minimum distance to nearest occupied layout/OCR bounding boxes in cardinal directions.

        Measured outward from the complete quantity declaration bounding box (dx1, dy1, dx2, dy2).
        Excludes the quantity declaration's own semantic text components (e.g. 'NET QTY:', 'g').
        """
        dx1, dy1, dx2, dy2 = declaration_bbox_px

        nearest: Dict[str, Optional[float]] = {
            "top": None,
            "bottom": None,
            "left": None,
            "right": None,
        }

        if not extraction or image_width <= 0 or image_height <= 0:
            return nearest

        # Gather external bounding boxes
        candidate_boxes = []

        # 1. Raw text blocks
        for block in extraction.raw_text_blocks:
            if block.bbox and len(block.bbox) == 4:
                b_px = self._convert_normalized_bbox_to_pixels(block.bbox, image_width, image_height)
                candidate_boxes.append((b_px, block.text))

        # 2. Other semantic declarations (MRP, generic name, consumer care, etc.)
        sem = extraction.semantic_extraction
        if sem.mrp and sem.mrp.bbox:
            candidate_boxes.append((self._convert_normalized_bbox_to_pixels(sem.mrp.bbox, image_width, image_height), "mrp"))
        if sem.generic_name and sem.generic_name.bbox:
            candidate_boxes.append((self._convert_normalized_bbox_to_pixels(sem.generic_name.bbox, image_width, image_height), "generic_name"))
        if sem.consumer_care and sem.consumer_care.bbox:
            candidate_boxes.append((self._convert_normalized_bbox_to_pixels(sem.consumer_care.bbox, image_width, image_height), "consumer_care"))
        for d in sem.dates:
            if d.bbox:
                candidate_boxes.append((self._convert_normalized_bbox_to_pixels(d.bbox, image_width, image_height), "date"))
        for ent in sem.manufacturer.entities:
            if ent.bbox:
                candidate_boxes.append((self._convert_normalized_bbox_to_pixels(ent.bbox, image_width, image_height), "manufacturer"))
        for s in sem.stickers:
            if s.bbox:
                candidate_boxes.append((self._convert_normalized_bbox_to_pixels(s.bbox, image_width, image_height), "sticker"))
        for ovt in sem.other_visible_text:
            if ovt.bbox:
                candidate_boxes.append((self._convert_normalized_bbox_to_pixels(ovt.bbox, image_width, image_height), "other_visible_text"))

        for (bx1, by1, bx2, by2), label in candidate_boxes:
            # Check overlap with declaration itself: if high overlap, it's the declaration's own text
            inter_x1 = max(bx1, dx1)
            inter_y1 = max(by1, dy1)
            inter_x2 = min(bx2, dx2)
            inter_y2 = min(by2, dy2)
            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                b_area = max(1, (bx2 - bx1) * (by2 - by1))
                if inter_area / b_area > 0.50:
                    continue  # Skip own declaration text

            # 1. Top candidate: strictly above declaration top (dy1)
            if by2 <= dy1:
                # Horizontally overlapping or near the declaration corridor
                if max(bx1, dx1 - 10) <= min(bx2, dx2 + 10):
                    dist = float(dy1 - by2)
                    if dist >= 0:
                        if nearest["top"] is None or dist < nearest["top"]:
                            nearest["top"] = dist

            # 2. Bottom candidate: strictly below declaration bottom (dy2)
            if by1 >= dy2:
                if max(bx1, dx1 - 10) <= min(bx2, dx2 + 10):
                    dist = float(by1 - dy2)
                    if dist >= 0:
                        if nearest["bottom"] is None or dist < nearest["bottom"]:
                            nearest["bottom"] = dist

            # 3. Left candidate: strictly to the left of declaration (dx1)
            if bx2 <= dx1:
                if max(by1, dy1 - 10) <= min(by2, dy2 + 10):
                    dist = float(dx1 - bx2)
                    if dist >= 0:
                        if nearest["left"] is None or dist < nearest["left"]:
                            nearest["left"] = dist

            # 4. Right candidate: strictly to the right of declaration (dx2)
            if bx1 >= dx2:
                if max(by1, dy1 - 10) <= min(by2, dy2 + 10):
                    dist = float(bx1 - dx2)
                    if dist >= 0:
                        if nearest["right"] is None or dist < nearest["right"]:
                            nearest["right"] = dist

        return nearest

    def detect_occupied_regions_from_image(
        self,
        image_bgr: np.ndarray,
        declaration_bbox_px: List[int],
        req_clearance_px: Dict[str, float],
        package_bbox_px: Optional[List[int]] = None,
    ) -> Tuple[Dict[str, Optional[float]], Dict[str, float], bool]:
        """Performs CV analysis in local ROI around quantity declaration to find nearest foreground elements.

        Measures clearance from the complete quantity declaration bounding box (dx1, dy1, dx2, dy2).
        Detects reflective foil glare, background texture, folds, and shadows to prevent uncertain spatial detection
        from converting into false compliance.

        Returns:
            Tuple of:
            - nearest_occupied: Dict[str, Optional[float]] of distance to detected print/edge
            - verified_clear_px: Dict[str, float] of verified unprinted clearance distance
            - is_uncertain_texture: bool indicating whether foil/texture/glare prevents confident analysis
        """
        img_h, img_w = image_bgr.shape[:2]
        dx1, dy1, dx2, dy2 = declaration_bbox_px

        req_top = req_clearance_px.get("top", 20.0)
        req_bot = req_clearance_px.get("bottom", 20.0)
        req_left = req_clearance_px.get("left", 40.0)
        req_right = req_clearance_px.get("right", 40.0)

        # Search window bounds: inspect at least 3.5x required clear space or minimum padding
        pad_top = max(int(req_top * 3.5), 100)
        pad_bot = max(int(req_bot * 3.5), 100)
        pad_left = max(int(req_left * 3.5), 120)
        pad_right = max(int(req_right * 3.5), 120)

        min_x = max(0, dx1 - pad_left)
        max_x = min(img_w, dx2 + pad_right)
        min_y = max(0, dy1 - pad_top)
        max_y = min(img_h, dy2 + pad_bot)

        if package_bbox_px and len(package_bbox_px) == 4:
            px1, py1, px2, py2 = package_bbox_px
            min_x = max(min_x, px1)
            max_x = min(max_x, px2)
            min_y = max(min_y, py1)
            max_y = min(max_y, py2)

        nearest_occupied: Dict[str, Optional[float]] = {"top": None, "bottom": None, "left": None, "right": None}
        verified_clear_px: Dict[str, float] = {
            "top": float(max(0, dy1 - min_y)),
            "bottom": float(max(0, max_y - dy2)),
            "left": float(max(0, dx1 - min_x)),
            "right": float(max(0, max_x - dx2)),
        }

        if min_x >= max_x or min_y >= max_y:
            return nearest_occupied, verified_clear_px, False

        roi = image_bgr[min_y:max_y, min_x:max_x]
        if roi.size == 0:
            return nearest_occupied, verified_clear_px, False

        # Convert to grayscale & binarize
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 40, 140)

        # Adaptive threshold foreground
        adapt = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4
        )
        combined = cv2.bitwise_or(edges, adapt)

        # Mask out own declaration bbox inside ROI
        local_dx1 = max(0, dx1 - min_x)
        local_dy1 = max(0, dy1 - min_y)
        local_dx2 = min(roi.shape[1], dx2 - min_x)
        local_dy2 = min(roi.shape[0], dy2 - min_y)

        # Check background texture & reflective foil characteristics in unprinted corridor
        mask_surround = np.ones((roi.shape[0], roi.shape[1]), dtype=np.uint8)
        mask_surround[local_dy1:local_dy2, local_dx1:local_dx2] = 0
        surround_pixels = gray[mask_surround > 0]
        surround_edges = edges[mask_surround > 0]

        glare_ratio = float(np.mean(surround_pixels >= 250)) if len(surround_pixels) > 0 else 0.0
        edge_density = float(np.mean(surround_edges > 0)) if len(surround_edges) > 0 else 0.0
        intensity_std = float(np.std(surround_pixels)) if len(surround_pixels) > 0 else 0.0

        # Reflective foil, excessive packaging crinkles, or heavy decorative graphics
        is_uncertain_texture = bool(glare_ratio > 0.18 or (edge_density > 0.30 and intensity_std > 45.0))

        # Mask declaration out of search space
        combined[local_dy1:local_dy2, local_dx1:local_dx2] = 0

        # Cardinal scanning strips originating from declaration boundaries
        cx1 = max(0, local_dx1)
        cx2 = min(roi.shape[1], local_dx2)
        cy1 = max(0, local_dy1)
        cy2 = min(roi.shape[0], local_dy2)

        # 1. TOP: scan upward from local_dy1
        if local_dy1 > 0 and cx1 < cx2:
            top_strip = combined[:local_dy1, cx1:cx2]
            y_indices, _ = np.where(top_strip > 0)
            if len(y_indices) > 0:
                nearest_y = int(np.max(y_indices))
                dist = float(local_dy1 - nearest_y)
                nearest_occupied["top"] = dist
                verified_clear_px["top"] = dist
            else:
                verified_clear_px["top"] = float(local_dy1)

        # 2. BOTTOM: scan downward from local_dy2
        if local_dy2 < roi.shape[0] and cx1 < cx2:
            bottom_strip = combined[local_dy2:, cx1:cx2]
            y_indices, _ = np.where(bottom_strip > 0)
            if len(y_indices) > 0:
                nearest_y = int(np.min(y_indices))
                dist = float(nearest_y)
                nearest_occupied["bottom"] = dist
                verified_clear_px["bottom"] = dist
            else:
                verified_clear_px["bottom"] = float(roi.shape[0] - local_dy2)

        # 3. LEFT: scan leftward from local_dx1
        if local_dx1 > 0 and cy1 < cy2:
            left_strip = combined[cy1:cy2, :local_dx1]
            _, x_indices = np.where(left_strip > 0)
            if len(x_indices) > 0:
                nearest_x = int(np.max(x_indices))
                dist = float(local_dx1 - nearest_x)
                nearest_occupied["left"] = dist
                verified_clear_px["left"] = dist
            else:
                verified_clear_px["left"] = float(local_dx1)

        # 4. RIGHT: scan rightward from local_dx2
        if local_dx2 < roi.shape[1] and cy1 < cy2:
            right_strip = combined[cy1:cy2, local_dx2:]
            _, x_indices = np.where(right_strip > 0)
            if len(x_indices) > 0:
                nearest_x = int(np.min(x_indices))
                dist = float(nearest_x)
                nearest_occupied["right"] = dist
                verified_clear_px["right"] = dist
            else:
                verified_clear_px["right"] = float(roi.shape[1] - local_dx2)

        return nearest_occupied, verified_clear_px, is_uncertain_texture

    def calculate_directional_clearance(
        self,
        declaration_bbox_px: List[int],
        layout_nearest: Dict[str, Optional[float]],
        image_nearest: Dict[str, Optional[float]],
        image_verified: Optional[Dict[str, float]],
        scale_v: float,
        scale_h: float,
        package_bbox_px: Optional[List[int]],
        img_w: int,
        img_h: int,
    ) -> Tuple[DirectionalClearancePx, DirectionalClearanceMm, bool]:
        """Calculates directional clearances in pixels and physical mm measured from declaration bbox.

        Uses directional calibration scales (vertical scale_v for top/bottom, horizontal scale_h for left/right).
        Clearances are physically bounded by detected obstacles or verified scanned clear boundaries.

        Returns:
            Tuple of (DirectionalClearancePx, DirectionalClearanceMm, is_confident)
        """
        dx1, dy1, dx2, dy2 = declaration_bbox_px

        # Package or image boundaries as ultimate limits from declaration edges
        bx1 = package_bbox_px[0] if package_bbox_px else 0
        by1 = package_bbox_px[1] if package_bbox_px else 0
        bx2 = package_bbox_px[2] if package_bbox_px else img_w
        by2 = package_bbox_px[3] if package_bbox_px else img_h

        bound_limits = {
            "top": float(max(0, dy1 - by1)),
            "bottom": float(max(0, by2 - dy2)),
            "left": float(max(0, dx1 - bx1)),
            "right": float(max(0, bx2 - dx2)),
        }

        clear_px: Dict[str, Optional[float]] = {}
        clear_mm: Dict[str, Optional[float]] = {}
        any_feature_detected = False

        for d in ["top", "bottom", "left", "right"]:
            candidates = []
            l_val = layout_nearest.get(d)
            if l_val is not None and l_val >= 0:
                candidates.append(l_val)
                any_feature_detected = True

            i_val = image_nearest.get(d)
            if i_val is not None and i_val >= 0:
                candidates.append(i_val)
                any_feature_detected = True

            if candidates:
                val_px = min(candidates)
            elif image_verified and d in image_verified:
                # Verified blank distance scanned in image, bounded by package border
                val_px = min(image_verified[d], bound_limits[d])
                any_feature_detected = True
            else:
                # Fallback to package boundary distance (when no image provided)
                val_px = bound_limits[d]

            clear_px[d] = round(val_px, 2)

            # Apply directional scales
            if d in ("top", "bottom"):
                clear_mm[d] = round(val_px * scale_v, 2)
            else:
                clear_mm[d] = round(val_px * scale_h, 2)

        px_obj = DirectionalClearancePx(
            top_px=clear_px["top"],
            bottom_px=clear_px["bottom"],
            left_px=clear_px["left"],
            right_px=clear_px["right"],
        )
        mm_obj = DirectionalClearanceMm(
            top_mm=clear_mm["top"],
            bottom_mm=clear_mm["bottom"],
            left_mm=clear_mm["left"],
            right_mm=clear_mm["right"],
        )
        return px_obj, mm_obj, any_feature_detected

    def evaluate_rule8_finding(
        self,
        field: str,
        numeral_measurement: Optional[NumeralMeasurement],
        extraction: Optional[LabelExtractionResult],
        calibration: Optional[CalibrationMetadata],
        image_bgr: Optional[np.ndarray],
        img_w: int,
        img_h: int,
    ) -> Rule8Finding:
        """Evaluates a single declaration against Rule 8 spatial clearance requirements."""
        # Scope Exclusion: Rule 8 applies ONLY to Net Quantity
        if field != "net_quantity":
            return Rule8Finding(
                field=field,
                numeral_bbox_px=numeral_measurement.numeral_bbox_px if numeral_measurement else None,
                measurement=None,
                required_clearance_mm=None,
                actual_clearance_px=None,
                actual_clearance_mm=None,
                direction_results=DirectionalResults(
                    top="not_applicable",
                    bottom="not_applicable",
                    left="not_applicable",
                    right="not_applicable",
                ),
                inspection_status="not_applicable",
                confidence=1.0,
                notes=f"Rule 8 spatial clearance requirement applies only to Net Quantity declarations; field '{field}' is excluded.",
            )

        # Missing measurement check
        if numeral_measurement is None or not numeral_measurement.numeral_bbox_px:
            return Rule8Finding(
                field=field,
                numeral_bbox_px=None,
                measurement=None,
                required_clearance_mm=None,
                actual_clearance_px=None,
                actual_clearance_mm=None,
                direction_results=DirectionalResults(
                    top="indeterminate",
                    bottom="indeterminate",
                    left="indeterminate",
                    right="indeterminate",
                ),
                inspection_status="indeterminate_missing_measurement",
                confidence=0.0,
                notes="Target numeral measurement or bounding box is missing from Module 2A.",
            )

        num_bbox = numeral_measurement.numeral_bbox_px
        h_mm = numeral_measurement.numeral_height_mm
        h_px = numeral_measurement.numeral_height_px
        quality = numeral_measurement.measurement_quality
        meas_conf = numeral_measurement.confidence

        meas_summary = Rule8MeasurementSummary(
            height_px=h_px,
            height_mm=h_mm,
            confidence=meas_conf,
            quality=quality,
        )

        # Required Clearances: Top=H, Bottom=H, Left=2H, Right=2H
        req_mm = DirectionalRequirement(
            top_mm=round(h_mm, 2),
            bottom_mm=round(h_mm, 2),
            left_mm=round(2.0 * h_mm, 2),
            right_mm=round(2.0 * h_mm, 2),
        )

        # Calibration Scales
        if calibration:
            scale_v = calibration.height_mm_per_px
            scale_h = calibration.width_mm_per_px
            pkg_bbox = calibration.package_bbox_px
        else:
            scale_v = (h_mm / max(1, h_px)) if h_px > 0 else 0.1
            scale_h = scale_v
            pkg_bbox = None

        # Determine complete quantity declaration bounding box (dx1, dy1, dx2, dy2)
        semantic_bbox_px = num_bbox
        if extraction and extraction.semantic_extraction.net_quantity.bbox:
            semantic_bbox_px = self._convert_normalized_bbox_to_pixels(
                extraction.semantic_extraction.net_quantity.bbox, img_w, img_h
            )
        elif numeral_measurement.semantic_region_normalized:
            semantic_bbox_px = self._convert_normalized_bbox_to_pixels(
                numeral_measurement.semantic_region_normalized, img_w, img_h
            )

        dx1 = min(semantic_bbox_px[0], num_bbox[0])
        dy1 = min(semantic_bbox_px[1], num_bbox[1])
        dx2 = max(semantic_bbox_px[2], num_bbox[2])
        dy2 = max(semantic_bbox_px[3], num_bbox[3])
        declaration_bbox_px = [dx1, dy1, dx2, dy2]

        req_clearance_px = {
            "top": req_mm.top_mm / max(1e-6, scale_v),
            "bottom": req_mm.bottom_mm / max(1e-6, scale_v),
            "left": req_mm.left_mm / max(1e-6, scale_h),
            "right": req_mm.right_mm / max(1e-6, scale_h),
        }

        # Detect nearest occupied regions from layout
        layout_nearest = self.detect_occupied_regions_from_layout(
            declaration_bbox_px, extraction, img_w, img_h
        )

        image_nearest = {"top": None, "bottom": None, "left": None, "right": None}
        image_verified = None
        is_uncertain_texture = False

        if image_bgr is not None:
            image_nearest, image_verified, is_uncertain_texture = self.detect_occupied_regions_from_image(
                image_bgr, declaration_bbox_px, req_clearance_px, pkg_bbox
            )

        actual_px, actual_mm, feature_detected = self.calculate_directional_clearance(
            declaration_bbox_px, layout_nearest, image_nearest, image_verified, scale_v, scale_h, pkg_bbox, img_w, img_h
        )

        # Evaluate threshold for each direction
        dir_res: Dict[str, DirectionStatus] = {}
        for d in ["top", "bottom", "left", "right"]:
            act_val = getattr(actual_mm, f"{d}_mm")
            req_val = getattr(req_mm, f"{d}_mm")

            if act_val is None:
                dir_res[d] = "indeterminate"
            elif act_val >= req_val:
                dir_res[d] = "above_threshold"
            else:
                dir_res[d] = "below_threshold"

        dir_results_obj = DirectionalResults(
            top=dir_res["top"],
            bottom=dir_res["bottom"],
            left=dir_res["left"],
            right=dir_res["right"],
        )

        # Uncertainty & Inspection Status Resolution
        # 1. Low measurement quality from Module 2A
        if quality in ("low", "failed") or meas_conf < 0.65:
            inspection_status = "indeterminate_low_confidence"
            notes = (
                f"Measurement quality is '{quality}' (confidence {meas_conf:.2f}). "
                "Preserving uncertainty without issuing definitive legal non-compliance order."
            )
        # 2. Reflective foil glare or packaging texture uncertainty
        elif is_uncertain_texture:
            inspection_status = "indeterminate_low_confidence"
            notes = (
                "Surrounding space exhibits reflective foil glare, texture, or folds. "
                "Preserving indeterminate status without converting spatial uncertainty into compliance."
            )
        # 3. No feature detected in layout or image
        elif not feature_detected and image_bgr is None:
            inspection_status = "indeterminate_low_confidence"
            notes = "No surrounding layout elements confidently detected to verify spatial boundaries."
        else:
            # 4. Deterministic verdict based on direction thresholds
            if all(v == "above_threshold" for v in dir_res.values()):
                inspection_status = "compliant"
                notes = "All 4 directional clear-space distances exceed statutory minimums (H top/bottom, 2H left/right per Rule 8(1) proviso)."
            elif any(v == "below_threshold" for v in dir_res.values()):
                inspection_status = "non_compliant"
                failed_dirs = [d for d, v in dir_res.items() if v == "below_threshold"]
                notes = f"Insufficient clear space detected in direction(s): {', '.join(failed_dirs)}."
            else:
                inspection_status = "indeterminate_low_confidence"
                notes = "Spatial clear area could not be resolved with high confidence."

        return Rule8Finding(
            field=field,
            declaration_bbox_px=declaration_bbox_px,
            numeral_bbox_px=num_bbox,
            semantic_bbox_px=semantic_bbox_px,
            measurement=meas_summary,
            required_clearance_mm=req_mm,
            actual_clearance_px=actual_px,
            actual_clearance_mm=actual_mm,
            direction_results=dir_results_obj,
            inspection_status=inspection_status,
            confidence=round(meas_conf * (0.95 if feature_detected and not is_uncertain_texture else 0.70), 2),
            notes=notes,
        )

    def render_rule8_debug_overlay(
        self,
        image_bgr: np.ndarray,
        findings: List[Rule8Finding],
        package_bbox_px: Optional[List[int]],
        scale_v: float = 0.1,
        scale_h: float = 0.1,
    ) -> str:
        """Renders diagnostic overlays with directional clearance lines and requirements."""
        overlay = image_bgr.copy()
        img_h, img_w = overlay.shape[:2]

        # 1. Package boundary (GREEN)
        if package_bbox_px and len(package_bbox_px) == 4:
            px1, py1, px2, py2 = package_bbox_px
            cv2.rectangle(overlay, (px1, py1), (px2, py2), (0, 255, 0), thickness=2)
            cv2.putText(
                overlay,
                "Package Boundary",
                (px1 + 4, max(18, py1 + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        for finding in findings:
            if finding.inspection_status == "not_applicable" or not finding.numeral_bbox_px:
                continue

            nx1, ny1, nx2, ny2 = finding.numeral_bbox_px
            decl_bbox = finding.declaration_bbox_px or finding.semantic_bbox_px or finding.numeral_bbox_px

            # 2. Complete Declaration BBox (BLUE)
            if decl_bbox:
                dx1, dy1, dx2, dy2 = decl_bbox
                cv2.rectangle(overlay, (dx1, dy1), (dx2, dy2), (255, 0, 0), thickness=2)
                cv2.putText(
                    overlay,
                    "Declaration: net_quantity",
                    (dx1, max(14, dy1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

            # 3. Numeral bbox (RED)
            cv2.rectangle(overlay, (nx1, ny1), (nx2, ny2), (0, 0, 255), thickness=2)
            cv2.putText(
                overlay,
                f"Numeral H={finding.measurement.height_mm if finding.measurement else ''}mm",
                (nx1, min(img_h - 4, ny2 + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

            # 4. Statutory Clearance Boundary (MAGENTA)
            # Rule 8(1) proviso requires H top/bottom, 2H left/right
            req_mm = finding.required_clearance_mm
            if decl_bbox and req_mm and scale_v > 0 and scale_h > 0:
                dx1, dy1, dx2, dy2 = decl_bbox
                req_top_px = int(round(req_mm.top_mm / scale_v))
                req_bot_px = int(round(req_mm.bottom_mm / scale_v))
                req_left_px = int(round(req_mm.left_mm / scale_h))
                req_right_px = int(round(req_mm.right_mm / scale_h))

                sc_x1 = max(0, dx1 - req_left_px)
                sc_y1 = max(0, dy1 - req_top_px)
                sc_x2 = min(img_w, dx2 + req_right_px)
                sc_y2 = min(img_h, dy2 + req_bot_px)
                cv2.rectangle(overlay, (sc_x1, sc_y1), (sc_x2, sc_y2), (255, 0, 255), thickness=1)
                cv2.putText(
                    overlay,
                    "Required Clear Zone [Rule 8(1) proviso]",
                    (sc_x1, max(12, sc_y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (255, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

            # 5. Draw Directional Clearance Rays from Declaration Box Edges
            act_px = finding.actual_clearance_px
            act_mm = finding.actual_clearance_mm
            d_res = finding.direction_results

            if act_px and act_mm and req_mm and decl_bbox:
                dx1, dy1, dx2, dy2 = decl_bbox
                cx = (dx1 + dx2) // 2
                cy = (dy1 + dy2) // 2

                # TOP: ray from (cx, dy1) upward
                top_color = (0, 255, 0) if d_res.top == "above_threshold" else (0, 0, 255)
                top_end_y = max(0, int(dy1 - (act_px.top_px or 0)))
                cv2.line(overlay, (cx, dy1), (cx, top_end_y), (255, 255, 0), thickness=2)
                cv2.line(overlay, (cx - 5, top_end_y), (cx + 5, top_end_y), top_color, thickness=2)
                cv2.putText(
                    overlay,
                    f"Top: {act_mm.top_mm}mm / req {req_mm.top_mm}mm",
                    (cx + 6, max(14, (dy1 + top_end_y) // 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    top_color,
                    1,
                    cv2.LINE_AA,
                )

                # BOTTOM: ray from (cx, dy2) downward
                bot_color = (0, 255, 0) if d_res.bottom == "above_threshold" else (0, 0, 255)
                bot_end_y = min(img_h, int(dy2 + (act_px.bottom_px or 0)))
                cv2.line(overlay, (cx, dy2), (cx, bot_end_y), (255, 255, 0), thickness=2)
                cv2.line(overlay, (cx - 5, bot_end_y), (cx + 5, bot_end_y), bot_color, thickness=2)
                cv2.putText(
                    overlay,
                    f"Bot: {act_mm.bottom_mm}mm / req {req_mm.bottom_mm}mm",
                    (cx + 6, min(img_h - 4, (dy2 + bot_end_y) // 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    bot_color,
                    1,
                    cv2.LINE_AA,
                )

                # LEFT: ray from (dx1, cy) leftward
                left_color = (0, 255, 0) if d_res.left == "above_threshold" else (0, 0, 255)
                left_end_x = max(0, int(dx1 - (act_px.left_px or 0)))
                cv2.line(overlay, (dx1, cy), (left_end_x, cy), (255, 255, 0), thickness=2)
                cv2.line(overlay, (left_end_x, cy - 5), (left_end_x, cy + 5), left_color, thickness=2)
                cv2.putText(
                    overlay,
                    f"Left: {act_mm.left_mm}mm / req {req_mm.left_mm}mm",
                    (max(4, left_end_x), cy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    left_color,
                    1,
                    cv2.LINE_AA,
                )

                # RIGHT: ray from (dx2, cy) rightward
                right_color = (0, 255, 0) if d_res.right == "above_threshold" else (0, 0, 255)
                right_end_x = min(img_w, int(dx2 + (act_px.right_px or 0)))
                cv2.line(overlay, (dx2, cy), (right_end_x, cy), (255, 255, 0), thickness=2)
                cv2.line(overlay, (right_end_x, cy - 5), (right_end_x, cy + 5), right_color, thickness=2)
                cv2.putText(
                    overlay,
                    f"Right: {act_mm.right_mm}mm / req {req_mm.right_mm}mm",
                    (min(img_w - 140, dx2 + 8), cy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    right_color,
                    1,
                    cv2.LINE_AA,
                )

        success, encoded_jpg = cv2.imencode(".jpg", overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not success:
            return ""
        b64_data = base64.b64encode(encoded_jpg).decode("utf-8")
        return f"data:image/jpeg;base64,{b64_data}"

    def evaluate(
        self,
        extraction: Optional[Any],
        measurements: Optional[Any],
        image_input: Optional[Union[str, np.ndarray, bytes]] = None,
        include_debug_image: bool = False,
    ) -> Rule8ResponseData:
        """Executes deterministic Rule 8 spatial clearance analysis on net quantity declaration."""
        parsed_extraction = self._parse_extraction_payload(extraction)
        calibration, meas_list, img_dims = self._parse_measurement_payload(measurements)
        image_bgr = self._decode_image(image_input)

        img_w, img_h = img_dims
        if image_bgr is not None:
            img_h, img_w = image_bgr.shape[:2]
        elif parsed_extraction and parsed_extraction.image.width > 0:
            img_w = parsed_extraction.image.width
            img_h = parsed_extraction.image.height

        meas_by_field: Dict[str, NumeralMeasurement] = {}
        for m in meas_list:
            meas_by_field[m.field] = m

        findings: List[Rule8Finding] = []

        # 1. Evaluate Net Quantity (primary target of Rule 8)
        net_qty_meas = meas_by_field.get("net_quantity")
        net_qty_finding = self.evaluate_rule8_finding(
            field="net_quantity",
            numeral_measurement=net_qty_meas,
            extraction=parsed_extraction,
            calibration=calibration,
            image_bgr=image_bgr,
            img_w=img_w,
            img_h=img_h,
        )
        findings.append(net_qty_finding)

        # 2. Explicitly evaluate other provided measurements as 'not_applicable'
        for f_name, m_val in meas_by_field.items():
            if f_name == "net_quantity":
                continue
            finding = self.evaluate_rule8_finding(
                field=f_name,
                numeral_measurement=m_val,
                extraction=parsed_extraction,
                calibration=calibration,
                image_bgr=image_bgr,
                img_w=img_w,
                img_h=img_h,
            )
            findings.append(finding)

        # Compute summary
        total_eval = len(findings)
        comp_count = sum(1 for f in findings if f.inspection_status == "compliant")
        non_comp_count = sum(1 for f in findings if f.inspection_status == "non_compliant")
        indet_count = sum(
            1 for f in findings if f.inspection_status in ("indeterminate_low_confidence", "indeterminate_missing_measurement")
        )
        na_count = sum(1 for f in findings if f.inspection_status == "not_applicable")

        summary = Rule8Summary(
            total_evaluated=total_eval,
            compliant_count=comp_count,
            non_compliant_count=non_comp_count,
            indeterminate_count=indet_count,
            not_applicable_count=na_count,
        )

        debug_img_b64 = None
        if include_debug_image and image_bgr is not None:
            pkg_bbox = calibration.package_bbox_px if calibration else None
            scale_v = calibration.height_mm_per_px if calibration else 0.1
            scale_h = calibration.width_mm_per_px if calibration else 0.1
            debug_img_b64 = self.render_rule8_debug_overlay(image_bgr, findings, pkg_bbox, scale_v, scale_h)

        return Rule8ResponseData(
            summary=summary,
            findings=findings,
            debug_image_base64=debug_img_b64,
        )
