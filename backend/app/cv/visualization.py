"""Debug visualization utility for Module 2A calibrated numeral size measurement.

Draws diagnostic overlays completely in-memory without persistent disk writes:
- GREEN: User-supplied package_bbox_px
- BLUE: Module 1 semantic numeral_region converted to pixels
- RED: Actual CV-detected numeral_bbox_px
"""

import base64
from typing import List, Optional, Tuple
import cv2
import numpy as np

from app.cv.preprocessing import convert_normalized_bbox_to_pixels
from app.schemas.measurement import NumeralMeasurement


def render_measurement_debug_overlay(
    image_bgr: np.ndarray,
    package_bbox_px: List[int],
    measurements: List[NumeralMeasurement],
    package_bbox_label: Optional[str] = None,
) -> str:
    """Renders diagnostic bounding boxes on a copy of the input image and returns a base64 JPEG string.

    Color coding (BGR):
    - GREEN (0, 255, 0): Package boundary bbox (auto or manual)
    - BLUE (255, 0, 0): Module 1 semantic region
    - RED (0, 0, 255): CV-detected numeral glyph bbox

    Returns:
        A data URI formatted base64 string: 'data:image/jpeg;base64,...'
    """
    overlay = image_bgr.copy()
    img_h, img_w = overlay.shape[:2]

    # 1. GREEN: Package bounding box
    if package_bbox_px and len(package_bbox_px) == 4:
        px1, py1, px2, py2 = package_bbox_px
        cv2.rectangle(overlay, (px1, py1), (px2, py2), (0, 255, 0), thickness=2)
        lbl_suffix = f" ({package_bbox_label})" if package_bbox_label else ""
        cv2.putText(
            overlay,
            f"Package BBox{lbl_suffix}: [{px1},{py1},{px2},{py2}]",
            (max(0, px1 + 4), max(18, py1 + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    # 2. Iterate measurements for BLUE (Semantic) & RED (Numeral glyph)
    for meas in measurements:
        # Convert normalized semantic region to pixel coordinates
        if meas.semantic_region_normalized and len(meas.semantic_region_normalized) == 4:
            sx1, sy1, sx2, sy2 = convert_normalized_bbox_to_pixels(
                meas.semantic_region_normalized,
                image_width=img_w,
                image_height=img_h,
            )
            # BLUE (255, 0, 0 in BGR)
            cv2.rectangle(overlay, (sx1, sy1), (sx2, sy2), (255, 0, 0), thickness=2)
            cv2.putText(
                overlay,
                f"Semantic: {meas.field}",
                (max(0, sx1), max(14, sy1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 0, 0),
                1,
                cv2.LINE_AA,
            )

        # RED (0, 0, 255 in BGR) for refined numeral glyph
        if meas.numeral_bbox_px and len(meas.numeral_bbox_px) == 4:
            nx1, ny1, nx2, ny2 = meas.numeral_bbox_px
            cv2.rectangle(overlay, (nx1, ny1), (nx2, ny2), (0, 0, 255), thickness=2)
            cv2.putText(
                overlay,
                f"Glyph '{meas.numeral}': {meas.numeral_height_px}px ({meas.numeral_height_mm}mm)",
                (max(0, nx1), min(img_h - 4, ny2 + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

    # Encode in-memory to JPEG
    success, encoded_jpg = cv2.imencode(
        ".jpg",
        overlay,
        [int(cv2.IMWRITE_JPEG_QUALITY), 85],
    )
    if not success:
        return ""

    b64_data = base64.b64encode(encoded_jpg).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_data}"
