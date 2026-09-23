"""Numeral glyph isolation and bounding box detection.

Extracts and isolates the actual printed numeral glyphs from within a semantic
numeral region (e.g. isolating '1' from '1 kg', or '500' from '500 g').
"""

from typing import NamedTuple, Optional, Tuple
import cv2
import numpy as np

from app.cv.preprocessing import binarize_numeral_crop, enhance_and_upscale_crop


class NumeralDetectionResult(NamedTuple):
    numeral_bbox_px: list[int]  # [x_min, y_min, x_max, y_max] in image coordinates
    numeral_height_px: int
    numeral_width_px: int
    confidence: float
    quality: str  # "good", "moderate", "low", "failed"
    binarization_method: str
    notes: Optional[str] = None


def detect_numeral_glyphs(
    crop_bgr: np.ndarray,
    crop_origin_px: Tuple[int, int],
    target_numeral_text: str = "",
) -> NumeralDetectionResult:
    """Isolates numeral glyphs within a cropped semantic region.

    Args:
        crop_bgr: Cropped BGR image of the semantic region.
        crop_origin_px: (px_x_min, px_y_min) offset in original image coordinates.
        target_numeral_text: String representing target numeral (e.g. '1', '500', '120.00').

    Returns:
        NumeralDetectionResult containing pixel bounding box and height.
    """
    crop_h, crop_w = crop_bgr.shape[:2]
    if crop_h <= 1 or crop_w <= 1:
        return NumeralDetectionResult(
            numeral_bbox_px=[crop_origin_px[0], crop_origin_px[1], crop_origin_px[0] + crop_w, crop_origin_px[1] + crop_h],
            numeral_height_px=crop_h,
            numeral_width_px=crop_w,
            confidence=0.1,
            quality="failed",
            binarization_method="none",
            notes="Crop region is too small to segment (<2px).",
        )

    # 1. Upscale and enhance
    enhanced_gray, scale_factor = enhance_and_upscale_crop(crop_bgr, target_min_height=120)
    up_h, up_w = enhanced_gray.shape[:2]

    # 2. Binarize (foreground = 255, background = 0)
    binary_mask, strategy_name, binarization_score = binarize_numeral_crop(enhanced_gray)

    # 3. Connected components analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    candidates = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        x = stats[i, cv2.CC_STAT_WIDTH]
        comp_x = stats[i, cv2.CC_STAT_LEFT]
        comp_y = stats[i, cv2.CC_STAT_TOP]
        comp_w = stats[i, cv2.CC_STAT_WIDTH]
        comp_h = stats[i, cv2.CC_STAT_HEIGHT]

        # Ignore tiny speckles (< 10 px in upscaled space) or frame-filling borders
        if area < 12 or area > 0.85 * (up_h * up_w):
            continue

        # Check plausible text height: between 15% and 96% of crop height
        if 0.15 * up_h <= comp_h <= 0.98 * up_h and comp_w <= 0.95 * up_w:
            candidates.append({
                "x1": comp_x,
                "y1": comp_y,
                "x2": comp_x + comp_w,
                "y2": comp_y + comp_h,
                "w": comp_w,
                "h": comp_h,
                "area": area,
            })

    # If no candidate components found, fallback to central bounding estimate
    if not candidates:
        fallback_h = max(1, int(round(crop_h * 0.75)))
        fallback_y1 = crop_origin_px[1] + int(round(crop_h * 0.12))
        fallback_y2 = fallback_y1 + fallback_h
        fallback_x1 = crop_origin_px[0]
        fallback_x2 = crop_origin_px[0] + crop_w

        return NumeralDetectionResult(
            numeral_bbox_px=[fallback_x1, fallback_y1, fallback_x2, fallback_y2],
            numeral_height_px=fallback_h,
            numeral_width_px=crop_w,
            confidence=0.35,
            quality="low",
            binarization_method=strategy_name,
            notes="No distinct connected glyphs isolated; used approximate region estimate.",
        )

    # Sort candidate glyphs horizontally (left to right)
    candidates.sort(key=lambda c: c["x1"])

    # Determine expected numeral glyph count
    num_digits = len([ch for ch in target_numeral_text if ch.isdigit()])
    if num_digits == 0:
        num_digits = 1

    # In labels like "1 kg" or "500 g", numerals usually appear before the unit.
    # We select the leading glyph cluster corresponding to the numeral count.
    selected_components = candidates[:max(1, min(num_digits, len(candidates)))]

    # If the digits are tall and uniform, take their combined bounding box
    glyph_x1 = min(c["x1"] for c in selected_components)
    glyph_y1 = min(c["y1"] for c in selected_components)
    glyph_x2 = max(c["x2"] for c in selected_components)
    glyph_y2 = max(c["y2"] for c in selected_components)

    # Map back to original unscaled crop coordinates
    orig_x1 = int(round(glyph_x1 / scale_factor))
    orig_y1 = int(round(glyph_y1 / scale_factor))
    orig_x2 = int(round(glyph_x2 / scale_factor))
    orig_y2 = int(round(glyph_y2 / scale_factor))

    # Clamp to original crop boundaries
    orig_x1 = max(0, min(crop_w - 1, orig_x1))
    orig_y1 = max(0, min(crop_h - 1, orig_y1))
    orig_x2 = max(orig_x1 + 1, min(crop_w, orig_x2))
    orig_y2 = max(orig_y1 + 1, min(crop_h, orig_y2))

    # Map to full image coordinates
    img_x1 = crop_origin_px[0] + orig_x1
    img_y1 = crop_origin_px[1] + orig_y1
    img_x2 = crop_origin_px[0] + orig_x2
    img_y2 = crop_origin_px[1] + orig_y2

    numeral_h_px = max(1, img_y2 - img_y1)
    numeral_w_px = max(1, img_x2 - img_x1)

    # Compute confidence
    fill_ratio = (numeral_h_px / float(crop_h)) if crop_h > 0 else 0.5
    confidence = float(min(0.96, max(0.60, binarization_score * 0.7 + (0.3 if 0.25 <= fill_ratio <= 0.95 else 0.1))))
    quality = "good" if confidence >= 0.82 else "moderate"

    return NumeralDetectionResult(
        numeral_bbox_px=[img_x1, img_y1, img_x2, img_y2],
        numeral_height_px=numeral_h_px,
        numeral_width_px=numeral_w_px,
        confidence=confidence,
        quality=quality,
        binarization_method=strategy_name,
        notes=f"Isolated {len(selected_components)} numeral glyph(s) with strategy '{strategy_name}'.",
    )
