"""Computer vision preprocessing pipeline for numeral size measurement.

Handles normalized coordinate conversion, safe cropping, contrast enhancement,
and multi-strategy binarization for text/numeral segmentation.
"""

from typing import Tuple
import cv2
import numpy as np


def convert_normalized_bbox_to_pixels(
    normalized_bbox: list[float],
    image_width: int,
    image_height: int,
) -> Tuple[int, int, int, int]:
    """Converts normalized [x_min, y_min, x_max, y_max] (0.0 to 1.0) to pixel coordinates.

    Clamps values to image boundaries and ensures valid width/height.
    """
    if len(normalized_bbox) != 4:
        raise ValueError(f"Expected 4 normalized coordinates, got {len(normalized_bbox)}")

    x1, y1, x2, y2 = normalized_bbox

    # Ensure ordering
    x_min_norm, x_max_norm = min(x1, x2), max(x1, x2)
    y_min_norm, y_max_norm = min(y1, y2), max(y1, y2)

    px_x_min = max(0, min(image_width - 1, int(round(x_min_norm * image_width))))
    px_y_min = max(0, min(image_height - 1, int(round(y_min_norm * image_height))))
    px_x_max = max(px_x_min + 1, min(image_width, int(round(x_max_norm * image_width))))
    px_y_max = max(px_y_min + 1, min(image_height, int(round(y_max_norm * image_height))))

    return px_x_min, px_y_min, px_x_max, px_y_max


def crop_semantic_region(
    image_bgr: np.ndarray,
    normalized_bbox: list[float],
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Crops the semantic region from the input image.

    Returns:
        (cropped_bgr, (px_x_min, px_y_min, px_x_max, px_y_max))
    """
    img_h, img_w = image_bgr.shape[:2]
    coords = convert_normalized_bbox_to_pixels(normalized_bbox, img_w, img_h)
    x1, y1, x2, y2 = coords
    crop = image_bgr[y1:y2, x1:x2].copy()
    return crop, coords


def enhance_and_upscale_crop(
    crop_bgr: np.ndarray,
    target_min_height: int = 120,
) -> Tuple[np.ndarray, float]:
    """Upscales small crops for subpixel glyph analysis and applies contrast enhancement.

    Returns:
        (enhanced_grayscale, scale_factor)
    """
    h, w = crop_bgr.shape[:2]
    scale = 1.0
    if h < target_min_height and h > 0:
        scale = float(target_min_height) / float(h)
        new_w = max(1, int(round(w * scale)))
        new_h = target_min_height
        resized = cv2.resize(crop_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    else:
        resized = crop_bgr

    if len(resized.shape) == 3 and resized.shape[2] == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized.copy()

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Mild Gaussian blur to suppress fine noise
    smoothed = cv2.GaussianBlur(enhanced, (3, 3), 0)

    return smoothed, scale


def score_binarization_candidate(binary_img: np.ndarray) -> float:
    """Scores a binary image candidate (white foreground = 255, black background = 0).

    Higher score indicates plausible typographic glyph structures.
    """
    total_pixels = binary_img.size
    white_pixels = cv2.countNonZero(binary_img)
    ratio = white_pixels / float(total_pixels)

    # Plausible text fill ratio is usually between 5% and 50%
    if ratio < 0.02 or ratio > 0.85:
        return 0.0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img, connectivity=8)
    if num_labels <= 1:
        return 0.0

    img_h, img_w = binary_img.shape[:2]
    plausible_components = 0
    total_comp_height = 0

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        comp_h = stats[i, cv2.CC_STAT_HEIGHT]
        comp_w = stats[i, cv2.CC_STAT_WIDTH]

        # Ignore tiny speckles (< 4 px) or huge blocks (> 90% of area)
        if area < 4 or area > 0.9 * total_pixels:
            continue

        # Check aspect ratio and relative height
        if 0.15 * img_h <= comp_h <= 0.98 * img_h and comp_w <= 0.9 * img_w:
            plausible_components += 1
            total_comp_height += comp_h

    if plausible_components == 0:
        # Fallback ratio score
        return float(1.0 - abs(ratio - 0.25))

    # Base score on component count and ideal foreground fill ratio (~20%-35%)
    fill_score = 1.0 - abs(ratio - 0.25) * 2.0
    comp_score = min(plausible_components / 5.0, 1.0)

    return float(max(0.1, fill_score * 0.6 + comp_score * 0.4))


def binarize_numeral_crop(gray: np.ndarray) -> Tuple[np.ndarray, str, float]:
    """Evaluates multiple thresholding strategies and selects the highest-scoring candidate.

    Guarantees that the returned binary mask has FOREGROUND (glyphs) as 255 and BACKGROUND as 0.

    Returns:
        (best_binary_mask, strategy_name, quality_score)
    """
    candidates = []

    # 1. Otsu thresholding (dark text on light background -> invert so glyphs are white)
    _, otsu_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    candidates.append((otsu_inv, "otsu_inverted", score_binarization_candidate(otsu_inv)))

    # 2. Otsu thresholding (light text on dark background -> standard binary)
    _, otsu_std = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidates.append((otsu_std, "otsu_standard", score_binarization_candidate(otsu_std)))

    # 3. Adaptive Gaussian thresholding (inverted)
    adapt_inv = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4
    )
    candidates.append((adapt_inv, "adaptive_gaussian_inv", score_binarization_candidate(adapt_inv)))

    # 4. Adaptive Gaussian thresholding (standard)
    adapt_std = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4
    )
    candidates.append((adapt_std, "adaptive_gaussian_std", score_binarization_candidate(adapt_std)))

    # Sort candidates by descending score
    candidates.sort(key=lambda x: x[2], reverse=True)
    best_mask, best_name, best_score = candidates[0]

    # Optional morphological closing to bridge broken numeral strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned_mask = cv2.morphologyEx(best_mask, cv2.MORPH_CLOSE, kernel)

    return cleaned_mask, best_name, best_score
