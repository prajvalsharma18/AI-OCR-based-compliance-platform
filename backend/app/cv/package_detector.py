"""Classical Computer Vision package boundary detector.

Estimates the visible package bounding box in pixel coordinates for real-world
calibration without using deep-learning models or neural networks.
"""

from typing import NamedTuple, Optional, Tuple
import cv2
import numpy as np


class PackageBoundaryResult(NamedTuple):
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    width_px: float
    height_px: float
    confidence: float
    quality: str  # "good", "moderate", "low", "failed"
    method: str
    notes: Optional[str] = None


def _extract_candidates_from_mask(
    mask: np.ndarray,
    img_w: int,
    img_h: int,
    method_name: str,
) -> list[dict]:
    """Finds external contours from a binary mask, applies morphological healing, and computes candidate metrics."""
    total_area = float(img_w * img_h)
    candidates = []

    # Morphological closing to bridge text and graphic elements into solid package body
    close_ksize = max(5, int(min(img_w, img_h) * 0.02) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_ksize, close_ksize))
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        contour_area = cv2.contourArea(cnt)
        bbox_area = float(w * h)
        area_ratio = bbox_area / total_area

        # Filter out obvious non-package elements (small text, icons, thin slivers, or entire frame border)
        if area_ratio < 0.10:
            continue
        if area_ratio > 0.99 and contour_area / total_area > 0.96:
            continue
        if w < int(img_w * 0.12) or h < int(img_h * 0.12):
            continue

        aspect_ratio = float(w) / float(h)
        if aspect_ratio < 0.15 or aspect_ratio > 6.0:
            continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = (contour_area / hull_area) if hull_area > 0 else 0.0
        extent = contour_area / bbox_area if bbox_area > 0 else 0.0

        # Boundary contact detection (within 1.5% margin of frame)
        margin_x = max(2, int(img_w * 0.015))
        margin_y = max(2, int(img_h * 0.015))
        touches_left = (x <= margin_x)
        touches_top = (y <= margin_y)
        touches_right = ((x + w) >= img_w - margin_x)
        touches_bottom = ((y + h) >= img_h - margin_y)
        borders_touched = sum([touches_left, touches_top, touches_right, touches_bottom])

        # Centeredness: distance from center normalized by image diagonal
        cx = x + w / 2.0
        cy = y + h / 2.0
        center_dist = np.hypot(cx - (img_w / 2.0), cy - (img_h / 2.0)) / np.hypot(img_w / 2.0, img_h / 2.0)

        # Approximate polygon to test for rectangular/quad package form
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
        is_quad = (len(approx) == 4)

        # Multi-factor score calculation
        # 1. Prominence score: package typically occupies 25% to 92% of frame
        if 0.25 <= area_ratio <= 0.92:
            prominence_score = 1.0 - abs(area_ratio - 0.60) * 0.5
        elif area_ratio < 0.25:
            prominence_score = max(0.1, (area_ratio - 0.10) / 0.15)
        else:
            prominence_score = max(0.2, (0.99 - area_ratio) / 0.07)

        # 2. Solidity score: packaged commodities have high solidity
        solidity_score = max(0.0, min(1.0, (solidity - 0.35) / 0.60))

        # 3. Centrality score: packages are typically centered
        centrality_score = max(0.0, 1.0 - (center_dist * 1.2))

        # 4. Border clearance score: packages with visible margins are more reliably isolated
        if borders_touched == 0:
            border_score = 1.0
        elif borders_touched == 1:
            border_score = 0.75
        elif borders_touched == 2:
            border_score = 0.45
        else:
            border_score = 0.15

        # 5. Extent / Rectangularity score
        rect_score = 1.0 if is_quad else max(0.3, min(1.0, extent))

        composite_score = (
            0.35 * prominence_score
            + 0.25 * solidity_score
            + 0.20 * border_score
            + 0.10 * centrality_score
            + 0.10 * rect_score
        )

        candidates.append({
            "bbox": (x, y, x + w, y + h),
            "width": w,
            "height": h,
            "area_ratio": area_ratio,
            "solidity": solidity,
            "score": composite_score,
            "is_quad": is_quad,
            "borders_touched": borders_touched,
            "method": method_name,
        })

    return candidates


def detect_package_boundary(image_bgr: np.ndarray) -> PackageBoundaryResult:
    """Detects visible physical package boundary in pixel coordinates using multi-strategy CV analysis.

    Strategies:
    1. Background contrast distance from image borders/corners.
    2. Adaptive thresholding and morphological gradient on smoothed image.
    3. Canny edge contour analysis with morphological closure.

    Returns:
        PackageBoundaryResult with pixel coordinates, confidence, and quality assessment.
    """
    if image_bgr is None or image_bgr.size == 0:
        return PackageBoundaryResult(
            x_min=0, y_min=0, x_max=0, y_max=0,
            width_px=0.0, height_px=0.0,
            confidence=0.0, quality="failed",
            method="invalid_image",
            notes="Input image is empty or invalid.",
        )

    img_h, img_w = image_bgr.shape[:2]
    if img_w < 50 or img_h < 50:
        return PackageBoundaryResult(
            x_min=0, y_min=0, x_max=img_w, y_max=img_h,
            width_px=float(img_w), height_px=float(img_h),
            confidence=0.10, quality="failed",
            method="too_small",
            notes=f"Image resolution {img_w}x{img_h} is too small for package boundary detection.",
        )

    candidates: list[dict] = []

    # ==========================================================================
    # STRATEGY 1: Background Color Contrast Distance
    # ==========================================================================
    # Sample border pixels (5% margin) to estimate background color
    margin_x = max(1, int(img_w * 0.05))
    margin_y = max(1, int(img_h * 0.05))

    top_strip = image_bgr[:margin_y, :]
    bottom_strip = image_bgr[-margin_y:, :]
    left_strip = image_bgr[:, :margin_x]
    right_strip = image_bgr[:, -margin_x:]

    border_pixels = np.vstack([
        top_strip.reshape(-1, 3),
        bottom_strip.reshape(-1, 3),
        left_strip.reshape(-1, 3),
        right_strip.reshape(-1, 3),
    ])

    bg_color = np.median(border_pixels, axis=0)  # Median BGR background estimate

    # Color difference from background in Lab space for perceptual uniformity
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2Lab)
    bg_lab = cv2.cvtColor(np.uint8([[bg_color]]), cv2.COLOR_BGR2Lab)[0, 0]

    color_dist = np.linalg.norm(lab.astype(np.float32) - bg_lab.astype(np.float32), axis=2)
    color_dist_norm = np.clip((color_dist / np.max(color_dist + 1e-5)) * 255, 0, 255).astype(np.uint8)

    # Smooth color distance
    blurred_dist = cv2.GaussianBlur(color_dist_norm, (11, 11), 0)
    _, bg_mask = cv2.threshold(blurred_dist, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    candidates.extend(_extract_candidates_from_mask(bg_mask, img_w, img_h, "color_contrast"))

    # ==========================================================================
    # STRATEGY 2: Canny Edge Contours on Heavy-Blurred Image
    # ==========================================================================
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred_gray = cv2.GaussianBlur(gray, (13, 13), 0)

    median_val = float(np.median(blurred_gray))
    lower_thresh = int(max(10, 0.60 * median_val))
    upper_thresh = int(min(250, 1.40 * median_val))
    edges = cv2.Canny(blurred_gray, lower_thresh, upper_thresh)

    edge_ksize = max(5, int(min(img_w, img_h) * 0.02) | 1)
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (edge_ksize, edge_ksize))
    dilated_edges = cv2.dilate(edges, edge_kernel, iterations=2)

    candidates.extend(_extract_candidates_from_mask(dilated_edges, img_w, img_h, "canny_edges"))

    # ==========================================================================
    # STRATEGY 3: Otsu Grayscale Thresholding (both polarities)
    # ==========================================================================
    _, otsu_mask1 = cv2.threshold(blurred_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_mask2 = cv2.bitwise_not(otsu_mask1)

    candidates.extend(_extract_candidates_from_mask(otsu_mask1, img_w, img_h, "otsu_direct"))
    candidates.extend(_extract_candidates_from_mask(otsu_mask2, img_w, img_h, "otsu_inverted"))

    # ==========================================================================
    # EVALUATION & SELECTION
    # ==========================================================================
    if not candidates:
        # Fallback: No candidate exceeded minimum package thresholds
        return PackageBoundaryResult(
            x_min=0,
            y_min=0,
            x_max=img_w,
            y_max=img_h,
            width_px=float(img_w),
            height_px=float(img_h),
            confidence=0.35,
            quality="low",
            method="frame_fill_fallback",
            notes="No isolated package boundary could be detected from background; frame fill fallback used with low confidence.",
        )

    # Sort candidates by composite score descending
    candidates.sort(key=lambda c: c["score"], reverse=True)
    best = candidates[0]

    bx1, by1, bx2, by2 = best["bbox"]
    bw = float(bx2 - bx1)
    bh = float(by2 - by1)
    score = best["score"]
    area_ratio = best["area_ratio"]

    # Quality and confidence grading
    if score >= 0.70 and best["borders_touched"] <= 1:
        confidence = min(0.95, round(0.80 + (score - 0.70) * 0.5, 2))
        quality = "good"
    elif score >= 0.50:
        confidence = min(0.82, round(0.68 + (score - 0.50) * 0.7, 2))
        quality = "moderate"
    else:
        confidence = min(0.65, round(0.40 + score * 0.5, 2))
        quality = "low"

    # If candidate touches all 4 borders or covers >97% of the image, warn that it occupies whole frame
    if best["borders_touched"] >= 3 or area_ratio >= 0.95:
        if quality == "good":
            quality = "moderate"
        confidence = min(confidence, 0.72)
        note = f"Package candidate occupies {area_ratio*100:.1f}% of frame touching image borders ({best['method']})."
    else:
        note = f"Detected package boundary via {best['method']} occupying {area_ratio*100:.1f}% of image."

    return PackageBoundaryResult(
        x_min=int(bx1),
        y_min=int(by1),
        x_max=int(bx2),
        y_max=int(by2),
        width_px=bw,
        height_px=bh,
        confidence=confidence,
        quality=quality,
        method=f"auto_{best['method']}",
        notes=note,
    )
