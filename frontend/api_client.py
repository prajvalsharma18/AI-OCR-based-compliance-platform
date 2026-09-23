"""API Client for SIH 2026 PS 26034 Compliance Checker Backend.

Communicates exclusively with FastAPI endpoints:
- GET  /health
- POST /api/v1/inspection
- POST /api/v1/report/pdf
"""

import logging
import os
from typing import Any, Dict, Optional, Tuple, Union
import requests

logger = logging.getLogger(__name__)

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8001").rstrip("/")


def check_health(timeout: float = 3.0) -> Dict[str, Any]:
    """Queries GET /health to determine backend connectivity."""
    url = f"{BACKEND_URL}/health"
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200:
            return {
                "connected": True,
                "status": "CONNECTED",
                "data": resp.json(),
            }
        return {
            "connected": False,
            "status": "OFFLINE",
            "error": f"Backend returned HTTP {resp.status_code}",
        }
    except requests.exceptions.Timeout:
        return {
            "connected": False,
            "status": "OFFLINE",
            "error": "Connection timed out connecting to FastAPI backend.",
        }
    except requests.exceptions.ConnectionError:
        return {
            "connected": False,
            "status": "OFFLINE",
            "error": f"Cannot connect to backend at {BACKEND_URL}. Ensure FastAPI is running on port 8001.",
        }
    except Exception as exc:
        return {
            "connected": False,
            "status": "OFFLINE",
            "error": f"Unexpected error checking backend health: {str(exc)}",
        }


def run_inspection(
    image_bytes: bytes,
    filename: str,
    brand_name: Optional[str] = None,
    generic_name: Optional[str] = None,
    package_type: Optional[str] = "retail",
    package_width_mm: Optional[float] = None,
    package_height_mm: Optional[float] = None,
    pdp_area_cm2: Optional[float] = None,
    rule_source_mode: str = "sih_ps_26034",
    timeout: float = 120.0,
) -> Dict[str, Any]:
    """Sends packaging image and optional metadata to POST /api/v1/inspection.

    Returns:
        Standard envelope dictionary:
        {"success": True, "data": {...}} or {"success": False, "error": {"code": ..., "message": ...}}
    """
    url = f"{BACKEND_URL}/api/v1/inspection"

    # Infer mime type
    ext = os.path.splitext(filename)[1].lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    content_type = mime_map.get(ext, "image/jpeg")

    files = {
        "image": (filename, image_bytes, content_type)
    }

    # Include only fields supplied by user
    data: Dict[str, Any] = {
        "rule_source_mode": rule_source_mode,
    }
    if brand_name and brand_name.strip():
        data["brand_name"] = brand_name.strip()
    if generic_name and generic_name.strip():
        data["generic_name"] = generic_name.strip()
    if package_type and package_type.strip():
        data["package_type"] = package_type.strip()
    if package_width_mm is not None and package_width_mm > 0:
        data["package_width_mm"] = str(package_width_mm)
    if package_height_mm is not None and package_height_mm > 0:
        data["package_height_mm"] = str(package_height_mm)
    if pdp_area_cm2 is not None and pdp_area_cm2 > 0:
        data["pdp_area_cm2"] = str(pdp_area_cm2)

    try:
        resp = requests.post(url, files=files, data=data, timeout=timeout)
        try:
            body = resp.json()
        except Exception:
            body = None

        if resp.status_code == 200 and body:
            return body

        # Handle 4xx / 5xx responses with structured error body if available
        if body and isinstance(body, dict) and "error" in body:
            return body

        error_message = f"Backend returned HTTP {resp.status_code}"
        if resp.text:
            error_message += f": {resp.text[:200]}"
        return {
            "success": False,
            "error": {
                "code": f"HTTP_{resp.status_code}",
                "message": error_message,
            },
        }

    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": {
                "code": "REQUEST_TIMEOUT",
                "message": "Inspection request timed out. The Vision LLM or CV measurement took too long.",
            },
        }
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": {
                "code": "BACKEND_UNAVAILABLE",
                "message": f"Inspection service is unavailable. Please verify that FastAPI is running on {BACKEND_URL}.",
            },
        }
    except Exception as exc:
        return {
            "success": False,
            "error": {
                "code": "CLIENT_ERROR",
                "message": f"Failed to execute inspection request: {str(exc)}",
            },
        }


def get_pdf_report(
    inspection_data: Dict[str, Any],
    image_base64: Optional[str] = None,
    timeout: float = 60.0,
) -> Tuple[bool, Union[bytes, str]]:
    """Calls POST /api/v1/report/pdf to generate and stream the statutory PDF report.

    Args:
        inspection_data: The Unified Inspection JSON data dictionary.
        image_base64: Optional base64 encoded packaging photo.
        timeout: Request timeout in seconds.

    Returns:
        (True, pdf_bytes) on success, or (False, error_message_str) on failure.
    """
    url = f"{BACKEND_URL}/api/v1/report/pdf"
    payload: Dict[str, Any] = {
        "inspection": inspection_data
    }
    if image_base64:
        payload["image_base64"] = image_base64

    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        if resp.status_code == 200:
            return True, resp.content

        # Error response
        try:
            err_json = resp.json()
            err_msg = err_json.get("error", {}).get("message", f"HTTP {resp.status_code}")
        except Exception:
            err_msg = f"HTTP {resp.status_code}: {resp.text[:200]}"
        return False, err_msg

    except requests.exceptions.Timeout:
        return False, "PDF generation timed out."
    except requests.exceptions.ConnectionError:
        return False, f"Could not connect to PDF report service at {BACKEND_URL}."
    except Exception as exc:
        return False, f"Error generating PDF report: {str(exc)}"
