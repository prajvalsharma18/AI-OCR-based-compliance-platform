"""Inspection source image storage and retrieval service.

SIH 2026 PS 26034: Automated Compliance Checker for Packaged Commodities

Provides secure, deterministic filesystem storage for the exact original package
image uploaded by the user at the start of the inspection workflow.

Guarantees:
- Exact original bytes preserved without any PIL re-encoding, compression, rotation, or conversion.
- Deterministic storage layout:
    inspections/
    └── <inspection_id>/
        ├── original_image.<ext>
        ├── inspection.json
        └── report.pdf
- Path traversal protection and strict inspection_id sanitization.
- SHA-256 cryptographic hashing of raw uploaded bytes before any downstream processing.
- Provenance verification: PDF report generator retrieves the exact stored source image.
"""

import hashlib
import json
import logging
from pathlib import Path
import re
import time
from typing import Any, Dict, Optional, Tuple, Union
from app.config import settings

logger = logging.getLogger(__name__)

# Allowed characters in inspection_id to prevent directory traversal
SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")

# Recognized safe image extensions and magic bytes
MAGIC_BYTES_MAP = {
    b"\x89PNG\r\n\x1a\n": ".png",
    b"\xff\xd8\xff": ".jpg",
    b"RIFF": ".webp",
    b"GIF87a": ".gif",
    b"GIF89a": ".gif",
    b"BM": ".bmp",
    b"MM\x00*": ".tiff",
    b"II*\x00": ".tiff",
}

DEFAULT_STORAGE_ROOT = Path(settings.REPORTS_DIR) / "inspections"


class ImageStorageError(Exception):
    """Base exception for image storage service operations."""

    def __init__(self, code: str, message: str, status_code: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class ImageStorageService:
    """Manages persistent inspection source image storage and cryptographic verification."""

    def __init__(self, storage_root: Optional[Union[str, Path]] = None) -> None:
        self.storage_root = Path(storage_root) if storage_root else DEFAULT_STORAGE_ROOT

    def sanitize_inspection_id(self, inspection_id: str) -> str:
        """Validates and sanitizes inspection_id against path traversal attacks."""
        if not inspection_id or not isinstance(inspection_id, str):
            raise ImageStorageError(
                code="INVALID_INSPECTION_ID",
                message="Inspection ID must be a non-empty string.",
                status_code=400,
            )

        clean_id = inspection_id.strip()
        if (
            not SAFE_ID_PATTERN.match(clean_id)
            or ".." in clean_id
            or "/" in clean_id
            or "\\" in clean_id
        ):
            raise ImageStorageError(
                code="PATH_TRAVERSAL_DETECTED",
                message=f"Invalid characters or traversal sequence detected in inspection ID: '{inspection_id}'",
                status_code=400,
            )

        # Verify resolved directory remains strictly within storage root
        target_path = (self.storage_root.resolve() / clean_id).resolve()
        base_path = self.storage_root.resolve()
        try:
            target_path.relative_to(base_path)
        except ValueError:
            raise ImageStorageError(
                code="PATH_TRAVERSAL_DETECTED",
                message=f"Path traversal detected: '{inspection_id}' resolves outside storage root.",
                status_code=400,
            )

        return clean_id

    def generate_inspection_id(self, prefix: str = "INSP") -> str:
        """Generates a deterministic unique inspection ID."""
        return f"{prefix}-{int(time.time() * 1000)}"

    def calculate_sha256(self, image_bytes: bytes) -> str:
        """Computes the SHA-256 cryptographic digest of the exact byte sequence."""
        return hashlib.sha256(image_bytes).hexdigest()

    def determine_safe_extension(self, filename: Optional[str], image_bytes: bytes) -> str:
        """Determines the canonical file extension from filename or raw byte magic headers."""
        if filename:
            suffix = Path(filename).suffix.lower()
            if suffix in (".jpg", ".jpeg"):
                return ".jpg"
            if suffix in (".png", ".webp", ".bmp", ".tiff", ".tif", ".gif"):
                return suffix

        # Fallback: inspect initial magic bytes
        for magic, ext in MAGIC_BYTES_MAP.items():
            if image_bytes.startswith(magic):
                return ext

        return ".jpg"

    def get_inspection_dir(self, inspection_id: str) -> Path:
        """Resolves the inspection directory path with traversal safety."""
        clean_id = self.sanitize_inspection_id(inspection_id)
        return (self.storage_root / clean_id).resolve()

    def save_original(
        self,
        image_bytes: bytes,
        filename: Optional[str],
        inspection_id: str,
    ) -> str:
        """Stores the exact original uploaded image bytes without modification.

        Args:
            image_bytes: Raw uncompressed bytes directly from the user's upload.
            filename: Original uploaded file name (for extension guidance).
            inspection_id: Unique inspection identifier.

        Returns:
            Absolute file path to the stored original image.
        """
        if not image_bytes or len(image_bytes) == 0:
            raise ImageStorageError(
                code="EMPTY_IMAGE_BYTES",
                message="Cannot save empty image bytes.",
                status_code=400,
            )

        clean_id = self.sanitize_inspection_id(inspection_id)
        insp_dir = self.storage_root / clean_id
        insp_dir.mkdir(parents=True, exist_ok=True)

        ext = self.determine_safe_extension(filename, image_bytes)
        target_file = insp_dir / f"original_image{ext}"

        # Write exact original bytes directly to disk
        target_file.write_bytes(image_bytes)
        logger.info(
            "Stored original inspection image for %s at %s (%d bytes)",
            clean_id,
            target_file,
            len(image_bytes),
        )

        return str(target_file.resolve())

    def get_original(self, inspection_id: str) -> Optional[str]:
        """Retrieves the file path of the stored original image for the inspection."""
        clean_id = self.sanitize_inspection_id(inspection_id)
        insp_dir = self.storage_root / clean_id
        if not insp_dir.is_dir():
            return None

        # Search for original_image.*
        candidates = list(insp_dir.glob("original_image.*"))
        if candidates:
            return str(candidates[0].resolve())

        return None

    def get_original_bytes(self, inspection_id: str) -> Optional[bytes]:
        """Reads and returns the exact stored bytes of the original package image."""
        path_str = self.get_original(inspection_id)
        if not path_str:
            return None

        file_path = Path(path_str)
        if file_path.exists():
            return file_path.read_bytes()

        return None

    def get_original_sha256(self, inspection_id: str) -> Optional[str]:
        """Computes SHA-256 digest of the stored original image file."""
        data = self.get_original_bytes(inspection_id)
        if data is None:
            return None
        return self.calculate_sha256(data)

    def exists(self, inspection_id: str) -> bool:
        """Returns True if a stored original image exists for the given inspection ID."""
        try:
            return self.get_original(inspection_id) is not None
        except Exception:
            return False

    def verify_hash(
        self,
        inspection_id: str,
        expected_sha256: Optional[str],
    ) -> Tuple[bool, str, Optional[str]]:
        """Verifies stored image hash against expected SHA-256.

        Returns:
            Tuple of (is_valid, verification_status, computed_hash)
            verification_status is one of: "MATCHED", "MISMATCH", "UNVERIFIED", "NOT_FOUND"
        """
        stored_bytes = self.get_original_bytes(inspection_id)
        if stored_bytes is None:
            return False, "NOT_FOUND", None

        computed = self.calculate_sha256(stored_bytes)

        if not expected_sha256:
            return True, "UNVERIFIED", computed

        if computed.lower() == expected_sha256.strip().lower():
            return True, "MATCHED", computed

        return False, "MISMATCH", computed

    def save_inspection_json(self, inspection_id: str, inspection_data: Any) -> str:
        """Persists the Unified Inspection JSON snapshot in the inspection folder."""
        clean_id = self.sanitize_inspection_id(inspection_id)
        insp_dir = self.storage_root / clean_id
        insp_dir.mkdir(parents=True, exist_ok=True)

        target_file = insp_dir / "inspection.json"

        if hasattr(inspection_data, "model_dump"):
            dumped = inspection_data.model_dump(mode="json")
        elif hasattr(inspection_data, "dict"):
            dumped = inspection_data.dict()
        elif isinstance(inspection_data, dict):
            dumped = inspection_data
        elif isinstance(inspection_data, str):
            dumped = json.loads(inspection_data)
        else:
            dumped = json.loads(str(inspection_data))

        target_file.write_text(json.dumps(dumped, indent=2), encoding="utf-8")
        return str(target_file.resolve())

    def get_inspection_json_path(self, inspection_id: str) -> Path:
        """Returns standard destination path for the inspection JSON snapshot."""
        clean_id = self.sanitize_inspection_id(inspection_id)
        insp_dir = self.storage_root / clean_id
        insp_dir.mkdir(parents=True, exist_ok=True)
        return (insp_dir / "inspection.json").resolve()

    def get_logical_source_path(self, inspection_id: str, filename: Optional[str] = None) -> str:
        """Returns a safe logical identifier for metadata (e.g. 'inspections/<id>/original_image.jpg')."""
        clean_id = self.sanitize_inspection_id(inspection_id)
        orig = self.get_original(clean_id)
        if orig:
            suffix = Path(orig).suffix
            return f"inspections/{clean_id}/original_image{suffix}"
        ext = Path(filename).suffix if filename else ".jpg"
        return f"inspections/{clean_id}/original_image{ext}"

    def get_report_path(self, inspection_id: str) -> Path:
        """Returns standard destination path for the inspection PDF report."""
        clean_id = self.sanitize_inspection_id(inspection_id)
        insp_dir = self.storage_root / clean_id
        insp_dir.mkdir(parents=True, exist_ok=True)
        return (insp_dir / "report.pdf").resolve()
