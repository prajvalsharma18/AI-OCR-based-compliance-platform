"""Image processing and validation utilities.

Handles file format validation, decoding verification, and byte preparation
in-memory without saving uploaded images permanently to disk.
"""

import io
from typing import NamedTuple, Set
from fastapi import UploadFile
from PIL import Image, UnidentifiedImageError

from app.config import settings


class ImageValidationResult(NamedTuple):
    content: bytes
    mime_type: str
    width: int
    height: int
    image_format: str


class ImageValidationError(Exception):
    """Raised when an uploaded file fails format, decoding, or size checks."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def get_supported_mime_types() -> Set[str]:
    """Returns normalized set of allowed image MIME types."""
    types = set(settings.ALLOWED_IMAGE_TYPES)
    # Ensure JPEG alias is recognized
    if "image/jpeg" in types:
        types.add("image/jpg")
    return types


async def validate_and_load_image(file: UploadFile) -> ImageValidationResult:
    """Validates an uploaded image file completely in-memory.

    Checks:
    1. File presence and filename.
    2. Declared and detected MIME type against supported formats.
    3. File size constraints.
    4. Image integrity via Pillow decode.

    Returns:
        ImageValidationResult containing image bytes, MIME type, dimensions, and format.

    Raises:
        ImageValidationError: If any validation check fails.
    """
    if not file or not file.filename:
        raise ImageValidationError(
            code="MISSING_FILE",
            message="No image file was provided in the upload."
        )

    content_type = (file.content_type or "").lower().strip()
    if content_type == "image/jpg":
        content_type = "image/jpeg"

    supported_types = get_supported_mime_types()
    if content_type not in supported_types:
        allowed_list = ", ".join(sorted(settings.ALLOWED_IMAGE_TYPES))
        raise ImageValidationError(
            code="UNSUPPORTED_MEDIA_TYPE",
            message=f"Unsupported image type '{file.content_type}'. Allowed types: {allowed_list}."
        )

    # Read bytes asynchronously
    try:
        content = await file.read()
    except Exception as exc:
        raise ImageValidationError(
            code="READ_ERROR",
            message=f"Failed to read image stream: {str(exc)}"
        ) from exc

    if not content or len(content) == 0:
        raise ImageValidationError(
            code="EMPTY_FILE",
            message="Uploaded image file is empty (0 bytes)."
        )

    max_bytes = settings.max_image_size_bytes
    if len(content) > max_bytes:
        max_mb = settings.MAX_IMAGE_SIZE_MB
        raise ImageValidationError(
            code="FILE_TOO_LARGE",
            message=f"File size ({len(content) / (1024 * 1024):.2f} MB) exceeds maximum allowed limit of {max_mb:.1f} MB."
        )

    # Decode and verify using Pillow in-memory
    try:
        image_stream = io.BytesIO(content)
        with Image.open(image_stream) as img:
            img.verify()

        # Reopen to read dimensions and format after verify() closes the stream
        image_stream.seek(0)
        with Image.open(image_stream) as img:
            width, height = img.size
            img_format = (img.format or "").upper()

        if width <= 0 or height <= 0:
            raise ImageValidationError(
                code="INVALID_DIMENSIONS",
                message="Image dimensions must be greater than zero."
            )

    except (UnidentifiedImageError, ValueError, OSError) as exc:
        raise ImageValidationError(
            code="CORRUPTED_IMAGE",
            message=f"Image file is corrupted or cannot be decoded: {str(exc)}"
        ) from exc

    return ImageValidationResult(
        content=content,
        mime_type=content_type,
        width=width,
        height=height,
        image_format=img_format or "JPEG"
    )
