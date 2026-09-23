"""Unit tests for ImageStorageService.

SIH 2026 PS 26034: Automated Compliance Checker for Packaged Commodities
Inspection Source Image Storage and Provenance Testing
"""

import hashlib
import json
from pathlib import Path
import pytest

from app.services.image_storage_service import ImageStorageError, ImageStorageService


@pytest.fixture
def storage_service(tmp_path):
    """ImageStorageService instance sandboxed to tmp_path."""
    return ImageStorageService(storage_root=tmp_path / "reports" / "inspections")


@pytest.fixture
def sample_png_bytes():
    """Valid small PNG byte sequence."""
    return (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0"
        b"\x00\x00\x03\x01\x01\x00\x18\xdd\x8d\xb0\x00\x00\x00\x00IEND\xaeB`\x82"
    )


@pytest.fixture
def sample_jpg_bytes():
    """Valid minimal JPEG byte sequence."""
    return b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xdb\x00C\x00\xff\xd9"


# 1. Original bytes saved exactly
def test_1_original_bytes_saved_exactly(storage_service, sample_png_bytes):
    insp_id = "INSP-TEST-001"
    saved_path = storage_service.save_original(sample_png_bytes, "test.png", insp_id)
    assert Path(saved_path).exists()
    assert Path(saved_path).read_bytes() == sample_png_bytes


# 2. Stored bytes == uploaded bytes
def test_2_stored_bytes_equal_uploaded_bytes(storage_service, sample_jpg_bytes):
    insp_id = "INSP-TEST-002"
    storage_service.save_original(sample_jpg_bytes, "photo.jpg", insp_id)
    retrieved = storage_service.get_original_bytes(insp_id)
    assert retrieved == sample_jpg_bytes


# 3. SHA-256 correct
def test_3_sha256_calculation(storage_service, sample_png_bytes):
    computed = storage_service.calculate_sha256(sample_png_bytes)
    expected = hashlib.sha256(sample_png_bytes).hexdigest()
    assert computed == expected

    storage_service.save_original(sample_png_bytes, "img.png", "INSP-TEST-003")
    assert storage_service.get_original_sha256("INSP-TEST-003") == expected


# 4. Original filename retained / safe extension determined
def test_4_safe_extension_determination(storage_service, sample_png_bytes, sample_jpg_bytes):
    # From magic bytes
    ext_png = storage_service.determine_safe_extension(None, sample_png_bytes)
    assert ext_png == ".png"

    ext_jpg = storage_service.determine_safe_extension(None, sample_jpg_bytes)
    assert ext_jpg == ".jpg"

    # From filename suffix
    ext_from_fn = storage_service.determine_safe_extension("upload.jpeg", sample_jpg_bytes)
    assert ext_from_fn == ".jpg"


# 5. Invalid / unsupported extension handled safely
def test_5_unknown_extension_fallback(storage_service):
    raw_arbitrary = b"CUSTOMDATA12345"
    ext = storage_service.determine_safe_extension("unknown.xyz", raw_arbitrary)
    assert ext in (".xyz", ".jpg")


# 6. Path traversal rejected
def test_6_path_traversal_rejected(storage_service, sample_png_bytes):
    malicious_ids = [
        "../traversal",
        "..\\windows_traversal",
        "nested/../../secret",
        "valid/id",
        "valid\\id",
        "/absolute/path",
        "C:\\Windows\\System32",
        "id with spaces",
        "id;injection",
    ]
    for bad_id in malicious_ids:
        with pytest.raises(ImageStorageError) as exc_info:
            storage_service.sanitize_inspection_id(bad_id)
        assert exc_info.value.code in ("PATH_TRAVERSAL_DETECTED", "INVALID_INSPECTION_ID")


# 7. Inspection ID sanitization
def test_7_inspection_id_sanitization(storage_service):
    assert storage_service.sanitize_inspection_id("INSP-12345_abc") == "INSP-12345_abc"
    with pytest.raises(ImageStorageError):
        storage_service.sanitize_inspection_id("")


# 8. Storage path stays inside base directory
def test_8_storage_path_stays_inside_base_dir(storage_service):
    insp_dir = storage_service.get_inspection_dir("INSP-CLEAN-001")
    assert storage_service.storage_root.resolve() in insp_dir.parents or insp_dir == storage_service.storage_root.resolve()


# 9. get_original works
def test_9_get_original_retrieval(storage_service, sample_png_bytes):
    insp_id = "INSP-TEST-009"
    storage_service.save_original(sample_png_bytes, "img.png", insp_id)
    path = storage_service.get_original(insp_id)
    assert path is not None
    assert Path(path).exists()
    assert storage_service.exists(insp_id) is True


# 10. Missing original image handled
def test_10_missing_original_image(storage_service):
    assert storage_service.get_original("INSP-NONEXISTENT") is None
    assert storage_service.get_original_bytes("INSP-NONEXISTENT") is None
    assert storage_service.get_original_sha256("INSP-NONEXISTENT") is None
    assert storage_service.exists("INSP-NONEXISTENT") is False


# 11. Report path resolution works
def test_11_report_path_resolution(storage_service):
    report_path = storage_service.get_report_path("INSP-TEST-011")
    assert report_path.name == "report.pdf"
    assert "INSP-TEST-011" in str(report_path)


# 12. Inspection JSON path resolution works and saves snapshot
def test_12_inspection_json_path_resolution_and_save(storage_service):
    json_path = storage_service.get_inspection_json_path("INSP-TEST-012")
    assert json_path.name == "inspection.json"

    data = {"inspection_id": "INSP-TEST-012", "status": "PASS"}
    saved = storage_service.save_inspection_json("INSP-TEST-012", data)
    assert Path(saved).exists()
    assert json.loads(Path(saved).read_text(encoding="utf-8")) == data


# 13. Logical source path generation
def test_13_logical_source_path(storage_service, sample_png_bytes):
    insp_id = "INSP-TEST-013"
    storage_service.save_original(sample_png_bytes, "package.png", insp_id)
    logical = storage_service.get_logical_source_path(insp_id, "package.png")
    assert logical == f"inspections/{insp_id}/original_image.png"
