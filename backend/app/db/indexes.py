"""MongoDB index definitions for the inspections collection.

Ensures performance, uniqueness of inspection_id, and rapid date/status filtering
without creating redundant or heavy indexes.
"""

import logging
from typing import Optional
from pymongo import ASCENDING, DESCENDING, IndexModel
from pymongo.database import Database
from pymongo.errors import PyMongoError

logger = logging.getLogger(__name__)

COLLECTION_INSPECTIONS = "inspections"


def get_inspection_indexes() -> list[IndexModel]:
    """Returns the list of index definitions for the inspections collection."""
    return [
        # 1. Unique index on inspection_id
        IndexModel(
            [("inspection_id", ASCENDING)],
            unique=True,
            name="idx_inspection_id_unique",
        ),
        # 2. Descending index on created_at for chronologically sorted history
        IndexModel(
            [("created_at", DESCENDING)],
            name="idx_created_at_desc",
        ),
        # 3. Brand name index for quick text/prefix lookups
        IndexModel(
            [("metadata.brand_name", ASCENDING)],
            name="idx_metadata_brand_name",
        ),
        # 4. Generic commodity name index
        IndexModel(
            [("metadata.generic_name", ASCENDING)],
            name="idx_metadata_generic_name",
        ),
        # 5. Package type index (retail, wholesale, combination_pack)
        IndexModel(
            [("metadata.package_type", ASCENDING)],
            name="idx_metadata_package_type",
        ),
        # 6. Non-compliant count index for filtering violations
        IndexModel(
            [("summary.non_compliant_findings_count", ASCENDING)],
            name="idx_summary_non_compliant",
        ),
        # 7. Review status index (Pending, Verified, Issue Raised, N/A)
        IndexModel(
            [("review.status", ASCENDING)],
            name="idx_review_status",
        ),
        # 8. Compound index for date-range and review status filtering
        IndexModel(
            [("created_at", DESCENDING), ("review.status", ASCENDING)],
            name="idx_created_at_review_status",
        ),
    ]


def ensure_indexes(db: Optional[Database]) -> bool:
    """Creates or updates all required indexes on the inspections collection.

    Args:
        db: The pymongo Database handle.

    Returns:
        bool: True if indexes were created or verified, False if database is unavailable.
    """
    if db is None:
        return False

    try:
        coll = db[COLLECTION_INSPECTIONS]
        indexes = get_inspection_indexes()
        coll.create_indexes(indexes)
        logger.info("Successfully verified MongoDB indexes on '%s' collection.", COLLECTION_INSPECTIONS)
        return True
    except PyMongoError as exc:
        logger.warning("Could not create indexes on '%s' collection: %s", COLLECTION_INSPECTIONS, exc)
        return False
    except Exception as exc:
        logger.warning("Unexpected error ensuring MongoDB indexes: %s", exc)
        return False
