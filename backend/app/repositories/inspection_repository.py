"""MongoDB Inspection Repository for SIH 2026 PS 26034.

Provides clean data access, index-backed queries, idempotency, and review updates
without exposing raw MongoDB query primitives to HTTP endpoints.
"""

from datetime import datetime, timezone
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from pymongo.database import Database
from pymongo.errors import PyMongoError

from app.db.indexes import COLLECTION_INSPECTIONS, ensure_indexes
from app.db.mongo import get_database

logger = logging.getLogger(__name__)


class InspectionRepositoryError(Exception):
    """Exception raised for repository and database access errors."""

    def __init__(self, message: str, code: str = "DATABASE_ERROR"):
        super().__init__(message)
        self.code = code
        self.message = message


class InspectionRepository:
    """Repository handling CRUD operations on the 'inspections' MongoDB collection."""

    def __init__(self, db: Optional[Database] = None):
        self._db = db

    def _get_db(self) -> Database:
        """Retrieves active database handle or raises InspectionRepositoryError if unavailable."""
        db = self._db if self._db is not None else get_database()
        if db is None:
            raise InspectionRepositoryError(
                code="MONGODB_UNAVAILABLE",
                message="MongoDB database handle is unavailable. Ensure MongoDB server is running.",
            )
        return db

    def _get_collection(self):
        """Returns the inspections collection handle."""
        return self._get_db()[COLLECTION_INSPECTIONS]

    def create_inspection(self, document: Dict[str, Any]) -> bool:
        """Persists or updates an inspection document idempotently by inspection_id.

        Args:
            document: Unified inspection document dictionary.

        Returns:
            bool: True on successful persistence.

        Raises:
            InspectionRepositoryError: On validation failure or database errors.
        """
        inspection_id = document.get("inspection_id")
        if not inspection_id or not isinstance(inspection_id, str):
            raise InspectionRepositoryError(
                code="INVALID_INSPECTION_ID",
                message="Document must contain a valid string 'inspection_id'.",
            )

        now_utc = datetime.now(timezone.utc)

        # Ensure timestamps are proper UTC datetimes
        created_at = document.get("created_at")
        if not isinstance(created_at, datetime):
            if isinstance(created_at, str):
                try:
                    document["created_at"] = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                except Exception:
                    document["created_at"] = now_utc
            else:
                document["created_at"] = now_utc

        document["updated_at"] = now_utc
        document["schema_version"] = document.get("schema_version", 1)

        # Default review block if missing
        if "review" not in document or not isinstance(document["review"], dict):
            document["review"] = {
                "status": "Pending",
                "notes": "",
                "updated_at": None,
            }

        try:
            coll = self._get_collection()
            # Idempotent upsert by unique inspection_id
            coll.replace_one(
                {"inspection_id": inspection_id},
                document,
                upsert=True,
            )
            logger.info("Successfully persisted inspection document in MongoDB: %s", inspection_id)
            return True
        except PyMongoError as exc:
            logger.error("MongoDB persistence failed for %s: %s", inspection_id, exc)
            raise InspectionRepositoryError(
                code="MONGO_WRITE_FAILED",
                message=f"Failed to persist inspection in MongoDB: {str(exc)}",
            ) from exc

    def get_inspection_by_id(self, inspection_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single complete inspection document by inspection_id.

        Args:
            inspection_id: The unique inspection ID string.

        Returns:
            Optional[dict]: Full inspection document or None if not found.
        """
        clean_id = inspection_id.strip()
        try:
            coll = self._get_collection()
            doc = coll.find_one({"inspection_id": clean_id})
            if not doc:
                return None

            # Convert internal ObjectId to string for JSON serialization
            if "_id" in doc:
                doc["_id"] = str(doc["_id"])
            return doc
        except PyMongoError as exc:
            logger.error("MongoDB lookup error for %s: %s", clean_id, exc)
            raise InspectionRepositoryError(
                code="MONGO_READ_FAILED",
                message=f"Failed to query MongoDB for {clean_id}: {str(exc)}",
            ) from exc

    def list_inspections(
        self,
        limit: int = 20,
        skip: int = 0,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        status: Optional[str] = None,
        brand_name: Optional[str] = None,
        package_type: Optional[str] = None,
        inspection_id: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Lists compact inspection records with optional date, brand, package, and status filters.

        Args:
            limit: Maximum records to return (capped at 100).
            skip: Offset of records to skip.
            date_from: Filter inspections created at or after this UTC timestamp.
            date_to: Filter inspections created at or before this UTC timestamp.
            status: Filter by review status (Pending, Verified, Issue Raised, N/A) or outcome (PASS, NON-COMPLIANT).
            brand_name: Substring or exact match on packaging brand name.
            package_type: Filter by package type (retail, wholesale, combination_pack).

        Returns:
            Tuple of (list_of_compact_items, total_count).
        """
        capped_limit = min(max(1, limit), 100)
        clean_skip = max(0, skip)

        query: Dict[str, Any] = {}

        if inspection_id and inspection_id.strip():
            query["inspection_id"] = inspection_id.strip()

        # Date range filter
        date_query: Dict[str, Any] = {}
        if date_from:
            date_query["$gte"] = date_from
        if date_to:
            date_query["$lte"] = date_to
        if date_query:
            query["created_at"] = date_query

        # Status filter
        if status and status.strip():
            clean_status = status.strip()
            if clean_status in ["Pending", "Verified", "Issue Raised", "N/A"]:
                query["review.status"] = clean_status
            elif clean_status.upper() == "NON-COMPLIANT":
                query["summary.non_compliant_findings_count"] = {"$gt": 0}
            elif clean_status.upper() == "PASS":
                query["summary.non_compliant_findings_count"] = 0

        # Brand name filter (case-insensitive substring)
        if brand_name and brand_name.strip():
            escaped_brand = re.escape(brand_name.strip())
            query["metadata.brand_name"] = {"$regex": escaped_brand, "$options": "i"}

        # Package type filter
        if package_type and package_type.strip():
            query["metadata.package_type"] = package_type.strip().lower()

        try:
            coll = self._get_collection()

            total = coll.count_documents(query)

            # Projection returning only fields necessary for list views
            projection = {
                "_id": 0,
                "inspection_id": 1,
                "created_at": 1,
                "metadata": 1,
                "summary": 1,
                "review": 1,
            }

            cursor = (
                coll.find(query, projection)
                .sort("created_at", -1)
                .skip(clean_skip)
                .limit(capped_limit)
            )

            items = []
            for doc in cursor:
                metadata = doc.get("metadata", {})
                summary = doc.get("summary", {})
                review = doc.get("review", {})

                created_at_val = doc.get("created_at")
                if not isinstance(created_at_val, datetime):
                    created_at_val = datetime.now(timezone.utc)

                items.append({
                    "inspection_id": doc.get("inspection_id", "N/A"),
                    "created_at": created_at_val,
                    "brand_name": metadata.get("brand_name"),
                    "generic_name": metadata.get("generic_name"),
                    "package_type": metadata.get("package_type", "retail"),
                    "pdp_area_cm2": metadata.get("pdp_area_cm2"),
                    "rule_source_mode": metadata.get("rule_source_mode", "sih_ps_26034"),
                    "total_declarations": summary.get("total_declarations_evaluated", 0),
                    "pass_count": summary.get("pass_findings_count", 0),
                    "non_compliant_count": summary.get("non_compliant_findings_count", 0),
                    "not_visible_count": summary.get("not_visible_declarations_count", 0),
                    "review_status": review.get("status", "Pending"),
                })

            return items, total

        except PyMongoError as exc:
            logger.error("MongoDB list query failed: %s", exc)
            raise InspectionRepositoryError(
                code="MONGO_QUERY_FAILED",
                message=f"Failed to query inspection history: {str(exc)}",
            ) from exc

    def update_review(
        self,
        inspection_id: str,
        review_status: str,
        notes: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Updates ONLY the manual LMO review status and notes for an existing inspection.

        Never mutates Rule 6, Rule 7, Rule 8, or summary metrics.

        Args:
            inspection_id: Target inspection identifier.
            review_status: One of 'Pending', 'Verified', 'Issue Raised', 'N/A'.
            notes: Optional review notes or instructions.

        Returns:
            Optional[dict]: Updated inspection document or None if not found.
        """
        clean_id = inspection_id.strip()
        allowed_statuses = {"Pending", "Verified", "Issue Raised", "N/A"}
        if review_status not in allowed_statuses:
            raise InspectionRepositoryError(
                code="INVALID_REVIEW_STATUS",
                message=f"Status '{review_status}' invalid. Allowed: {sorted(list(allowed_statuses))}",
            )

        now_utc = datetime.now(timezone.utc)
        update_fields: Dict[str, Any] = {
            "review.status": review_status,
            "review.updated_at": now_utc,
            "updated_at": now_utc,
        }
        if notes is not None:
            update_fields["review.notes"] = notes

        try:
            coll = self._get_collection()
            result = coll.update_one(
                {"inspection_id": clean_id},
                {"$set": update_fields},
            )
            if result.matched_count == 0:
                return None

            return self.get_inspection_by_id(clean_id)
        except PyMongoError as exc:
            logger.error("MongoDB review update failed for %s: %s", clean_id, exc)
            raise InspectionRepositoryError(
                code="MONGO_UPDATE_FAILED",
                message=f"Failed to update review for {clean_id}: {str(exc)}",
            ) from exc

    def count_inspections(self, query: Optional[Dict[str, Any]] = None) -> int:
        """Returns total document count matching query."""
        try:
            coll = self._get_collection()
            return coll.count_documents(query or {})
        except Exception as exc:
            logger.warning("Could not count inspections: %s", exc)
            return 0
