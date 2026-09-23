"""MongoDB connection manager for SIH 2026 PS 26034 Compliance Checker.

Maintains a single application-level MongoClient, provides database handles,
and exposes safe health verification without leaking credentials or crashing on outage.
"""

import logging
from typing import Any, Dict, Optional
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, PyMongoError, ServerSelectionTimeoutError

from app.config import settings

logger = logging.getLogger(__name__)

# Application-level singleton client
_mongo_client: Optional[MongoClient] = None


def get_mongo_client() -> Optional[MongoClient]:
    """Retrieves or lazily instantiates the application-level MongoClient singleton."""
    global _mongo_client

    if not settings.MONGODB_ENABLED:
        return None

    if _mongo_client is not None:
        return _mongo_client

    try:
        timeout_ms = getattr(settings, "MONGODB_TIMEOUT_MS", 2000)
        _mongo_client = MongoClient(
            settings.MONGODB_URI,
            serverSelectionTimeoutMS=timeout_ms,
            connectTimeoutMS=timeout_ms,
            socketTimeoutMS=timeout_ms,
            retryWrites=True,
            appname="SIH-Compliance-Checker",
        )
        return _mongo_client
    except Exception as exc:
        logger.warning("Failed to initialize MongoClient [%s]: %s", settings.MONGODB_URI, exc)
        return None


def set_mongo_client(client: Optional[MongoClient]) -> None:
    """Sets or overrides the MongoClient singleton (used for testing and dependency injection)."""
    global _mongo_client
    _mongo_client = client


def get_database(db_name: Optional[str] = None) -> Optional[Database]:
    """Returns the primary compliance database handle, or None if MongoDB is unavailable/disabled."""
    client = get_mongo_client()
    if client is None:
        return None
    target_db = db_name or settings.MONGODB_DATABASE
    try:
        return client[target_db]
    except Exception as exc:
        logger.warning("Could not access database '%s': %s", target_db, exc)
        return None


def check_mongo_connection() -> Dict[str, Any]:
    """Tests connectivity to MongoDB via administrative ping.

    Returns:
        dict: {"status": "connected" | "disconnected" | "disabled", "details": ...}
    """
    if not settings.MONGODB_ENABLED:
        return {
            "status": "disabled",
            "message": "MongoDB persistence is disabled via MONGODB_ENABLED=false.",
        }

    client = get_mongo_client()
    if client is None:
        return {
            "status": "disconnected",
            "message": "MongoClient could not be instantiated.",
        }

    try:
        # Run lightweight admin ping command
        client.admin.command("ping")
        return {
            "status": "connected",
            "database": settings.MONGODB_DATABASE,
        }
    except (ServerSelectionTimeoutError, ConnectionFailure) as exc:
        logger.debug("MongoDB ping failed (server unreachable): %s", exc)
        return {
            "status": "disconnected",
            "message": "MongoDB server unreachable or timed out.",
        }
    except PyMongoError as exc:
        logger.warning("MongoDB ping error: %s", exc)
        return {
            "status": "disconnected",
            "message": f"MongoDB error: {str(exc)}",
        }
    except Exception as exc:
        logger.warning("Unexpected error during MongoDB health check: %s", exc)
        return {
            "status": "disconnected",
            "message": str(exc),
        }


def close_mongo_client() -> None:
    """Closes the MongoClient singleton cleanly on application shutdown."""
    global _mongo_client
    if _mongo_client is not None:
        try:
            _mongo_client.close()
            logger.info("Closed MongoClient connection cleanly.")
        except Exception as exc:
            logger.warning("Error closing MongoClient: %s", exc)
        finally:
            _mongo_client = None
