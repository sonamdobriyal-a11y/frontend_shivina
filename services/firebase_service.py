"""Firebase helper utilities for recording unhealthy detections."""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from typing import Dict, List, Optional

import firebase_admin
from firebase_admin import credentials, firestore  # type: ignore

from config.settings import FirebaseSettings

_firebase_lock = threading.Lock()
_firebase_app: Optional[firebase_admin.App] = None
_firestore_client: Optional[firestore.Client] = None
_logger = logging.getLogger(__name__)


def initialize_firebase(settings: FirebaseSettings) -> Optional[firestore.Client]:
    """Initialise Firebase (Firestore) using the provided settings."""
    if not settings.enabled:
        return None

    global _firebase_app, _firestore_client

    with _firebase_lock:
        if _firestore_client is not None:
            return _firestore_client

        settings.validate()

        cred = credentials.Certificate(str(settings.credential_path))
        app_options = {}
        if settings.project_id:
            app_options["projectId"] = settings.project_id
        if settings.database_url:
            app_options["databaseURL"] = settings.database_url
        _firebase_app = firebase_admin.initialize_app(cred, options=app_options or None)
        _firestore_client = firestore.client(app=_firebase_app)
        return _firestore_client


def log_detection(
    detection: Dict,
    settings: FirebaseSettings,
) -> Optional[str]:
    """Store a detection event in Firestore and return the generated document ID."""
    if not settings.enabled:
        return None

    client = initialize_firebase(settings)
    if client is None:
        return None

    if "captured_at" not in detection:
        detection["captured_at"] = datetime.utcnow().isoformat()

    try:
        doc_ref = client.collection(settings.collection).document()
        doc_ref.set(detection)
        return doc_ref.id
    except Exception as exc:  # noqa: BLE001
        _logger.error("Failed to persist detection to Firestore: %s", exc)
        return None


def fetch_recent_detections(settings: FirebaseSettings, limit: int = 25) -> List[Dict]:
    """Fetch the most recent detection records from Firestore."""
    if not settings.enabled:
        return []

    client = initialize_firebase(settings)
    if client is None:
        return []

    try:
        query = (
            client.collection(settings.collection)
            .order_by("captured_at", direction=firestore.Query.DESCENDING)
            .limit(limit)
        )
        return [{**doc.to_dict(), "id": doc.id} for doc in query.stream()]
    except Exception as exc:  # noqa: BLE001
        _logger.error("Failed to fetch detections from Firestore: %s", exc)
        return []


def delete_detection(document_id: str, settings: FirebaseSettings) -> bool:
    """Delete a Firestore detection document by ID."""
    if not settings.enabled:
        return False

    client = initialize_firebase(settings)
    if client is None:
        return False

    try:
        doc_ref = client.collection(settings.collection).document(document_id)
        snapshot = doc_ref.get()
        if not snapshot.exists:
            return False
        doc_ref.delete()
        return True
    except Exception as exc:  # noqa: BLE001
        _logger.error("Failed to delete detection %s: %s", document_id, exc)
        return False


def update_motor_command(command_flags: Dict[str, bool], settings: FirebaseSettings) -> Dict[str, bool]:
    """Write the requested motor command flags into the Firestore collection."""
    if not settings.enabled:
        raise ValueError("Firebase integration is disabled for this deployment.")
    collection_name = (settings.motor_command_path or "").strip("/ ")
    if not collection_name:
        raise ValueError("FIREBASE_MOTOR_COMMAND_PATH must reference a Firestore collection.")

    client = initialize_firebase(settings)
    if client is None:
        raise RuntimeError("Firebase client failed to initialise.")

    expected_keys = ("forward", "reverse", "left", "right", "stop")
    payload = {key: bool(command_flags.get(key, False)) for key in expected_keys}
    if not any(payload.values()):
        raise ValueError("At least one motor control flag must be true.")
    payload["updated_at"] = datetime.utcnow().isoformat()

    document_name = (settings.motor_command_document or "").strip()
    try:
        collection_ref = client.collection(collection_name)
        if document_name:
            collection_ref.document(document_name).set(payload)
        else:
            collection_ref.add(payload)
        return payload
    except Exception as exc:  # noqa: BLE001
        _logger.error("Failed to update motor command in Firestore: %s", exc)
        raise
