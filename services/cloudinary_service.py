"""Cloudinary helper utilities for uploading unhealthy leaf detections."""

from __future__ import annotations

import uuid
from typing import Dict, Iterable, Optional

import cloudinary
import cloudinary.uploader
import cv2

from config.settings import CloudinarySettings


def configure_cloudinary(settings: CloudinarySettings) -> None:
    """Initialise the Cloudinary SDK with the provided credentials."""
    settings.validate()
    cloudinary.config(
        cloud_name=settings.cloud_name,
        api_key=settings.api_key,
        api_secret=settings.api_secret,
        secure=True,
    )


def _format_context(metadata: Optional[Dict[str, str]]) -> Optional[str]:
    if not metadata:
        return None
    return "|".join(f"{key}={value}" for key, value in metadata.items() if value is not None)


def upload_frame(
    frame,
    *,
    folder: str,
    public_id: Optional[str] = None,
    tags: Optional[Iterable[str]] = None,
    metadata: Optional[Dict[str, str]] = None,
) -> Dict:
    """
    Upload an OpenCV frame to Cloudinary as a JPEG image.

    Returns the Cloudinary response which contains the secure URL and other metadata.
    """
    if frame is None:
        raise ValueError("frame must not be None")

    success, encoded = cv2.imencode(".jpg", frame)
    if not success:
        raise RuntimeError("Failed to encode frame for upload.")

    public_id = public_id or f"unhealthy-leaf-{uuid.uuid4().hex}"
    upload_result = cloudinary.uploader.upload(
        encoded.tobytes(),
        folder=folder,
        public_id=public_id,
        resource_type="image",
        tags=list(tags) if tags else None,
        context=_format_context(metadata),
    )
    return upload_result


def delete_asset(public_id: str) -> bool:
    """Remove an asset from Cloudinary by its public ID."""
    if not public_id:
        return False
    response = cloudinary.uploader.destroy(public_id, invalidate=True)
    return response.get("result") in {"ok", "not_found"}
