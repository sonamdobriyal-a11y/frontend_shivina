"""Configuration helpers for the bean leaf classifier project."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence
import cv2

from dotenv import load_dotenv

load_dotenv()


def _split_env_list(value: str | None) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_bool_env(value: str | None, default: bool = True) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_int(value: str | None, default: int | None = None) -> int | None:
    if value is None:
        return default
    value = value.strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


BACKEND_NAME_MAP = {
    "any": cv2.CAP_ANY,
    "auto": cv2.CAP_ANY,
    "msmf": getattr(cv2, "CAP_MSMF", None),
    "dshow": getattr(cv2, "CAP_DSHOW", None),
    "directshow": getattr(cv2, "CAP_DSHOW", None),
    "v4l2": getattr(cv2, "CAP_V4L2", None),
}


def _parse_backend(value: str | None) -> int | None:
    if not value:
        return None
    token = value.strip()
    if not token:
        return None
    if token.isdigit():
        return int(token)
    lowered = token.lower()
    if lowered.startswith("0x"):
        try:
            return int(lowered, 16)
        except ValueError:
            return None
    return BACKEND_NAME_MAP.get(lowered, None)


@dataclass(frozen=True)
class CloudinarySettings:
    cloud_name: str
    api_key: str
    api_secret: str
    folder: str = field(default="unhealthy-leaves")

    @classmethod
    def from_env(cls) -> "CloudinarySettings":
        return cls(
            cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME", ""),
            api_key=os.getenv("CLOUDINARY_API_KEY", ""),
            api_secret=os.getenv("CLOUDINARY_API_SECRET", ""),
            folder=os.getenv("CLOUDINARY_FOLDER", "unhealthy-leaves"),
        )

    def validate(self) -> None:
        missing = [
            name
            for name, value in [
                ("CLOUDINARY_CLOUD_NAME", self.cloud_name),
                ("CLOUDINARY_API_KEY", self.api_key),
                ("CLOUDINARY_API_SECRET", self.api_secret),
            ]
            if not value
        ]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"Missing Cloudinary configuration environment variables: {joined}")


@dataclass(frozen=True)
class FirebaseSettings:
    credential_path: Path
    collection: str = field(default="detections")
    project_id: str | None = field(default=None)
    database_url: str | None = field(default=None)
    motor_command_path: str = field(default="rover_controls")
    motor_command_document: str = field(default="current")
    enabled: bool = field(default=True)

    @classmethod
    def from_env(cls) -> "FirebaseSettings":
        credential = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
        project_id = os.getenv("FIREBASE_PROJECT_ID", None)
        collection = os.getenv("FIREBASE_COLLECTION", "detections")
        database_url = os.getenv("FIREBASE_DATABASE_URL")
        motor_command_path = os.getenv("FIREBASE_MOTOR_COMMAND_PATH", "rover_controls")
        motor_command_document = os.getenv("FIREBASE_MOTOR_COMMAND_DOCUMENT", "current")
        enabled = _parse_bool_env(os.getenv("FIREBASE_ENABLED"), default=True)
        return cls(
            credential_path=Path(credential),
            collection=collection,
            project_id=project_id,
            database_url=database_url,
            motor_command_path=motor_command_path,
            motor_command_document=motor_command_document,
            enabled=enabled,
        )

    def validate(self) -> None:
        if not self.enabled:
            return
        if not self.credential_path or not self.credential_path.exists():
            raise FileNotFoundError(
                f"Firebase credential file not found: {self.credential_path or '<unset>'}"
            )


@dataclass(frozen=True)
class YOLOSettings:
    dataset_config: Path = field(default=Path("config/data_config.yaml"))
    base_checkpoint: str = field(default="yolo11n-cls.pt")
    trained_weights: Path | None = field(default=None)
    unhealthy_classes: Sequence[str] = field(
        default_factory=lambda: ["angular_leaf_spot", "bean_rust"]
    )
    confidence_threshold: float = field(default=0.6)
    capture_device: int = field(default=0)
    prediction_interval: float = field(default=0.5)
    capture_backend: int | None = field(default=None)
    auto_discover: bool = field(default=True)
    capture_scan_indices: Sequence[int] = field(default_factory=lambda: [0, 1, 2, 3])
    preferred_backends: Sequence[int] = field(
        default_factory=lambda: [
            backend
            for backend in (getattr(cv2, "CAP_DSHOW", None), getattr(cv2, "CAP_MSMF", None), cv2.CAP_ANY)
            if backend is not None
        ]
    )

    @classmethod
    def from_env(cls) -> "YOLOSettings":
        unhealthy = _split_env_list(os.getenv("UNHEALTHY_CLASSES"))
        trained_weights = os.getenv("YOLO_WEIGHTS_PATH")
        capture_backend = _parse_backend(os.getenv("CAPTURE_BACKEND"))
        capture_indices = [
            index
            for index in (
                _parse_int(token.strip())
                for token in os.getenv("CAPTURE_SCAN_INDICES", "0,1,2,3").split(",")
            )
            if index is not None
        ]
        preferred_backend_tokens = os.getenv("CAPTURE_PREFERRED_BACKENDS", "dshow,msmf,any")
        preferred_backends = [
            backend for backend in (_parse_backend(token) for token in preferred_backend_tokens.split(",")) if backend
        ]
        return cls(
            dataset_config=Path(os.getenv("YOLO_DATA_CONFIG", "config/data_config.yaml")),
            base_checkpoint=os.getenv("YOLO_BASE_MODEL", "yolo11n-cls.pt"),
            trained_weights=Path(trained_weights) if trained_weights else None,
            unhealthy_classes=unhealthy or ["angular_leaf_spot", "bean_rust"],
            confidence_threshold=float(os.getenv("DETECTION_CONFIDENCE", "0.6")),
            capture_device=_parse_int(os.getenv("CAPTURE_DEVICE_INDEX"), default=0) or 0,
            prediction_interval=float(os.getenv("PREDICTION_INTERVAL_SECONDS", "0.5")),
            capture_backend=capture_backend,
            auto_discover=_parse_bool_env(os.getenv("CAPTURE_AUTO_DISCOVER"), default=True),
            capture_scan_indices=capture_indices or [0, 1, 2, 3],
            preferred_backends=preferred_backends
            or [
                backend
                for backend in (
                    getattr(cv2, "CAP_DSHOW", None),
                    getattr(cv2, "CAP_MSMF", None),
                    cv2.CAP_ANY,
                )
                if backend is not None
            ],
        )


@dataclass(frozen=True)
class AppSettings:
    cloudinary: CloudinarySettings
    firebase: FirebaseSettings
    yolo: YOLOSettings

    @classmethod
    def from_env(cls) -> "AppSettings":
        cloudinary_settings = CloudinarySettings.from_env()
        firebase_settings = FirebaseSettings.from_env()
        yolo_settings = YOLOSettings.from_env()
        return cls(
            cloudinary=cloudinary_settings,
            firebase=firebase_settings,
            yolo=yolo_settings,
        )
