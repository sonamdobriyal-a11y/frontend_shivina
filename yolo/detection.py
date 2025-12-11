"""Live detection pipeline for identifying unhealthy bean leaves."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import cv2
from ultralytics import YOLO

from config.settings import CloudinarySettings, FirebaseSettings, YOLOSettings
from services.cloudinary_service import configure_cloudinary, upload_frame
from services.firebase_service import initialize_firebase, log_detection
from yolo.training import ensure_local_checkpoint


@dataclass
class DetectionRecord:
    """Simple DTO representing a captured unhealthy plant frame."""

    class_name: str
    confidence: float
    captured_at: str
    cloudinary_url: str
    cloudinary_public_id: str
    firebase_document_id: Optional[str] = None


class DetectionStore:
    """Thread-safe in-memory store used by the dashboard to display recent events."""

    def __init__(self, max_size: int = 100) -> None:
        self._records: Deque[DetectionRecord] = deque(maxlen=max_size)
        self._lock = threading.Lock()

    def add(self, record: DetectionRecord) -> None:
        with self._lock:
            self._records.appendleft(record)

    def list(self) -> List[DetectionRecord]:
        with self._lock:
            return list(self._records)

    def clear(self) -> None:
        with self._lock:
            self._records.clear()

    def remove(self, predicate: Callable[[DetectionRecord], bool]) -> int:
        """Remove records matching the predicate. Returns the number of items removed."""
        with self._lock:
            records = list(self._records)
            filtered = [record for record in records if not predicate(record)]
            removed = len(records) - len(filtered)
            self._records = deque(filtered, maxlen=self._records.maxlen)
        return removed


class LiveDetector:
    """Handle webcam capture, run YOLO classification, and persist unhealthy detections."""

    def __init__(
        self,
        *,
        yolo_settings: YOLOSettings,
        cloudinary_settings: CloudinarySettings,
        firebase_settings: FirebaseSettings,
        detection_store: DetectionStore,
        cooldown_seconds: float = 3.0,
    ) -> None:
        self._yolo_settings = yolo_settings
        self._cloudinary_settings = cloudinary_settings
        self._firebase_settings = firebase_settings
        self._detection_store = detection_store
        self._cooldown_seconds = cooldown_seconds

        self._model: YOLO | None = None
        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self._last_capture: Dict[str, float] = {}
        self._capture_backend: int | None = None
        self._active_device_index: int = yolo_settings.capture_device

        self._latest_frame = None
        self._latest_prediction: Optional[Dict[str, Any]] = None
        self._frame_lock = threading.Lock()

        self._log = logging.getLogger(self.__class__.__name__)
        try:
            self._tz = ZoneInfo("Asia/Kolkata")
        except ZoneInfoNotFoundError:
            self._log.warning("ZoneInfo Asia/Kolkata not found. Falling back to UTC+05:30 offset.")
            self._tz = timezone(timedelta(hours=5, minutes=30))

    def _open_capture(self) -> cv2.VideoCapture:
        """Attempt to open the capture device using preferred backend order."""
        candidate_backends: List[int] = []
        if self._capture_backend is not None:
            candidate_backends.append(self._capture_backend)
        if (
            self._yolo_settings.capture_backend is not None
            and self._yolo_settings.capture_backend not in candidate_backends
        ):
            candidate_backends.append(self._yolo_settings.capture_backend)

        for backend in self._yolo_settings.preferred_backends:
            if backend is not None and backend not in candidate_backends:
                candidate_backends.append(backend)

        if cv2.CAP_ANY not in candidate_backends:
            candidate_backends.append(cv2.CAP_ANY)

        candidate_indices = [self._yolo_settings.capture_device]
        if self._yolo_settings.auto_discover:
            for idx in self._yolo_settings.capture_scan_indices:
                if idx not in candidate_indices:
                    candidate_indices.append(idx)

        for backend in candidate_backends:
            for device_index in candidate_indices:
                capture = (
                    cv2.VideoCapture(device_index)
                    if backend in (None, cv2.CAP_ANY)
                    else cv2.VideoCapture(device_index, backend)
                )
                if not capture.isOpened():
                    capture.release()
                    continue

                success = False
                for _ in range(8):
                    ret, frame = capture.read()
                    if ret and frame is not None:
                        success = True
                        break

                if success:
                    self._capture_backend = backend if backend is not None else cv2.CAP_ANY
                    self._active_device_index = device_index
                    self._log.info(
                        "Opened capture device %s using backend %s",
                        device_index,
                        self._capture_backend,
                    )
                    return capture

                capture.release()

        raise RuntimeError(
            "Unable to open a camera that returns frames. Verify hardware availability or update "
            "CAPTURE_SCAN_INDICES / CAPTURE_PREFERRED_BACKENDS."
        )

    def get_latest_frame_bytes(self) -> Optional[bytes]:
        with self._frame_lock:
            if self._latest_frame is None:
                return None
            success, buffer = cv2.imencode(".jpg", self._latest_frame)
        if not success:
            return None
        return buffer.tobytes()

    def get_latest_prediction(self) -> Optional[Dict[str, Any]]:
        with self._frame_lock:
            if self._latest_prediction is None:
                return None
            return dict(self._latest_prediction)

    def _load_model(self) -> YOLO:
        if self._model is not None:
            return self._model

        weights_candidate = (
            str(self._yolo_settings.trained_weights)
            if self._yolo_settings.trained_weights
            else self._yolo_settings.base_checkpoint
        )
        try:
            ensured = ensure_local_checkpoint(weights_candidate, destination=Path("."))
            weights_source = str(ensured)
        except Exception as exc:  # noqa: BLE001
            self._log.warning(
                "Unable to ensure local weights for live detector (%s). Using %s directly.",
                exc,
                weights_candidate,
            )
            weights_source = weights_candidate
        self._model = YOLO(weights_source)
        return self._model

    def update_weights(self, weights_path: str | Path) -> None:
        """Hot-swap the model weights used for live detection."""
        self._model = YOLO(str(weights_path))
        self._yolo_settings = replace(
            self._yolo_settings, trained_weights=Path(weights_path), base_checkpoint=str(weights_path)
        )
        self._log.info("Live detector weights updated: %s", weights_path)

    def start(self) -> None:
        if self._running.is_set():
            return

        configure_cloudinary(self._cloudinary_settings)
        initialize_firebase(self._firebase_settings)
        self._model = self._load_model()

        # Verify the capture device can be opened before spinning up the background thread.
        test_capture = self._open_capture()
        test_capture.release()

        with self._frame_lock:
            self._latest_frame = None
            self._latest_prediction = None

        self._running.set()
        self._thread = threading.Thread(target=self._run_loop, name="live-detector", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        self._thread = None
        with self._frame_lock:
            self._latest_frame = None
            self._latest_prediction = None

    def is_running(self) -> bool:
        return self._running.is_set()

    def _run_loop(self) -> None:
        capture = self._open_capture()
        try:
            while self._running.is_set():
                success, frame = capture.read()
                if not success:
                    time.sleep(0.25)
                    continue
                try:
                    self._evaluate_frame(frame)
                except Exception as exc:  # noqa: BLE001 - log and continue
                    self._log.exception("Live detection error: %s", exc)

                time.sleep(self._yolo_settings.prediction_interval)
        finally:
            capture.release()

    def _evaluate_frame(self, frame) -> None:
        model = self._load_model()
        unhealthy_classes = set(self._yolo_settings.unhealthy_classes)

        display_frame = frame.copy()
        prediction_details: Dict[str, Any] | None = None

        results = model(frame, verbose=False)
        for result in results:
            if not hasattr(result, "probs") or result.probs is None:
                continue

            class_index = int(result.probs.top1)
            confidence = float(result.probs.top1conf)
            class_name = self._resolve_class_name(model, class_index)

            is_unhealthy = class_name in unhealthy_classes and confidence >= self._yolo_settings.confidence_threshold
            label = f"{class_name} {confidence * 100:.1f}%"
            color = (0, 0, 255) if is_unhealthy else (0, 200, 0)
            cv2.putText(
                display_frame,
                label,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                color,
                2,
                cv2.LINE_AA,
            )

            captured_at_ist = datetime.now(self._tz).isoformat()
            prediction_details = {
                "class_name": class_name,
                "confidence": confidence,
                "is_unhealthy": is_unhealthy,
                "captured_at": captured_at_ist,
            }

            if is_unhealthy:
                current_time = time.monotonic()
                last_capture = self._last_capture.get(class_name, 0.0)
                if current_time - last_capture >= self._cooldown_seconds:
                    timestamp = captured_at_ist
                    upload_response = upload_frame(
                        frame,
                        folder=self._cloudinary_settings.folder,
                        tags=["unhealthy", class_name],
                        metadata={
                            "class_name": class_name,
                            "captured_at": timestamp,
                            "confidence": f"{confidence:.3f}",
                        },
                    )

                    detection_payload = {
                        "class_name": class_name,
                        "confidence": confidence,
                        "captured_at": timestamp,
                        "cloudinary_url": upload_response.get("secure_url"),
                        "cloudinary_public_id": upload_response.get("public_id"),
                    }
                    doc_id = log_detection(detection_payload, self._firebase_settings)
                    record = DetectionRecord(
                        class_name=class_name,
                        confidence=confidence,
                        captured_at=timestamp,
                        cloudinary_url=detection_payload["cloudinary_url"],
                        cloudinary_public_id=detection_payload["cloudinary_public_id"],
                        firebase_document_id=doc_id,
                    )
                    self._detection_store.add(record)
                    self._last_capture[class_name] = current_time
                    self._log.info(
                        "Captured %s leaf (confidence %.3f). Cloudinary asset: %s",
                        class_name,
                        confidence,
                        detection_payload["cloudinary_url"],
                    )
            break

        if prediction_details is None:
            cv2.putText(
                display_frame,
                "No prediction",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            prediction_details = {
                "class_name": None,
                "confidence": None,
                "is_unhealthy": False,
                "captured_at": datetime.now(self._tz).isoformat(),
            }

        with self._frame_lock:
            self._latest_frame = display_frame
            self._latest_prediction = prediction_details

    @staticmethod
    def _resolve_class_name(model: YOLO, class_index: int) -> str:
        names = getattr(model, "names", None)
        if isinstance(names, dict):
            return names.get(class_index, str(class_index))
        if isinstance(names, Sequence):
            if 0 <= class_index < len(names):
                return str(names[class_index])
        return str(class_index)
