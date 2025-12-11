"""Shared application state for the dashboard service."""

from __future__ import annotations

import logging
import threading
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, Optional

from config.settings import AppSettings
from yolo.training import train_yolo_classifier

log = logging.getLogger(__name__)


class DashboardState:
    """Tracks training jobs triggered from the dashboard."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="dashboard-worker")
        self._training_future: Optional[Future] = None
        self._training_lock = threading.Lock()
        self.training_status: Dict[str, Any] = {
            "running": False,
            "last_run": None,
            "last_model_path": None,
            "error": None,
        }

    def start_training(self, request_payload: Optional[Dict[str, Any]] = None) -> bool:
        with self._training_lock:
            if self.training_status["running"]:
                return False

            self.training_status.update({"running": True, "error": None})
            self._training_future = self._executor.submit(self._run_training, request_payload or {})
            return True

    def _run_training(self, payload: Dict[str, Any]) -> None:
        try:
            summary = train_yolo_classifier(
                config_path=self.settings.yolo.dataset_config,
                model_variant=self.settings.yolo.base_checkpoint,
                epochs=int(payload.get("epochs", 30)),
                batch_size=int(payload.get("batch", 32)),
                image_size=int(payload.get("imgsz", 224)),
                project="runs/classify",
            )
            best_weight = summary.get("best_weight") or summary.get("last_weight")
            if best_weight:
                self.training_status["last_model_path"] = str(best_weight)
            self.training_status["last_run"] = str(summary.get("save_dir"))
            log.info("Training finished successfully.")
        except Exception as exc:  # noqa: BLE001
            log.error("Training failed: %s", exc)
            log.debug("Traceback:\n%s", traceback.format_exc())
            self.training_status["error"] = str(exc)
        finally:
            self.training_status["running"] = False

    def get_training_future(self) -> Optional[Future]:
        return self._training_future
