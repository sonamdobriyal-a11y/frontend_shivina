"""Utilities for training a YOLO11 classification model against the bean leaf dataset."""

from __future__ import annotations

import datetime as _dt
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import requests
import yaml
from ultralytics import YOLO


log = logging.getLogger(__name__)

YOLO_MODEL_BASE_URL = "https://github.com/ultralytics/assets/releases/latest/download"


def ensure_local_checkpoint(model_name: str, destination: Path | None = None) -> Path:
    """
    Ensure a YOLO checkpoint exists locally.

    If `model_name` points to an existing path, it is returned. Otherwise we download the file into
    the provided `destination` (defaults to project root).
    """
    candidate_path = Path(model_name)
    if candidate_path.exists():
        return candidate_path

    destination = destination or Path(".")
    destination.mkdir(parents=True, exist_ok=True)

    target_path = destination / candidate_path.name
    if target_path.exists():
        return target_path

    download_url = f"{YOLO_MODEL_BASE_URL}/{candidate_path.name}"
    log.info("Downloading YOLO checkpoint %s to %s", candidate_path.name, target_path)

    response = requests.get(download_url, stream=True, timeout=60)
    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to download checkpoint from {download_url} (status {response.status_code})"
        )

    with target_path.open("wb") as fh:
        shutil.copyfileobj(response.raw, fh)

    return target_path


def _load_data_config(config_path: Path) -> Dict[str, Any]:
    """Load the Ultralytics data configuration file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Data config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    required_keys = {"path", "train", "val"}
    if not required_keys.issubset(config):
        missing = ", ".join(sorted(required_keys - set(config)))
        raise ValueError(f"Data config missing expected keys: {missing}")

    return config


def train_yolo_classifier(
    config_path: str | Path = "config/data_config.yaml",
    model_variant: str = "yolo11n-cls.pt",
    epochs: int = 50,
    batch_size: int = 32,
    image_size: int = 224,
    learning_rate: Optional[float] = None,
    device: Optional[str] = None,
    project: str = "runs/classify",
    experiment_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Trigger YOLO11 classification training.

    Returns a dictionary containing the key artifacts (weights path, metrics directory, etc.).
    """
    config_path = Path(config_path)
    data_config = _load_data_config(config_path)

    dataset_root = (config_path.parent / data_config["path"]).resolve()
    train_dir = dataset_root / data_config["train"]
    val_dir = dataset_root / data_config["val"]

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"Expected train ({train_dir}) and val ({val_dir}) directories to exist."
        )

    experiment_name = experiment_name or f"bean-leaf-{_dt.datetime.now():%Y%m%d-%H%M%S}"

    model_source: str | Path = model_variant
    try:
        model_source = ensure_local_checkpoint(model_variant, destination=Path("."))
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "Could not ensure local checkpoint for %s (%s). Falling back to Ultralytics hub.",
            model_variant,
            exc,
        )

    model = YOLO(model_source)

    train_kwargs: Dict[str, Any] = {
        "data": str(dataset_root),
        "epochs": epochs,
        "batch": batch_size,
        "imgsz": image_size,
        "project": project,
        "name": experiment_name,
    }

    if learning_rate is not None:
        train_kwargs["lr0"] = learning_rate
    if device is not None:
        train_kwargs["device"] = device

    log.info(
        "Starting YOLO11 training with data=%s, epochs=%s, batch=%s, imgsz=%s, project=%s, name=%s",
        train_kwargs["data"],
        epochs,
        batch_size,
        image_size,
        project,
        experiment_name,
    )

    results = model.train(**train_kwargs)

    # Ultralytics stores artifacts under runs/classify/<experiment_name>/...
    save_dir = Path(getattr(model.trainer, "save_dir", Path(project) / experiment_name))
    weights_dir = save_dir / "weights"
    best_weight = weights_dir / "best.pt"
    last_weight = weights_dir / "last.pt"

    summary: Dict[str, Any] = {
        "results": results,
        "save_dir": save_dir,
        "weights_dir": weights_dir,
        "best_weight": best_weight if best_weight.exists() else None,
        "last_weight": last_weight if last_weight.exists() else None,
        "experiment_name": experiment_name,
        "model_variant": model_variant,
    }

    log.info("Training complete. Artifacts stored at %s", save_dir)
    return summary
