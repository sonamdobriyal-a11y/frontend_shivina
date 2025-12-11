"""Blueprint wrapping dashboard UI endpoints."""

from __future__ import annotations

import logging
from typing import Any, Dict

from flask import Blueprint, jsonify, render_template, request

from dashboard_service.state import DashboardState
from services.cloudinary_service import configure_cloudinary, delete_asset as delete_cloudinary_asset
from services.firebase_service import (
    delete_detection as delete_firebase_detection,
    fetch_recent_detections,
    update_motor_command,
)

log = logging.getLogger(__name__)


def create_blueprint(state: DashboardState) -> Blueprint:
    bp = Blueprint("dashboard_service", __name__)

    @bp.route("/", methods=["GET"])
    def dashboard_view() -> str:
        return render_template("dashboard.html")

    @bp.route("/api/status", methods=["GET"])
    def status():
        recent = fetch_recent_detections(state.settings.firebase, limit=1)
        latest = recent[0] if recent else None
        return jsonify(
            {
                "training": state.training_status,
                "latest_detection": latest,
                "motor_control": {
                    "enabled": state.settings.firebase.enabled and bool(state.settings.firebase.database_url),
                    "path": state.settings.firebase.motor_command_path,
                },
            }
        )

    @bp.route("/api/detections", methods=["GET"])
    def get_detections():
        limit_raw = request.args.get("limit")
        try:
            limit = max(1, min(100, int(limit_raw))) if limit_raw else 25
        except ValueError:
            limit = 25
        detections = fetch_recent_detections(state.settings.firebase, limit=limit)
        return jsonify(detections)

    @bp.route("/api/detections/<string:document_id>", methods=["DELETE"])
    def delete_detection(document_id: str):
        if not state.settings.firebase.enabled:
            return jsonify({"message": "Firebase storage is disabled in this deployment."}), 400

        payload = request.get_json(silent=True) or {}
        cloudinary_id = payload.get("cloudinary_id") or payload.get("cloudinary_public_id")

        response_payload: Dict[str, Any] = {
            "firebase_deleted": False,
            "cloudinary_deleted": False,
        }

        try:
            response_payload["firebase_deleted"] = delete_firebase_detection(document_id, state.settings.firebase)
        except Exception as exc:  # noqa: BLE001
            log.error("Failed to delete detection %s from Firebase: %s", document_id, exc)
            return jsonify({"message": f"Failed to delete Firestore record: {exc}"}), 500

        if cloudinary_id:
            try:
                configure_cloudinary(state.settings.cloudinary)
                response_payload["cloudinary_deleted"] = delete_cloudinary_asset(cloudinary_id)
            except Exception as exc:  # noqa: BLE001
                log.error("Failed to delete Cloudinary asset %s: %s", cloudinary_id, exc)
                return jsonify({"message": f"Failed to delete Cloudinary asset: {exc}"}), 500

        return jsonify(response_payload)

    @bp.route("/api/train", methods=["POST"])
    def trigger_training():
        payload = request.get_json(silent=True) or {}
        started = state.start_training(payload)
        if not started:
            return jsonify({"message": "Training already running"}), 409
        return jsonify({"message": "Training started"}), 202

    @bp.route("/api/motor-control", methods=["POST"])
    def motor_control():
        payload = request.get_json(silent=True) or {}
        flags = {
            "forward": bool(payload.get("forward")),
            "reverse": bool(payload.get("reverse")),
            "left": bool(payload.get("left")),
            "right": bool(payload.get("right")),
            "stop": bool(payload.get("stop")),
        }
        if not any(flags.values()):
            return jsonify({"message": "One of forward, reverse, left, right, or stop must be true."}), 400
        try:
            applied = update_motor_command(flags, state.settings.firebase)
        except ValueError as exc:
            return jsonify({"message": str(exc)}), 400
        except Exception as exc:  # noqa: BLE001
            return jsonify({"message": f"Failed to update motor command: {exc}"}), 500
        return jsonify({"message": "Command sent.", "controls": applied}), 200

    return bp
