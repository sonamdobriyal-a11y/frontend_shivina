"""Dashboard service Flask application factory."""

from __future__ import annotations

import logging
from pathlib import Path

from flask import Flask

from config.settings import AppSettings
from dashboard_service.routes import create_blueprint
from dashboard_service.state import DashboardState


def create_app() -> Flask:
    base_dir = Path(__file__).resolve().parents[1]
    app = Flask(
        __name__,
        template_folder=str(base_dir / "templates"),
        static_folder=str(base_dir / "static"),
    )
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    settings = AppSettings.from_env()
    state = DashboardState(settings)

    app.register_blueprint(create_blueprint(state))

    return app
