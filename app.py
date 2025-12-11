from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from dashboard_service import create_app


def configure_logging() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def prepare_application():
    configure_logging()
    load_dotenv(dotenv_path=Path(".") / ".env", override=False)
    return create_app()


application = prepare_application()


def main() -> None:
    host = os.getenv("FLASK_RUN_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_RUN_PORT", "5000"))
    debug_flag = os.getenv("FLASK_DEBUG", "1").strip().lower() in {"1", "true", "yes", "on"}
    application.run(host=host, port=port, debug=debug_flag)


if __name__ == "__main__":
    main()
