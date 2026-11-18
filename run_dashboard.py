"""Run the Intelliginet Face Tracker dashboard.

Usage:
    python run_dashboard.py

This script loads the project's configuration (config.json) and starts the
Flask dashboard defined in `web/server.py`.  The dashboard provides a simple
web UI that displays the number of unique visitors, entry/exit totals, a
gallery of registered faces, and a table of recent events.
"""

from src.config_loader import load_config
from web.server import run_dashboard


def main() -> None:
    # Load configuration (falls back to defaults if config.json is missing)
    cfg = load_config()
    # Launch the Flask app on the host/port specified in the config
    run_dashboard(cfg)


if __name__ == "__main__":
    main()

