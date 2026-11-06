"""Lightweight Flask dashboard for unique counts, faces, and recent events."""

from __future__ import annotations

from pathlib import Path

from flask import Flask, abort, render_template, send_from_directory

from src.database import FaceDatabase


def create_app(config: dict) -> Flask:
    root = Path(__file__).resolve().parent
    app = Flask(__name__, template_folder=str(root / "templates"), static_folder=str(root / "static"))
    db = FaceDatabase(config["database_path"])
    workspace = Path.cwd()

    @app.teardown_appcontext
    def _close_db(_exc: BaseException | None) -> None:
        return None

    @app.route("/")
    def index():
        faces = db.list_faces()
        events = db.recent_events(80)
        counts = db.event_counts()
        return render_template(
            "index.html",
            unique=db.unique_visitor_count(),
            entries=counts["entry"],
            exits=counts["exit"],
            faces=faces,
            events=events,
        )

    @app.route("/media/<path:rel>")
    def media(rel: str):
        path = (workspace / rel).resolve()
        try:
            path.relative_to(workspace.resolve())
        except ValueError:
            abort(403)
        if not path.exists() or not path.is_file():
            abort(404)
        return send_from_directory(path.parent, path.name)

    return app


def run_dashboard(config: dict) -> None:
    app = create_app(config)
    host = config.get("dashboard_host", "127.0.0.1")
    port = int(config.get("dashboard_port", 7860))
    print(f"Dashboard: http://{host}:{port}")
    app.run(host=host, port=port, debug=False)
