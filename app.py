"""CLI entry point for the Intelligent Face Tracker.

Examples:
    python app.py --source data/sample.mp4
    python app.py --source rtsp://user:pass@ip:554/stream
    python app.py --dashboard
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config_loader import load_config
from src.pipeline import FaceTrackerPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Intelligent Face Tracker")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    parser.add_argument(
        "--source",
        default=None,
        help="Video file, webcam index, or RTSP URL (overrides config)",
    )
    parser.add_argument("--dashboard", action="store_true", help="Launch the review dashboard")
    parser.add_argument("--display", action="store_true", help="Show an OpenCV preview window")
    parser.add_argument("--no-output", action="store_true", help="Do not write annotated video")
    return parser.parse_args()


def resolve_source(cfg: dict, cli_source: str | None) -> str:
    if cli_source:
        return cli_source
    if cfg.get("rtsp_url"):
        return str(cfg["rtsp_url"])
    return str(cfg.get("video_source") or "data/sample.mp4")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.display:
        cfg["display"] = True

    if args.dashboard:
        from web.server import run_dashboard

        run_dashboard(cfg)
        return

    source = resolve_source(cfg, args.source)
    if not source.lower().startswith(("rtsp://", "rtsps://", "http://", "https://")) and not source.isdigit():
        if not Path(source).exists():
            raise SystemExit(
                f"Video source not found: {source}\n"
                "Place the sample clip at data/sample.mp4 or pass --source PATH\n"
                "Download helper: python scripts/download_assets.py"
            )

    pipeline = FaceTrackerPipeline(cfg)
    try:
        summary = pipeline.process_source(source, write_output=not args.no_output)
        print(
            "\nDone.\n"
            f"  Unique visitors : {summary['unique_visitors']}\n"
            f"  Entries         : {summary['events']['entry']}\n"
            f"  Exits           : {summary['events']['exit']}\n"
            f"  Frames          : {summary['frames']}\n"
            f"  Events log      : {cfg['events_log']}\n"
            f"  Database        : {cfg['database_path']}\n"
        )
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
