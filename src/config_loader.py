"""Load and validate config.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULTS: dict[str, Any] = {
    "video_source": "data/sample.mp4",
    "rtsp_url": "",
    "skip_frames": 2,
    "yolo_model": "models/yolov8n-face.pt",
    "yolo_confidence": 0.45,
    "yolo_imgsz": 640,
    "insightface_model": "buffalo_sc",
    "embedding_threshold": 0.42,
    "bbox_expand_ratio": 0.25,
    "tracker_max_age": 25,
    "tracker_min_hits": 2,
    "tracker_iou_threshold": 0.3,
    "device": "auto",
    "database_path": "data/tracker.db",
    "log_dir": "logs",
    "events_log": "logs/events.log",
    "output_video": "logs/annotated_output.mp4",
    "output_max_width": 1280,
    "display": False,
    "dashboard_host": "127.0.0.1",
    "dashboard_port": 7860,
}


def load_config(path: str | Path = "config.json") -> dict[str, Any]:
    """Return merged configuration. Missing keys fall back to DEFAULTS."""
    cfg_path = Path(path)
    data: dict[str, Any] = {}
    if cfg_path.exists():
        with cfg_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    merged = {**DEFAULTS, **data}
    merged["skip_frames"] = max(0, int(merged["skip_frames"]))
    return merged
