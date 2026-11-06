"""Filesystem logging: events.log plus cropped entry/exit images."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


class EventLogger:
    """Writes structured system events and cropped face images."""

    def __init__(self, log_dir: str | Path, events_log: str | Path) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        (self.log_dir / "entries").mkdir(parents=True, exist_ok=True)
        (self.log_dir / "exits").mkdir(parents=True, exist_ok=True)
        (self.log_dir / "faces").mkdir(parents=True, exist_ok=True)

        self.events_path = Path(events_log)
        self.events_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger("face_tracker")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        self.logger.propagate = False

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler = logging.FileHandler(self.events_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

    def info(self, message: str) -> None:
        self.logger.info(message)

    def warning(self, message: str) -> None:
        self.logger.warning(message)

    def error(self, message: str) -> None:
        self.logger.error(message)

    def save_crop(
        self,
        event_type: str,
        face_id: str,
        timestamp: datetime,
        crop: np.ndarray,
    ) -> str:
        """Save a cropped face under logs/{entries|exits}/YYYY-MM-DD/."""
        folder = "entries" if event_type == "entry" else "exits"
        day_dir = self.log_dir / folder / timestamp.strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{face_id}_{timestamp.strftime('%H%M%S_%f')}.jpg"
        path = day_dir / filename
        cv2.imwrite(str(path), crop)
        return str(path).replace("\\", "/")

    def save_face_thumbnail(self, face_id: str, crop: np.ndarray) -> str:
        path = self.log_dir / "faces" / f"{face_id}.jpg"
        cv2.imwrite(str(path), crop)
        return str(path).replace("\\", "/")
