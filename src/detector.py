"""YOLO-based face detection (Ultralytics YOLOv8-face)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ultralytics import YOLO


class FaceDetector:
    """Runs YOLOv8 face detection and returns xyxy boxes with scores."""

    def __init__(self, model_path: str, confidence: float, imgsz: int, device: str) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"YOLO face model not found at {path}. Run: python scripts/download_assets.py"
            )
        self.model = YOLO(str(path))
        self.confidence = confidence
        self.imgsz = imgsz
        self.device = device

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Return a list of {bbox: [x1,y1,x2,y2], conf: float}."""
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        detections: list[dict] = []
        if not results:
            return detections
        boxes = results[0].boxes
        if boxes is None or boxes.xyxy is None:
            return detections
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        for box, conf in zip(xyxy, confs):
            detections.append(
                {
                    "bbox": box.astype(float),
                    "conf": float(conf),
                }
            )
        return detections
