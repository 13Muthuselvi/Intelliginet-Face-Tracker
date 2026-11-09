"""YOLO-based face detection (Ultralytics YOLOv8-face)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ultralytics import YOLO


def keep_face_box(bbox: np.ndarray, min_side: float) -> bool:
    """Drop tiny boxes and boxes that are far from a face shape."""
    x1, y1, x2, y2 = [float(v) for v in bbox]
    width = x2 - x1
    height = y2 - y1
    if width < min_side or height < min_side or height <= 0:
        return False
    aspect = width / height
    return 0.45 <= aspect <= 2.2


class FaceDetector:
    """Runs YOLOv8 face detection and returns xyxy boxes with scores."""

    def __init__(
        self,
        model_path: str,
        confidence: float,
        imgsz: int,
        device: str,
        min_box_side: float = 16.0,
    ) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"YOLO face model not found at {path}. Run: python scripts/download_assets.py"
            )
        self.model = YOLO(str(path))
        self.confidence = confidence
        self.imgsz = imgsz
        self.device = device
        self.min_box_side = min_box_side

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
            bbox = box.astype(float)
            if not keep_face_box(bbox, self.min_box_side):
                continue
            detections.append(
                {
                    "bbox": bbox,
                    "conf": float(conf),
                }
            )
        return detections
