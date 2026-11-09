"""InsightFace embeddings generated from YOLO crops.

YOLO owns detection. InsightFace is used only to produce a recognition embedding
from a padded face crop so the two modules stay independent, as required.
"""

from __future__ import annotations

import numpy as np
from insightface.app import FaceAnalysis


def resolve_onnx_providers(device: str) -> list[str]:
    if device.startswith("cuda"):
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def face_crop_is_usable(
    det_score: float,
    kps: np.ndarray | None,
    bbox: np.ndarray,
    min_det_score: float,
) -> bool:
    """Accept a crop only when InsightFace is confident it is a real face.

    Five-point landmarks are left eye, right eye, nose, and mouth corners.
    A shoulder or floor patch tends to have a low score or collapsed eyes.
    """
    if det_score < min_det_score:
        return False
    if kps is None or len(kps) < 3:
        return True
    points = np.asarray(kps, dtype=float)
    eye_dist = float(np.linalg.norm(points[0] - points[1]))
    x1, y1, x2, y2 = [float(v) for v in bbox]
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    if eye_dist < 0.18 * width:
        return False
    eye_y = float((points[0][1] + points[1][1]) / 2.0)
    nose_y = float(points[2][1])
    if eye_y > y1 + 0.72 * height:
        return False
    if nose_y < eye_y:
        return False
    return True


class FaceRecognizer:
    """Wraps InsightFace FaceAnalysis recognition on cropped faces."""

    def __init__(self, model_name: str, device: str, min_det_score: float = 0.55) -> None:
        self.min_det_score = min_det_score
        providers = resolve_onnx_providers(device)
        # ctx_id >= 0 selects GPU in InsightFace; -1 forces CPU.
        ctx_id = 0 if device.startswith("cuda") else -1
        self.app = FaceAnalysis(name=model_name, providers=providers)
        # Crops are already face-centric from YOLO, so allow a lower det thresh.
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640), det_thresh=0.2)

    def embed(self, crop_bgr: np.ndarray) -> np.ndarray | None:
        """Return L2-normalized embedding, or None if the crop has no usable face."""
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        h, w = crop_bgr.shape[:2]
        if h < 32 or w < 32:
            return None
        faces = self.app.get(crop_bgr)
        if not faces:
            return None
        faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
        face = faces[0]
        if not face_crop_is_usable(
            float(getattr(face, "det_score", 0.0)),
            getattr(face, "kps", None),
            np.asarray(face.bbox, dtype=float),
            self.min_det_score,
        ):
            return None
        return np.asarray(face.normed_embedding, dtype=np.float32)
