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


class FaceRecognizer:
    """Wraps InsightFace FaceAnalysis recognition on cropped faces."""

    def __init__(self, model_name: str, device: str) -> None:
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
        return np.asarray(faces[0].normed_embedding, dtype=np.float32)
