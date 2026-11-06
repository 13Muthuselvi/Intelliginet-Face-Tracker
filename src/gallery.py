"""In-memory face gallery: auto-register new embeddings, match returning visitors."""

from __future__ import annotations

import uuid

import numpy as np


class FaceGallery:
    """Stores L2-normalized embeddings keyed by Face ID."""

    def __init__(self, match_threshold: float) -> None:
        self.match_threshold = match_threshold
        self._ids: list[str] = []
        self._matrix: np.ndarray | None = None  # shape (N, D)

    def load(self, rows: list[tuple[str, bytes]]) -> None:
        for face_id, blob in rows:
            embedding = np.frombuffer(blob, dtype=np.float32)
            self._append(face_id, embedding)

    def match(self, embedding: np.ndarray) -> tuple[str | None, float]:
        """Return (face_id, score) if cosine similarity exceeds the threshold."""
        if self._matrix is None or embedding is None:
            return None, 0.0
        vec = embedding.astype(np.float32).reshape(-1)
        scores = self._matrix @ vec
        idx = int(np.argmax(scores))
        score = float(scores[idx])
        if score >= self.match_threshold:
            return self._ids[idx], score
        return None, score

    def register(self, embedding: np.ndarray) -> str:
        face_id = f"F{uuid.uuid4().hex[:8].upper()}"
        self._append(face_id, embedding)
        return face_id

    def _append(self, face_id: str, embedding: np.ndarray) -> None:
        vec = embedding.astype(np.float32).reshape(1, -1)
        self._ids.append(face_id)
        if self._matrix is None:
            self._matrix = vec
        else:
            # Pad / trim if a mixed model somehow produced a different dim.
            dim = self._matrix.shape[1]
            if vec.shape[1] != dim:
                if vec.shape[1] > dim:
                    vec = vec[:, :dim]
                else:
                    pad = np.zeros((1, dim - vec.shape[1]), dtype=np.float32)
                    vec = np.concatenate([vec, pad], axis=1)
            self._matrix = np.vstack([self._matrix, vec])

    def __len__(self) -> int:
        return len(self._ids)
