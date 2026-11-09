"""In-memory face gallery: auto-register new embeddings, match returning visitors."""

from __future__ import annotations

import uuid

import numpy as np


def choose_identity(
    ranked: list[tuple[str, float]],
    blocked: set[str],
    match_threshold: float,
    same_face_threshold: float,
) -> tuple[str, str | None, float]:
    """Pick an identity that is not already on another visible person.

    Returns (action, face_id, score) where action is "match", "new", or "duplicate".
    A very high score against an ID already on screen is a second box on the same
    person. A lower score against that same ID is a different person.
    """
    if not ranked:
        return "new", None, 0.0
    best_id, best_score = ranked[0]
    if best_id not in blocked and best_score >= match_threshold:
        return "match", best_id, best_score
    if best_id in blocked and best_score >= same_face_threshold:
        return "duplicate", best_id, best_score
    for face_id, score in ranked:
        if face_id in blocked:
            continue
        if score >= match_threshold:
            return "match", face_id, score
        break
    return "new", None, best_score


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
        ranked = self.ranked(embedding)
        if not ranked:
            return None, 0.0
        face_id, score = ranked[0]
        if score >= self.match_threshold:
            return face_id, score
        return None, score

    def ranked(self, embedding: np.ndarray) -> list[tuple[str, float]]:
        """All gallery faces sorted by cosine similarity, highest first."""
        if self._matrix is None or embedding is None:
            return []
        vec = embedding.astype(np.float32).reshape(-1)
        scores = self._matrix @ vec
        order = np.argsort(-scores)
        return [(self._ids[int(i)], float(scores[int(i)])) for i in order]

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
