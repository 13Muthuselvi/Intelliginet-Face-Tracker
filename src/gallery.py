"""In-memory face gallery: auto-register new embeddings, match returning visitors."""

from __future__ import annotations

import uuid

import numpy as np


def l2_normalize(embedding: np.ndarray) -> np.ndarray:
    vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return vec
    return vec / norm


def cosine_scores(gallery: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Cosine similarity: both sides must be L2-normalized. Range [-1, 1]."""
    return gallery @ l2_normalize(query)


def choose_identity(
    ranked: list[tuple[str, float]],
    blocked: set[str],
    match_threshold: float,
    same_face_threshold: float,
    match_margin: float = 0.05,
) -> tuple[str, str | None, float]:
    """Pick an identity that is not already on another visible person.

    Match only if cosine >= threshold and the winner is clearly better than
    the next unused face (margin). That keeps matching correct at a low cutoff.
    """
    if not ranked:
        return "new", None, 0.0
    best_id, best_score = ranked[0]
    if best_id in blocked and best_score >= same_face_threshold:
        return "duplicate", best_id, best_score

    unblocked = [(face_id, score) for face_id, score in ranked if face_id not in blocked]
    if not unblocked:
        return "new", None, best_score
    face_id, score = unblocked[0]
    if score < match_threshold:
        return "new", None, score
    if len(unblocked) > 1:
        second = unblocked[1][1]
        if second >= match_threshold and (score - second) < match_margin:
            return "new", None, score
    return "match", face_id, score


class FaceGallery:
    """Stores L2-normalized embeddings keyed by Face ID."""

    def __init__(self, match_threshold: float, match_margin: float = 0.05) -> None:
        self.match_threshold = match_threshold
        self.match_margin = match_margin
        self._ids: list[str] = []
        self._matrix: np.ndarray | None = None  # shape (N, D)

    def load(self, rows: list[tuple[str, bytes]]) -> None:
        for face_id, blob in rows:
            embedding = np.frombuffer(blob, dtype=np.float32)
            self._append(face_id, embedding)

    def match(self, embedding: np.ndarray) -> tuple[str | None, float]:
        """Return (face_id, score) if cosine similarity is a clear match."""
        action, face_id, score = choose_identity(
            self.ranked(embedding),
            set(),
            self.match_threshold,
            same_face_threshold=1.1,
            match_margin=self.match_margin,
        )
        if action == "match":
            return face_id, score
        return None, score

    def ranked(self, embedding: np.ndarray) -> list[tuple[str, float]]:
        """All gallery faces sorted by cosine similarity, highest first."""
        if self._matrix is None or embedding is None:
            return []
        scores = cosine_scores(self._matrix, embedding)
        order = np.argsort(-scores)
        return [(self._ids[int(i)], float(scores[int(i)])) for i in order]

    def register(self, embedding: np.ndarray) -> str:
        face_id = f"F{uuid.uuid4().hex[:8].upper()}"
        self._append(face_id, embedding)
        return face_id

    def _append(self, face_id: str, embedding: np.ndarray) -> None:
        vec = l2_normalize(embedding).reshape(1, -1)
        self._ids.append(face_id)
        if self._matrix is None:
            self._matrix = vec
            return
        dim = self._matrix.shape[1]
        if vec.shape[1] != dim:
            if vec.shape[1] > dim:
                vec = vec[:, :dim]
            else:
                pad = np.zeros((1, dim - vec.shape[1]), dtype=np.float32)
                vec = np.concatenate([vec, pad], axis=1)
            vec = l2_normalize(vec).reshape(1, -1)
        self._matrix = np.vstack([self._matrix, vec])

    def __len__(self) -> int:
        return len(self._ids)
