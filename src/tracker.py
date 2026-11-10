"""ByteTrack for face boxes: high-score match, then low-score recovery, Kalman motion."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
    inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def center_distance_ratio(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ac = np.array([(box_a[0] + box_a[2]) / 2.0, (box_a[1] + box_a[3]) / 2.0])
    bc = np.array([(box_b[0] + box_b[2]) / 2.0, (box_b[1] + box_b[3]) / 2.0])
    scale = max(box_a[2] - box_a[0], box_a[3] - box_a[1], box_b[2] - box_b[0], box_b[3] - box_b[1], 1.0)
    return float(np.linalg.norm(ac - bc) / scale)


def association_score(box_a: np.ndarray, box_b: np.ndarray) -> float:
    overlap = iou(box_a, box_b)
    if overlap > 0:
        return overlap
    dist = center_distance_ratio(box_a, box_b)
    if dist < 1.6:
        return 0.45 * (1.6 - dist) / 1.6
    return 0.0


def _xyxy_to_cxcywh(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in box]
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0, max(1.0, x2 - x1), max(1.0, y2 - y1)], dtype=float)


def _cxcywh_to_xyxy(vec: np.ndarray) -> np.ndarray:
    cx, cy, w, h = vec[:4]
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=float)


class _KalmanBBox:
    """Constant-velocity Kalman on (cx, cy, w, h). Helps ByteTrack across skip_frames."""

    def __init__(self, bbox: np.ndarray) -> None:
        self.mean = np.zeros(8, dtype=float)
        self.mean[:4] = _xyxy_to_cxcywh(bbox)
        self.cov = np.eye(8, dtype=float)
        self.cov[:4, :4] *= 10.0
        self.cov[4:, 4:] *= 100.0
        self._f = np.eye(8, dtype=float)
        for i in range(4):
            self._f[i, i + 4] = 1.0
        self._h = np.eye(4, 8, dtype=float)
        self._q = np.eye(8, dtype=float)
        self._q[:4, :4] *= 1.0
        self._q[4:, 4:] *= 1.0
        self._r = np.eye(4, dtype=float) * 1.0

    def predict(self) -> np.ndarray:
        self.mean = self._f @ self.mean
        self.mean[2] = max(1.0, self.mean[2])
        self.mean[3] = max(1.0, self.mean[3])
        self.cov = self._f @ self.cov @ self._f.T + self._q
        return _cxcywh_to_xyxy(self.mean)

    def update(self, bbox: np.ndarray) -> np.ndarray:
        z = _xyxy_to_cxcywh(bbox)
        y = z - self._h @ self.mean
        s = self._h @ self.cov @ self._h.T + self._r
        k = self.cov @ self._h.T @ np.linalg.inv(s)
        self.mean = self.mean + k @ y
        self.cov = (np.eye(8) - k @ self._h) @ self.cov
        self.mean[2] = max(1.0, self.mean[2])
        self.mean[3] = max(1.0, self.mean[3])
        return _cxcywh_to_xyxy(self.mean)


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    face_id: str | None = None
    embedding: np.ndarray | None = None
    counted_in_scene: bool = False
    last_crop: np.ndarray | None = field(default=None, repr=False)
    confirmed: bool = False
    rejected: bool = False
    failed_embeds: int = 0
    kalman: _KalmanBBox | None = field(default=None, repr=False)


def _match(
    tracks: list[Track],
    detections: list[dict],
    iou_threshold: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))
    score = np.zeros((len(tracks), len(detections)), dtype=float)
    for i, track in enumerate(tracks):
        for j, det in enumerate(detections):
            score[i, j] = iou(track.bbox, np.asarray(det["bbox"], dtype=float))
    row_ind, col_ind = linear_sum_assignment(1.0 - score)
    matched: list[tuple[int, int]] = []
    used_t: set[int] = set()
    used_d: set[int] = set()
    for r, c in zip(row_ind, col_ind):
        if score[r, c] >= iou_threshold:
            matched.append((r, c))
            used_t.add(r)
            used_d.add(c)
    unmatched_t = [i for i in range(len(tracks)) if i not in used_t]
    unmatched_d = [j for j in range(len(detections)) if j not in used_d]
    return matched, unmatched_t, unmatched_d


class ByteTracker:
    """ByteTrack: associate high-confidence boxes first, then leftover tracks with low-confidence boxes.

    Detection stays YOLO. This class only links boxes over time so skip_frames still works.
    """

    def __init__(
        self,
        max_age: int,
        min_hits: int,
        iou_threshold: float,
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        track_new_thresh: float | None = None,
    ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.track_new_thresh = track_high_thresh if track_new_thresh is None else track_new_thresh
        self.tracks: list[Track] = []
        self._next_id = 1

    def _apply_match(self, track: Track, det: dict) -> None:
        box = np.asarray(det["bbox"], dtype=float)
        if track.kalman is None:
            track.kalman = _KalmanBBox(box)
        track.bbox = track.kalman.update(box)
        track.hits += 1
        track.time_since_update = 0
        if track.hits >= self.min_hits:
            track.confirmed = True

    def update(self, detections: list[dict]) -> tuple[list[Track], list[Track]]:
        for track in self.tracks:
            track.age += 1
            track.time_since_update += 1
            if track.kalman is not None:
                track.bbox = track.kalman.predict()

        high: list[dict] = []
        low: list[dict] = []
        for det in detections:
            conf = float(det.get("conf", 1.0))
            if conf >= self.track_high_thresh:
                high.append(det)
            elif conf >= self.track_low_thresh:
                low.append(det)

        matched_high, unmatched_t, unmatched_high = _match(self.tracks, high, self.iou_threshold)
        for ti, di in matched_high:
            self._apply_match(self.tracks[ti], high[di])

        leftover_tracks = [self.tracks[i] for i in unmatched_t]
        matched_low, _, _ = _match(leftover_tracks, low, self.iou_threshold)
        for local_i, di in matched_low:
            self._apply_match(leftover_tracks[local_i], low[di])

        for j in unmatched_high:
            det = high[j]
            if float(det.get("conf", 1.0)) < self.track_new_thresh:
                continue
            box = np.asarray(det["bbox"], dtype=float)
            track = Track(track_id=self._next_id, bbox=box, kalman=_KalmanBBox(box))
            self._next_id += 1
            self.tracks.append(track)

        active: list[Track] = []
        exited: list[Track] = []
        for track in self.tracks:
            if track.time_since_update > self.max_age:
                if track.confirmed and track.face_id and track.counted_in_scene:
                    exited.append(track)
            else:
                active.append(track)
        self.tracks = active
        return active, exited


# Older tests and imports used this name.
IoUTracker = ByteTracker
