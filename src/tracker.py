"""IoU + centroid tracker (SORT-style) for face boxes across frames."""

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
    """Euclidean center distance divided by the larger box's max side."""
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


class IoUTracker:
    """Associates detections to tracks; drops tracks after max_age missed updates."""

    def __init__(self, max_age: int, min_hits: int, iou_threshold: float) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(self, detections: list[dict]) -> tuple[list[Track], list[Track]]:
        for track in self.tracks:
            track.age += 1
            track.time_since_update += 1

        det_boxes = [np.asarray(d["bbox"], dtype=float) for d in detections]
        matched_track_idx: set[int] = set()
        matched_det_idx: set[int] = set()

        if self.tracks and det_boxes:
            score = np.zeros((len(self.tracks), len(det_boxes)), dtype=float)
            for i, track in enumerate(self.tracks):
                for j, box in enumerate(det_boxes):
                    score[i, j] = association_score(track.bbox, box)
            row_ind, col_ind = linear_sum_assignment(1.0 - score)
            for r, c in zip(row_ind, col_ind):
                if score[r, c] >= self.iou_threshold:
                    track = self.tracks[r]
                    track.bbox = det_boxes[c]
                    track.hits += 1
                    track.time_since_update = 0
                    if track.hits >= self.min_hits:
                        track.confirmed = True
                    matched_track_idx.add(r)
                    matched_det_idx.add(c)

        for j, box in enumerate(det_boxes):
            if j in matched_det_idx:
                continue
            self.tracks.append(Track(track_id=self._next_id, bbox=box))
            self._next_id += 1

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
