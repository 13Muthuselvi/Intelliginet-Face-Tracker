"""Logic tests that do not require YOLO or InsightFace weights."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gallery import FaceGallery
from src.tracker import IoUTracker


def test_gallery_registers_then_matches() -> None:
    gallery = FaceGallery(match_threshold=0.4)
    vec = np.zeros(16, dtype=np.float32)
    vec[0] = 1.0
    face_id = gallery.register(vec)
    matched, score = gallery.match(vec)
    assert matched == face_id
    assert score > 0.99
    other = np.zeros(16, dtype=np.float32)
    other[1] = 1.0
    matched_other, _ = gallery.match(other)
    assert matched_other is None


def test_tracker_entry_and_exit() -> None:
    tracker = IoUTracker(max_age=2, min_hits=1, iou_threshold=0.3)
    box = {"bbox": np.array([10.0, 10.0, 50.0, 50.0])}
    active, exited = tracker.update([box])
    assert len(active) == 1 and not exited
    active[0].confirmed = True
    active[0].face_id = "FTEST"
    active[0].counted_in_scene = True
    _, exited = tracker.update([])
    assert not exited
    _, exited = tracker.update([])
    assert not exited
    _, exited = tracker.update([])
    assert len(exited) == 1
    assert exited[0].face_id == "FTEST"


if __name__ == "__main__":
    test_gallery_registers_then_matches()
    test_tracker_entry_and_exit()
    print("ok")
