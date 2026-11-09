"""Logic tests that do not require YOLO or InsightFace weights."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detector import keep_face_box
from src.gallery import FaceGallery, choose_identity
from src.recognizer import face_crop_is_usable
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


def test_one_id_cannot_cover_two_people() -> None:
    ranked = [("FSHARED", 0.47), ("FOTHER", 0.20)]
    action, face_id, _ = choose_identity(ranked, {"FSHARED"}, 0.42, 0.60)
    assert action == "new" and face_id is None


def test_second_box_on_same_person_is_suppressed() -> None:
    ranked = [("FSAME", 0.81)]
    action, face_id, score = choose_identity(ranked, {"FSAME"}, 0.42, 0.60)
    assert action == "duplicate" and face_id == "FSAME" and score > 0.8


def test_box_shape_and_landmarks_reject_non_faces() -> None:
    assert keep_face_box(np.array([0.0, 0.0, 40.0, 48.0]), 16)
    assert not keep_face_box(np.array([0.0, 0.0, 80.0, 10.0]), 16)
    bbox = np.array([0.0, 0.0, 100.0, 120.0])
    good_kps = np.array([[30.0, 40.0], [70.0, 40.0], [50.0, 70.0], [35.0, 95.0], [65.0, 95.0]])
    assert face_crop_is_usable(0.8, good_kps, bbox, 0.55)
    assert not face_crop_is_usable(0.2, good_kps, bbox, 0.55)
    collapsed = np.array([[50.0, 40.0], [52.0, 40.0], [51.0, 70.0], [40.0, 95.0], [60.0, 95.0]])
    assert not face_crop_is_usable(0.9, collapsed, bbox, 0.55)


if __name__ == "__main__":
    test_gallery_registers_then_matches()
    test_tracker_entry_and_exit()
    test_one_id_cannot_cover_two_people()
    test_second_box_on_same_person_is_suppressed()
    test_box_shape_and_landmarks_reject_non_faces()
    print("ok")
