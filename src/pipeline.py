"""End-to-end detection → tracking → recognition → logging loop."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.database import FaceDatabase
from src.detector import FaceDetector
from src.device import resolve_device
from src.event_logger import EventLogger
from src.gallery import FaceGallery
from src.recognizer import FaceRecognizer
from src.tracker import IoUTracker, Track
from src.video_source import open_capture


def clip_bbox(bbox: np.ndarray, width: int, height: int, expand: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox.tolist()
    bw, bh = x2 - x1, y2 - y1
    x1 -= bw * expand
    y1 -= bh * expand
    x2 += bw * expand
    y2 += bh * expand
    x1 = int(max(0, np.floor(x1)))
    y1 = int(max(0, np.floor(y1)))
    x2 = int(min(width - 1, np.ceil(x2)))
    y2 = int(min(height - 1, np.ceil(y2)))
    return x1, y1, x2, y2


class FaceTrackerPipeline:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.device = resolve_device(config.get("device", "auto"))
        self.skip_frames = int(config.get("skip_frames", 0))
        self.expand = float(config.get("bbox_expand_ratio", 0.25))

        self.logger = EventLogger(config["log_dir"], config["events_log"])
        self.db = FaceDatabase(config["database_path"])
        self.detector = FaceDetector(
            config["yolo_model"],
            float(config["yolo_confidence"]),
            int(config["yolo_imgsz"]),
            self.device,
        )
        self.recognizer = FaceRecognizer(config["insightface_model"], self.device)
        self.tracker = IoUTracker(
            max_age=int(config["tracker_max_age"]),
            min_hits=int(config["tracker_min_hits"]),
            iou_threshold=float(config["tracker_iou_threshold"]),
        )
        self.gallery = FaceGallery(float(config["embedding_threshold"]))
        self.gallery.load(self.db.load_gallery())

        self.latest_frame: np.ndarray | None = None
        self.frame_index = 0
        self.running = False
        # Face IDs with an unmatched ENTRY this run (not track-refcount).
        self._open: set[str] = set()
        self._pending_exit: dict[str, tuple[Track, datetime, int]] = {}
        self._session_events = {"entry": 0, "exit": 0}
        self.exit_grace_frames = int(config.get("exit_grace_frames", 60))

        self.logger.info(
            f"Pipeline ready | device={self.device} | skip_frames={self.skip_frames} "
            f"| gallery_size={len(self.gallery)} | exit_grace_frames={self.exit_grace_frames}"
        )

    def _crop(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = clip_bbox(bbox, w, h, self.expand)
        return frame[y1:y2, x1:x2].copy()

    def _identify(self, track: Track, crop: np.ndarray, timestamp: datetime) -> None:
        embedding = self.recognizer.embed(crop)
        if embedding is None:
            self.logger.warning(f"Embedding skipped for track={track.track_id} (no usable face in crop)")
            return
        track.embedding = embedding
        matched_id, score = self.gallery.match(embedding)
        if matched_id:
            track.face_id = matched_id
            self.db.touch_face(matched_id, timestamp)
            self.logger.info(
                f"RECOGNITION | track={track.track_id} | face_id={matched_id} | cosine={score:.3f}"
            )
            return

        face_id = self.gallery.register(embedding)
        track.face_id = face_id
        thumb_path = self.logger.save_face_thumbnail(face_id, crop)
        self.db.register_face(face_id, embedding.astype(np.float32).tobytes(), timestamp, thumb_path)
        self.logger.info(
            f"REGISTRATION | track={track.track_id} | face_id={face_id} | unique_visitors={len(self.gallery)}"
        )

    def _mark_present(self, track: Track, timestamp: datetime) -> None:
        """Log at most one ENTRY per Face ID until a flushed EXIT."""
        face_id = track.face_id
        if not face_id:
            return
        track.counted_in_scene = True

        if face_id in self._pending_exit:
            self._pending_exit.pop(face_id, None)
            self._open.add(face_id)
            self.logger.info(
                f"TRACKING | face_id={face_id} | track={track.track_id} resumed (exit cancelled)"
            )
            return

        if face_id in self._open:
            return

        self._open.add(face_id)
        self._log_event("entry", track, timestamp)
        self.logger.info(f"TRACKING | face_id={face_id} | track={track.track_id} started")

    def _mark_absent(self, track: Track, timestamp: datetime, still_visible: bool) -> None:
        """Hold EXIT until grace frames pass, so a flickering track does not double-count."""
        face_id = track.face_id
        if not face_id or face_id not in self._open:
            return
        track.counted_in_scene = False
        if still_visible:
            return
        if face_id in self._pending_exit:
            return
        self._pending_exit[face_id] = (track, timestamp, self.frame_index)
        self.logger.info(
            f"TRACKING | face_id={face_id} | track={track.track_id} "
            f"pending-exit frame={self.frame_index}"
        )

    def _flush_pending_exits(self, timestamp: datetime, force: bool = False) -> None:
        due: list[str] = []
        for face_id, (_, _, start_frame) in self._pending_exit.items():
            if force or (self.frame_index - start_frame) >= self.exit_grace_frames:
                due.append(face_id)
        for face_id in due:
            track, event_ts, _ = self._pending_exit.pop(face_id)
            if face_id not in self._open:
                continue
            self._open.discard(face_id)
            self._log_event("exit", track, event_ts)
            self.logger.info(f"TRACKING | face_id={face_id} | track={track.track_id} ended")

    def _log_event(self, event_type: str, track: Track, timestamp: datetime) -> None:
        crop = track.last_crop
        image_path = None
        if crop is not None and crop.size > 0:
            image_path = self.logger.save_crop(event_type, track.face_id or "UNKNOWN", timestamp, crop)
        self.db.insert_event(
            face_id=track.face_id or "UNKNOWN",
            event_type=event_type,
            timestamp=timestamp,
            image_path=image_path,
            track_id=track.track_id,
            frame_index=self.frame_index,
        )
        self._session_events[event_type] = self._session_events.get(event_type, 0) + 1
        self.logger.info(
            f"{event_type.upper()} | face_id={track.face_id} | track={track.track_id} "
            f"| frame={self.frame_index} | image={image_path}"
        )

    def _annotate(self, frame: np.ndarray, tracks: list[Track]) -> np.ndarray:
        canvas = frame.copy()
        unique = len(self._open) or len(self.gallery)
        cv2.rectangle(canvas, (8, 8), (420, 78), (16, 16, 16), -1)
        cv2.putText(
            canvas,
            f"Unique visitors: {unique}",
            (18, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (80, 220, 120),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"Frame: {self.frame_index}",
            (18, 66),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        for track in tracks:
            if not track.confirmed:
                continue
            x1, y1, x2, y2 = track.bbox.astype(int)
            color = (60, 200, 90) if track.face_id else (40, 180, 255)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            label = track.face_id or f"T{track.track_id}"
            cv2.putText(
                canvas,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
        return canvas

    def process_source(self, source: str, write_output: bool = True) -> dict:
        cap = open_capture(source)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

        max_w = int(self.config.get("output_max_width") or 0)
        out_w, out_h = width, height
        if max_w and width > max_w:
            scale = max_w / float(width)
            out_w, out_h = max_w, int(height * scale)

        writer = None
        if write_output and self.config.get("output_video"):
            out_path = Path(self.config["output_video"])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (out_w, out_h))

        self.logger.info(f"SOURCE | {source} | {width}x{height} @ {fps:.1f} FPS")
        self.running = True
        self._open.clear()
        self._pending_exit.clear()
        self._session_events = {"entry": 0, "exit": 0}
        if self.config.get("fresh_events"):
            self.db.clear_events()
            self.logger.info("Cleared previous ENTRY/EXIT rows (faces kept for re-id)")
        active: list[Track] = []

        try:
            while self.running:
                ok, frame = cap.read()
                if not ok:
                    break
                timestamp = datetime.now()
                detect_this_frame = self.frame_index % (self.skip_frames + 1) == 0
                exited: list[Track] = []
                if detect_this_frame:
                    detections = self.detector.detect(frame)
                    self.logger.info(
                        f"DETECTION | frame={self.frame_index} | faces={len(detections)}"
                    )
                    active, exited = self.tracker.update(detections)

                for track in active:
                    if detect_this_frame and track.time_since_update == 0:
                        track.last_crop = self._crop(frame, track.bbox)
                        if track.confirmed and track.face_id is None:
                            self._identify(track, track.last_crop, timestamp)

                visible_ids = {t.face_id for t in active if t.confirmed and t.face_id}
                for track in exited:
                    self._mark_absent(track, timestamp, still_visible=track.face_id in visible_ids)
                for track in active:
                    if track.confirmed and track.face_id:
                        self._mark_present(track, timestamp)
                self._flush_pending_exits(timestamp)

                annotated = self._annotate(frame, active)
                self.latest_frame = annotated
                if writer is not None:
                    frame_out = annotated if (out_w, out_h) == (width, height) else cv2.resize(annotated, (out_w, out_h))
                    writer.write(frame_out)
                if self.config.get("display"):
                    cv2.imshow("Intelligent Face Tracker", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        self.logger.info("Interrupted by user (q)")
                        break
                self.frame_index += 1
        finally:
            # Flush remaining occupants so ENTRY/EXIT stay paired on EOF or interrupt.
            now = datetime.now()
            for track in list(self.tracker.tracks):
                self._mark_absent(track, now, still_visible=False)
            self._flush_pending_exits(now, force=True)
            cap.release()
            if writer is not None:
                writer.release()
            if self.config.get("display"):
                cv2.destroyAllWindows()
            self.running = False

        summary = {
            "frames": self.frame_index,
            "unique_visitors": self._session_events["entry"],
            "gallery_size": self.db.unique_visitor_count(),
            "events": dict(self._session_events),
            "events_all_runs": self.db.event_counts(),
        }
        self.logger.info(
            f"COMPLETE | frames={summary['frames']} | unique={summary['unique_visitors']} "
            f"| entries={summary['events']['entry']} | exits={summary['events']['exit']}"
        )
        return summary

    def close(self) -> None:
        self.db.close()
