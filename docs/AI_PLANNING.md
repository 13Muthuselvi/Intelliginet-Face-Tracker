# AI Planning Document — Intelligent Face Tracker

This document records the app-building workflow used for the Katomaran hackathon task.

## 1. Planning the app

The brief asks for a **unique visitor counter** that:

1. Reads a video file now, and an RTSP camera at interview time.
2. Detects faces with **YOLO**.
3. Builds embeddings with a SOTA recognizer (**InsightFace**), not `face_recognition`.
4. Auto-registers a new Face ID on first sight.
5. Tracks the same person across frames until they leave.
6. Writes **exactly one ENTRY and one EXIT** per appearance, with a cropped image and timestamp.
7. Stores metadata in a database **and** on disk (`events.log` + dated image folders).
8. Counts unique visitors without incrementing on re-identification.

Planning decisions:

| Concern | Choice | Why |
| --- | --- | --- |
| Detection | YOLOv8n-face | Required YOLO family; nano keeps CPU feasible |
| Recognition | InsightFace `buffalo_sc` | ArcFace embeddings, lighter than `buffalo_l` on CPU |
| Tracking | IoU + Hungarian (SORT-style) | Stable IDs between detections; no extra GPU tracker weights |
| Database | SQLite + WAL | Zero ops, interview-portable, survives crashes |
| Config | `config.json` `skip_frames` | Explicitly required |
| UI | Optional Flask dashboard | Brief says UI is an added advantage |
| Input | File path or RTSP URL | Same pipeline, different `cv2.VideoCapture` backend |

Processing loop (one frame):

```
read frame
  → if frame_index % (skip_frames+1) == 0: YOLO detect
  → associate boxes to tracks (IoU)
  → for a new confirmed track: InsightFace embed → match gallery or register
  → first confirmation with a Face ID: ENTRY log + image + DB row
  → track lost for max_age frames: EXIT log + image + DB row
  → overlay Face ID and unique count; optionally write annotated MP4
```

## 2. Feature list

- Load settings from `config.json` (skip interval, thresholds, paths, device).
- Open a local video, webcam index, or RTSP URL.
- YOLOv8 face detection with confidence filter.
- Skip N frames between detection cycles to control compute.
- SORT-style multi-object tracking so the same person keeps a track ID.
- InsightFace embedding on the YOLO crop (detection and recognition stay decoupled).
- Cosine match against an in-memory gallery loaded from SQLite.
- Auto-register unknown faces (`F` + 8 hex chars).
- Persist embeddings so a later run can re-identify the same people.
- ENTRY/EXIT image store: `logs/entries/YYYY-MM-DD/` and `logs/exits/YYYY-MM-DD/`.
- `logs/events.log` for detection, embedding, registration, tracking, entry, exit.
- Unique visitor count = number of rows in `faces` (re-id does not insert a new face).
- Annotated output video `logs/annotated_output.mp4`.
- Flush open tracks as EXIT on Ctrl+C / end-of-file so logs stay paired.
- Optional dashboard: unique count, gallery, recent events.
- Asset downloader for YOLO weights and the Drive sample folder.

## 3. Compute load estimate

Assumptions: 1280×720, ~2 concurrent faces, `skip_frames=2` (detect every 3rd frame), `yolov8n-face` + `buffalo_sc`.

| Stage | CPU (approx.) | GPU (approx.) |
| --- | --- | --- |
| YOLO n-face @ 640 | 40–90 ms | 4–8 ms |
| InsightFace embed / new or unmatched crop | 30–80 ms | 5–12 ms |
| Tracking + I/O | < 2 ms | < 1 ms |
| **Effective throughput** | **8–15 FPS** processed display; detections at 1/3 rate | **25–40+ FPS** |

Memory (steady state):

- Python + OpenCV + Ultralytics: ~0.8–1.2 GB RAM
- InsightFace buffalo_sc: ~200–400 MB extra RAM
- GPU VRAM if CUDA: ~1.0–1.5 GB

`skip_frames` is the main knob. Raising it from 0 → 2 cuts detector cost by ~3× and is the recommended default for laptop CPUs. `buffalo_l` roughly doubles recognition cost; keep `buffalo_sc` unless accuracy on the sample video is insufficient.

## 4. Architecture

```
                    config.json
                         │
                         ▼
                      app.py
                         │
            ┌────────────┼────────────┐
            ▼            ▼            ▼
      video_source   pipeline     Flask dashboard
            │            │
            ▼            ▼
         OpenCV    FaceDetector (YOLOv8)
                         │
                         ▼
                   IoUTracker
                         │
                         ▼
              FaceRecognizer (InsightFace)
                         │
                         ▼
                    FaceGallery
                    ↙         ↘
            FaceDatabase     EventLogger
            (SQLite WAL)     events.log + crops
```

## 5. Interview notes

- Swap `video_source` for `--source rtsp://...` or set `rtsp_url` in `config.json`.
- Do not use the `face_recognition` package.
- Unique count is `SELECT COUNT(*) FROM faces`, not entry-event count (re-entry of the same person must not increase uniqueness).
- One ENTRY and one EXIT are emitted per **track appearance**, not per frame.
