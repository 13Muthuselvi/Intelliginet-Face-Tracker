# Intelligent Face Tracker

AI-driven unique visitor counter: detect faces with YOLO, recognize them with InsightFace, auto-register newcomers, track each person until they leave the frame, and log every entry/exit with a timestamped crop in both SQLite and the filesystem.

## Features

- **Detection** — YOLOv8-face (`yolov8n-face.pt`)
- **Recognition** — InsightFace ArcFace embeddings (`buffalo_sc`)
- **Tracking** — ByteTrack (high-score match, then low-score recovery + Kalman)
- **Auto-registration** — unknown faces get a stable `Face ID`
- **Unique count** — re-identification does not increment the visitor total
- **Logging** — `logs/events.log`, cropped images, SQLite (`data/tracker.db`)
- **Inputs** — sample video file for development; RTSP URL for the interview
- **Config** — `skip_frames` and thresholds in `config.json`
- **Optional UI** — Flask dashboard for counts, gallery, and events

## Architecture

```mermaid
flowchart LR
    SRC[Video / RTSP] --> CAP[OpenCV capture]
    CAP --> SKIP{skip_frames gate}
    SKIP --> YOLO[YOLOv8-face]
    YOLO --> TRK[ByteTrack]
    TRK --> REC[InsightFace embed]
    REC --> GAL{Gallery match}
    GAL -->|new| REG[Register Face ID]
    GAL -->|known| RID[Reuse Face ID]
    REG --> LOG[events.log + crops + SQLite]
    RID --> LOG
    TRK -->|lost track| EXIT[EXIT event]
    TRK -->|confirmed| ENTRY[ENTRY event]
```

Full planning, feature list, and CPU/GPU estimates: [docs/AI_PLANNING.md](docs/AI_PLANNING.md).

## Setup

Python 3.10–3.13 (this machine uses 3.13). A GPU is optional.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
python scripts/download_assets.py
```

`download_assets.py` fetches:

1. `models/yolov8n-face.pt`
2. The sample folder from [Google Drive](https://drive.google.com/drive/folders/15YCN3CYb97GyIoNUV6NJxGNIf-rUBFUJ?usp=sharing) into `data/drive_sample/`, and copies the first video to `data/sample.mp4` when possible.

If Drive download is blocked, copy the provided video to `data/sample.mp4` yourself.

First InsightFace run downloads `buffalo_sc` into `~/.insightface/` (needs network once).

### Run on the sample video

```bash
python app.py --source data/sample.mp4
```

### Run on a live RTSP camera (interview)

```bash
python app.py --source rtsp://USER:PASS@CAMERA_IP:554/Streaming/Channels/101
```

Or set `rtsp_url` in `config.json` and run `python app.py`.

Press `q` in the preview window if you launched with `--display`.

### Review results

```bash
python app.py --dashboard
```

Open http://127.0.0.1:7860

Outputs:

| Path | Contents |
| --- | --- |
| `logs/events.log` | Detection, recognition, registration, tracking, entry, exit |
| `logs/entries/YYYY-MM-DD/` | One cropped image per ENTRY |
| `logs/exits/YYYY-MM-DD/` | One cropped image per EXIT |
| `logs/faces/` | Thumbnail per registered Face ID |
| `logs/annotated_output.mp4` | Overlay video |
| `data/tracker.db` | `faces` + `events` tables |

Unique visitors:

```bash
python -c "from src.database import FaceDatabase; print(FaceDatabase('data/tracker.db').unique_visitor_count())"
```

## Sample `config.json`

```json
{
  "video_source": "data/sample.mp4",
  "rtsp_url": "",
  "skip_frames": 2,
  "yolo_model": "models/yolov8n-face.pt",
  "yolo_confidence": 0.50,
  "yolo_imgsz": 640,
  "insightface_model": "buffalo_sc",
  "embedding_threshold": 0.30,
  "match_margin": 0.05,
  "min_det_score": 0.55,
  "min_box_side": 16,
  "same_face_threshold": 0.60,
  "bbox_expand_ratio": 0.25,
  "tracker_max_age": 40,
  "tracker_min_hits": 2,
  "tracker_iou_threshold": 0.15,
  "track_high_thresh": 0.50,
  "track_low_thresh": 0.10,
  "track_new_thresh": 0.50,
  "exit_grace_frames": 180,
  "device": "auto"
}
```

`skip_frames` is the number of frames to skip **between detection cycles**. `2` means YOLO runs on frames 0, 3, 6, …

## Assumptions

- The Drive sample is a single-camera, mostly frontal or three-quarter view stream.
- A “visitor” is a unique face embedding cluster, not a legal identity. Glasses, masks, or extreme pose can split or merge IDs.
- ENTRY/EXIT are **per Face ID visit**, not per track ID. Track flicker does not add events. A second pair is logged only after a flushed EXIT (they were gone longer than `exit_grace_frames`). Unique count still does not increase on re-id.
- Detection uses YOLO only; InsightFace is not the primary detector. It may run a light inner detector on the crop solely to align for ArcFace.
- `embedding_threshold` 0.30 is the cosine cutoff for `buffalo_sc` (vectors are L2-normalized first). A match also needs a 0.05 gap over the next-best face (`match_margin`) so two similar people are not merged. `same_face_threshold` 0.60 stops a second box from copying an ID already visible. `min_det_score` drops shoulder and floor detections.
- CPU-only laptops should keep `skip_frames >= 2` and `buffalo_sc` (not `buffalo_l`).
- Annotated video uses the `mp4v` codec for portability; install extra codecs only if your player needs them.
- No cloud APIs; everything runs locally so an offline interview machine still works after the first model download.

## Sample output

Committed under [`sample_output/`](sample_output/) from `data/sample.mp4` (`video_sample1.mp4`, 8s 4K crowd clip). That folder is what reviewers should open on GitHub (logs, crops, SQLite). The 40 MB source video is gitignored; restore it with `python scripts/download_assets.py`.

| Metric | Value |
| --- | --- |
| Frames processed | 240 |
| Unique visitors | 20 |
| ENTRY events | 20 |
| EXIT events | 20 (one pair per unique visitor) |

| Path | Contents |
| --- | --- |
| `sample_output/events.log` | Detection, recognition, tracking, ENTRY, EXIT |
| `sample_output/tracker.db` | `faces` (20) + `events` (20+20) |
| `sample_output/entries/2026-11-09/` | Cropped ENTRY images |
| `sample_output/exits/2026-11-09/` | Cropped EXIT images |
| `sample_output/faces/` | Thumbnail per Face ID |

Re-identification does not add a new `faces` row. If the same Face ID is seen again after a short drop, the pending EXIT is cancelled so the count stays one entry / one exit. A local run also writes `logs/annotated_output.mp4`.

```bash
python app.py --source data/sample.mp4 --fresh-events
```

## Demo video

**Demo:** https://www.loom.com/share/b2d74f2840144398b258064217f1d01a

Walkthrough (~5 min): `config.json` / `skip_frames`, the sample video, matching ENTRY/EXIT logs, and cropped face images. The committed sample counts 20 unique visitors after non-face boxes are rejected. Interview camera: `python app.py --source rtsp://...`.

## Project layout

```
app.py
config.json
src/           detector, recognizer, tracker, gallery, db, logger, pipeline
web/           optional dashboard
scripts/       model + Drive downloader
docs/          AI planning and prompt log
logs/          runtime logs and crops
data/          SQLite + sample video (local)
sample_output/ committed logs, crops, and tracker.db
models/        yolov8n-face.pt
```

---

This project is a part of a hackathon run by https://katomaran.com
