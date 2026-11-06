# Intelligent Face Tracker

AI-driven unique visitor counter: detect faces with YOLO, recognize them with InsightFace, auto-register newcomers, track each person until they leave the frame, and log every entry/exit with a timestamped crop in both SQLite and the filesystem.

## Features

- **Detection** — YOLOv8-face (`yolov8n-face.pt`)
- **Recognition** — InsightFace ArcFace embeddings (`buffalo_sc`)
- **Tracking** — IoU + Hungarian assignment (SORT-style)
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
    YOLO --> TRK[IoU tracker]
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

Python 3.10–3.12 recommended. A GPU is optional.

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
  "insightface_model": "buffalo_sc",
  "embedding_threshold": 0.35,
  "tracker_max_age": 15,
  "tracker_min_hits": 2,
  "device": "auto"
}
```

`skip_frames` is the number of frames to skip **between detection cycles**. `2` means YOLO runs on frames 0, 3, 6, …

## Assumptions

- The Drive sample is a single-camera, mostly frontal or three-quarter view stream.
- A “visitor” is a unique face embedding cluster, not a legal identity. Glasses, masks, or extreme pose can split or merge IDs.
- ENTRY/EXIT are **per appearance** (one track life-cycle). If the same person walks out and later returns, that is a second entry/exit pair, but unique count stays the same.
- Detection uses YOLO only; InsightFace is not the primary detector. It may run a light inner detector on the crop solely to align for ArcFace.
- `embedding_threshold` 0.42 is a starting cosine cutoff for `buffalo_sc`. Tune on the sample video if you see splits (raise) or collisions (lower).
- CPU-only laptops should keep `skip_frames >= 2` and `buffalo_sc` (not `buffalo_l`).
- Annotated video uses the `mp4v` codec for portability; install extra codecs only if your player needs them.
- No cloud APIs; everything runs locally so an offline interview machine still works after the first model download.

## Sample output

Checked-in snapshot from `data/sample.mp4` (`video_sample1.mp4`, 8s 4K crowd clip) is in `sample_output/`:

| Metric | Value |
| --- | --- |
| Frames processed | 240 |
| Unique visitors | 30 |
| ENTRY events | 32 |
| EXIT events | 32 (paired with entries) |

Re-identification does not add a new `faces` row; a second ENTRY for the same Face ID is a return into the frame. Inspect `sample_output/events.log`, `sample_output/tracker.db`, and the crop folders. A full local run also writes `logs/annotated_output.mp4`.

## Demo video

Add your Loom / YouTube walkthrough URL here before submitting the Google Form:

**Demo:** _TODO — record a 3–5 minute walkthrough (setup, `config.json`, live run, logs, DB, unique count). Submissions without this video are not reviewed._

Suggested outline:

1. Show `config.json` and `skip_frames`.
2. Run `python app.py --source data/sample.mp4`.
3. Open `logs/events.log`, an entry crop, and SQLite (`faces` / `events`).
4. Show unique count not increasing on re-identification.
5. Mention RTSP: `python app.py --source rtsp://...`.

## Project layout

```
app.py
config.json
src/           detector, recognizer, tracker, gallery, db, logger, pipeline
web/           optional dashboard
scripts/       model + Drive downloader
docs/          AI planning and prompt log
logs/          runtime logs and crops
data/          SQLite + sample video
models/        yolov8n-face.pt
```

---

This project is a part of a hackathon run by https://katomaran.com
