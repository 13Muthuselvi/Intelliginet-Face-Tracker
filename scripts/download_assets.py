"""Download the YOLOv8-face weights and optionally the hackathon sample video folder."""

from __future__ import annotations

import argparse
from pathlib import Path

import requests

YOLO_FACE_URL = (
    "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.pt"
)
DRIVE_FOLDER = (
    "https://drive.google.com/drive/folders/15YCN3CYb97GyIoNUV6NJxGNIf-rUBFUJ?usp=sharing"
)


def download_yolo(dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"YOLO model already present: {dest}")
        return
    print(f"Downloading {YOLO_FACE_URL}")
    with requests.get(YOLO_FACE_URL, stream=True, timeout=120) as response:
        response.raise_for_status()
        tmp = dest.with_suffix(".part")
        with tmp.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if chunk:
                    handle.write(chunk)
        tmp.replace(dest)
    print(f"Saved {dest} ({dest.stat().st_size} bytes)")


def download_drive_folder(target: Path) -> None:
    try:
        import gdown
    except ImportError as exc:
        raise SystemExit("Install gdown (pip install gdown) to fetch the Drive folder.") from exc
    target.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Drive folder into {target}")
    gdown.download_folder(url=DRIVE_FOLDER, output=str(target), quiet=False)
    videos = list(target.rglob("*.mp4")) + list(target.rglob("*.avi"))
    preferred = [p for p in videos if "sample" in p.name.lower()]
    chosen = (preferred or videos)
    if chosen:
        sample = Path("data/sample.mp4")
        sample.parent.mkdir(parents=True, exist_ok=True)
        sample.write_bytes(chosen[0].read_bytes())
        print(f"Copied {chosen[0]} -> {sample}")
    else:
        print("No video files found in the Drive folder. Place one at data/sample.mp4 manually.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-video", action="store_true", help="Only download YOLO weights")
    args = parser.parse_args()
    download_yolo(Path("models/yolov8n-face.pt"))
    if not args.skip_video:
        download_drive_folder(Path("data/drive_sample"))


if __name__ == "__main__":
    main()
