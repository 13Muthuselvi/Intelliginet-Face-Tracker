"""Open a video file or RTSP URL with OpenCV."""

from __future__ import annotations

import cv2


def open_capture(source: str) -> cv2.VideoCapture:
    """
    `source` may be a filesystem path, webcam index as string ('0'), or RTSP URL.
    RTSP uses the FFmpeg backend with a short buffer so interview live streams stay fresh.
    """
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    elif source.lower().startswith(("rtsp://", "rtsps://", "http://", "https://")):
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")
    return cap
