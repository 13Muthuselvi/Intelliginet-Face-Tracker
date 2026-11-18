"""Benchmark & Accuracy Evaluation tool for the Intelligent Face Tracker.

Compares tracker performance and output metrics against ground-truth visitor counts
to compute Precision, Recall, Accuracy, Tracking Stability, and Processing FPS.

Usage:
    python evaluate.py
    python evaluate.py --ground-truth 22 --source data/sample.mp4
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.config_loader import load_config
from src.pipeline import FaceTrackerPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Face Tracker Accuracy & Performance")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    parser.add_argument(
        "--source",
        default="data/sample.mp4",
        help="Path to evaluation video file",
    )
    parser.add_argument(
        "--ground-truth",
        type=int,
        default=22,
        help="Known actual unique visitors in the test video (default: 22 for data/sample.mp4)",
    )
    return parser.parse_args()


def run_evaluation(config_path: str, source: str, ground_truth: int) -> None:
    print("=" * 65)
    print("  INTELLIGENT FACE TRACKER - ACCURACY & PERFORMANCE EVALUATION")
    print("=" * 65)

    if not Path(source).exists():
        raise SystemExit(f"Evaluation source not found: {source}")

    cfg = load_config(config_path)
    if cfg is None:
        cfg = {}

    # Run in headless mode without saving video for fast benchmark
    cfg["display"] = False
    cfg["fresh_events"] = True

    print(f"\n[1/3] Initializing pipeline...")
    print(f"      Command           : python evaluate.py")
    print(f"      Source Video      : {source}")
    print(f"      Ground Truth Count: {ground_truth} unique visitors")
    print(f"      Detector Model    : {cfg.get('yolo_model')}")
    print(f"      Recognizer Model  : {cfg.get('insightface_model')}")

    pipeline = FaceTrackerPipeline(cfg)

    print(f"\n[2/3] Processing video stream...")
    start_time = time.perf_counter()
    try:
        summary = pipeline.process_source(source, write_output=False)
    finally:
        pipeline.close()
    elapsed_time = time.perf_counter() - start_time

    # Metric calculations
    detected_visitors = summary["unique_visitors"]
    entries = summary["events"]["entry"]
    exits = summary["events"]["exit"]
    total_frames = summary["frames"]
    fps = total_frames / elapsed_time if elapsed_time > 0 else 0.0

    # True Positives: accurately identified visitors up to ground truth
    true_positives = min(detected_visitors, ground_truth)
    false_positives = max(0, detected_visitors - ground_truth)
    false_negatives = max(0, ground_truth - detected_visitors)

    # Standard metrics
    precision = (true_positives / (true_positives + false_positives)) * 100 if (true_positives + false_positives) > 0 else 0.0
    recall = (true_positives / (true_positives + false_negatives)) * 100 if (true_positives + false_negatives) > 0 else 0.0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = (1.0 - abs(detected_visitors - ground_truth) / ground_truth) * 100 if ground_truth > 0 else 0.0

    # Tracking event consistency (pairing of entry and exit)
    event_diff = abs(entries - exits)
    tracking_consistency = (1.0 - (event_diff / max(entries, 1))) * 100

    print(f"\n[3/3] Results & Metrics:")
    print("-" * 65)
    print(f"  Total Frames Processed : {total_frames}")
    print(f"  Total Processing Time  : {elapsed_time:.2f} seconds")
    print(f"  Processing Throughput  : {fps:.2f} FPS")
    print("-" * 65)
    print(f"  Ground Truth Visitors  : {ground_truth}")
    print(f"  System Detected Count  : {detected_visitors}")
    print(f"  Total Entries Logged   : {entries}")
    print(f"  Total Exits Logged     : {exits}")
    print("-" * 65)
    print(f"  Counting Accuracy      : {accuracy:.1f}%")
    print(f"  Precision              : {precision:.1f}%")
    print(f"  Recall (Sensitivity)   : {recall:.1f}%")
    print(f"  F1-Score               : {f1_score:.1f}%")
    print(f"  Tracking Consistency   : {tracking_consistency:.1f}% (Entry/Exit parity)")
    print("=" * 65)


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args.config, args.source, args.ground_truth)
