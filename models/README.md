Models directory:
- `yolov8n-face.pt` — YOLOv8 face detector weights
- `buffalo_sc/` — InsightFace lightweight model pack (`det_500m.onnx`, `w600k_mbf.onnx`)

Download both models automatically via:
```bash
python scripts/download_assets.py --skip-video
```

Sources:
- YOLOv8 Face: https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.pt
- Buffalo_sc: https://github.com/deepinsight/insightface/releases/download/model-zoo/buffalo_sc.zip
