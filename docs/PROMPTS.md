# Prompts used while building this project

The brief states that prompts may be reviewed in the interview. This file is the working log of how the application was specified and generated.

## Prompt 1 — Scope from the problem statement

Help me implement Katomaran’s Intelligent Face Tracker hackathon task from the PDF: YOLO face detection, InsightFace embeddings, auto-registration, tracking until exit, entry/exit logs with cropped images and timestamps in both a database and the filesystem, unique visitor counting, `config.json` skip-frames, README with planning/architecture, and RTSP support for the interview. No `face_recognition` library.

## Prompt 2 — Architecture constraints

Write modular Python (not a notebook): detector, recognizer, tracker, gallery, SQLite, event logger, pipeline, CLI, optional Flask dashboard. Detection must be YOLO; recognition must be InsightFace on YOLO crops so the two stages stay independent.

## Prompt 3 — Logging contract

Every appearance must produce exactly one entry event and one exit event. Store crops under `logs/entries/YYYY-MM-DD/` and `logs/exits/YYYY-MM-DD/`. `events.log` must include detection, embedding/registration, tracking, entry, and exit. Database writes should use WAL so an interrupt does not corrupt the file. Flush remaining tracks as exits when the video ends.

## Prompt 4 — Delivery artifacts

Produce `README.md` with setup, assumptions, sample `config.json`, AI planning, architecture diagram, compute estimates, and the line: “This project is a part of a hackathon run by https://katomaran.com”. Include a download script for yolov8n-face weights and the Drive sample folder.
