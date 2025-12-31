# Eye-Controlled Mouse using Computer Vision

A real-time eye-controlled mouse system using webcam input, MediaPipe Face Mesh, and blink detection.

## Features
- Smooth eye-based cursor control
- Blink-to-click interaction
- Dead-zone & smoothing for stability
- ESC key to exit safely

## Tech Stack
- Python
- OpenCV
- MediaPipe
- PyAutoGUI

## How It Works
- Tracks iris position relative to eye center
- Converts gaze displacement to relative cursor movement
- Uses Eye Aspect Ratio (EAR) for blink detection

## Run
```bash
pip install -r requirements.txt
python src/inference.py
