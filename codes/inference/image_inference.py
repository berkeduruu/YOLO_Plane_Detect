"""
Single-image YOLO inference.

Runs a one-off prediction on a still image using a trained YOLO model.
Useful for quick sanity checks on model weights and input resolution.

Usage:
    python image_inference.py
"""

from ultralytics import YOLO

# --- Configuration (update paths before running) ---
MODEL_PATH = "path/to/model/best.pt"
IMAGE_PATH = "path/to/image.jpg"
INFERENCE_SIZE = (1280, 736)
# ---

model = YOLO(MODEL_PATH)
results = model.predict(IMAGE_PATH, imgsz=INFERENCE_SIZE)
