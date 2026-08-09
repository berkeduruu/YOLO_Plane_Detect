"""
Export a YOLO model to TensorRT engine format.

Converts a trained PyTorch (.pt) YOLO model into a TensorRT engine for
faster GPU inference on NVIDIA hardware.

Usage:
    python export_tensorrt_engine.py
"""

from ultralytics import YOLO

# --- Configuration (update paths before running) ---
MODEL_PATH = "path/to/model/best.pt"
EXPORT_IMAGE_SIZE = [720]
EXPORT_DEVICE = 0  # GPU index
# ---

model = YOLO(MODEL_PATH)
model.export(format="engine", imgsz=EXPORT_IMAGE_SIZE, device=EXPORT_DEVICE)
