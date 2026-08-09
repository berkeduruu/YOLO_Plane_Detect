"""
Video instance segmentation with YOLO masks.

Processes an input video frame by frame, runs YOLO segmentation on each frame,
overlays semi-transparent masks with contour outlines, and saves the result
to an output video file.

Usage:
    python video_segmentation.py
"""

import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

# --- Configuration (update paths before running) ---
MODEL_PATH = "path/to/model/best.pt"
INPUT_VIDEO_PATH = "path/to/input.mp4"
OUTPUT_VIDEO_PATH = "path/to/output.mp4"
CONF_THRESHOLD = 0.1
MASK_COLOR = (0, 255, 255)  # Yellow (BGR)
MASK_ALPHA = 0.4
INFERENCE_SIZE = [1280, 736]
# ---


def draw_segmentation(frame, results, color, alpha=0.4):
    """Draw segmentation masks with semi-transparent fill and contour outlines."""
    if results.masks is None:
        return frame

    overlay = frame.copy()

    for mask in results.masks:
        polygon = mask.xy[0].astype(np.int32)
        if len(polygon) < 3:
            continue
        cv2.fillPoly(overlay, [polygon], color)

    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    for mask in results.masks:
        polygon = mask.xy[0].astype(np.int32)
        if len(polygon) >= 3:
            cv2.polylines(frame, [polygon], True, color, 2)

    return frame


def main():
    """Process a video with YOLO segmentation and save annotated output."""
    print(f"Loading segmentation model '{MODEL_PATH}'...")
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video '{INPUT_VIDEO_PATH}'. Check the file path.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

    if not out.isOpened():
        print("Error: Could not create output video. Check write permissions or codec.")
        cap.release()
        return

    print("Starting video segmentation...")

    for _ in tqdm(range(total_frames), desc="Processing video"):
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            frame, conf=CONF_THRESHOLD, verbose=False, imgsz=INFERENCE_SIZE
        )
        frame = draw_segmentation(frame, results[0], MASK_COLOR, MASK_ALPHA)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("\nDone.")
    print(f"Output saved to '{OUTPUT_VIDEO_PATH}'.")


if __name__ == "__main__":
    main()
