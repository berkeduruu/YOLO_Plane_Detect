"""
Video object detection with YOLO bounding boxes.

Processes an input video frame by frame, runs YOLO detection on each frame,
draws bounding boxes on detected objects, and writes the annotated result
to an output video file.

Usage:
    python video_detection.py
"""

import cv2
from ultralytics import YOLO
from tqdm import tqdm

# --- Configuration (update paths before running) ---
MODEL_PATH = "path/to/model/best.pt"
INPUT_VIDEO_PATH = "path/to/input.mp4"
OUTPUT_VIDEO_PATH = "path/to/output.mp4"
CONF_THRESHOLD = 0.1
BOX_COLOR = (0, 255, 255)  # Yellow (BGR)
INFERENCE_SIZE = [1280, 736]
# ---


def main():
    """Process a video with standard YOLO detection and save annotated output."""
    print(f"Loading model '{MODEL_PATH}'...")
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

    print("Starting video processing...")

    for _ in tqdm(range(total_frames), desc="Processing video"):
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            frame, conf=CONF_THRESHOLD, verbose=False, imgsz=INFERENCE_SIZE
        )

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("\nDone.")
    print(f"Output saved to '{OUTPUT_VIDEO_PATH}'.")


if __name__ == "__main__":
    main()
