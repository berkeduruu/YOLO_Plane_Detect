"""
Video detection with dynamic Region of Interest (ROI).

Improves detection continuity across consecutive frames. When a target is
found, subsequent frames are cropped to a padded ROI around the detections
(forced to 16:9 aspect ratio) so the model keeps searching near the last known
position. If nothing is detected inside the ROI for SCAN_TIMEOUT seconds, the
pipeline falls back to a full-frame scan.

This does not increase throughput — frame rate stays the same. The goal is to
reduce missed detections between back-to-back frames in fast-moving scenes.

Usage:
    python video_dynamic_roi_detection.py
"""

import cv2
import time
from ultralytics import YOLO
from tqdm import tqdm

# --- Configuration (update paths before running) ---
MODEL_PATH = "path/to/model/best.pt"
INPUT_VIDEO_PATH = "path/to/input.mp4"
OUTPUT_VIDEO_PATH = "path/to/output.mp4"
CONF_THRESHOLD = 0.40

# Dynamic ROI settings
ROI_PADDING = 150
SCAN_TIMEOUT = 0.1  # seconds before falling back to full-frame scan
TARGET_ASPECT_RATIO = 16.0 / 9.0

# Visualization
DETECTION_COLOR = (0, 255, 0)  # Green — detection boxes
ROI_COLOR = (255, 0, 0)        # Blue — active ROI
# ---


def force_aspect_ratio(box, target_aspect, frame_width, frame_height):
    """Expand or shrink a box around its center to match the target aspect ratio."""
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1

    if w <= 0 or h <= 0:
        return box

    current_aspect = w / h
    cx = x1 + w / 2
    cy = y1 + h / 2

    if current_aspect > target_aspect:
        new_h = w / target_aspect
        new_w = w
    else:
        new_w = h * target_aspect
        new_h = h

    new_x1 = cx - new_w / 2
    new_y1 = cy - new_h / 2
    new_x2 = cx + new_w / 2
    new_y2 = cy + new_h / 2

    final_x1 = max(0, int(new_x1))
    final_y1 = max(0, int(new_y1))
    final_x2 = min(frame_width, int(new_x2))
    final_y2 = min(frame_height, int(new_y2))

    return (final_x1, final_y1, final_x2, final_y2)


def main():
    """Run dynamic-ROI video detection and save annotated output."""
    print("Loading model...")
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video '{INPUT_VIDEO_PATH}'.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps_video, (width, height))

    active_roi = None
    last_seen_timestamp = 0

    print("Starting dynamic-ROI video processing...")

    for _ in tqdm(range(total_frames), desc="Processing video"):
        ret, frame = cap.read()
        if not ret:
            break

        scan_area_to_draw = None

        if active_roi and (time.time() - last_seen_timestamp < SCAN_TIMEOUT):
            roi_x1, roi_y1, roi_x2, roi_y2 = active_roi
            scan_area_to_draw = active_roi
            frame_to_process = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        else:
            active_roi = None
            frame_to_process = frame
            roi_x1, roi_y1 = 0, 0

        results = model.predict(frame_to_process, conf=CONF_THRESHOLD, verbose=False)

        detected_boxes_info = []

        if len(results[0].boxes) > 0:
            last_seen_timestamp = time.time()

            min_x1, min_y1 = float("inf"), float("inf")
            max_x2, max_y2 = float("-inf"), float("-inf")

            for box in results[0].boxes:
                bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]

                full_x1, full_y1 = bx1 + roi_x1, by1 + roi_y1
                full_x2, full_y2 = bx2 + roi_x1, by2 + roi_y1

                detected_boxes_info.append({
                    "box": (full_x1, full_y1, full_x2, full_y2),
                    "conf": conf,
                    "class_name": class_name,
                })

                min_x1, min_y1 = min(min_x1, full_x1), min(min_y1, full_y1)
                max_x2, max_y2 = max(max_x2, full_x2), max(max_y2, full_y2)

            initial_roi = (
                max(0, min_x1 - ROI_PADDING),
                max(0, min_y1 - ROI_PADDING),
                min(width, max_x2 + ROI_PADDING),
                min(height, max_y2 + ROI_PADDING),
            )
            active_roi = force_aspect_ratio(
                initial_roi, TARGET_ASPECT_RATIO, width, height
            )

        if scan_area_to_draw:
            rx1, ry1, rx2, ry2 = scan_area_to_draw
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), ROI_COLOR, 2)

        for info in detected_boxes_info:
            x1, y1, x2, y2 = info["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), DETECTION_COLOR, 2)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"\nDone. Output saved to '{OUTPUT_VIDEO_PATH}'.")


if __name__ == "__main__":
    main()
