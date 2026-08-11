"""
Video detection with dynamic ROI and dual-model ensemble verification.

Combines dynamic Region of Interest (ROI) tracking with a two-model ensemble.
Both models run on the same scan area (full frame or ROI). Detections are
matched by IoU; only pairs that pass individual and average confidence gates
are accepted. This reduces false positives compared to single-model ROI
detection while keeping search continuity across consecutive frames.

Usage:
    python video_dynamic_roi_ensemble_detection.py
"""

import time
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

# --- Configuration (update paths before running) ---
MODEL_A_PATH = "path/to/model_a/best.pt"
MODEL_B_PATH = "path/to/model_b/best.pt"
INPUT_VIDEO_PATH = "path/to/input.mp4"
OUTPUT_VIDEO_PATH = "path/to/output.mp4"

# Ensemble thresholds
# Both models run at this low conf to capture as many candidates as possible
DETECT_CONF = 0.10

# A detection is accepted only when:
#   - Both models have a matching detection (IoU)
#   - Each model's conf >= MIN_INDIVIDUAL_CONF
#   - Average conf >= MIN_AVERAGE_CONF
MIN_INDIVIDUAL_CONF = 0.25
MIN_AVERAGE_CONF = 0.45
IOU_THRESHOLD = 0.20

# Dynamic ROI settings
ROI_PADDING = 150
SCAN_TIMEOUT = 0.1  # seconds before falling back to full-frame scan
TARGET_ASPECT_RATIO = 16.0 / 9.0

# Visualization
ENSEMBLE_COLOR = (0, 255, 0)   # Green — accepted detection (both models agree)
REJECTED_COLOR = (0, 0, 255)  # Red — rejected detection (single model only)
ROI_COLOR = (255, 0, 0)       # Blue — active ROI
TEXT_COLOR = (255, 255, 255)
DRAW_REJECTED = True          # Draw rejected detections for debugging
# ---


def calculate_iou(box1, box2):
    """Compute Intersection over Union for two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def match_detections(detections_a, detections_b):
    """
    Match detections from two models using IoU greedy assignment.

    Returns:
        matched: [(det_a, det_b), ...] matched pairs
        unmatched_a: [det_a, ...] detections seen only by model A
        unmatched_b: [det_b, ...] detections seen only by model B
    """
    if not detections_a or not detections_b:
        return [], detections_a, detections_b

    iou_matrix = np.zeros((len(detections_a), len(detections_b)))
    for i, det_a in enumerate(detections_a):
        for j, det_b in enumerate(detections_b):
            iou_matrix[i, j] = calculate_iou(det_a["box"], det_b["box"])

    matched = []
    used_a = set()
    used_b = set()

    while True:
        if iou_matrix.size == 0:
            break
        max_iou = iou_matrix.max()
        if max_iou < IOU_THRESHOLD:
            break

        idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
        i, j = idx[0], idx[1]

        matched.append((detections_a[i], detections_b[j]))
        used_a.add(i)
        used_b.add(j)

        iou_matrix[i, :] = 0
        iou_matrix[:, j] = 0

    unmatched_a = [
        detections_a[i] for i in range(len(detections_a)) if i not in used_a
    ]
    unmatched_b = [
        detections_b[j] for j in range(len(detections_b)) if j not in used_b
    ]

    return matched, unmatched_a, unmatched_b


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


def extract_detections(results, model, roi_x1, roi_y1):
    """Extract detections from model results and map coordinates to full frame."""
    detections = []
    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            bx1, by1, bx2, by2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]

            full_x1, full_y1 = bx1 + roi_x1, by1 + roi_y1
            full_x2, full_y2 = bx2 + roi_x1, by2 + roi_y1

            detections.append({
                "box": (full_x1, full_y1, full_x2, full_y2),
                "conf": conf,
                "class_name": class_name,
            })
    return detections


def main():
    """Run ensemble dynamic-ROI video detection and save annotated output."""
    print(f"Loading model A: {MODEL_A_PATH}")
    model_a = YOLO(MODEL_A_PATH)

    print(f"Loading model B: {MODEL_B_PATH}")
    model_b = YOLO(MODEL_B_PATH)

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
    fps_history = deque(maxlen=10)

    total_accepted = 0
    total_rejected = 0

    print("Starting ensemble + dynamic-ROI video processing...")
    print(f"  -> Model A: {MODEL_A_PATH}")
    print(f"  -> Model B: {MODEL_B_PATH}")
    print(f"  -> Detect conf: {DETECT_CONF}")
    print(f"  -> Min individual conf: {MIN_INDIVIDUAL_CONF}")
    print(f"  -> Min average conf: {MIN_AVERAGE_CONF}")
    print(f"  -> IoU match threshold: {IOU_THRESHOLD}")

    for _ in tqdm(range(total_frames), desc="Processing video"):
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        scan_mode = "FULL FRAME SCAN"
        scan_area_to_draw = None

        if active_roi and (time.time() - last_seen_timestamp < SCAN_TIMEOUT):
            scan_mode = "ROI SCAN"
            roi_x1, roi_y1, roi_x2, roi_y2 = active_roi
            scan_area_to_draw = active_roi
            frame_to_process = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        else:
            active_roi = None
            frame_to_process = frame
            roi_x1, roi_y1 = 0, 0

        results_a = model_a.predict(frame_to_process, conf=DETECT_CONF, verbose=False)
        results_b = model_b.predict(frame_to_process, conf=DETECT_CONF, verbose=False)

        detections_a = extract_detections(results_a, model_a, roi_x1, roi_y1)
        detections_b = extract_detections(results_b, model_b, roi_x1, roi_y1)

        matched, unmatched_a, unmatched_b = match_detections(detections_a, detections_b)

        accepted_detections = []
        rejected_detections = []

        for det_a, det_b in matched:
            conf_a = det_a["conf"]
            conf_b = det_b["conf"]
            avg_conf = (conf_a + conf_b) / 2.0

            if (
                conf_a >= MIN_INDIVIDUAL_CONF
                and conf_b >= MIN_INDIVIDUAL_CONF
                and avg_conf >= MIN_AVERAGE_CONF
            ):
                best_det = det_a if conf_a >= conf_b else det_b
                accepted_detections.append({
                    "box": best_det["box"],
                    "conf_a": conf_a,
                    "conf_b": conf_b,
                    "avg_conf": avg_conf,
                    "class_name": best_det["class_name"],
                })
            else:
                best_det = det_a if conf_a >= conf_b else det_b
                rejected_detections.append({
                    "box": best_det["box"],
                    "conf_a": conf_a,
                    "conf_b": conf_b,
                    "reason": f"A:{conf_a:.2f} B:{conf_b:.2f} avg:{avg_conf:.2f}",
                })

        for det in unmatched_a:
            rejected_detections.append({
                "box": det["box"],
                "conf_a": det["conf"],
                "conf_b": 0.0,
                "reason": f"Only A: {det['conf']:.2f}",
            })

        for det in unmatched_b:
            rejected_detections.append({
                "box": det["box"],
                "conf_a": 0.0,
                "conf_b": det["conf"],
                "reason": f"Only B: {det['conf']:.2f}",
            })

        total_accepted += len(accepted_detections)
        total_rejected += len(rejected_detections)

        if accepted_detections:
            last_seen_timestamp = time.time()

            min_x1, min_y1 = float("inf"), float("inf")
            max_x2, max_y2 = float("-inf"), float("-inf")

            for det in accepted_detections:
                x1, y1, x2, y2 = det["box"]
                min_x1, min_y1 = min(min_x1, x1), min(min_y1, y1)
                max_x2, max_y2 = max(max_x2, x2), max(max_y2, y2)

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

        if DRAW_REJECTED:
            for det in rejected_detections:
                x1, y1, x2, y2 = det["box"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), REJECTED_COLOR, 1)
                cv2.putText(
                    frame,
                    det["reason"],
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    REJECTED_COLOR,
                    1,
                )

        for det in accepted_detections:
            x1, y1, x2, y2 = det["box"]
            label = f"{det['class_name']} A:{det['conf_a']:.2f} B:{det['conf_b']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), ENSEMBLE_COLOR, 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                ENSEMBLE_COLOR,
                2,
            )

        cv2.putText(
            frame,
            f"{scan_mode} | ENSEMBLE",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            TEXT_COLOR,
            2,
            cv2.LINE_AA,
        )

        stats_text = (
            f"Accepted: {len(accepted_detections)} | "
            f"Rejected: {len(rejected_detections)}"
        )
        cv2.putText(
            frame,
            stats_text,
            (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            TEXT_COLOR,
            2,
            cv2.LINE_AA,
        )

        end_time = time.time()
        processing_time = end_time - start_time
        current_fps = 1 / processing_time if processing_time > 0 else 0
        fps_history.append(current_fps)
        avg_fps = sum(fps_history) / len(fps_history)

        fps_text = f"FPS: {avg_fps:.2f}"
        cv2.putText(
            frame,
            fps_text,
            (width - 200, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            TEXT_COLOR,
            2,
            cv2.LINE_AA,
        )

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"\nDone. Output saved to '{OUTPUT_VIDEO_PATH}'.")
    print(f"Total accepted detections: {total_accepted}")
    print(f"Total rejected detections: {total_rejected}")
    if total_accepted + total_rejected > 0:
        acceptance_rate = total_accepted / (total_accepted + total_rejected) * 100
        print(f"Acceptance rate: {acceptance_rate:.1f}%")


if __name__ == "__main__":
    main()
