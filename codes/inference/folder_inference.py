"""
Batch folder inference with YOLO label export.

Runs detection on every image in a folder and writes YOLO-format label files
(normalized class cx cy w h). Optionally saves annotated images for review.

Usage:
    python folder_inference.py
"""

from pathlib import Path

import cv2
from ultralytics import YOLO
from tqdm import tqdm

# --- Configuration (update paths before running) ---
MODEL_PATH = "path/to/model/best.pt"
INPUT_FOLDER = "path/to/images"
OUTPUT_LABELS_DIR = "path/to/output_labels"
OUTPUT_VIS_DIR = "path/to/output_vis"
SAVE_ANNOTATED = False
CONF_THRESHOLD = 0.25
INFERENCE_SIZE = (1280, 736)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
# ---


def collect_images(input_folder: Path) -> list[Path]:
    """Return sorted image paths from a folder."""
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(input_folder.glob(f"*{ext}"))
    return sorted(images)


def write_yolo_labels(label_path: Path, boxes) -> None:
    """Write YOLO detection labels (class cx cy w h, normalized)."""
    lines = []
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls[0])
            cx, cy, w, h = box.xywhn[0].tolist()
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def main():
    """Run batch inference on a folder and export YOLO labels."""
    input_folder = Path(INPUT_FOLDER)
    labels_dir = Path(OUTPUT_LABELS_DIR)
    vis_dir = Path(OUTPUT_VIS_DIR)

    if not input_folder.is_dir():
        print(f"Error: Input folder not found: '{INPUT_FOLDER}'")
        return

    image_paths = collect_images(input_folder)
    if not image_paths:
        print(f"Error: No images found in '{INPUT_FOLDER}'")
        return

    labels_dir.mkdir(parents=True, exist_ok=True)
    if SAVE_ANNOTATED:
        vis_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model '{MODEL_PATH}'...")
    model = YOLO(MODEL_PATH)

    total_detections = 0

    for image_path in tqdm(image_paths, desc="Processing images"):
        results = model.predict(
            str(image_path),
            conf=CONF_THRESHOLD,
            imgsz=INFERENCE_SIZE,
            verbose=False,
        )

        boxes = results[0].boxes
        detection_count = len(boxes) if boxes is not None else 0
        total_detections += detection_count

        label_path = labels_dir / f"{image_path.stem}.txt"
        write_yolo_labels(label_path, boxes)

        if SAVE_ANNOTATED:
            annotated = results[0].plot()
            out_path = vis_dir / image_path.name
            cv2.imwrite(str(out_path), annotated)

    print(f"\nDone. Processed {len(image_paths)} images.")
    print(f"Labels saved to: '{labels_dir}'")
    print(f"Total detections: {total_detections}")
    if SAVE_ANNOTATED:
        print(f"Annotated images saved to: '{vis_dir}'")


if __name__ == "__main__":
    main()
