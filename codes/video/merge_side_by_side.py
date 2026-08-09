"""
Merge two videos into a single side-by-side comparison video.

Horizontally concatenates two input videos frame by frame. When
NORMALIZE_RESOLUTION is enabled, both videos are resized to the smaller
shared resolution before merging. A vertical separator line is drawn
between the two panels.

Usage:
    python merge_side_by_side.py
"""

import cv2
import numpy as np
from tqdm import tqdm

# --- Configuration (update paths before running) ---
LEFT_VIDEO_PATH = "path/to/left_video.mp4"
RIGHT_VIDEO_PATH = "path/to/right_video.mp4"
OUTPUT_VIDEO_PATH = "path/to/output_side_by_side.mp4"

# Set to True to resize both videos to the smaller shared resolution
NORMALIZE_RESOLUTION = True

# Visualization
SEPARATOR_COLOR = (255, 255, 255)  # White
# ---


def main():
    """Merge two videos side by side and write the combined output."""
    cap_left = cv2.VideoCapture(LEFT_VIDEO_PATH)
    cap_right = cv2.VideoCapture(RIGHT_VIDEO_PATH)

    if not cap_left.isOpened():
        print(f"Error: Could not open left video '{LEFT_VIDEO_PATH}'.")
        return
    if not cap_right.isOpened():
        print(f"Error: Could not open right video '{RIGHT_VIDEO_PATH}'.")
        return

    width_left = int(cap_left.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_left = int(cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_left = cap_left.get(cv2.CAP_PROP_FPS)

    width_right = int(cap_right.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_right = int(cap_right.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if NORMALIZE_RESOLUTION:
        target_width = min(width_left, width_right)
        target_height = min(height_left, height_right)
        target_fps = fps_left

        print(f"Left video resolution:  {width_left}x{height_left}")
        print(f"Right video resolution: {width_right}x{height_right}")
        print(
            f"Output resolution:      {target_width * 2}x{target_height} "
            f"@ {target_fps:.2f} FPS"
        )
    else:
        target_width = width_left
        target_height = height_left
        target_fps = fps_left

    total_frames = min(
        int(cap_left.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(cap_right.get(cv2.CAP_PROP_FRAME_COUNT)),
    )

    combined_width = target_width * 2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        OUTPUT_VIDEO_PATH, fourcc, target_fps, (combined_width, target_height)
    )

    if not out.isOpened():
        print("Error: Could not create output video.")
        cap_left.release()
        cap_right.release()
        return

    print("Merging videos...")

    for _ in tqdm(range(total_frames), desc="Processing"):
        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()

        if not ret_left or not ret_right:
            break

        if NORMALIZE_RESOLUTION:
            if frame_left.shape[1] != target_width or frame_left.shape[0] != target_height:
                frame_left = cv2.resize(
                    frame_left, (target_width, target_height), interpolation=cv2.INTER_AREA
                )
            if frame_right.shape[1] != target_width or frame_right.shape[0] != target_height:
                frame_right = cv2.resize(
                    frame_right, (target_width, target_height), interpolation=cv2.INTER_AREA
                )

        combined_frame = np.concatenate((frame_left, frame_right), axis=1)
        cv2.line(
            combined_frame,
            (target_width, 0),
            (target_width, target_height),
            SEPARATOR_COLOR,
            2,
        )

        out.write(combined_frame)

    cap_left.release()
    cap_right.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"\nDone. Output saved to '{OUTPUT_VIDEO_PATH}'.")


if __name__ == "__main__":
    main()
