# YOLO Plane Detect v0.1

Fixed-wing aircraft detection for the TEKNOFEST Fighting UAV competition, built with custom-trained YOLOv11 models. This repository includes training notebooks, inference scripts, export utilities, and reference results for aerial target detection.

## Dataset Access

The training dataset is **not included** in this repository. If you need access, please send a request to **[berkeduruu@gmail.com](mailto:berkeduruu@gmail.com)**.

## Building Your Own Dataset

Two companion tools from the same author can help you go from raw video to a labeled YOLO dataset:

| Tool | Repository | Purpose |
|------|------------|---------|
| **Video Frame Grabber** | [berkeduruu/Video-Frame-Grabber](https://github.com/berkeduruu/Video-Frame-Grabber) | Extract frames from video files with precise time-range control, multiple extraction modes, and image filters — ideal for turning flight footage into training images. |
| **YOLO Supported Annotation Tool** | [berkeduruu/YOLO_Supported_Annotation_Tool](https://github.com/berkeduruu/YOLO_Supported_Annotation_Tool) | Annotate extracted frames in YOLO format. If you already have a trained model, it can pre-annotate frames automatically so you can spot model errors early and correct them while building a new dataset. |

**Suggested workflow:** extract frames with Video Frame Grabber → annotate (and refine model mistakes) with the Annotation Tool → train with [`codes/train/YoloPlane.ipynb`](codes/train/YoloPlane.ipynb).

## Repository Layout

```
YOLO_Plane_Detect/
├── codes/
│   ├── inference/     # Image & video inference scripts
│   ├── export/        # TensorRT engine export
│   ├── train/         # Colab training notebook
│   └── video/         # Video post-processing utilities
├── models/            # Trained model weights (if provided)
└── assets/            # Demo media
```

See [`codes/README.md`](codes/README.md) for a full script reference.

## Jetson Deployment

Models in this project were tested on **Jetson Orin Nano Super** and **Jetson Orin NX 16 GB** under competition conditions.

### Without DeepStream (OpenCV / Ultralytics pipeline)

When inference runs outside a DeepStream pipeline, model size should match the board:

| Board | Recommended model |
|-------|-------------------|
| Jetson Orin Nano Super | **Nano** (YOLOv11n) |
| Jetson Orin NX 16 GB | **Small** (YOLOv11s) — runs reliably |

### With DeepStream (GStreamer pipeline)

A **DeepStream + GStreamer** setup uses the Jetson hardware far more efficiently than pulling frames through OpenCV-style loops. In our experience, this architecture delivers noticeably better real-world performance on the same boards.

| Board | Recommended models |
|-------|-------------------|
| Jetson Orin Nano Super | **Small** or **Medium** (Super can also be evaluated) |
| Jetson Orin NX 16 GB | **Medium** runs comfortably; larger variants are worth exploring |

For production-ready DeepStream pipeline templates (camera/UDP inputs, YOLO integration, streaming output, concurrent recording), see:

**[berkeduruu/jetson-deepstream-yolo-pipelines](https://github.com/berkeduruu/jetson-deepstream-yolo-pipelines)**

## Dynamic ROI Detection

[`codes/inference/video_dynamic_roi_detection.py`](codes/inference/video_dynamic_roi_detection.py) implements a **dynamic Region of Interest (ROI)** strategy to improve detection continuity across consecutive frames.

### How it works

1. **Full-frame scan** — When no target is being tracked, every frame is processed at full resolution.
2. **ROI lock-on** — After a detection, a padded ROI (forced to 16:9) is placed around the target and subsequent frames are cropped to that region before inference.
3. **Timeout fallback** — If nothing is detected inside the ROI for a short period (`SCAN_TIMEOUT`), the pipeline returns to full-frame scanning.

### Why it helps

In fast-moving aerial scenes, a target can be detected in one frame and missed in the next because it shifts to a different part of the image or occupies fewer pixels. By keeping the search area centered on the last known position, the ROI approach reduces the *"saw it once, lost it on the next frame"* problem and makes back-to-back detections more consistent.

> **Note:** This ROI logic is about **detection reliability**, not throughput. Frame rate stays the same — the goal is fewer missed targets between consecutive frames, not higher FPS.

### Demo

In the video below, the **blue rectangle** marks the active ROI search area. **Green boxes** are detections.

<video src="assets/dynamic_roi.mp4" controls loop muted autoplay width="100%"></video>

Place your ROI demo video at `assets/dynamic_roi.mp4` for the preview above to appear on GitHub.

## Quick Start

1. Update placeholder paths in the script you want to run (`path/to/model/best.pt`, etc.).
2. **Inference:** `python codes/inference/video_detection.py`
3. **Training:** open `codes/train/YoloPlane.ipynb` in Google Colab.
4. **TensorRT export:** `python codes/export/export_tensorrt_engine.py`

## License

See [LICENSE](LICENSE).
