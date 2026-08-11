# YOLO Plane Detect v1

<div align="center">

[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB)](https://www.python.org/)
[![YOLOv11](https://img.shields.io/badge/YOLO-v11-00A6D6)](https://docs.ultralytics.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8)](https://opencv.org/)
[![License](https://img.shields.io/badge/license-GPL--3.0-blue)](LICENSE)
[![Version](https://img.shields.io/badge/version-v1-orange)](https://github.com/berkeduruu/YOLO_Plane_Detect)

Fixed-wing aircraft detection for the TEKNOFEST Fighting UAV competition, built with custom-trained YOLOv11 models. This repository includes training notebooks, inference scripts, export utilities, and reference results for aerial target detection.

### Contributors

| [Berke DURU](https://github.com/berkeduruu) | [Tuna ÜNÜVAR](https://github.com/tunaunuvar) | Yiğit AVCIOĞLU | Yağız YUNGUL |
|:---:|:---:|:---:|:---:|
| [berkeduruu@gmail.com](mailto:berkeduruu@gmail.com) | [unuvartuna00@gmail.com](mailto:unuvartuna00@gmail.com) | — | — |

</div>

---

## Demo — YOLO Lock-On

Example competition footage with YOLO target lock-on on fixed-wing aircraft:

<table>
  <tr>
    <td width="50%"><video src="https://github.com/user-attachments/assets/ef8e3192-4cc0-4f32-b6cb-491e925e3b9d" controls muted playsinline width="100%"></video></td>
    <td width="50%"><video src="https://github.com/user-attachments/assets/ef98dbec-0755-4568-b2a1-4a8455cea856" controls muted playsinline width="100%"></video></td>
  </tr>
</table>

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
├── codes/             # Inference, export, training, video utilities
├── models/            # Model weights, training logs, charts
├── data/              # Dataset layout guide and data.yaml template
└── assets/            # Demo media
```

See [`codes/README.md`](codes/README.md) for a full script reference.

Trained weights (`models/N_new720p.pt`, `models/S_new720p.pt`) and training logs (`models/results.csv`, `models/results.png`) are included. See [`models/README.md`](models/README.md) for the model reference.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For **TensorRT engine export** on an NVIDIA GPU:

```bash
pip install -r requirements-export.txt
```

| Environment | Notes |
|-------------|-------|
| **General inference / training** | `requirements.txt` is enough. `ultralytics` pulls in PyTorch and most dependencies automatically. |
| **NVIDIA CUDA GPU** | If the default PyTorch wheel does not match your CUDA version, install the matching build from [pytorch.org](https://pytorch.org). |
| **AMD GPU (ROCm)** | YOLO runs on AMD GPUs with a ROCm PyTorch build — not CPU-only. Install the ROCm wheel from [pytorch.org](https://pytorch.org) (select ROCm) before or after `requirements.txt`. |
| **CPU-only** | Inference and training work without a GPU; expect lower throughput. Skip `requirements-export.txt` and TensorRT export. |
| **TensorRT export** | NVIDIA GPU only. Export must run on the same GPU architecture you plan to deploy on. |
| **Jetson** | Use JetPack-matched PyTorch and system TensorRT; export on the target device. See [Jetson Deployment](#jetson-deployment) and [jetson-deepstream-yolo-pipelines](https://github.com/berkeduruu/jetson-deepstream-yolo-pipelines). |
| **Google Colab** | The training notebook installs `ultralytics` directly — no local venv needed. |

On **NVIDIA CUDA** and **AMD ROCm** GPUs, **FP16 inference** (`half=True` in `model.predict()` or export) is recommended for higher throughput when your GPU supports it. See [`models/README.md`](models/README.md#inference-recommendations).

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
| Jetson Orin Nano Super | **Small** |
| Jetson Orin NX 16 GB | **Medium** runs comfortably |

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

In the clip below, the **blue rectangle** marks the active ROI search area. **Green boxes** are detections.

<video src="https://github.com/user-attachments/assets/d355c2d1-823e-4994-a910-8ff87155be75" controls muted playsinline width="100%"></video>

## Ensemble ROI Detection

[`codes/inference/video_dynamic_roi_ensemble_detection.py`](codes/inference/video_dynamic_roi_ensemble_detection.py) extends dynamic ROI detection with **dual-model ensemble verification**.

### How it works

1. **Same ROI pipeline** — Full-frame scan, ROI lock-on, and timeout fallback behave like [`video_dynamic_roi_detection.py`](codes/inference/video_dynamic_roi_detection.py).
2. **Two-model inference** — Both models run on the same scan area (full frame or ROI) at a low `DETECT_CONF` to collect candidates.
3. **IoU matching** — Detections from model A and model B are paired by greedy IoU assignment.
4. **Confidence gating** — A pair is accepted only when each model's confidence exceeds `MIN_INDIVIDUAL_CONF` and their average exceeds `MIN_AVERAGE_CONF`. Unmatched or low-confidence detections are rejected.

### Why it helps

Single-model ROI detection can still produce false positives in cluttered aerial scenes. Requiring agreement from two independently trained models filters many spurious detections while keeping the continuity benefits of ROI tracking.

> **Note:** Running two models per frame is slower than single-model ROI. Tune `DRAW_REJECTED` and confidence thresholds for your latency budget.

## Quick Start

1. Install dependencies (see [Installation](#installation)).
2. Update placeholder paths in the script you want to run (`path/to/model/best.pt`, etc.).
3. **Single image:** `python codes/inference/image_inference.py`
4. **Video inference:** `python codes/inference/video_detection.py`
5. **Folder batch inference + label export:** `python codes/inference/folder_inference.py`
6. **Training:** open `codes/train/YoloPlane.ipynb` in Google Colab.
7. **TensorRT export:** `python codes/export/export_tensorrt_engine.py`

## License

See [LICENSE](LICENSE).
