# Codes

Utility scripts for YOLO-based aircraft detection, video processing, and model export.

All scripts use placeholder paths (`path/to/...`) in their configuration sections.
Update these paths before running.

## Directory layout

```
codes/
├── inference/          # Model inference scripts
│   ├── image_inference.py
│   ├── video_detection.py
│   ├── video_segmentation.py
│   └── video_dynamic_roi_detection.py
├── export/             # Model export utilities
│   └── export_tensorrt_engine.py
├── train/              # Training notebooks
│   └── YoloPlane.ipynb
└── video/              # Video post-processing
    └── merge_side_by_side.py
```

## Scripts

### Inference (`inference/`)

| Script | Description |
|--------|-------------|
| `image_inference.py` | Run a single YOLO prediction on one image. Quick test for model weights and resolution. |
| `video_detection.py` | Process a video with bounding-box detection; draw boxes and save annotated output. |
| `video_segmentation.py` | Process a video with instance segmentation; overlay semi-transparent masks. |
| `video_dynamic_roi_detection.py` | Video detection with a dynamic ROI that tracks the last known target position to reduce missed detections between consecutive frames (not a throughput optimization). |

### Export (`export/`)

| Script | Description |
|--------|-------------|
| `export_tensorrt_engine.py` | Export a `.pt` YOLO model to TensorRT `.engine` format for faster NVIDIA GPU inference. |

### Training (`train/`)

| Notebook | Description |
|----------|-------------|
| `YoloPlane.ipynb` | Colab notebook for dataset prep, train/val split, optional hyperparameter tuning, and YOLO11 plane-detector training. |

### Video (`video/`)

| Script | Description |
|--------|-------------|
| `merge_side_by_side.py` | Combine two videos into one side-by-side comparison clip with an optional resolution-normalization step. |

## Migration from old filenames

| Old name | New name |
|----------|----------|
| `YOLO.py` | `inference/image_inference.py` |
| `YOLO_video.py` | `inference/video_detection.py` |
| `YOLO_video_seg.py` | `inference/video_segmentation.py` |
| `YOLO_roi.py` | `inference/video_dynamic_roi_detection.py` |
| `create_engine.py` | `export/export_tensorrt_engine.py` |
| `YOLO_merge_videos_dynamic.py` | `video/merge_side_by_side.py` |
| `Compare_Merge.py` | `video/merge_side_by_side.py` (set `NORMALIZE_RESOLUTION = False`) |
