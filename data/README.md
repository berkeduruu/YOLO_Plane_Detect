# Dataset Guide

The training dataset is **not included** in this repository. For access, contact **[berkeduruu@gmail.com](mailto:berkeduruu@gmail.com)**.

This folder documents the expected YOLO dataset layout and provides a [`data.yaml.example`](data.yaml.example) template for Ultralytics training.

## Raw Dataset Layout

After extracting or collecting labeled frames:

```
dataset/
├── images/
│   ├── frame_0001.jpg
│   └── ...
└── labels/
    ├── frame_0001.txt
    └── ...
```

Each image must have a matching label file with the same stem (e.g. `frame_0001.jpg` → `frame_0001.txt`).

## Train / Validation Split Layout

The training notebook ([`codes/train/YoloPlane.ipynb`](../codes/train/YoloPlane.ipynb)) splits data into:

```
dataset_split/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── data.yaml
```

Copy [`data.yaml.example`](data.yaml.example) to your split folder as `data.yaml` and update `path` to the absolute path of `dataset_split`.

## Label Format

YOLO detection format — one line per object:

```
class_id cx cy w h
```

All coordinates are **normalized** (0–1) relative to image width and height.

| Field | Description |
|-------|-------------|
| `class_id` | Integer class index |
| `cx`, `cy` | Box center x and y |
| `w`, `h` | Box width and height |

This project uses a **single class**:

| Class ID | Name |
|----------|------|
| 0 | `plane` |

### Negative Samples

Images with no aircraft should have an **empty** `.txt` file (or a file with no lines). This helps the model learn background scenes.

## Building a Dataset

For the full data-building workflow (frame extraction → annotation → training), see the main [README](../README.md#building-your-own-dataset).

## Training

After preparing `dataset_split/data.yaml`:

```bash
yolo train data=/path/to/dataset_split/data.yaml model=yolo11n.pt imgsz=640
```

Or open the Colab notebook for the full pipeline (subset, tuning, augmentation).
