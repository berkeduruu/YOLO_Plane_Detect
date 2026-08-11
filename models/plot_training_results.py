"""
Generate training curve charts from models/results.csv.

Usage:
    python plot_training_results.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_CSV = SCRIPT_DIR / "results.csv"
CHARTS_DIR = SCRIPT_DIR / "charts"


def plot_training_loss(df: pd.DataFrame, output_path: Path) -> None:
    """Plot training box, classification, and DFL loss vs epoch."""
    plt.figure(figsize=(10, 6))
    plt.plot(df["epoch"], df["train/box_loss"], label="box loss")
    plt.plot(df["epoch"], df["train/cls_loss"], label="cls loss")
    plt.plot(df["epoch"], df["train/dfl_loss"], label="dfl loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_validation_loss(df: pd.DataFrame, output_path: Path) -> None:
    """Plot validation box, classification, and DFL loss vs epoch."""
    plt.figure(figsize=(10, 6))
    plt.plot(df["epoch"], df["val/box_loss"], label="box loss")
    plt.plot(df["epoch"], df["val/cls_loss"], label="cls loss")
    plt.plot(df["epoch"], df["val/dfl_loss"], label="dfl loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_detection_metrics(df: pd.DataFrame, output_path: Path) -> None:
    """Plot precision, recall, mAP50, and mAP50-95 vs epoch."""
    plt.figure(figsize=(10, 6))
    plt.plot(df["epoch"], df["metrics/precision(B)"], label="precision")
    plt.plot(df["epoch"], df["metrics/recall(B)"], label="recall")
    plt.plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP50")
    plt.plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP50-95")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Detection Metrics")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_summary_card(df: pd.DataFrame, output_path: Path) -> None:
    """Plot final-epoch metric summary as a bar chart."""
    last = df.iloc[-1]
    metrics = {
        "Precision": last["metrics/precision(B)"],
        "Recall": last["metrics/recall(B)"],
        "mAP50": last["metrics/mAP50(B)"],
        "mAP50-95": last["metrics/mAP50-95(B)"],
    }

    plt.figure(figsize=(8, 5))
    bars = plt.bar(list(metrics.keys()), list(metrics.values()), color="#2ecc71")
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title(f"Final Epoch Metrics (epoch {int(last['epoch'])})")
    plt.grid(True, axis="y", alpha=0.3)

    for bar, value in zip(bars, metrics.values()):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    """Read results.csv and write chart PNGs to models/charts/."""
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(f"Results file not found: {RESULTS_CSV}")

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RESULTS_CSV)

    plot_training_loss(df, CHARTS_DIR / "train_loss.png")
    plot_validation_loss(df, CHARTS_DIR / "val_loss.png")
    plot_detection_metrics(df, CHARTS_DIR / "detection_metrics.png")
    plot_summary_card(df, CHARTS_DIR / "final_metrics_summary.png")

    print(f"Charts saved to: {CHARTS_DIR}")


if __name__ == "__main__":
    main()
