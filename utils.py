"""Utilities for loading images and saving model-evaluation artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from skimage.color import gray2rgb
from skimage.io import imread
from skimage.transform import resize
from sklearn.metrics import auc, confusion_matrix, roc_curve

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _as_rgb(image: np.ndarray) -> np.ndarray:
    """Convert grayscale/RGBA images to a three-channel RGB array."""
    if image.ndim == 2:
        return gray2rgb(image)
    if image.ndim == 3 and image.shape[2] == 4:
        return image[:, :, :3]
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    raise ValueError(f"Unsupported image shape: {image.shape}")


def load_image(image_path: str | Path, target_size: tuple[int, int] = (40, 40)) -> np.ndarray:
    """Read, convert, resize, and normalize one image to ``float32`` RGB."""
    image = _as_rgb(np.asarray(imread(image_path)))
    resized = resize(
        image,
        (target_size[0], target_size[1], 3),
        anti_aliasing=True,
        preserve_range=False,
    )
    return np.asarray(resized, dtype=np.float32)


def _image_paths(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def _category_directories(data_dir: Path, category: str, splits: Sequence[str]) -> list[Path]:
    """Support both ``root/category`` and ``root/split/category`` layouts."""
    split_dirs = [data_dir / split / category for split in splits]
    existing_split_dirs = [path for path in split_dirs if path.is_dir()]
    if existing_split_dirs:
        return existing_split_dirs
    direct_dir = data_dir / category
    return [direct_dir] if direct_dir.is_dir() else []


def load_and_preprocess_data(
    data_dir: str | Path,
    categories: Sequence[str],
    target_size: tuple[int, int] = (40, 40),
    splits: Sequence[str] = ("train", "test"),
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Load labeled images from a conventional cat/dog directory structure.

    The loader accepts either ``data_dir/<category>`` or
    ``data_dir/<split>/<category>`` and returns flattened normalized RGB arrays.
    """
    data_dir = Path(data_dir)
    samples: list[np.ndarray] = []
    labels: list[int] = []
    errors: list[str] = []

    for class_index, category in enumerate(categories):
        category_dirs = _category_directories(data_dir, category, splits)
        paths = [path for directory in category_dirs for path in _image_paths(directory)]
        if verbose:
            print(f"  {category}: {len(paths)} image(s)")
        for image_path in paths:
            try:
                samples.append(load_image(image_path, target_size).ravel())
                labels.append(class_index)
            except (OSError, ValueError) as exc:
                errors.append(f"{image_path}: {exc}")

    if not samples:
        expected = " or ".join(
            [f"{data_dir}/{category}" for category in categories]
            + [f"{data_dir}/<split>/<category>" for category in categories]
        )
        raise FileNotFoundError(
            "No supported images were found. Expected a structure such as " + expected
        )

    if errors and verbose:
        print(f"Skipped {len(errors)} unreadable image(s).")
        for error in errors[:5]:
            print(f"  - {error}")

    X = np.asarray(samples, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    if verbose:
        print(f"Loaded {len(y)} image(s); feature matrix shape: {X.shape}")
    return X, y


def _ensure_parent(save_path: str | Path) -> Path:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: Sequence[str],
    save_path: str | Path = "results/confusion_matrix.png",
) -> Path:
    """Save a readable confusion-matrix heatmap and return its path."""
    path = _ensure_parent(save_path)
    matrix = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
    )
    ax.set_title("Confusion Matrix", weight="bold")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: str | Path = "results/roc_curve.png",
) -> Path:
    """Save the ROC curve for the positive class and return its path."""
    path = _ensure_parent(save_path)
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#2563eb", lw=2.5, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color="#94a3b8", lw=1.5, linestyle="--")
    ax.set(xlim=(0, 1), ylim=(0, 1.05), xlabel="False Positive Rate", ylabel="True Positive Rate")
    ax.set_title("Receiver Operating Characteristic", weight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_metrics(metrics: dict, save_path: str | Path = "results/metrics.json") -> Path:
    """Persist evaluation metrics as formatted JSON."""
    path = _ensure_parent(save_path)
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
