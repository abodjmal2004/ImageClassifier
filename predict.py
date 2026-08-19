"""Predict whether an image contains a cat or a dog."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread

from utils import load_image

DEFAULT_CATEGORIES = ["cat", "dog"]
DEFAULT_IMAGE_SIZE = (40, 40)


def load_model_bundle(model_path: str | Path) -> tuple[object, list[str], tuple[int, int]]:
    """Load the current model bundle while remaining compatible with legacy .pkl files."""
    bundle = joblib.load(model_path)
    if isinstance(bundle, dict) and "model" in bundle:
        return (
            bundle["model"],
            list(bundle.get("categories", DEFAULT_CATEGORIES)),
            tuple(bundle.get("image_size", DEFAULT_IMAGE_SIZE)),
        )
    return bundle, DEFAULT_CATEGORIES, DEFAULT_IMAGE_SIZE


def predict_image(
    image_path: str | Path,
    model_path: str | Path = "artifacts/svm_model.joblib",
    output_path: str | Path = "results/prediction.png",
) -> dict:
    """Run inference and save an annotated preview image."""
    try:
        model, categories, image_size = load_model_bundle(model_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Model not found at '{model_path}'. Run train.py before predicting."
        ) from exc

    original = np.asarray(imread(image_path))
    features = load_image(image_path, image_size).reshape(1, -1)
    prediction_index = int(model.predict(features)[0])
    predicted_class = categories[prediction_index]

    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        confidence = float(probabilities[prediction_index])

    result = {
        "image": str(image_path),
        "prediction": predicted_class,
        "confidence": confidence,
    }
    confidence_label = f" ({confidence:.1%} confidence)" if confidence is not None else ""
    print(f"Prediction: {predicted_class.upper()}{confidence_label}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(original, cmap="gray" if original.ndim == 2 else None)
    ax.set_title(f"Prediction: {predicted_class.upper()}{confidence_label}", weight="bold")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved preview to {output}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classify one image as cat or dog.")
    parser.add_argument("--image", type=Path, required=True, help="Path to an image file.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("artifacts/svm_model.joblib"),
        help="Path to a trained model bundle.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/prediction.png"),
        help="Path for the annotated prediction preview.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    predict_image(args.image, args.model_path, args.output)


if __name__ == "__main__":
    main()
