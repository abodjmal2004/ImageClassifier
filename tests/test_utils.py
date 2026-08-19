from pathlib import Path

import numpy as np
from PIL import Image

from utils import load_and_preprocess_data, load_image


def _write_image(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array).save(path)


def test_load_image_normalizes_grayscale_to_rgb(tmp_path: Path) -> None:
    image_path = tmp_path / "gray.png"
    _write_image(image_path, np.full((12, 10), 128, dtype=np.uint8))

    image = load_image(image_path, target_size=(8, 8))

    assert image.shape == (8, 8, 3)
    assert image.dtype == np.float32
    assert 0.0 <= float(image.min()) <= float(image.max()) <= 1.0


def test_load_and_preprocess_data_supports_train_test_layout(tmp_path: Path) -> None:
    for split in ("train", "test"):
        for label, value in (("cat", 40), ("dog", 200)):
            folder = tmp_path / split / label
            folder.mkdir(parents=True)
            _write_image(folder / f"{label}.png", np.full((10, 10, 3), value, dtype=np.uint8))

    features, labels = load_and_preprocess_data(
        tmp_path,
        categories=("cat", "dog"),
        target_size=(6, 6),
        verbose=False,
    )

    assert features.shape == (4, 6 * 6 * 3)
    assert sorted(set(labels.tolist())) == [0, 1]
