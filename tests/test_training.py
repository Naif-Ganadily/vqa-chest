"""Fast, offline tests for reproducibility and feature-engineering.

These deliberately avoid downloading the dataset or the DistilBERT weights so
they stay fast and runnable in CI. Heavier end-to-end checks belong behind the
``slow`` marker.
"""

import numpy as np
import torch
from PIL import Image

from src.pipelines.feature_eng_pipeline import build_img_transform
from src.utils import set_seed


def test_set_seed_makes_torch_reproducible():
    set_seed(123)
    first = torch.rand(5)
    set_seed(123)
    second = torch.rand(5)
    assert torch.equal(first, second)


def test_set_seed_makes_numpy_reproducible():
    set_seed(7)
    first = np.random.rand(5)
    set_seed(7)
    second = np.random.rand(5)
    assert np.allclose(first, second)


def test_img_transform_output_shape():
    config = {"img_size": 224}
    transform = build_img_transform(config)
    img = Image.new("RGB", (300, 400), color=(120, 120, 120))

    tensor = transform(img)

    assert tensor.shape == (3, 224, 224)
