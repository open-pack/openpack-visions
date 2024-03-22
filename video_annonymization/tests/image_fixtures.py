from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

_SAMPLE_DIR = Path(__file__).parents[2] / "samples"


@pytest.fixture
def sample_image_path() -> Path:
    return Path(_SAMPLE_DIR, "openpack-sample-raw-1.png")


@pytest.fixture
def sample_image(sample_image_path: Path) -> np.ndarray:
    img = cv2.imread(str(sample_image_path))
    return img


@pytest.fixture
def sample_image_tensor(sample_image: np.ndarray) -> torch.Tensor:
    image_rgb = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
    image_tensor = torch.from_numpy(image_rgb)
    return image_tensor
