import cv2
import numpy as np
import torch

COLOR_RGBA_FOREGROUND = (0, 0, 0, 255)
COLOR_RGBA_BACKGROUND = (0, 0, 0, 0)


def image_bgr_to_tensor(image_bgr: np.ndarray) -> torch.Tensor:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
    image_tensor = torch.from_numpy(image_rgb)
    return image_tensor


def tensor_to_image_bgr(image_tensor: torch.Tensor) -> np.ndarray:
    image_rgb = image_tensor.numpy().transpose(1, 2, 0)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return image_bgr


def apply_mask_to_image(img_bgra: np.ndarray, mask_bgra: np.ndarray) -> np.ndarray:
    """
    Return:
        BGRA image
    """
    img_bgra[mask_bgra[:, :, 3] > 0] = 0  # remove color at where mask is applied
    img_masked = img_bgra + mask_bgra
    return img_masked
