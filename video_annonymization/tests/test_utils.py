import cv2
import numpy as np

from video_annonymization.utils import apply_mask_to_image

COLOR_RGBA_FOREGROUND = (0, 0, 0, 255)


def test_apply_mask_to_image():
    bbox = (0, 0, 50, 50)  # Mask area (xywh)
    img_bgra = np.full((100, 100, 4), 255, dtype=np.uint8)
    mask_bgra = np.zeros((100, 100, 4), dtype=np.uint8)
    mask_bgra[bbox[1] : (bbox[1] + bbox[3]), bbox[0] : (bbox[0] + bbox[2]), :] = (
        COLOR_RGBA_FOREGROUND
    )
    img_masked_expected = np.full((100, 100, 4), 255, dtype=np.uint8)
    img_masked_expected[bbox[1] : (bbox[1] + bbox[3]), bbox[0] : (bbox[0] + bbox[2]), :3] = 0

    img_masked = apply_mask_to_image(img_bgra, mask_bgra)

    # cv2.imwrite("./output.png", img_masked)
    np.testing.assert_array_equal(img_masked, img_masked_expected)
