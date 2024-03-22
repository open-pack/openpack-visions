import cv2
import numpy as np
import torch

from video_annonymization.segmentation.dataclass import SegmentationMask, SegmentationSingleFrame
from video_annonymization.segmentation.deeplab import DeepLabWeapper, draw_segmentation_mask

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pytest_plugins = [
    "video_annonymization.tests.image_fixtures",
]


def test_smoke_DeepLabWrapper(
    sample_image: np.ndarray,
    sample_image_tensor: torch.Tensor,
):
    pose_estimator = DeepLabWeapper(device=_DEVICE)

    outputs = pose_estimator(sample_image_tensor.to(_DEVICE))
    assert isinstance(outputs[0], SegmentationSingleFrame)
    assert isinstance(outputs[0].masks[0], SegmentationMask)
    np.testing.assert_array_equal(outputs[0].masks[0].mask.shape, sample_image.shape[:2])

    img_output = draw_segmentation_mask(sample_image, outputs[0])
    cv2.imwrite("./output_seg.png", img_output)
