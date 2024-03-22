import cv2
import numpy as np
import torch

from video_annonymization.pose_estimation.keypoint_rcnn import KeypointRCnnWeapper, draw_keypoint

pytest_plugins = [
    "video_annonymization.tests.image_fixtures",
]


_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_smoke_KeypointRCnnWeapper(
    sample_image: np.ndarray,
    sample_image_tensor: torch.Tensor,
):
    pose_estimator = KeypointRCnnWeapper(device=_DEVICE)

    x = [sample_image_tensor.to(_DEVICE)]
    outputs = pose_estimator(x)

    img_output = draw_keypoint(sample_image, outputs[0])
    cv2.imwrite("./output.png", img_output)
