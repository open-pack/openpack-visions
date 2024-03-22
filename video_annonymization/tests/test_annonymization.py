import cv2
import numpy as np
import torch

from video_annonymization.annonymization import annonymize_single_image
from video_annonymization.pose_estimation.keypoint_rcnn import KeypointRCnnWeapper
from video_annonymization.segmentation.deeplab import DEEPLAB_PERSON_CLASS_NAME, DeepLabWeapper

pytest_plugins = [
    "video_annonymization.tests.image_fixtures",
]

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_smoke_annonymize_single_image(
    sample_image: np.ndarray,
):
    # init models
    pose_estimator = KeypointRCnnWeapper(device=_DEVICE)
    segmenetation_model = DeepLabWeapper(
        device=_DEVICE, target_classes=(DEEPLAB_PERSON_CLASS_NAME,)
    )

    # apply annonymization
    image_out = annonymize_single_image(sample_image, pose_estimator, segmenetation_model)

    np.testing.assert_array_equal(image_out.shape, (720, 1280, 3))
    cv2.imwrite("./output_annonymized.png", image_out)
