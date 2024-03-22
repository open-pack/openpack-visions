import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection import (
    KeypointRCNN_ResNet50_FPN_Weights,
    keypointrcnn_resnet50_fpn,
)

from video_annonymization.pose_estimation.dataclass import PoseOutput, PoseSingleFrame
from video_annonymization.pose_estimation.wrapper import PoseEstimationWrapper

SCORE_KEY_NAME = "scores"
KEYPOINTS_KEY_NAME = "keypoints"
KEYPOINTS_SCORE_KEY_NAME = "keypoints_scores"

MSCOCO_NUM_JOINTS = 17
MSCOCO_SKELETON = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (0, 5),
    (0, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 13),
    (12, 14),
    (13, 15),
    (14, 16),
]
MSCOCO_NOSE_JOINT_INDEX = 0
MSCOCO_LEFT_SHOULDER_JOINT_INDEX = 5
MSCOCO_RIGHT_SHOULDER_JOINT_INDEX = 6


class KeypointRCnnWeapper(PoseEstimationWrapper):
    def __init__(self, device: torch.device):
        # Model
        weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
        model = keypointrcnn_resnet50_fpn(weights=weights, progress=False).to(device)
        model = model.eval()
        self.model = model

        # Preprocess
        self.transforms = weights.transforms()

        # minimum confidence score for the person bbox
        self.detection_threshold: float = 0.80

    def __call__(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        # Preprocess
        preproc_outputs = [self.transforms(x) for x in inputs]

        # Inference
        inf_outputs = self.inference(preproc_outputs)

        # Postprocess
        postproc_outputs = self.postprocess(inf_outputs)

        return postproc_outputs

    def inference(self, inputs: list[torch.Tensor]) -> list[PoseSingleFrame]:
        outputs = self.model(inputs)
        return outputs

    def postprocess(self, outputs: list[dict]) -> list[PoseSingleFrame]:
        pose_multi_frames = []
        for output_per_image in outputs:
            pose_single_frame = PoseSingleFrame()
            for det_idx in range(len(output_per_image[SCORE_KEY_NAME])):
                bbox_score = output_per_image[SCORE_KEY_NAME][det_idx]
                if bbox_score < self.detection_threshold:
                    continue
                pose_output = PoseOutput(
                    joints=output_per_image[KEYPOINTS_KEY_NAME][det_idx]
                    .detach()
                    .cpu()
                    .numpy()[:, :2],
                    scores=np.ones(
                        MSCOCO_NUM_JOINTS
                    ),  # keypoint_scores of this models are not probablity.
                )
                pose_single_frame.poses.append(pose_output)
            pose_multi_frames.append(pose_single_frame)
        return pose_multi_frames


def draw_keypoint(image_bgr: np.ndarray, pose_single_frame: PoseSingleFrame) -> np.ndarray:
    # OpenCV(BGR) to tensor
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
    image_tensor = torch.from_numpy(image_rgb)

    keypoints = np.stack([pose.joints for pose in pose_single_frame.poses])
    keypoints = torch.from_numpy(keypoints)
    image_tensor = torchvision.utils.draw_keypoints(
        image_tensor,
        keypoints,
        connectivity=MSCOCO_SKELETON,
        colors="blue",
        radius=4,
        width=3,
    )

    # tensor to OpenCV(BGR)
    image_rgb_out = image_tensor.detach().numpy().transpose(1, 2, 0)
    image_bgr_out = cv2.cvtColor(image_rgb_out, cv2.COLOR_RGB2BGR)
    return image_bgr_out
