#!/opt/conda/bin/python
""" Add black masks around faces of subjects.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import tqdm
from loguru import logger

from video_annonymization.pose_estimation.dataclass import PoseSingleFrame
from video_annonymization.pose_estimation.keypoint_rcnn import (
    MSCOCO_LEFT_SHOULDER_JOINT_INDEX,
    MSCOCO_NOSE_JOINT_INDEX,
    MSCOCO_RIGHT_SHOULDER_JOINT_INDEX,
    KeypointRCnnWeapper,
)
from video_annonymization.pose_estimation.wrapper import PoseEstimationWrapper
from video_annonymization.segmentation.dataclass import SegmentationMask
from video_annonymization.segmentation.deeplab import DEEPLAB_PERSON_CLASS_NAME, DeepLabWeapper
from video_annonymization.segmentation.wrapper import SegmentationWrapper
from video_annonymization.utils import (
    COLOR_RGBA_FOREGROUND,
    apply_mask_to_image,
    image_bgr_to_tensor,
)

KEYPOINT_THRESHOLD = 0.2
BBOX_CONFIDENCE_THRESHOLD = 0.2

FACE_JOINT_INDICES = (
    MSCOCO_NOSE_JOINT_INDEX,
    MSCOCO_LEFT_SHOULDER_JOINT_INDEX,
    MSCOCO_RIGHT_SHOULDER_JOINT_INDEX,
)

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"`{_DEVICE}` is selected as a compute device.")


def exist_valid_keypoint(pose_single_frame: PoseSingleFrame) -> bool:
    result = False
    for pose in pose_single_frame.poses:
        kpt_score = pose.scores[MSCOCO_NOSE_JOINT_INDEX]
        if kpt_score <= KEYPOINT_THRESHOLD:
            result = True
    return not result


def convert_binary_mask_to_rgba_mask(
    mask: np.ndarray, color_rgba: tuple[int, int, int, int] = COLOR_RGBA_FOREGROUND
) -> np.ndarray:
    """Convert binary mask to RGBA mask.
    Args:
        mask: shape=[H, W]
    """
    assert mask.ndim == 2, f"expected shape is [H, W] but got {mask.shape}"
    height, width = mask.shape

    mask_rgba = np.zeros((height, width, 4)).astype(np.uint8)  # RGBA
    positives = np.where(mask == 1)
    for row, col in zip(*positives):
        mask_rgba[row, col] = color_rgba
    return mask_rgba


def generate_face_annonymization_mask(
    pose_single_frame: PoseSingleFrame,
    segmentation_mask: SegmentationMask,
    img_width: int,
    img_height: int,
    mask_width: int = 150,
    mask_height: int = 150,
) -> np.ndarray:
    """Generate ellipse-shape mask around face.
    The mask is generated based on the keypoints of face and segmentation mask for `person`.
    The mask for face annonymization is created by extracting only the elliptical part of the
    segmentation mask. The center of the ellipse is the average of the keypoints of the face
    and the width and height of the ellipse are defined by `mask_width` and `mask_height`.

    Returns:
        BGRA image
    """
    assert segmentation_mask.class_name == DEEPLAB_PERSON_CLASS_NAME

    # Specify the area of the faces
    face_area_mask = np.zeros((img_width, img_height, 4)).astype(np.uint8)  # RGBA
    num_face_mask = 0
    for pose in pose_single_frame.poses:
        valid_kpts = np.array(
            [pose.joints[j, :2] for j in FACE_JOINT_INDICES if pose.scores[j] > KEYPOINT_THRESHOLD]
        )
        if len(valid_kpts) != len(FACE_JOINT_INDICES):
            continue
        pt_center = (int(valid_kpts[:, 0].mean()), int(valid_kpts[:, 1].mean()))

        # shift center of ellipse to the upper side because this center is shifted to the lower side
        # because this point is calculated based on the nose and shoulders.
        shift = int(mask_height * (0.5 - 0.3))
        pt0 = (pt_center[0], pt_center[1] - shift)
        cv2.ellipse(face_area_mask, (pt0, (mask_width, mask_height), 0), (1, 1, 1, 1), thickness=-1)
        num_face_mask += 1

    # Extract areas of the people faces from the person segmentation mask
    person_mask_rgba = convert_binary_mask_to_rgba_mask(segmentation_mask.mask)
    if num_face_mask == 0:
        return person_mask_rgba
    face_annonymization_mask = person_mask_rgba * face_area_mask
    return face_annonymization_mask


def annonymize_single_image(
    image: np.ndarray,
    pose_estimator: PoseEstimationWrapper,
    segmentation_model: SegmentationWrapper,
) -> np.ndarray:
    """Apply annoymization to a single image."""
    height, width = image.shape[:2]
    image_tensor = image_bgr_to_tensor(image).to(_DEVICE)

    # model inference
    pose_single_frame = pose_estimator([image_tensor])[0]
    segmentation_single_frame = segmentation_model(image_tensor)[0]

    # anonymization mask generation
    face_annonymization_mask = generate_face_annonymization_mask(
        pose_single_frame, segmentation_single_frame.masks[0], height, width
    )

    # Apply the mask to the image
    image_bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    image_annonymized = apply_mask_to_image(image_bgra, face_annonymization_mask)
    image_annonymized = cv2.cvtColor(image_annonymized, cv2.COLOR_BGRA2BGR)
    return image_annonymized


def annonymize_video(
    input_video_path: Path,
    output_dir: Path,
    pose_estimator: PoseEstimationWrapper,
    segmentation_model: SegmentationWrapper,
):
    """Apply annonymization to a video."""
    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {input_video_path}")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Video information: {frame_width=}, {frame_height=}, {fps=}, {total_frames=}")

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video_path = output_dir / f"{input_video_path.stem}_annonymized.mp4"
    video = cv2.VideoWriter(str(output_video_path), fourcc, fps, (frame_width, frame_height))
    logger.info(f"Export an annonymized video to {output_video_path}")

    pbar = tqdm.tqdm(total=total_frames)
    while cap.isOpened():
        try:
            ret, frame = cap.read()
            if not ret:
                break

            frame_annonymized = annonymize_single_image(frame, pose_estimator, segmentation_model)
            video.write(frame_annonymized)
            pbar.update(1)
        except KeyboardInterrupt:
            break

    cap.release()
    video.release()
    logger.info("Finish annonymization.")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply annonymization to a video.")
    parser.add_argument("-i", "--input", type=Path, help="Input vidoe path.")
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=Path("."), help="Output directory path."
    )
    return parser


def main(args: argparse.Namespace):
    # init models
    pose_estimator = KeypointRCnnWeapper(device=_DEVICE)
    segmenetation_model = DeepLabWeapper(
        device=_DEVICE, target_classes=(DEEPLAB_PERSON_CLASS_NAME,)
    )

    # apply annonymization
    annonymize_video(args.input, args.output_dir, pose_estimator, segmenetation_model)


if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)
