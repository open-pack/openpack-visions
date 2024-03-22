import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights, deeplabv3_resnet101

from video_annonymization.segmentation.dataclass import SegmentationMask, SegmentationSingleFrame

SEGMENTATION_RESULTS_KEY_NAME = "out"
SCORE_THREHOLD = 0.5


DEEPLAB_PERSON_CLASS_NAME = "person"


class DeepLabWeapper:
    def __init__(self, device: torch.device, target_classes: tuple[str] | None = None):
        if target_classes is None:
            target_classes = (DEEPLAB_PERSON_CLASS_NAME,)
        self.target_classes: tuple[str] = target_classes

        # Model
        weights = DeepLabV3_ResNet101_Weights.DEFAULT
        model = deeplabv3_resnet101(weights=weights, progress=False).to(device)
        model = model.eval()
        self.model = model
        self.class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}

        # Preprocess
        self.transforms = weights.transforms()

    def __call__(self, inputs: torch.Tensor) -> list[SegmentationSingleFrame]:
        # Preprocess
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(0)
        original_image_sizes: list[tuple[int, int]] = [tuple(img.size())[1:] for img in inputs]
        preproc_outputs = self.transforms(inputs)

        # Inference
        inf_outputs = self.inference(preproc_outputs)

        # Postprocess
        postproc_outputs = self.postprocess(inf_outputs, original_image_sizes)

        return postproc_outputs

    def inference(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.model(inputs)
        return outputs

    def postprocess(
        self, outputs: list[dict], original_image_sizes: list[tuple]
    ) -> list[SegmentationSingleFrame]:
        prediction = outputs[SEGMENTATION_RESULTS_KEY_NAME]
        num_images = prediction.shape[0]
        normalized_masks = prediction.softmax(dim=1)

        output_multi_frames = []
        for image_idx in range(num_images):
            segmentation_single_frame = SegmentationSingleFrame()
            for target_cls_name in self.target_classes:
                # extract mask of the target class
                target_cls_idx = self.class_to_idx[target_cls_name]
                mask = normalized_masks[image_idx, target_cls_idx]

                # Resize mask to original image size
                original_img_size = original_image_sizes[image_idx]
                mask = torch.functional.F.interpolate(
                    mask.unsqueeze(0).unsqueeze(0),
                    size=original_img_size,
                    mode="bilinear",
                    align_corners=False,
                )
                mask = mask.detach().cpu().numpy().squeeze()

                # binarize mask and create output dataclass
                binary_mask = np.where(mask >= SCORE_THREHOLD, 1, 0)
                segmentation_mask = SegmentationMask(target_cls_name, binary_mask)
                segmentation_single_frame.masks.append(segmentation_mask)
            output_multi_frames.append(segmentation_single_frame)

        return output_multi_frames


def draw_segmentation_mask(image_bgr: np.ndarray, seg_mask: SegmentationSingleFrame) -> np.ndarray:
    # OpenCV(BGR) to tensor
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
    image_tensor = torch.from_numpy(image_rgb)

    masks = np.stack([seg_mask.mask.astype(bool) for seg_mask in seg_mask.masks])
    image_tensor = torchvision.utils.draw_segmentation_masks(
        image_tensor, masks=torch.from_numpy(masks), alpha=0.8
    )

    # tensor to OpenCV(BGR)
    image_rgb_out = image_tensor.detach().numpy().transpose(1, 2, 0)
    image_bgr_out = cv2.cvtColor(image_rgb_out, cv2.COLOR_RGB2BGR)
    return image_bgr_out
