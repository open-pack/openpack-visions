from abc import ABCMeta, abstractmethod

import torch

from video_annonymization.segmentation.dataclass import SegmentationSingleFrame


class SegmentationWrapper(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, inputs: torch.Tensor) -> list[SegmentationSingleFrame]:
        pass

    @abstractmethod
    def inference(self, inputs: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def postprocess(
        self, outputs: list[dict], original_image_sizes: list[tuple]
    ) -> list[SegmentationSingleFrame]:
        pass
