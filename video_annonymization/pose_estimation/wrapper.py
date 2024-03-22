from abc import ABCMeta, abstractmethod

import torch

from video_annonymization.pose_estimation.dataclass import PoseSingleFrame


class PoseEstimationWrapper(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, inputs: list[torch.Tensor]) -> list[PoseSingleFrame]:
        pass

    @abstractmethod
    def inference(self, inputs: list[torch.Tensor]) -> list[dict]:
        pass

    @abstractmethod
    def postprocess(self, outputs: list[dict]) -> list[PoseSingleFrame]:
        pass
