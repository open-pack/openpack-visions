from typing import List

import numpy as np
from attrs import define, field


@define
class SegmentationMask:
    class_name: str
    mask: np.ndarray


@define
class SegmentationSingleFrame:
    masks: List[SegmentationMask] = field(factory=list)
