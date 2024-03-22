from typing import List, Optional

import numpy as np
from attrs import define, field


@define
class Bbox:
    xywh: tuple[int, int, int, int]
    score: float


@define
class PoseOutput:
    joints: np.ndarray
    scores: float
    bbox: Optional[Bbox] = None


@define
class PoseSingleFrame:
    poses: List[PoseOutput] = field(factory=list)
