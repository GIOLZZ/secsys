import numpy as np
from dataclasses import dataclass


@dataclass
class YoloDeceteResults:
    """Yolo检测结果"""
    boxes: list[list[int]] = []
    clss: list[int] = []
    confs: list[float] = []
    masks: list[np.ndarray] = []
