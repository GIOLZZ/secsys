import numpy as np
from typing import List
# from dataclasses import dataclass


# @dataclass
class YoloDetectResults:
    """Yolo检测结果"""
    def __init__(self):
        self.boxes: List[List[int]] = []
        self.clss: List[int] = []
        self.confs: List[float] = []
        self.masks: List[np.ndarray] = []
        self.keypoints: List[List[int, float]] = []
        self.xyxyxyxy: List[List[int]] = []
