import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np

from algo.yolov8.detect import YoloDecete


class CameraInsider:
    """相机内参"""
    def __init__(self, fx, fy, cx, cy):
        """
        Args:
            fx (float): x轴焦距
            fy (float): y轴焦距
            cx (float): x轴中点
            cy (float): y轴中点
        """
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy


class Decete3DResults:
    """全局3d检测结果"""
    def __init__(self):
        self.boxes2d: list[list[int]] = []
        self.clss: list[int] = []
        self.confs: list[float] = []
        self.coors3d: list[list[float]] = []


class YoloDecete3D(YoloDecete):
    """3d检测"""
    def __init__(self, model_path: str, camera_insider: CameraInsider, device: str='cuda'):
        """
        Args:
            model_path (str): 模型路径
            camera_insider (CameraInsider): 相机内参
            device (str, optional): 设备. Defaults to 'cuda'.
        """
        super().__init__(model_path, task='det', device=device)
        self.camera_insider = camera_insider
        
    def detect3d(self, color_image: np.ndarray, depth_image: np.ndarray, depth_hs: int=1, conf: float=0.6, iou: float=0.1) -> Decete3DResults:
        """
        Args:
            color_image (np.ndarray): 输入图像
            depth_image (np.ndarray): 深度图像
            depth_hs (int, optional): 深度分辨率. Defaults to 1.
            conf (float, optional): 置信度. Defaults to 0.6.
            iou (float, optional): iou. Defaults to 0.1.
        
        Returns:
            Decete3DResults: 检测结果
        """
        img_h, img_w = depth_image.shape[:2]
        results = self.detect(color_image, conf=conf, iou=iou)
        clss = results.clss
        if len(clss) == 0:
            return Decete3DResults()
        boxes = results.boxes
        confs = results.confs

        det3d_reasults = Decete3DResults()
        for i, bbox in enumerate(boxes):
            u, v = (bbox[0] + bbox[2]) // 2, bbox[1]
            if u < 0:
                u = 0
            elif u > img_w-1:
                u = img_w-1
            if v < 0:
                v = 0
            elif v > img_h-1:
                v = img_h-1

            z = depth_image[v, u] / depth_hs
            if z <= 0:
                continue
            x = (u - self.camera_insider.cx) * z / self.camera_insider.fx
            y = (v - self.camera_insider.cy) * z / self.camera_insider.fy

            det3d_reasults.coors3d.append([x, y, z])
            det3d_reasults.boxes2d.append(bbox)
            det3d_reasults.clss.append(clss[i])
            det3d_reasults.confs.append(confs[i])
        
        return det3d_reasults
