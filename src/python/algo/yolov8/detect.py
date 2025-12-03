import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import numpy as np

from algo.yolov8.results import YoloDeceteResults
from algo.yolov8.enums import DevicePlatform, ModelTask


class YoloDecete:
    """Yolo检测"""
    def __init__(self, model_path: str, task: ModelTask, device: DevicePlatform=DevicePlatform.CPU):
        """
        Args:
            model_path (str): path
            task (ModelTask): 模型任务
            device (DevicePlatform, optional): 设备. Defaults to DevicePlatform.CPU.
        """
        if device == DevicePlatform.CPU or device == DevicePlatform.CUDA:
            from ultralytics import YOLO
            from algo.yolov8.cuda.predict import DetectionPredictor, SegmentationPredictor

            self.model = YOLO(model_path)
            self.task = task
            self.model(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False, device=device.value)

            if self.task == ModelTask.DET:
                self.predictor = DetectionPredictor(self.model, device.value)
            elif self.task == ModelTask.SEG:
                self.predictor = SegmentationPredictor(self.model, device.value)
        
        elif device == DevicePlatform.ASCEND:
            pass

        elif device == DevicePlatform.RKNN:
            pass

    def detect(self, image: np.ndarray, conf: float=0.25, iou: float=0.1) -> YoloDeceteResults:
        """
        Args:
            image (np.ndarray): 输入图像
            conf (float, optional): 置信度. Defaults to 0.25.
            iou (float, optional): iou. Defaults to 0.1.
        
        Returns:
            YoloDeceteResults: 检测结果
        """
        results = self.predictor.predict(image, conf, iou)

        return results


if __name__ == "__main__":
    det = YoloDecete('/home/xmv/secsys/models/best.pt', task=ModelTask.DET, device=DevicePlatform.CPU)
