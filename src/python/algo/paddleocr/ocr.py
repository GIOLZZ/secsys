import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from algo.paddleocr.enums import DevicePlatform
from algo.paddleocr.results import OCRResults


class PaddleOCR:
    def __init__(self, model_path: str, device: DevicePlatform=DevicePlatform.CPU, character_dict_path: str=''):
        """
        Args:
            model_path (str): path
            device (DevicePlatform, optional): 推理设备. Defaults to DevicePlatform.CPU.
            character_dict_path (str, optional): 字典路径. Defaults to ''.
        """
        if device == DevicePlatform.CPU or device == DevicePlatform.CUDA:
            from paddleocr import PaddleOCR

            self.ocr_model = PaddleOCR(rec_model_dir=model_path)
        
        elif device == DevicePlatform.RKNN:
            from algo.paddleocr.rknn.paddleocr_rknn import PaddleOCRRknn

            self.ocr_model = PaddleOCRRknn(model_path, character_dict_path=character_dict_path)

    def ocr(self, img) -> OCRResults:
        ocr_results = OCRResults()
        results = self.ocr_model.ocr(img, det=False, cls=False)[0][0]
        ocr_results.text = results[0]
        ocr_results.conf = results[1]

        return ocr_results
        