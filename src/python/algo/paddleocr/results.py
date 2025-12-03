from dataclasses import dataclass


@dataclass
class OCRResults:
    """ocr识别结果"""
    text: str = ''
    conf: float = 0.0
        
