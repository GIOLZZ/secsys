from enum import Enum


class DevicePlatform(Enum):
    """推理平台"""
    CPU = 'cpu'
    CUDA = 'cuda'
    ASCEND = 'ascend'
    RKNN = 'rknn'


class ModelTask(Enum):
    """模型推理任务"""
    DET = 'detect'
    SEG = 'segment'
    POSE = 'pose'
    OBB = 'obb'
    