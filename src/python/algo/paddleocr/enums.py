from enum import Enum


class DevicePlatform(Enum):
    """推理平台"""
    CPU = 'cpu'
    CUDA = 'cuda'
    RKNN = 'rknn'

