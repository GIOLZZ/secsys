import math
import numpy as np
import cv2


def postprocessing_obb(outputs: np.ndarray,
                       conf_thres: float,
                       iou_thres: float,
                       class_number: int,
                       imgz: int,
                       image_shape: tuple):
    """
    极速版 YOLOv8-obb 后处理
    :param outputs:     原始模型输出 (1, 5+cls+1, N)
    :param conf_thres:  置信度阈值
    :param iou_thres:   旋转框NMS阈值
    :param class_number:类别数
    :param imgz:        模型输入尺寸（正方形）
    :param image_shape: 原图 (height, width)
    :return:            (list[ corners4 ], list[cls], list[score])
    """
    # 解析输出并置信度过滤（向量化）
    outputs = np.squeeze(outputs)                # (5+cls+1, N)
    boxes = outputs[:4, :].T                     # (N, 4)  x y w h
    angles = outputs[-1, :]                      # (N,)
    cls_scores = outputs[4:4+class_number, :].T  # (N, cls)

    scores = np.max(cls_scores, axis=1)
    labels = np.argmax(cls_scores, axis=1)
    valid_mask = scores > conf_thres
    if valid_mask.sum() == 0:
        return [], [], []

    boxes = boxes[valid_mask]
    angles = angles[valid_mask]
    scores = scores[valid_mask]
    labels = labels[valid_mask]

    # 角度 → 度，准备 cv2 格式
    angles_deg = angles * 180 / np.pi
    rboxes_for_cv2 = np.hstack([boxes, angles_deg.reshape(-1, 1)])  # (M,5)

    # 用 OpenCV 做旋转框 NMS
    keep = nms_rotated_cv2(rboxes_for_cv2, scores, iou_thres)
    rboxes = rboxes_for_cv2[keep]
    scores = scores[keep]
    labels = labels[keep]

    # 计算 4 个角点
    box_list = get_obb_box(rboxes, (imgz, imgz), image_shape)
    cls_list = labels.astype(int).tolist()
    score_list = scores.tolist()

    return box_list, cls_list, score_list

def nms_rotated_cv2(rboxes: np.ndarray, scores: np.ndarray, iou_thres: float):
    """
    :param rboxes: (N,5)  [x,y,w,h,angle_deg]  numpy
    :param scores: (N,)   numpy
    :param iou_thres: float
    :return: 保留索引  np.ndarray[int]
    """
    # 1. 构造真正的 cv2.RotatedRect 列表
    rects = []
    for (x, y, w, h, a) in rboxes.astype(float):
        rects.append(cv2.RotatedRect((x, y), (w, h), a))

    # 2. 调用 OpenCV NMS
    indices = cv2.dnn.NMSBoxesRotated(rects, scores.tolist(),
                                      score_threshold=0.0,
                                      nms_threshold=iou_thres)
    return indices.flatten() if len(indices) else np.empty(0, int)

def get_obb_box(rboxes, input_size, original_size):
    """
    Args:
        boxes: 一个包含多个框的列表，每个框是 (x, y, w, h, a)
        input_size: 输入图像的尺寸 (input_width, input_height)
        original_size: 原始图像的尺寸 (original_width, original_height)
    """
    input_width, input_height = input_size
    original_width, original_height = original_size
    box_list = []
    for box in rboxes:
        x, y, w, h, r = box[0], box[1], box[2], box[3], np.deg2rad(box[4])
        corner = rotate_box(x, y, w, h, r)
        
        # 映射到原图大小
        corner[:, 0] = (corner[:, 0] / input_width) * original_width
        corner[:, 1] = (corner[:, 1] / input_height) * original_width - (original_width-original_height)/2

        corner = corner.astype(int).tolist()
        box_list.append(corner)
    
    return box_list

def rotate_box(x, y, w, h, r):
    """
    计算旋转框的四个角点坐标。
    Args:
        x, y: 旋转框的中心坐标
        w, h: 旋转框的宽度和高度
        r: 旋转角度（弧度）
    Returns:
        四个角点的坐标，按照顺时针顺序排列
    """
    # 计算框的四个角点
    cos_r = math.cos(r)
    sin_r = math.sin(r)
    
    # 相对于框中心的四个角点坐标
    corners = np.array([
        [-w / 2, -h / 2],  # 左上角
        [w / 2, -h / 2],   # 右上角
        [w / 2, h / 2],    # 右下角
        [-w / 2, h / 2],   # 左下角
    ])

    # 旋转矩阵
    rotation_matrix = np.array([
        [cos_r, -sin_r],
        [sin_r, cos_r]
    ])
    
    # 旋转并平移到(x, y)
    rotated_corners = np.dot(corners, rotation_matrix.T) + np.array([x, y])
    
    return rotated_corners