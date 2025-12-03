import sys
sys.path.append('/home/xmv/tiercel-core-serving/python')

import math
import cv2
import numpy as np
from shapely import Polygon


plate_number_conf = 0.5
chinese_char_conf = 0.4

# 车牌类型代码
plate_type_code = {0: '02', 1: '52', 2: '01', 3: '51', 4: '01'}


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# 将关键点顺序从左上角开始排序
def handle_key_point(key_points):
    if [0, 0] not in key_points:
        if (key_points[0][0] > key_points[1][0] and key_points[0][0] >
            key_points[2][0]) and (
                key_points[3][0] > key_points[1][0] and key_points[0][
            0] > key_points[2][0]):
            key_points[0], key_points[1] = key_points[1], \
            key_points[0]
            key_points[2], key_points[3] = key_points[3], \
            key_points[2]
    return key_points

def blacken_right_percent(image: np.ndarray, percent: float) -> np.ndarray:
    """
    将图片右侧 percent% 的区域涂黑。

    参数
    ----
    image : np.ndarray
        输入图像，形状为 (H, W, C) 或 (H, W)，数据类型不限。
    percent : float
        右侧需要涂黑的比例，0~100。例如 20 表示右侧 20% 区域变黑。

    返回
    ----
    np.ndarray
        处理后的图像，与输入同形状、同 dtype。
    """
    if not (0 <= percent <= 100):
        raise ValueError("percent 必须在 0~100 之间")

    w = image.shape[1]
    split = int(w * (100 - percent) / 100)  # 左侧保留的宽度
    image_copy = image.copy()
    image_copy[:, split:] = 0  # 涂黑右侧
    return image_copy

def is_chinese_char(ch: str) -> bool:
    """
    判断单个字符是否为中文汉字。
    注意：仅支持 BMP 区（基本多语言平面），不含扩展区。
    """
    return '\u4e00' <= ch <= '\u9fff'


class PlateRecognition:
    def __init__(self, platform='CUDA'):
        self.platform = platform

        self.carplte_model = None
        self.paddleOCR_model = None
    
    def init_model(self, carplte_model_path, paddleOCR_path, character_dict_path='ppocr_keys_v1.txt'):
        if self.platform == 'CUDA':
            from ultralytics import YOLO
            from paddleocr import PaddleOCR
            device = 'cpu'
            self.carplte_model = YOLO(carplte_model_path).to(device=device)
            self.paddleOCR_model = PaddleOCR(rec_model_dir=paddleOCR_path)
        
        elif self.platform == 'RKNN':
            from utils.yolov8_rknn.yolo_rknn_det import YoloRKNN
            from utils.paddleOCR_rknn.paddleOCR_rknn.paddleOCR_rknn import PaddleOCRRknn
            
            self.carplte_model = YoloRKNN(carplte_model_path, 'pose')
            self.paddleOCR_model = PaddleOCRRknn(paddleOCR_path, character_dict_path)
    
    def detect(self, car_img):
        boxs, keypoints, classIds, scores = None, None, None, None

        # 检测车牌
        if self.platform == 'CUDA':
            results = self.carplte_model(car_img)
            boxs = results[0].boxes.xyxy.int().tolist()
            keypoints = results[0].keypoints.xy.int().tolist()
            classIds = results[0].boxes.cls.int().tolist()
            scores = results[0].boxes.conf.tolist()

        elif self.platform == 'RKNN':
            results = self.carplte_model.detect(car_img)
            boxs = results['boxs']
            keypoints = results['keypoints']
            classIds = results['classIds']
            scores = results['scores']

        if not classIds:
            return None

        # 获取需要识别的车牌位置
        if len(classIds) > 1:
            area_max_index = self.get_area_max(boxs)
        else:
            area_max_index = 0
        
        # 获取车牌图片
        plate_img = self.get_plate_img(car_img, boxs[area_max_index], keypoints[area_max_index], classIds[area_max_index])

        # 车牌识别
        if self.platform == 'CUDA':
            output = self.paddleOCR_model.ocr(plate_img, det=False, cls=False)[0][0]
        elif self.platform == 'RKNN':
            output = self.paddleOCR_model.ocr(plate_img)[0]

        if output[1] > plate_number_conf:
            plate_number = output[0]
            plate_number = plate_number.upper()  # 确保大写
            plate_number = plate_number.replace('O', '0').replace('I', '1').replace(' ', '').replace(':', '').replace('·', '')

            chinese_char_conf = self.get_chinese_char_conf(plate_img, plate_number)

            return (plate_number, output[1], chinese_char_conf, plate_type_code[classIds[area_max_index]])
        else:
            return None

    def get_area_max(self, boxs):
        areas = [Polygon([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]]).area for box in boxs]
        area_max_index = areas.index(max(areas))
        
        return area_max_index
    
    def get_plate_img(self, car_img, box, keypoints, cls):
        # keypoints = [[int(keypoint[0]), int(keypoint[1])] for keypoint in keypoints if keypoint[2] > 0.8]
        keypoints = [[int(keypoint[0]), int(keypoint[1])] for keypoint in keypoints]

        if len(keypoints) == 4:
            keypoints = handle_key_point(keypoints)
            keypoints_poly = Polygon(keypoints)
            # keypoints_poly = keypoints_poly.buffer(0)   # 修复自交的"蝴蝶结"形状
            keypoints = list(keypoints_poly.exterior.coords)[:-1]
            
            box_poly = Polygon([
                [box[0], box[1]], 
                [box[2], box[1]], 
                [box[2], box[3]], 
                [box[0], box[3]]
            ])

            # 使用关键点提取车牌
            if keypoints_poly.area > box_poly.area / 8:
                print('使用keypoint提取车牌')
                h = int(min(distance(keypoints[0], keypoints[1]), distance(keypoints[0], keypoints[-1])))
                # 单层
                if cls != 4:
                    w = int(h * 3.2)
                # 双层
                else:
                    w = int(h * 1.8)

                # print(keypoints)
                src_points = np.float32(keypoints)
                dst_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
                M = cv2.getPerspectiveTransform(src_points, dst_points)
                plate_img = cv2.warpPerspective(car_img, M, (w, h))
            # 使用box框提取车牌
            else:
                print('使用Bbox提取车牌')
                plate_img = car_img[box[1]:box[3], box[0]:box[2]]
        
        # 使用box框提取车牌
        else:
            print('使用Bbox提取车牌')
            plate_img = car_img[box[1]:box[3], box[0]:box[2]]
        
        return plate_img

    def get_chinese_char_conf(self, plate_img, plate_number):
        """
        模糊字符删除
        Args:
            plate_img (np.array): 车牌
            plate_number (str): 车牌号
        
        Return:
            conf (float): 单汉字置信度
        """
        if len(plate_number) > 1:
            if is_chinese_char(plate_number[1]):
                chinese_char_tf = 2
            elif is_chinese_char(plate_number[0]):
                chinese_char_tf = 1
            else:
                chinese_char_tf = 0
        else:
            if is_chinese_char(plate_number[0]):
                chinese_char_tf = 1
            else:
                chinese_char_tf = 0

        if not chinese_char_tf:
            return 0

        plate_img_blacken = blacken_right_percent(plate_img, 82)    # 遮住数字和字母
        if self.platform == 'CUDA':
            output_blacken = self.paddleOCR_model.ocr(plate_img_blacken, det=False, cls=False)[0][0]
        elif self.platform == 'RKNN':
            output_blacken = self.paddleOCR_model.ocr(plate_img_blacken)[0]
        
        # # 单汉字置信度不🦶 chinese_char_conf
        # if output_blacken[1] < chinese_char_conf:
            # plate_number = plate_number[chinese_char_tf:]

        return output_blacken[1]


if __name__ == '__main__':
    test = PlateRecognition('CUDA')
    test.init_model(
        '/home/xmv/tiercel-core-serving/python/utils/plate_recognition/weights/carplate_pose_20250721.pt',
        '/home/xmv/tiercel-core-serving/python/utils/paddleOCR_rknn/weights/ch_PP-OCRv3_rec_infer_250909_T',
        '/home/xmv/tiercel-core-serving/python/utils/paddleOCR_rknn/weights/ppocr_keys_v1.txt'
    )

    img = cv2.imread('/home/xmv/tiercel-core-serving/python/utils/plate_recognition/che.jpg')
    text = test.detect(img)
    print(text)
