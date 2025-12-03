import numpy as np
from typing import List, Dict
from enum import Enum

from alg_det.uav.uav_not_yield.utils import ana_det_result


del_vanish_time = 10000    # 目标消失时间（删除）阈值(s)


class ObjLables(Enum):
    """检测对象labels"""
    car = 0
    truck = 1
    person = 2
    mini_truck = 3
    bus = 4
    policecar = 5


class ObjInfo:
    def __init__(self, id, label, track_time_max=60000):
        """
        单对象信息管理类
        Args:
            id (int) : obj跟踪id
            label (str) : obj标签
            track_time_max (int, Optional) : 最大保存时间(ms)(避免驻车一直保持信息)
        """
        self.id = id
        self.label = label
        self.track_time_max = track_time_max

        self.img_keys: List[int] = []    # 图像对应的key
        self.bboxs: List[List] = []
        self.timestamps: List[float] = []
        self.track_points: List[List] = []  # 轨迹点
        self.frame_number = 0   # 保存的帧数
        
        self.outtime = 0    # 对象的消失时间(ms)
        self.vio_tf = False # 是否违规

        self.vio_img = None
        self.vio_bbox = None
        self.vio_timestamp = None

        self.report_tf = False  # 是否已上报

    def update(self, img_key, bbox, timestamp):
        """
        更新obj info
        Args:
            img_key (int): 图像对应的key
            bbox (List[List[int]]): bbox
            timestamp (float): 时间戳(ms)
        """
        if self.frame_number > 2:
            if self.timestamps[-1] - self.timestamps[0] > self.track_time_max:
                self.img_keys.pop(0)
                self.bboxs.pop(0)
                self.track_points.pop(0)
                self.timestamps.pop(0)

        self.img_keys.append(img_key)
        self.bboxs.append(bbox)
        track_point = [int(bbox[0] + (bbox[2] - bbox[0]) / 4 * 3), int((bbox[1] + bbox[3]) / 2)]
        self.track_points.append(track_point)
        self.timestamps.append(timestamp)

        self.outtime = 0
        self.frame_number = len(self.img_keys)


class TrackObjInfos:
    def __init__(self):
        """多目标跟踪对象信息管理类"""
        self.obj_infos: Dict[int, ObjInfo] = {}

        self.imgs: Dict[int, np.ndarray] = {}
        self.img_number = 0
        
    def update(self, det_results, img, timestamp):
        """
        更新objs info
        Args:
            det_results (): 上层的检测结果
            img (np.ndarray): img
            timestamp (float): 时间戳(ms)
        """
        if len(det_results) == 0:
            return
        
        # 更新图像info列表
        self.imgs[self.img_number] = img

        # 更新每个obj的消失时间
        for obj_id in self.obj_infos.keys():
            self.obj_infos[obj_id].outtime = timestamp - self.obj_infos[obj_id].timestamps[-1]

        # 更新最新info到每个obj
        for det_result in det_results:
            id, label, bbox = ana_det_result(det_result)
            if id in self.obj_infos.keys():
                self.obj_infos[id].update(self.img_number, bbox, timestamp)
            else:
                self.obj_infos[id] = ObjInfo(id, label)
                self.obj_infos[id].update(self.img_number, bbox, timestamp)

        # 对于消失超时或已上报的obj进行删除
        del_obj_keys = []
        for obj_id in self.obj_infos.keys():
            if self.obj_infos[obj_id].outtime > del_vanish_time or self.obj_infos[obj_id].report_tf:
                del_obj_keys.append(obj_id)
        for del_key in del_obj_keys:
            del self.obj_infos[del_key]

        # 维护图像储存列表 self.imgs
        all_img_keys = [] # 目前还在被引用的img的key
        for obj_info in self.obj_infos.values():
            img_keys = obj_info.img_keys
            all_img_keys.extend(img_keys)
        all_img_keys = list(set(all_img_keys))  # 去重
        del_img_keys = []
        for img_key in self.imgs.keys():
            if img_key not in all_img_keys:
                del_img_keys.append(img_key)
        for del_key in del_img_keys:
            del self.imgs[del_key]

        self.img_number += 1
        if self.img_number > 9999:
            self.img_number = 0


