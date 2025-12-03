import json

from alg_det.uav.uav_not_yield.algo import NotYieldAlgo
from alg_det.uav.uav_not_yield.track_obj_info import TrackObjInfos
from alg_det.uav.uav_not_yield.utils import report
from utils.alg_log import get_logger
logger = get_logger(__name__)


class UavNotYield(NotYieldAlgo):
    def __init__(self, index):
        super().__init__()
        self.index = index
        self.event_queue = None
        self.camera_id = None

        self.img_h, self.img_w = None, None

        self.track_obj_infos = TrackObjInfos()
    
    def set_config(self, index, config_path, event_queue):
        # 消息通道
        self.event_queue = event_queue

        config = json.load(open(config_path, 'r'))        
        self.camera_id = config['cameraListParam']['camera'][self.index]['deviceId']
    
    async def detect(self, index, detection_type, frame, det_results, timestamp, road_model=None):
        """更新跟踪对象信息"""
        self.track_obj_infos.update(det_results, frame, timestamp)

        """算法检测"""
        self.algo_run(frame, det_results, road_model)

        """构建上报信息"""
        is_report, report_events = report(self.track_obj_infos, self.camera_id, ["108"])

        if is_report and self.event_queue is not None:
            self.event_queue.put(report_events)

    def algo_run(self, frame, det_results, road_model):
        if frame is None or road_model is None:
            logger.info('*****frame is None or road_model is None*****')
            return
        
        if len(det_results) == 0:
            logger.info('*****检测结果为空*****')
            return
        
        road_model = json.loads(road_model)

        if "zebra" not in road_model.keys():
            logger.info('*****建模没有斑马线*****')
            return

        self.img_h, self.img_w = frame.shape[:2]
        zebra_polygons = road_model["zebra"]
        for zebra_polygon in zebra_polygons:
            zebra_polygon_points = zebra_polygon["polygon"]["coordinates"]
            zebra_polygon_points = [[int(zebra_polygon_point[0] * self.img_w), int(zebra_polygon_point[1] * self.img_h)] for zebra_polygon_point in zebra_polygon_points]
            for car_id, car_info in self.track_obj_infos.obj_infos.items():
                if car_info.outtime > 0:
                    continue
                if car_info.label in ("person", "zebra", "trailor"):
                    continue
                if len(car_info.track_points) < 2:
                    continue
                if car_info.vio_tf == True:
                    continue

                for person_id, person_info in self.track_obj_infos.obj_infos.items():
                    if person_info.outtime > 0:
                        continue
                    if person_info.label != "person":
                        continue
                    if len(person_info.track_points) < 2:
                        continue

                    logger.info("*****同时发现人和车，进行检测中*****")
                    
                    viola_id = self.run(person_info.id, person_info.track_points, person_info.bboxs[-1], car_info.id, car_info.track_points, car_info.bboxs[-1], zebra_polygon_points)
                    if viola_id is None:
                        continue
                    
                    self.track_obj_infos.obj_infos[viola_id].vio_img = frame
                    self.track_obj_infos.obj_infos[viola_id].vio_bbox = self.track_obj_infos.obj_infos[viola_id].bboxs[-1]
                    self.track_obj_infos.obj_infos[viola_id].vio_timestamp = self.track_obj_infos.obj_infos[viola_id].timestamps[-1]
                    self.track_obj_infos.obj_infos[viola_id].vio_tf = True

                    logger.info(f"*****{viola_id} 违规*****")
                    break
