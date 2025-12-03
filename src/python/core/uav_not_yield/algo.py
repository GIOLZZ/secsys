from typing import List

from shapely import Point, Polygon


Expansion_factor = 2    # 车box向两边扩展倍数


def get_polygon_range(polygon_points: List[List]):
    x_min, x_max = None, None
    y_min, y_max = None, None
    for point in polygon_points:
        if x_min is None:
            x_min = point[0]
            x_max = point[0]
            y_min = point[1]
            y_max = point[1]
        
        else:
            if point[0] < x_min:
                x_min = point[0]
            if point[0] > x_max:
                x_max = point[0]

            if point[1] < y_min:
                y_min = point[1]
            if point[1] > y_max:
                y_max = point[1]
    
    return x_min, x_max, y_min, y_max


class NotYieldAlgo:
    def __init__(self):
        self.incivility_to_be_identified = {}
        
    def run(self, person_id, person_track_points: List[List], person_bbox: List[List], car_id, car_track_points: List[List], car_bbox: List[List], zebra_polygon_points: List[List]):
        zebra_x_min, zebra_x_max, zebra_y_min, zebra_y_max = get_polygon_range(zebra_polygon_points)
        
        # 向两侧扩展Expansion_factor倍的车宽
        car_left_bbox = (
            (int(car_bbox[0] - (car_bbox[2] - car_bbox[0]) * Expansion_factor), car_bbox[1]),
            (car_bbox[0], car_bbox[1]),
            (car_bbox[0], car_bbox[3]),
            (int(car_bbox[0] - (car_bbox[2] - car_bbox[0]) * Expansion_factor), car_bbox[3])
        )
        car_right_bbox = (
            (car_bbox[2], car_bbox[1]),
            (int(car_bbox[2] + (car_bbox[2] - car_bbox[0]) * Expansion_factor), car_bbox[1]),
            (int(car_bbox[2] + (car_bbox[2] - car_bbox[0]) * Expansion_factor), car_bbox[3]),
            (car_bbox[2], car_bbox[3])
        )

        # 判断车辆方向
        if car_track_points[-1][1] - car_track_points[-2][1] < 0:
            direction_y = "-"   # 向上
        else:
            direction_y = "+"   # 向下

        # 车辆不满足不礼让行为条件
        if not (zebra_x_min < car_track_points[-1][0] < zebra_x_max):
            return None
        
        person_jiao_point = (int((person_bbox[0] + person_bbox[2]) / 2), person_bbox[3])

        # 车辆向上且在斑马线下方 or 向下且在斑马线上方, 此时分析可能要不礼让行人的车辆
        if (direction_y == "-" and car_track_points[-1][1] > zebra_y_min) or (direction_y == "+" and car_track_points[-1][1] < zebra_y_max):
            # 人不在斑马线上
            if not Point(person_jiao_point).within(Polygon(zebra_polygon_points)):
                return None
            
            # 人在车辆左侧范围内
            if car_left_bbox[0][0] < person_jiao_point[0] < car_left_bbox[1][0]:
                # 判断人行走的方向, 行人从左往右, 车辆才有可能不礼让行人
                if person_track_points[-1][0] - person_track_points[-2][0] < 0:
                    return None

                if car_id not in self.incivility_to_be_identified.keys():
                    self.incivility_to_be_identified[car_id] = {"person_ids": [person_id], "fan_wei": (car_left_bbox[0][0], car_bbox[2])}
                else:
                    self.incivility_to_be_identified[car_id]["person_ids"].append(person_id)
            
            # 人在车辆右侧范围内
            elif car_right_bbox[0][0] < person_jiao_point[0] < car_right_bbox[1][0]:
                # 判断人行走的方向, 行人从左往右, 车辆才有可能不礼让行人
                if person_track_points[-1][0] - person_track_points[-2][0] > 0:
                    return None

                if car_id not in self.incivility_to_be_identified.keys():
                    self.incivility_to_be_identified[car_id] = {"person_ids": [person_id], "fan_wei": (car_bbox[0], car_right_bbox[1][0])}
                else:
                    self.incivility_to_be_identified[car_id]["person_ids"].append(person_id)

        # 车辆向上且在斑马线上方 or 向下且在斑马线下方, 此时分析已经不礼让行人的车辆
        elif (direction_y == "-" and car_track_points[-1][1] < zebra_y_min) or (direction_y == "+" and car_track_points[-1][1] > zebra_y_max):
            if car_id not in self.incivility_to_be_identified.keys():
                return None
            
            person_ids = self.incivility_to_be_identified[car_id]["person_ids"]
            fan_wei = self.incivility_to_be_identified[car_id]["fan_wei"]

            if person_id not in person_ids:
                return None

            # 判断人是否还在原来的地方范围
            if Point(person_jiao_point).within(Polygon(zebra_polygon_points)) and (fan_wei[0] < person_jiao_point[0] < fan_wei[1]):
                del self.incivility_to_be_identified[car_id]

                return car_id

