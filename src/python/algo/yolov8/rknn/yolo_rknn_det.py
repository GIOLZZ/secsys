import sys
sys.path.append('/home/xmv/tiercel-core-serving/python/gaosu_road_modeling/yolov8_rknn')

import numpy as np
import cv2, math
import torch
import torchvision
import torch.nn.functional as F

from itertools import product as product
from shapely.geometry import Polygon
from rknnlite.api import RKNNLite

from py_utils.coco_utils import COCO_test_helper


SEG_OBJ_MAX_DETECT = 300


class DetectBox_obb:
    def __init__(self, classId, score, xmin, ymin, xmax, ymax,angle):
        self.classId = classId
        self.score = score
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.angle=angle

class DetectBox:
    def __init__(self, classId, score, xmin, ymin, xmax, ymax, keypoint):
        self.classId = classId
        self.score = score
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.keypoint = keypoint


def letterbox_resize(image, size, bg_color):
    """
    letterbox_resize the image according to the specified size
    :param image: input image, which can be a NumPy array or file path
    :param size: target size (width, height)
    :param bg_color: background filling data 
    :return: processed image
    """
    if isinstance(image, str):
        image = cv2.imread(image)

    target_width, target_height = size
    image_height, image_width, _ = image.shape

    # Calculate the adjusted image size
    aspect_ratio = min(target_width / image_width, target_height / image_height)
    new_width = int(image_width * aspect_ratio)
    new_height = int(image_height * aspect_ratio)

    # Use cv2.resize() for proportional scaling
    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Create a new canvas and fill it
    result_image = np.ones((target_height, target_width, 3), dtype=np.uint8) * bg_color
    offset_x = (target_width - new_width) // 2
    offset_y = (target_height - new_height) // 2
    result_image[offset_y:offset_y + new_height, offset_x:offset_x + new_width] = image
    return result_image, aspect_ratio, offset_x, offset_y

def rotate_rectangle(x1, y1, x2, y2, a):
    # 计算中心点坐标
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    # 将角度转换为弧度
    # a = math.radians(a)
    # 对每个顶点进行旋转变换
    x1_new = int((x1 - cx) * math.cos(a) - (y1 - cy) * math.sin(a) + cx)
    y1_new = int((x1 - cx) * math.sin(a) + (y1 - cy) * math.cos(a) + cy)

    x2_new = int((x2 - cx) * math.cos(a) - (y2 - cy) * math.sin(a) + cx)
    y2_new = int((x2 - cx) * math.sin(a) + (y2 - cy) * math.cos(a) + cy)

    x3_new = int((x1 - cx) * math.cos(a) - (y2 - cy) * math.sin(a) + cx)
    y3_new = int((x1 - cx) * math.sin(a) + (y2 - cy) * math.cos(a) + cy)

    x4_new =int( (x2 - cx) * math.cos(a) - (y1 - cy) * math.sin(a) + cx)
    y4_new =int( (x2 - cx) * math.sin(a) + (y1 - cy) * math.cos(a) + cy)
    return [(x1_new, y1_new), (x3_new, y3_new),(x2_new, y2_new) ,(x4_new, y4_new)]



def intersection(g, p):
    g=np.asarray(g)
    p=np.asarray(p)
    g = Polygon(g[:8].reshape((4, 2)))
    p = Polygon(p[:8].reshape((4, 2)))
    if not g.is_valid or not p.is_valid:
        return 0
    inter = Polygon(g).intersection(Polygon(p)).area
    union = g.area + p.area - inter
    if union == 0:
        return 0
    else:
        return inter/union

def NMS_obb(detectResult, nmsThresh):
    predBoxs = []

    sort_detectboxs = sorted(detectResult, key=lambda x: x.score, reverse=True)
    for i in range(len(sort_detectboxs)):
        xmin1 = sort_detectboxs[i].xmin
        ymin1 = sort_detectboxs[i].ymin
        xmax1 = sort_detectboxs[i].xmax
        ymax1 = sort_detectboxs[i].ymax
        classId = sort_detectboxs[i].classId
        angle = sort_detectboxs[i].angle
        p1=rotate_rectangle(xmin1, ymin1, xmax1, ymax1, angle)
        p1=np.array(p1).reshape(-1)
        
        if sort_detectboxs[i].classId != -1:
            predBoxs.append(sort_detectboxs[i])
            for j in range(i + 1, len(sort_detectboxs), 1):
                if classId == sort_detectboxs[j].classId:
                    xmin2 = sort_detectboxs[j].xmin
                    ymin2 = sort_detectboxs[j].ymin
                    xmax2 = sort_detectboxs[j].xmax
                    ymax2 = sort_detectboxs[j].ymax
                    angle2 = sort_detectboxs[j].angle
                    p2=rotate_rectangle(xmin2, ymin2, xmax2, ymax2, angle2)
                    p2=np.array(p2).reshape(-1)
                    iou=intersection(p1, p2)
                    if iou > nmsThresh:
                        sort_detectboxs[j].classId = -1
    return predBoxs

def IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2):
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)

    innerWidth = xmax - xmin
    innerHeight = ymax - ymin

    innerWidth = innerWidth if innerWidth > 0 else 0
    innerHeight = innerHeight if innerHeight > 0 else 0

    innerArea = innerWidth * innerHeight

    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    total = area1 + area2 - innerArea

    return innerArea / total


def NMS_pose(detectResult, nmsThresh):
    predBoxs = []

    sort_detectboxs = sorted(detectResult, key=lambda x: x.score, reverse=True)

    for i in range(len(sort_detectboxs)):
        xmin1 = sort_detectboxs[i].xmin
        ymin1 = sort_detectboxs[i].ymin
        xmax1 = sort_detectboxs[i].xmax
        ymax1 = sort_detectboxs[i].ymax
        classId = sort_detectboxs[i].classId

        if sort_detectboxs[i].classId != -1:
            predBoxs.append(sort_detectboxs[i])
            for j in range(i + 1, len(sort_detectboxs), 1):
                if classId == sort_detectboxs[j].classId:
                    xmin2 = sort_detectboxs[j].xmin
                    ymin2 = sort_detectboxs[j].ymin
                    xmax2 = sort_detectboxs[j].xmax
                    ymax2 = sort_detectboxs[j].ymax
                    iou = IOU(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2)
                    if iou > nmsThresh:
                        sort_detectboxs[j].classId = -1
    return predBoxs


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x, axis=-1):
    # 将输入向量减去最大值以提高数值稳定性
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def process_obb(out, model_w, model_h, stride, angle_feature, index, class_num, objectThresh, scale_w=1, scale_h=1):
    angle_feature=angle_feature.reshape(-1)
    xywh=out[:,:64,:]
    conf=sigmoid(out[:,64:,:])
    out=[]
    conf=conf.reshape(-1)
    for ik in range(model_h*model_w*class_num):
        if conf[ik]>objectThresh:
            w=ik%model_w
            h=(ik%(model_w*model_h))//model_w
            c=ik//(model_w*model_h)
            xywh_=xywh[0,:,(h*model_w)+w] #[1,64,1]
            xywh_=xywh_.reshape(1,4,16,1)
            data=np.array([i for i in range(16)]).reshape(1,1,16,1)
            xywh_=softmax(xywh_,2)
            xywh_ = np.multiply(data, xywh_)
            xywh_ = np.sum(xywh_, axis=2, keepdims=True).reshape(-1)
            xywh_add=xywh_[:2]+xywh_[2:]
            xywh_sub=(xywh_[2:]-xywh_[:2])/2
            angle_feature_= (angle_feature[index+(h*model_w)+w]-0.25)*3.1415927410125732
            angle_feature_cos=math.cos(angle_feature_)
            angle_feature_sin=math.sin(angle_feature_)
            xy_mul1=xywh_sub[0] * angle_feature_cos
            xy_mul2=xywh_sub[1] * angle_feature_sin
            xy_mul3=xywh_sub[0] * angle_feature_sin
            xy_mul4=xywh_sub[1] * angle_feature_cos
            xy=xy_mul1-xy_mul2,xy_mul3+xy_mul4
            xywh_1=np.array([(xy_mul1-xy_mul2)+w+0.5,(xy_mul3+xy_mul4)+h+0.5,xywh_add[0],xywh_add[1]])
            xywh_=xywh_1*stride
            xmin = (xywh_[0] - xywh_[2] / 2) * scale_w
            ymin = (xywh_[1] - xywh_[3] / 2) * scale_h
            xmax = (xywh_[0] + xywh_[2] / 2) * scale_w
            ymax = (xywh_[1] + xywh_[3] / 2) * scale_h
            box = DetectBox_obb(c,conf[ik], xmin, ymin, xmax, ymax,angle_feature_)
            out.append(box)
    return out

def process_pose(out, keypoints, index, model_w, model_h, stride, class_num, objectThresh, scale_w=1, scale_h=1):
    xywh=out[:,:64,:]
    conf=sigmoid(out[:,64:,:])
    out=[]
    for h in range(model_h):
        for w in range(model_w):
            for c in range(class_num):
                if conf[0,c,(h*model_w)+w]>objectThresh:
                    xywh_=xywh[0,:,(h*model_w)+w] #[1,64,1]
                    xywh_=xywh_.reshape(1,4,16,1)
                    data=np.array([i for i in range(16)]).reshape(1,1,16,1)
                    xywh_=softmax(xywh_,2)
                    xywh_ = np.multiply(data, xywh_)
                    xywh_ = np.sum(xywh_, axis=2, keepdims=True).reshape(-1)

                    xywh_temp=xywh_.copy()
                    xywh_temp[0]=(w+0.5)-xywh_[0]
                    xywh_temp[1]=(h+0.5)-xywh_[1]
                    xywh_temp[2]=(w+0.5)+xywh_[2]
                    xywh_temp[3]=(h+0.5)+xywh_[3]

                    xywh_[0]=((xywh_temp[0]+xywh_temp[2])/2)
                    xywh_[1]=((xywh_temp[1]+xywh_temp[3])/2)
                    xywh_[2]=(xywh_temp[2]-xywh_temp[0])
                    xywh_[3]=(xywh_temp[3]-xywh_temp[1])
                    xywh_=xywh_*stride

                    xmin=(xywh_[0] - xywh_[2] / 2) * scale_w
                    ymin = (xywh_[1] - xywh_[3] / 2) * scale_h
                    xmax = (xywh_[0] + xywh_[2] / 2) * scale_w
                    ymax = (xywh_[1] + xywh_[3] / 2) * scale_h
                    keypoint=keypoints[...,(h*model_w)+w+index] 
                    keypoint[...,0:2]=keypoint[...,0:2]//1
                    box = DetectBox(c,conf[0,c,(h*model_w)+w], xmin, ymin, xmax, ymax,keypoint)
                    out.append(box)

    return out

def filter_boxes_seg(boxes, box_confidences, box_class_probs, seg_part, conf):
    """Filter boxes with object threshold.
    """
    box_confidences = box_confidences.reshape(-1)
    candidate, class_num = box_class_probs.shape

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score * box_confidences >= conf)
    scores = (class_max_score * box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]
    seg_part = (seg_part * box_confidences.reshape(-1, 1))[_class_pos]

    return boxes, classes, scores, seg_part

def filter_boxes(boxes, box_confidences, box_class_probs, conf):
    """Filter boxes with object threshold.
    """
    box_confidences = box_confidences.reshape(-1)
    candidate, class_num = box_class_probs.shape

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score* box_confidences >= conf)
    scores = (class_max_score* box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]

    return boxes, classes, scores

def dfl(position):
    # Distribution Focal Loss (DFL)
    x = torch.tensor(position)
    n,c,h,w = x.shape
    p_num = 4
    mc = c//p_num
    y = x.reshape(n,p_num,mc,h,w)
    y = y.softmax(2)
    acc_metrix = torch.tensor(range(mc)).float().reshape(1,1,mc,1,1)
    y = (y*acc_metrix).sum(2)
    return y.numpy()

def box_process(position, imgz):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([imgz // grid_h, imgz //grid_w]).reshape(1, 2, 1, 1)

    position = dfl(position)
    box_xy  = grid +0.5 -position[:,0:2,:,:]
    box_xy2 = grid +0.5 +position[:,2:4,:,:]
    xyxy = np.concatenate((box_xy*stride, box_xy2*stride), axis=1)

    return xyxy

def post_process(input_data, imgz=640, conf=0.5, iou=0.8):
    # input_data[0], input_data[4], and input_data[8] are detection box information
    # input_data[1], input_data[5], and input_data[9] are category score information
    # input_data[2], input_data[6], and input_data[10] are confidence score information
    # input_data[3], input_data[7], and input_data[11] are segmentation information
    # input_data[12] is the proto information
    proto = input_data[-1]
    boxes, scores, classes_conf, seg_part = [], [], [], []
    defualt_branch=3
    pair_per_branch = len(input_data)//defualt_branch

    for i in range(defualt_branch):
        boxes.append(box_process(input_data[pair_per_branch*i], imgz))
        classes_conf.append(input_data[pair_per_branch*i+1])
        scores.append(np.ones_like(input_data[pair_per_branch*i+1][:,:1,:,:], dtype=np.float32))
        seg_part.append(input_data[pair_per_branch*i+3])

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0,2,3,1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]
    seg_part = [sp_flatten(_v) for _v in seg_part]

    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)
    seg_part = np.concatenate(seg_part)

    # filter according to threshold
    boxes, classes, scores, seg_part = filter_boxes_seg(boxes, scores, classes_conf, seg_part, conf)

    zipped = zip(boxes, classes, scores, seg_part)
    sort_zipped = sorted(zipped, key=lambda x: (x[2]), reverse=True)
    result = zip(*sort_zipped)

    max_nms = 30000
    n = boxes.shape[0]  # number of boxes
    if not n:
        return None, None, None, None
    elif n > max_nms:  # excess boxes
        boxes, classes, scores, seg_part = [np.array(x[:max_nms]) for x in result]
    else:
        boxes, classes, scores, seg_part = [np.array(x) for x in result]

    # nms
    nboxes, nclasses, nscores, nseg_part = [], [], [], []
    agnostic = 0
    max_wh = 7680
    c = classes * (0 if agnostic else max_wh)
    ids = torchvision.ops.nms(torch.tensor(boxes, dtype=torch.float32) + torch.tensor(c, dtype=torch.float32).unsqueeze(-1),
                              torch.tensor(scores, dtype=torch.float32), iou)
    real_keeps = ids.tolist()[:SEG_OBJ_MAX_DETECT]
    nboxes.append(boxes[real_keeps])
    nclasses.append(classes[real_keeps])
    nscores.append(scores[real_keeps])
    nseg_part.append(seg_part[real_keeps])

    if not nclasses and not nscores:
        return None, None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)
    seg_part = np.concatenate(nseg_part)

    ph, pw = proto.shape[-2:]
    proto = proto.reshape(seg_part.shape[-1], -1)
    seg_img = np.matmul(seg_part, proto)
    seg_img = sigmoid(seg_img)
    seg_img = seg_img.reshape(-1, ph, pw)

    seg_threadhold = 0.5

    # crop seg outside box
    seg_img = F.interpolate(torch.tensor(seg_img)[None], torch.Size([640, 640]), mode='bilinear', align_corners=False)[0]
    seg_img_t = _crop_mask(seg_img,torch.tensor(boxes) )

    seg_img = seg_img_t.numpy()
    seg_img = seg_img > seg_threadhold
    return boxes, classes, scores, seg_img
        
def _crop_mask(masks, boxes):
    """
    "Crop" predicted masks by zeroing out everything not in the predicted bbox.
    Vectorized by Chong (thanks Chong).

    Args:
        - masks should be a size [h, w, n] tensor of masks
        - boxes should be a size [n, 4] tensor of bbox coords in relative point form
    """

    n, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(1,1,n)
    r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # rows shape(1,w,1)
    c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # cols shape(h,1,1)
    
    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

def nms_boxes(boxes, scores, iou):
    """Suppress non-maximal boxes.
    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


class YoloRKNN:
    def __init__(self, model_path, task, imgz=640):
        self.rknn = RKNNLite(verbose=False)
        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            print('Load RKNN model \"{}\" failed!'.format(model_path))
            exit(ret)
        ret = self.rknn.init_runtime()
        if ret != 0:
            print('Init runtime environment failed!')
            exit(ret)

        self.task = task
        if self.task == 'seg':
            self.co_helper = COCO_test_helper(enable_letter_box=True)

        self.iou = None
        self.conf = None
        self.imgz = imgz

        if self.task == 'obb':
            self.obb_class_num = self.rknn.rknn_runtime.get_outputs()[0].shape[1] - 64  # obb模型类别数量
        elif self.task == 'pose':
            self.pose_class_num = self.rknn.rknn_runtime.get_outputs()[0].shape[1] - 64  # pose模型类别数量
    
    def detect(self, img, iou=0.1, conf=0.25):
        # if self.task not in ('obb', 'seg', 'det', 'pose'):
        #     print('检测模式错误')
        #     return None

        if img is None:
            return None
        
        self.iou = iou
        self.conf = conf
        
        # 数据预处理
        infer_img, aspect_ratio, offset_x, offset_y = self.pretreatment(img)

        # 推理、后处理
        if self.task == 'obb':
            results = self.obb_detect(infer_img, aspect_ratio, offset_x, offset_y)
        elif self.task == 'seg':
            results = self.seg_detect(infer_img)
        elif self.task == 'det':
            results = self.det_detect(infer_img)
        elif self.task == 'pose':
            results = self.pose_detect(infer_img, aspect_ratio, offset_x, offset_y)

        return results
    
    # 数据预处理
    def pretreatment(self, img):
        if self.task in ('obb', 'pose'):
            img, aspect_ratio, offset_x, offset_y = letterbox_resize(img, (self.imgz, self.imgz), 114)  # letterbox缩放
        elif self.task in ('det', 'seg'):
            img = self.co_helper.letter_box(im= img.copy(), new_shape=(self.imgz, self.imgz), pad_color=(114, 114, 114))
            aspect_ratio, offset_x, offset_y = None, None, None
            
        infer_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)            # BGR2RGB
        infer_img = np.expand_dims(infer_img, axis=0)   # (1, 640, 640, 3)

        return infer_img, aspect_ratio, offset_x, offset_y
    
        
    # obb检测
    def obb_detect(self, infer_img, aspect_ratio, offset_x, offset_y):
        results = self.rknn.inference(inputs=[infer_img])
        boxs, classIds, scores = self.obb_postprocessing(results, aspect_ratio, offset_x, offset_y)
        
        return {'boxs': boxs, 'classIds': classIds, 'scores': scores}

    # seg检测
    def seg_detect(self, infer_img):
        outputs = self.rknn.inference(inputs=[infer_img])
        boxs, segs, classIds, scores = self.seg_postprocessing(outputs)
        
        return {'boxs': boxs, 'segs': segs, 'classIds': classIds, 'scores': scores}
    
    def det_detect(self, infer_img):
        outputs = self.rknn.inference(inputs=[infer_img])
        boxs, classIds, scores = self.det_postprocessing(outputs)

        return {'boxs': boxs, 'classIds': classIds, 'scores': scores}

    def pose_detect(self, infer_img, aspect_ratio, offset_x, offset_y):
        results = self.rknn.inference(inputs=[infer_img])
        boxs, keypoints, classIds, scores = self.pose_postprocessing(results, aspect_ratio, offset_x, offset_y)

        return {'boxs': boxs, 'keypoints': keypoints, 'classIds': classIds, 'scores': scores}

    # obb结果后处理
    def obb_postprocessing(self, results, aspect_ratio, offset_x, offset_y):
        outputs=[]
        for x in results[:-1]:
            index,stride=0,0
            if x.shape[2]==20:
                stride=32
                # index=20*4*20*4+20*2*20*2
                index = 6400 + 1600
            if x.shape[2]==40:
                stride=16
                # index=20*4*20*4
                index = 6400
            if x.shape[2]==80:
                stride=8
                index=0
            feature = x.reshape(1, self.obb_class_num + 64, -1)
            output = process_obb(feature, x.shape[3], x.shape[2], stride, results[-1], index, self.obb_class_num, self.conf)
            outputs = outputs + output
        predbox = NMS_obb(outputs, self.iou)
        
        boxs, classIds, scores = [], [], []
        for index in range(len(predbox)):
            xmin = int((predbox[index].xmin-offset_x)/aspect_ratio)
            ymin = int((predbox[index].ymin-offset_y)/aspect_ratio)
            xmax = int((predbox[index].xmax-offset_x)/aspect_ratio)
            ymax = int((predbox[index].ymax-offset_y)/aspect_ratio)
            classId = predbox[index].classId
            score = predbox[index].score
            angle = predbox[index].angle
            xyxyxyxy = rotate_rectangle(xmin,ymin,xmax,ymax,angle)

            boxs.append(xyxyxyxy)
            classIds.append(classId)
            scores.append(score)
        
        return boxs, classIds, scores

    # seg结果后处理
    def seg_postprocessing(self, outputs):
        boxes, classIds, scores, seg_img = post_process(outputs, self.imgz, self.conf, self.iou)
        if boxes is not None:
            real_boxs = self.co_helper.get_real_box(boxes)
            real_segs = self.co_helper.get_real_seg(seg_img)
        else:
            return [], [], [], []
        
        return real_boxs.tolist(), real_segs, classIds.tolist(), scores.tolist()

    def det_postprocessing(self, outputs):
        boxes, scores, classes_conf = [], [], []
        defualt_branch=3
        pair_per_branch = len(outputs)//defualt_branch
        # Python 忽略 score_sum 输出
        for i in range(defualt_branch):
            boxes.append(box_process(outputs[pair_per_branch*i]))
            classes_conf.append(outputs[pair_per_branch*i+1])
            scores.append(np.ones_like(outputs[pair_per_branch*i+1][:,:1,:,:], dtype=np.float32))

        def sp_flatten(_in):
            ch = _in.shape[1]
            _in = _in.transpose(0,2,3,1)
            return _in.reshape(-1, ch)

        boxes = [sp_flatten(_v) for _v in boxes]
        classes_conf = [sp_flatten(_v) for _v in classes_conf]
        scores = [sp_flatten(_v) for _v in scores]

        boxes = np.concatenate(boxes)
        classes_conf = np.concatenate(classes_conf)
        scores = np.concatenate(scores)

        # filter according to threshold
        boxes, classes, scores = filter_boxes(boxes, scores, classes_conf, self.conf)

        # nms
        nboxes, nclasses, nscores = [], [], []
        for c in set(classes):
            inds = np.where(classes == c)
            b = boxes[inds]
            c = classes[inds]
            s = scores[inds]
            keep = nms_boxes(b, s, self.iou)

            if len(keep) != 0:
                nboxes.append(b[keep])
                nclasses.append(c[keep])
                nscores.append(s[keep])

        if not nclasses and not nscores:
            return [], [], []

        boxes = np.concatenate(nboxes)
        classes = np.concatenate(nclasses)
        scores = np.concatenate(nscores)

        return boxes, classes, scores

    def pose_postprocessing(self, results, aspect_ratio, offset_x, offset_y):
        outputs=[]
        keypoints=results[3]
        for x in results[:3]:
            index,stride=0,0
            if x.shape[2]==20:
                stride=32
                # index=20*4*20*4+20*2*20*2
                index = 6400 + 1600
            if x.shape[2]==40:
                stride=16
                # index=20*4*20*4
                index = 6400
            if x.shape[2]==80:
                stride=8
                index=0
            feature=x.reshape(1, self.pose_class_num + 64, -1)
            output=process_pose(feature, keypoints, index, x.shape[3], x.shape[2], stride, self.pose_class_num, self.conf)
            outputs=outputs+output
        predbox = NMS_pose(outputs, self.iou)

        boxs, keypointss, classIds, scores = [], [], [], []
        for i in range(len(predbox)):
            xmin = int((predbox[i].xmin-offset_x)/aspect_ratio)
            ymin = int((predbox[i].ymin-offset_y)/aspect_ratio)
            xmax = int((predbox[i].xmax-offset_x)/aspect_ratio)
            ymax = int((predbox[i].ymax-offset_y)/aspect_ratio)
            classId = predbox[i].classId
            score = predbox[i].score

            keypoints =predbox[i].keypoint.reshape(-1, 3) #keypoint [x, y, conf]
            keypoints[...,0]=(keypoints[...,0]-offset_x)/aspect_ratio
            keypoints[...,1]=(keypoints[...,1]-offset_y)/aspect_ratio

            boxs.append([xmin, ymin, xmax, ymax])
            classIds.append(classId)
            scores.append(score)
            keypointss.append(keypoints.tolist())
        
        return boxs, keypointss, classIds, scores
    
    # 释放模型
    def release(self):
        self.rknn.release()
        self.rknn = None
            

if __name__ == '__main__':
    test = YoloRKNN('/home/xmv/tiercel-core-serving/python/gaosu_road_modeling/weights/carplate_pose_20250609.rknn', 'pose')
    img = cv2.imread('/home/xmv/tiercel-core-serving/python/gaosu_road_modeling/yolov8_rknn/2_3111.jpg')
    
    for _ in range(10):
        results = test.detect(img)
        print(results)
        
    test.release()