import os
import acl
import numpy as np
import cv2
import time
from .postprocessing_obb import postprocessing_obb
from .postprocessing_seg import postprocessing_seg


ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    
    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))    
    dw, dh = (new_shape[1] - new_unpad[0])/2, (new_shape[0] - new_unpad[1])/2  # wh padding 
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    return im


class YoloAscend:
    def __init__(self, model_path, task, device_id=0):
        self.task = task
        self.device_id = device_id

        self.imgz = None            # 输入数据大小
        self.outputs_shape = []    # 输出数据形状
        self.class_number = None    # 类别数量
        self.height, self.width = None, None

        # 初始化昇腾环境
        ret = acl.init()
        if ret == 0:
            print("昇腾环境初始化成功")
        else:
            print(f"昇腾环境初始化失败, code: {ret}")
        ret = acl.rt.set_device(self.device_id)
        if ret == 0:
            print(f"加载设备:{self.device_id} 成功")
        else:
            print(f"加载设备:{self.device_id} 失败, code: {ret}")

        # 加载模型
        self.model_id, ret = acl.mdl.load_from_file(model_path)
        if ret == 0:
            print("模型加载成功")
        else:
            print(f"模型加载失败, code: {ret}")

        self.model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        if ret == 0:
            print("get_desc 成功")
        else:
            print(f"get_desc 失败, code: {ret}")

        # 创建输入输出数据
        self.input_dataset, self.input_data = self.prepare_dataset('input')
        self.output_dataset, self.output_data = self.prepare_dataset('output')

        print('输入数据大小:', self.imgz)
        print('输出数据形状:', self.outputs_shape)
        print('类别数量:', self.class_number)

    # 检测
    def detect(self, img, iou=0.1, conf=0.25):
        # if self.task not in ('obb', 'seg'):
        #     print('检测模式错误')
        #     return None

        if img is None:
            return None
        
        self.height, self.width = img.shape[:2]

        # 预处理
        input_img = self.pretreatment(img, (self.imgz, self.imgz))

        # 推理
        outputs = self.forward(input_img)

        # 后处理
        if self.task == 'obb':
            outputs = outputs[0].reshape(*self.outputs_shape[0])
            boxes, class_ids, scores = postprocessing_obb(outputs, conf, iou, self.class_number, self.imgz, (self.width, self.height))

            return {'boxs': boxes, 'classIds': class_ids, 'scores': scores}
        elif self.task == 'seg':
            outputs_1 = outputs[0].reshape(*self.outputs_shape[0])
            outputs_2 = outputs[1].reshape(*self.outputs_shape[1])
            boxes, class_ids, scores, mask_maps = postprocessing_seg([outputs_1, outputs_2], conf, iou, self.imgz, (self.width, self.height))

            return {'segs': mask_maps, 'boxs': boxes, 'classIds': class_ids, 'scores': scores}
    
    def pretreatment(self, image, input_shape):
        if self.task == 'obb':
            image_rize = letterbox(image, (self.imgz, self.imgz))
        elif self.task == 'seg':
            image_rize = cv2.resize(image, input_shape)
        image_rize = cv2.cvtColor(image_rize, cv2.COLOR_BGR2RGB)
        input = image_rize.transpose(2, 0, 1).astype(dtype=np.float32)  # HWC2CHW
        input = input / 255.0
        input = np.expand_dims(input, 0)
    
        return input
    
    def forward(self, input):
        # 拷贝所有输入到设备
        bytes_data = input.tobytes()
        bytes_ptr = acl.util.bytes_to_ptr(bytes_data)
        ret = acl.rt.memcpy(
            self.input_data[0]["buffer"], 
            self.input_data[0]["size"], 
            bytes_ptr, 
            len(bytes_data), 
            ACL_MEMCPY_HOST_TO_DEVICE
        )
        
        # 执行推理
        ret = acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset)
        # 获取所有输出
        outputs = []
        for i in range(len(self.output_data)):
            buffer_host, _ = acl.rt.malloc_host(self.output_data[i]["size"])
            ret = acl.rt.memcpy(
                buffer_host,
                self.output_data[i]["size"],
                self.output_data[i]["buffer"],
                self.output_data[i]["size"],
                ACL_MEMCPY_DEVICE_TO_HOST
            )
            bytes_out = acl.util.ptr_to_bytes(buffer_host, self.output_data[i]["size"])
            data = np.frombuffer(bytes_out, dtype=np.float32)
            outputs.append(data)
            acl.rt.free_host(buffer_host)
            
        return outputs

    def prepare_dataset(self, io_type):
        # 根据类型获取输入/输出数量
        if io_type == "input":
            io_num = acl.mdl.get_num_inputs(self.model_desc)
            get_size_func = acl.mdl.get_input_size_by_index

            dims = acl.mdl.get_input_dims(self.model_desc, 0)
            if dims[1] != 0:
                print(f"get_input_dims errer code: {dims[1]}")
            self.imgz = dims[0]['dims'][-1]

        else:
            io_num = acl.mdl.get_num_outputs(self.model_desc)
            get_size_func = acl.mdl.get_output_size_by_index
            
            dims = acl.mdl.get_output_dims(self.model_desc, 0)

            self.outputs_shape.append(dims[0]['dims'])
            if self.task == 'obb':
                self.class_number = dims[0]['dims'][1] - 5
            elif self.task == 'seg':
                self.class_number = dims[0]['dims'][1] - 36

                dims = acl.mdl.get_output_dims(self.model_desc, 1)
                self.outputs_shape.append(dims[0]['dims'])

        # 创建数据集
        dataset = acl.mdl.create_dataset()
        buffers = []
        for i in range(io_num):
            # 获取内存大小并分配
            buffer_size = get_size_func(self.model_desc, i)

            buffer, _ = acl.rt.malloc(buffer_size, ACL_MEM_MALLOC_HUGE_FIRST)
            # 绑定数据缓冲区
            data_buffer = acl.create_data_buffer(buffer, buffer_size)
            acl.mdl.add_dataset_buffer(dataset, data_buffer)
            buffers.append({
                "buffer": buffer, 
                "data": data_buffer, 
                "size": buffer_size
            })

        return dataset, buffers


    def __del__(self):
        # 释放资源
        for data in self.input_data + self.output_data:
            acl.destroy_data_buffer(data["data"])
            acl.rt.free(data["buffer"])
        acl.mdl.destroy_dataset(self.input_dataset)
        acl.mdl.destroy_dataset(self.output_dataset)
        acl.mdl.destroy_desc(self.model_desc)
        acl.mdl.unload(self.model_id)
        acl.rt.reset_device(self.device_id)
        acl.finalize()
