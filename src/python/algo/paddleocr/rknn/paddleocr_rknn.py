import cv2
import numpy as np

from rknnlite.api import RKNNLite
from utils.rec_postprocess import CTCLabelDecode
from utils import operators


REC_INPUT_SHAPE = [48, 120] # h,w
PRE_PROCESS_CONFIG = [ 
    {
        'NormalizeImage': {
            'std': [1, 1, 1],
            'mean': [0, 0, 0],
            'scale': '1./255.',
            'order': 'hwc'
        }
    }
]


class PaddleOCRRknn:
    def __init__(self, model_path, character_dict_path):
        self.model = RKNNLite(verbose=False)
        ret = self.model.load_rknn(model_path)
        if ret != 0:
            print('Load RKNN model \"{}\" failed!'.format(model_path))
            exit(ret)
        ret = self.model.init_runtime()
        if ret != 0:
            print('Init runtime environment failed!')
            exit(ret)

        self.preprocess_funct = []
        for item in PRE_PROCESS_CONFIG:
            for key in item:
                pclass = getattr(operators, key)
                p = pclass(**item[key])
                self.preprocess_funct.append(p)

        postprocess_config = {
            'CTCLabelDecode':{
                "character_dict_path": character_dict_path,
                "use_space_char": True
                }   
        }

        self.ctc_postprocess = CTCLabelDecode(**postprocess_config['CTCLabelDecode'])
    
    def ocr(self, img, det=False, cls=False):
        # img = expand_image_with_similar_color(img)
        input_img = self.preprocess({'image':img})
        output = self.model.inference(inputs=[input_img])
        preds = output[0].astype(np.float32)
        output = self.ctc_postprocess(preds)

        return [output]
    
    def preprocess(self, img):
        for p in self.preprocess_funct:
            img = p(img)
        img = cv2.resize(img['image'], (REC_INPUT_SHAPE[1], REC_INPUT_SHAPE[0]))
        img = np.expand_dims(img, 0)
        
        return img
    

if __name__ == "__main__":
    import os

    paddleocr = PaddleOCRRknn(
        "/home/xmv/tiercel-core-serving/python/utils/paddleOCR_rknn/weights/v3_T_48_120_250910.rknn",
        "/home/xmv/tiercel-core-serving/python/utils/paddleOCR_rknn/weights/ppocr_keys_v1.txt"
    )

    imgs_path = '/home/xmv/tiercel-core-serving/python/utils/paddleOCR_rknn/test_imgs'
    img_names = os.listdir(imgs_path)
    for name in img_names:
        print(name)
        img_path = os.path.join(imgs_path, name)
        img_cp = cv2.imread(img_path)

        output = paddleocr.ocr(img_cp)
        print(output)
