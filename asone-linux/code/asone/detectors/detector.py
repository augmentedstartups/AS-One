import os
import cv2

from yolov5 import YOLOv5Detector 
from yolov6 import YOLOv6Detector 
from yolov7 import YOLOv7Detector
from yolor import YOLOrDetector
from yolox import YOLOxDetector
from utils.weights_path import get_weight_path
from utils.cfg_path import get_cfg_path


class Detector:
    def __init__(self, model_flag: int, use_cuda=True):
        self.detector = self._select_detector(model_flag, use_cuda)
     

    def _select_detector(self, model_flag, cuda):
        # Get required weight using model_flag
        onnx, weight = get_weight_path(model_flag)
       
        if model_flag in range(0, 20):
            _detector = YOLOv5Detector(weights=weight,
                                       use_onnx=onnx,
                                       use_cuda=cuda)
        elif model_flag in range(20, 26):
            _detector = YOLOv6Detector(weights=weight,
                                       use_onnx=onnx,
                                       use_cuda=cuda)
        elif model_flag in range(26, 40):
            _detector = YOLOv7Detector(weights=weight,
                                       use_onnx=onnx,
                                       use_cuda=cuda)
        elif model_flag in range(40, 50):
            # Get Configuration file for Yolor
            if model_flag in [40, 42, 44, 46, 48]: 
                cfg = get_cfg_path(model_flag)
            else:
                cfg = None
            _detector = YOLOrDetector(weights=weight,
                                      cfg=cfg,
                                      use_onnx=onnx,
                                      use_cuda=cuda)
        
        elif model_flag in range(50, 64):
            _detector = YOLOxDetector(model_name=model_name,
                                      exp_file=exp,
                                      weights=weight,
                                      use_onnx=onnx,
                                      use_cuda=cuda)
            
          
        return _detector

    def get_detector(self, image):
        return self.detector.detect(image)
            

if __name__ == '__main__':
    
    # Initialize YOLOv6 object detector
    model_type = 27
    result = Detector(model_flag=model_type, use_cuda=True)
    img = cv2.imread('/home/ajmair/benchmarking/asone/asone-linux/test.jpeg')
    pred = result.get_detector(img)
    print(pred)
   
  