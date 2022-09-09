from yolov5 import YOLOv5Detector 
from yolov6 import YOLOv6Detector 
from yolov7 import YOLOv7Detector
from yolor import YOLOrDetector
from yolox import YOLOxDetector

import os
import cv2

# Detectors


class Detector:
    def __init__(self, model_type: int, use_cuda=True):

        self.detector = self._select_detector(model_type, use_cuda)
     
    def _select_detector(self, model_type, cuda):

        if model_type in range(0,32):
            onnx = False
        else:
            onnx = True
        
        
        if model_type == 0:
            weights = os.path.join('yolov5','weights','yolov5n.pt')
            _detector = YOLOv5Detector(weights=weights,
                                        use_onnx=onnx,
                                        use_cuda=cuda)
        
        return _detector

    def get_detector(self, image):
        return self.detector.detect(image)
            

if __name__ == '__main__':
    
    # Initialize YOLOv6 object detector
    model_type = 0
    result = Detector(model_type, use_cuda=True)
    img = cv2.imread('/home/ajmair/benchmarking/asone/asone-linux/test.jpeg')
    pred = result.get_detector(img)
   
  