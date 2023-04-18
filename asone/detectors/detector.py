import cv2

from asone.detectors.yolov5 import YOLOv5Detector
from asone.detectors.yolov6 import YOLOv6Detector
from asone.detectors.yolov7 import YOLOv7Detector
from asone.detectors.yolor import YOLOrDetector
from asone.detectors.yolox import YOLOxDetector
from asone.detectors.easyocr_detector import TextDetector

from asone.detectors.utils.weights_path import get_weight_path
from asone.detectors.utils.cfg_path import get_cfg_path
from asone.detectors.utils.exp_name import get_exp__name
from .yolov8 import YOLOv8Detector


class Detector:
    def __init__(self,
                 model_flag: int,
                 weights: str = None,
                 use_cuda: bool = True,
                 recognizer:int = None):
        
        self.model = self._select_detector(model_flag, weights, use_cuda, recognizer)
    def _select_detector(self, model_flag, weights, cuda, recognizer):
        # Get required weight using model_flag
        mlmodel = False
        if weights and weights.split('.')[-1] == 'onnx':
            onnx = True
            weight = weights
        elif weights and weights.split('.')[-1] == 'mlmodel':
            onnx = False
            weight = weights
            mlmodel = True    
        elif weights:
            onnx = False
            weight = weights
        else:
            mlmodel, onnx, weight = get_weight_path(model_flag)
        
        if model_flag in range(0, 20):
            _detector = YOLOv5Detector(weights=weight,
                                       use_onnx=onnx,
                                       use_cuda=cuda)
        elif model_flag in range(20, 34):
            _detector = YOLOv6Detector(weights=weight,
                                       use_onnx=onnx,
                                       use_cuda=cuda)
        elif model_flag in range(34, 48):
            _detector = YOLOv7Detector(weights=weight,
                                       use_onnx=onnx,
                                       use_cuda=cuda)
        elif model_flag in range(48, 58):
            # Get Configuration file for Yolor
            if model_flag in range(48, 57, 2):
                cfg = get_cfg_path(model_flag)
            else:
                cfg = None
            _detector = YOLOrDetector(weights=weight,
                                      cfg=cfg,
                                      use_onnx=onnx,
                                      use_cuda=cuda)

        elif model_flag in range(58, 72):
            # Get exp file and corresponding model for pytorch only
            if model_flag in range(58, 71, 2):
                exp, model_name = get_exp__name(model_flag)
            else:
                exp = model_name = None
            _detector = YOLOxDetector(model_name=model_name,
                                      exp_file=exp,
                                      weights=weight,
                                      use_onnx=onnx,
                                      use_cuda=cuda)
        elif model_flag in range(72, 82):
            # Get exp file and corresponding model for pytorch only
            _detector = YOLOv8Detector(weights=weight,
                                       use_onnx=onnx,
                                       use_cuda=cuda)
        # Get TextDetector model
        elif model_flag  in range(82, 85):
            _detector = TextDetector(detect_network=weight, use_cuda=cuda)
        
        elif model_flag in range(120, 131):
            # Get exp file and corresponding model for coreml only
            _detector = YOLOv5Detector(weights=weight,
                                       use_onnx=onnx,
                                       mlmodel=mlmodel,
                                       use_cuda=cuda)
        elif model_flag in range(131, 139):
            # Get exp file and corresponding model for coreml only
            _detector = YOLOv7Detector(weights=weight,
                                       use_onnx=onnx,
                                       mlmodel=mlmodel,
                                       use_cuda=cuda)
        elif model_flag in range(139, 144):
            # Get exp file and corresponding model for coreml only
            _detector = YOLOv8Detector(weights=weight,
                                       use_onnx=onnx,
                                       mlmodel=mlmodel,
                                       use_cuda=cuda)
        return _detector

    def get_detector(self):
        return self.model

    def detect(self,
               image: list,
               return_image=False,
               **kwargs: dict):
        return self.model.detect(image,return_image,**kwargs)


if __name__ == '__main__':

    # Initialize YOLOv6 object detector
    model_type = 56
    result = Detector(model_flag=model_type, use_cuda=True)
    img = cv2.imread('asone/asone-linux/test.jpeg')
    pred = result.get_detector(img)
    print(pred)
