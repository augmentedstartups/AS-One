import time
import cv2
import numpy as np
import os
import onnxruntime
import torch
import sys
from .yolov6_utils import xywh2xyxy, prepare_input, process_output 


class YOLOv6:
    def __init__(self, weights=
                       os.path.join
                       (os.path.dirname
                       (os.path.abspath(__file__)), './weights/yolov5s.onnx'),
                 use_cuda=True, use_onnx=True) -> None:

        if use_onnx:
            if use_cuda:
                providers = [
                            'CUDAExecutionProvider',
                            'CPUExecutionProvider'
                            ]
            else:
                providers = ['CPUExecutionProvider']
      
        self.session = onnxruntime.InferenceSession(weights,
                                                    providers = providers)
        # Get Model Input
        
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        
        # Input shape
        
        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        
        # Get Model Output
        
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    def detect(self, image: list,
               conf_thres: float = 0.7,
               iou_thres: float = 0.5,
               classes: int = None,
               input_shape=(640, 640),
               max_det: int = 1000) -> list:
        
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres

        # Prepare Input
        img_height, img_width = image.shape[:2]
        input_tensor = prepare_input(image, img_height, img_width)
    
        # Perform Inference on the Image
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})[0] 

        # Process Output
        self.boxes, self.scores, self.class_ids = process_output(outputs, img_height, img_width )
        return self.boxes, self.scores, self.class_ids 


# if __name__ == '__main__':
#     from imread_from_url import imread_from_url

#     model_path = "../models/yolov6s.onnx"

#     # Initialize YOLOv6 object detector
#     yolov6_detector = YOLOv6(model_path, conf_thres=0.3, iou_thres=0.5)

#     img = imread_from_url(
#         "https://upload.wikimedia.org/wikipedia/commons/a/af/Motorcyclists_lane_splitting_in_Bangkok%2C_Thailand.jpg")

#     # Detect Objects
#     yolov6_detector(img)

#     # Draw detections
#     combined_img = yolov6_detector.draw_detections(img)
#     cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
#     cv2.imshow("Output", combined_img)
    # cv2.waitKey(0)

    


