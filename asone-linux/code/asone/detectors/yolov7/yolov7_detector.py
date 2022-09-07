from utils.yolov7_utils import (prepare_input, process_output,
                               draw_detections, non_max_suppression)
from models.experimental import attempt_load

import onnxruntime
import torch
import os
import cv2
import sys
import numpy as np


class YOLOv7Detector:
    def __init__(self,
                 weights=None,
                 use_onnx=False,
                 use_cuda=True):

        self.use_onnx = use_onnx
        self.device = 'cuda' if use_cuda else 'cpu'
        if weights == None:
            weights = os.path.join("weights", "yolov5n.pt")
        #If incase weighst is a list of paths then select path at first index
        weights = str(weights[0] if isinstance(weights, list) else weights)
        
        # Load Model
        self.model = self.load_model(use_cuda, weights)  

    def load_model(self, use_cuda, weights, fp16=False):
        # Device: CUDA and if fp16=True only then half precision floating point works  
        self.fp16 = fp16 & ((not self.use_onnx or self.use_onnx) and self.device != 'cpu')
        # Load onnx 
        if self.use_onnx:
            if use_cuda:
                providers = ['CUDAExecutionProvider','CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
            model = onnxruntime.InferenceSession(weights, providers=providers)
        #Load Pytorch
        else: 
            model = attempt_load(weights, map_location=self.device)
            model.half() if self.fp16 else model.float()
        return model


    def detect(self, image: list,
               conf_thres: float = 0.25,
               iou_thres: float = 0.45,
               classes: int = None,
               agnostic_nms: bool = False,
               input_shape=(640, 640)) -> list:
        # Preprocess input image and also copying original image for later use
        original_image = image.copy()
        img_height, img_width = original_image.shape[:2]
        processed_image = prepare_input(image, input_shape)
        
        # Perform Inference on the Image
        if self.use_onnx:
        # Run ONNX model 
            input_name = self.model.get_inputs()[0].name
            prediction = self.model.run([self.model.get_outputs()[0].name], {
                                 input_name: processed_image})
        # Run Pytorch model
        else:
            processed_image = torch.from_numpy(processed_image).to(self.device)
            # Change image floating point precision if fp16 set to true
            processed_image = processed_image.half() if self.fp16 else processed_image.float() 
            prediction = self.model(processed_image, augment=False)[0]
     
        # Postprocess prediction
        if self.use_onnx:
            detection = process_output(prediction,
                                       original_image.shape[:2],
                                       input_shape,
                                       conf_thres,
                                       iou_thres)
        else:
            detection = non_max_suppression(prediction,
                                            conf_thres,
                                            iou_thres,
                                            classes,
                                            agnostic_nms)[0]
            
            detection = detection.detach().cpu().numpy()
            # Rescaling Bounding Boxes
            detection[:, :4] /= np.array([input_shape[1], input_shape[0], input_shape[1], input_shape[0]])
            detection[:, :4] *= np.array([img_width, img_height, img_width, img_height])

        image_info = {
            'width': original_image.shape[1],
            'height': original_image.shape[0],
        }

        self.boxes = detection[:, :4]
        self.scores = detection[:, 4:5]
        self.class_ids = detection[:, 5:6]
        
        return detection, image_info

    def draw_detections(self, image, draw_scores=True, mask_alpha=0.4):
        return draw_detections(image, self.boxes, self.scores,
                               self.class_ids, mask_alpha)

if __name__ == '__main__':
    model_path = sys.argv[1]
    # Initialize YOLOv7 object detector
    yolov7_detector = YOLOv7Detector(model_path, use_onnx=True, use_cuda=False)
    img = cv2.imread('/home/ajmair/benchmarking/asone/asone-linux/test.jpeg')
    # Detect Objects
    result =  yolov7_detector.detect(img)
    print(result)
    bbox_drawn = yolov7_detector.draw_detections(img) 
    cv2.imwrite("result.jpg", bbox_drawn)