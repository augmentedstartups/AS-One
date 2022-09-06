from utils.yolov5_utils import (non_max_suppression,
                                scale_coords, letterbox,
                                draw_detections)
from models.experimental import attempt_load

import numpy as np
import torch
import onnxruntime
import os
import sys
import cv2

class YOLOv5Detector:
    def __init__(self,
                 weights=None, use_onnx=True):
        self.use_onnx = use_onnx
        if weights == None:
            weights = os.path.join("weights", "yolov5n.pt")
        #If incase weighst is a list of paths then select path at first index
        self.weights = str(weights[0] if isinstance(weights, list) else weights)
        # Load Model
        self.model = self.load_model()
        
    def load_model(self, fp16=False, use_cuda=True):
        self.use_cuda = use_cuda
        self.device = 'cuda' if self.use_cuda else 'cpu'
        # Device: CUDA and if fp16=True only then half precision floating point works  
        self.fp16 = fp16 & ((not self.use_onnx or self.use_onnx) and self.device != 'cpu')
        # Load onnx 
        if self.use_onnx:
            if self.use_cuda:
                providers = ['CUDAExecutionProvider','CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
                model = onnxruntime.InferenceSession(self.weights, providers=providers)
        else: #Load Pytorch
            model = attempt_load(self.weights, device=self.device, inplace=True, fuse=True)
            model.half() if self.fp16 else model.float()
        return model

    def image_preprocessing(self,
                            image: list,
                            input_shape=(640, 640))-> list:

        self.original_image = image.copy()
        image = letterbox(image, input_shape, stride=32, auto=False)[0]
        image = image.transpose((2, 0, 1))[::-1]
        image = np.ascontiguousarray(image, dtype=np.float32)
        image /= 255  # 0 - 255 to 0.0 - 1.0
        if len(image.shape) == 3:
            image = image[None]  # expand for batch dim  
        self.image = image
        return self.original_image, self.image

    def detect(self, 
               image: list,
               conf_thres: float = 0.25,
               iou_thres: float = 0.45,
               classes: int = None,
               agnostic_nms: bool = False,
               input_shape=(640, 640),
               max_det: int = 1000) -> list:
     
        # Image Preprocessing
        original_image, processed_image = self.image_preprocessing(image, input_shape)
        
        # Inference
        if self.use_onnx:
            # Input names of ONNX model on which it is exported   
            input_name = self.model.get_inputs()[0].name
            # Run onnx model 
            pred = self.model.run([self.model.get_outputs()[0].name], {input_name: processed_image})[0]
        else:
            processed_image = torch.from_numpy(processed_image).to(self.device)
            # Change image floating point precision if fp16 set to true
            processed_image = processed_image.half() if self.fp16 else processed_image.float() 
            pred = self.model(processed_image, augment=False, visualize=False)[0]
       
        # Post Processing
        if isinstance(pred, np.ndarray):
            pred = torch.tensor(pred, device=self.device)
        predictions = non_max_suppression(pred, conf_thres, 
                                          iou_thres, classes, 
                                          agnostic_nms, 
                                          max_det=max_det)
     
        for i, prediction in enumerate(predictions):  # per image
            if len(prediction):
                prediction[:, :4] = scale_coords(
                    processed_image.shape[2:], prediction[:, :4], original_image.shape).round()
                predictions[i] = prediction
        detections = predictions[0].cpu().numpy()
        image_info = {
            'width': original_image.shape[1],
            'height': original_image.shape[0],
        }

        self.boxes = detections[:, :4]
        self.scores = detections[:, 4:5]
        self.class_ids = detections[:, 5:6]
        return detections, image_info

    def draw_detections(self, image, draw_scores=True, mask_alpha=0.4):
        return draw_detections(image, self.boxes, self.scores,
                               self.class_ids, mask_alpha)

if __name__ == '__main__':
    
    # model_path = sys.argv[1]
    # Initialize YOLOv6 object detector
    yolov5_detector = YOLOv5Detector(use_onnx=False)
    img = cv2.imread('/home/ajmair/benchmarking/asone/asone-linux/test.jpeg')
    # Detect Objects
    result =  yolov5_detector.detect(img)
    print(result)
    bbox_drawn = yolov5_detector.draw_detections(img)
    cv2.imwrite("myoutput.jpg", bbox_drawn)
 