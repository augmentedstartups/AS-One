from models.yolov5_utils import (non_max_suppression,
                                scale_coords, letterbox,
                                draw_detections)
from models.experimental import attempt_load

import numpy as np
import torch
import onnxruntime
import os
import cv2

class YOLOv5Detector:
    def __init__(self,
                 weights = os.path.join(os.path.dirname(
                           os.path.abspath(__file__)), './weights/yolov5s.onnx')) -> None:
      
        self.weights = str(weights[0] if isinstance(weights, list) else weights)
        self.weights_type = self.weights.split("/")[-1].split(".")[1]
          # FP16
        self.pt = True if self.weights_type == "pt" else False
        self.onnx = True if self.pt is not True else False


    def load_model(self, fp16=True, use_cuda=False):
        self.use_cuda = use_cuda
        self.device = 'cuda' if self.use_cuda else 'cpu'
        self.fp16 = fp16 & ((self.pt or self.onnx) and self.device != 'cpu')
        print(self.fp16)
    
        # Load onnx 
        if self.onnx:
            if self.use_cuda:
                providers = [
                            'CUDAExecutionProvider',
                            'CPUExecutionProvider'
                            ]
            else:
                providers = ['CPUExecutionProvider']
            self.model = onnxruntime.InferenceSession(self.weights, providers=providers)
        else: #Load Pytorch
            model = attempt_load(self.weights, device=self.device, inplace=True, fuse=True)
            model.half() if self.fp16 else model.float()
            self.model = model
        return self.model

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
        
        self.model = self.load_model()
        image0, img = self.image_preprocessing(image, input_shape)
        
        # Inference
        if self.onnx:   
            input_name = self.model.get_inputs()[0].name
            pred = self.model.run([self.model.get_outputs()[0].name], {input_name: img})[0]
        else:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.fp16 else img.float() 
            pred = self.model(img, augment=False, visualize=False)[0]
       
        # Post Processing
        if isinstance(pred, np.ndarray):
            pred = torch.tensor(pred, device=self.device)
        pred = non_max_suppression(
                                pred, conf_thres, 
                                iou_thres, classes, 
                                agnostic_nms, 
                                max_det=max_det)
     
        for i, det in enumerate(pred):  # per image
            if len(det):
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], image0.shape).round()
                pred[i] = det
        dets = pred[0].cpu().numpy()
        image_info = {
            'width': image0.shape[1],
            'height': image0.shape[0],
        }

        self.boxes = dets[:, :4]
        self.scores = dets[:, 4:5]
        self.class_ids = dets[:, 5:6]
        return dets, image_info

    def draw_detections(self, image, draw_scores=True, mask_alpha=0.4):
        return draw_detections(image, self.boxes, self.scores,
                               self.class_ids, mask_alpha)

if __name__ == '__main__':
    model_path = "/home/ajmair/benchmarking/asone/asone-linux/code/asone/detectors/yolov5/weights/yolov5n.onnx"
    # Initialize YOLOv6 object detector
    yolov5_detector = YOLOv5Detector(model_path)
    img = cv2.imread('/home/ajmair/benchmarking/asone/asone-linux/test.jpeg')
    # Detect Objects
    result =  yolov5_detector.detect(img)
    print(result)
    bbox_drawn = yolov5_detector.draw_detections(img)
    cv2.imwrite("myoutput.jpg", bbox_drawn)
 