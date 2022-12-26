import os
import sys
import onnxruntime
import torch
from asone.utils import get_names
import numpy as np
import warnings
from asone.detectors.yolov7.yolov7.utils.yolov7_utils import (prepare_input,
                                 process_output,
                                 non_max_suppression)
from asone.detectors.yolov7.yolov7.models.experimental import attempt_load
from asone import utils

sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov7'))
class YOLOv7Detector:
    def __init__(self,
                 weights=None,
                 use_onnx=False,
                 use_cuda=True):
        self.use_onnx = use_onnx
        self.device = 'cuda' if use_cuda else 'cpu'

        #If incase weighst is a list of paths then select path at first index

        weights = str(weights[0] if isinstance(weights, list) else weights)

        if not os.path.exists(weights):
            utils.download_weights(weights)
            

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
               input_shape=(640, 640),
               filter_classes:list=None) -> list:
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

            with torch.no_grad():
                prediction = self.model(processed_image, augment=False)[0]
                
        detection = []
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

        if len(detection) > 0:
            self.boxes = detection[:, :4]
            self.scores = detection[:, 4:5]
            self.class_ids = detection[:, 5:6]

        if filter_classes:
            class_names = get_names()

            filter_class_idx = []
            if filter_classes:
                for _class in filter_classes:
                    if _class.lower() in class_names:
                        filter_class_idx.append(class_names.index(_class.lower()))
                    else:
                        warnings.warn(f"class {_class} not found in model classes list.")

            detection = detection[np.in1d(detection[:,5].astype(int), filter_class_idx)]

        return detection, image_info
