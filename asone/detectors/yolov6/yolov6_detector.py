import os
import sys
from asone.utils import get_names
import numpy as np
import warnings
import torch
import onnxruntime

from asone import utils
from asone.detectors.yolov6.yolov6.utils.yolov6_utils import (prepare_input, load_pytorch,
                                                              non_max_suppression, process_and_scale_boxes) 
sys.path.append(os.path.dirname(__file__))  

class YOLOv6Detector:
    def __init__(self,
                 weights=None,
                 use_onnx=False,
                 use_cuda=True):

        self.use_onnx = use_onnx
        self.device = 'cuda' if use_cuda else 'cpu'

        if not os.path.exists(weights):
            utils.download_weights(weights)
        #If incase weighst is a list of paths then select path at first index
        weights = str(weights[0] if isinstance(weights, list) else weights)
        
        # Load Model
        self.model = self.load_model(use_cuda, weights)
        
        if use_onnx:
            # Get Some ONNX model details 
            self.input_shape, self.input_height, self.input_width = self.ONNXModel_detail(self.model)
            self.input_names, self.output_names = self.ONNXModel_names(self.model)


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
            model = load_pytorch(weights, map_location=self.device)
            model.half() if self.fp16 else model.float()
        return model

    def ONNXModel_detail(self, model):
         # Get Model Input
        model_inputs = model.get_inputs()
        # Input shape
        input_shape = model_inputs[0].shape
        input_height = input_shape[2]
        input_width = input_shape[3]
        
        return input_shape, input_height, input_width

    def ONNXModel_names(self, model):
        # Get Model Input
        model_inputs = model.get_inputs()
        input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        # Get Model Output
        model_outputs = model.get_outputs()
        output_names = [model_outputs[i].name for i in range(len(model_outputs))]
        
        return input_names, output_names  
        
    def detect(self, image: list,
               input_shape: tuple = (640, 640),
               conf_thres: float = 0.25,
               iou_thres: float = 0.45,
               max_det: int = 1000,
               filter_classes: bool = None,
               agnostic_nms: bool = True,
               with_p6: bool = False,
               return_image=False) -> list:
        
        # Prepare Input
        img_height, img_width = image.shape[:2]
        processed_image = prepare_input(image, input_shape[0], input_shape[1])
        
        # Perform Inference on the Image
        if self.use_onnx:
        # Run ONNX model 
            prediction = self.model.run(self.output_names,
                                    {self.input_names[0]: processed_image})[0] 
        # Run Pytorch model
        else:
            processed_image = torch.from_numpy(processed_image).to(self.device)
            # Change image floating point precision if fp16 set to true
            processed_image = processed_image.half() if self.fp16 else processed_image.float() 
            prediction = self.model(processed_image)[0]

        # Post Procesing, non-max-suppression and rescaling
        if self.use_onnx:
            # Process ONNX Output
            
            boxes, scores, class_ids = process_and_scale_boxes(prediction, img_height, img_width,
                                                   input_shape[1], input_shape[0])
            detection = []
            for box in range(len(boxes)):
                pred = np.append(boxes[box], scores[box])
                pred = np.append(pred, class_ids[box])
                detection.append(pred)
            detection = np.array(detection)
        else:
            detection = non_max_suppression(prediction,
                                    conf_thres,
                                    iou_thres,
                                    agnostic=agnostic_nms, 
                                    max_det=max_det)[0]
            
            detection = detection.detach().cpu().numpy()
            detection[:, :4] /= np.array([input_shape[1], input_shape[0], input_shape[1], input_shape[0]])
            detection[:, :4] *= np.array([img_width, img_height, img_width, img_height])
            
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
    
        image_info = {
            'width': image.shape[1],
            'height': image.shape[0],
        }

        if return_image:
            return detection, image
        else: 
            return detection, image_info