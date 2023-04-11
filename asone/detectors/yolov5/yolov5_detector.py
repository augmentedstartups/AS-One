import os
from asone.utils import get_names
import numpy as np
import warnings
import torch
from PIL import Image
import onnxruntime
import coremltools as ct
from asone.detectors.yolov5.yolov5.utils.yolov5_utils import (non_max_suppression,
                                                              scale_coords,
                                                              letterbox)
from asone.detectors.yolov5.yolov5.models.experimental import attempt_load
from asone import utils
from asone.detectors.utils.coreml_utils import yolo_to_xyxy, generalize_output_format, scale_bboxes


class YOLOv5Detector:
    def __init__(self,
                 weights=None,
                 use_onnx=False,
                 mlmodel=False,
                 use_cuda=True):

        self.use_onnx = use_onnx
        self.mlmodel = mlmodel
        self.device = 'cuda' if use_cuda else 'cpu'

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
        #Load coreml 
        elif self.mlmodel:
            model = ct.models.MLModel(weights)
        #Load Pytorch
        else: 
            model = attempt_load(weights, device=self.device, inplace=True, fuse=True)
            model.half() if self.fp16 else model.float()
        return model

    def image_preprocessing(self,
                            image: list,
                            input_shape=(640, 640))-> list:

        original_image = image.copy()
        image = letterbox(image, input_shape, stride=32, auto=False)[0]
        image = image.transpose((2, 0, 1))[::-1]
        image = np.ascontiguousarray(image, dtype=np.float32)
        image /= 255  # 0 - 255 to 0.0 - 1.0
        if len(image.shape) == 3:
            image = image[None]  # expand for batch dim  
        return original_image, image
    
    def detect(self, image: list,
               input_shape: tuple = (640, 640),
               conf_thres: float = 0.25,
               iou_thres: float = 0.45,
               max_det: int = 1000,
               filter_classes: bool = None,
               agnostic_nms: bool = True,
               with_p6: bool = False, 
               return_image=False) -> list:
     
        # Image Preprocessing
        original_image, processed_image = self.image_preprocessing(image, input_shape)
        
        # Inference
        if self.use_onnx:
            # Input names of ONNX model on which it is exported   
            input_name = self.model.get_inputs()[0].name
            # Run onnx model 
            pred = self.model.run([self.model.get_outputs()[0].name], {input_name: processed_image})[0]
            # Run mlmodel   
        
        elif self.mlmodel:
            h ,w = image.shape[:2]
            pred = self.model.predict({"image":Image.fromarray(image).resize(input_shape)})
            xyxy = yolo_to_xyxy(pred['coordinates'], input_shape)
            out = generalize_output_format(xyxy, pred['confidence'], conf_thres)
            if out != []:
                detections = scale_bboxes(out, image.shape[:2], input_shape)
            else:
                detections = np.empty((0, 6))
            
            if filter_classes:
                class_names = get_names()

                filter_class_idx = []
                if filter_classes:
                    for _class in filter_classes:
                        if _class.lower() in class_names:
                            filter_class_idx.append(class_names.index(_class.lower()))
                        else:
                            warnings.warn(f"class {_class} not found in model classes list.")

                detections = detections[np.in1d(detections[:,5].astype(int), filter_class_idx)]
            
            return detections, {'width':w, 'height':h}
            # Run Pytorch model  
        else:
            processed_image = torch.from_numpy(processed_image).to(self.device)
            # Change image floating point precision if fp16 set to true
            processed_image = processed_image.half() if self.fp16 else processed_image.float() 
            pred = self.model(processed_image, augment=False, visualize=False)[0]
            
        # Post Processing
        if isinstance(pred, np.ndarray):
            pred = torch.tensor(pred, device=self.device)
        # print(pred)
        predictions = non_max_suppression(pred, conf_thres, 
                                          iou_thres, 
                                          agnostic=agnostic_nms, 
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

        if filter_classes:
            class_names = get_names()

            filter_class_idx = []
            if filter_classes:
                for _class in filter_classes:
                    if _class.lower() in class_names:
                        filter_class_idx.append(class_names.index(_class.lower()))
                    else:
                        warnings.warn(f"class {_class} not found in model classes list.")

            detections = detections[np.in1d(detections[:,5].astype(int), filter_class_idx)]

        if return_image:
            return detections, original_image
        else: 
            return detections, image_info

 