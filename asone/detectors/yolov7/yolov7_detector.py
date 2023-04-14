import os
import sys
import onnxruntime
import torch
import coremltools as ct
from asone.utils import get_names
import numpy as np
import warnings
from asone.detectors.yolov7.yolov7.utils.yolov7_utils import (prepare_input,
                                 process_output,
                                 non_max_suppression)
from asone.detectors.yolov7.yolov7.models.experimental import attempt_load
from asone import utils
from PIL import Image
from asone.detectors.utils.coreml_utils import yolo_to_xyxy, generalize_output_format, scale_bboxes

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov7'))
class YOLOv7Detector:
    def __init__(self,
                 weights=None,
                 use_onnx=False,
                 mlmodel=False,
                 use_cuda=True):
        self.use_onnx = use_onnx
        self.mlmodel = mlmodel
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
        # Load coreml
        elif self.mlmodel:
            model = ct.models.MLModel(weights)
        #Load Pytorch
        else: 
            model = attempt_load(weights, map_location=self.device)
            model.half() if self.fp16 else model.float()
        return model


    def detect(self, image: list,
               input_shape: tuple = (640, 640),
               conf_thres: float = 0.25,
               iou_thres: float = 0.45,
               max_det: int = 1000,
               filter_classes: bool = None,
               agnostic_nms: bool = True,
               with_p6: bool = False,
               return_image=False) -> list:

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
        # Run Coreml model 
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
                                            agnostic=agnostic_nms)[0]
            
            detection = detection.detach().cpu().numpy()
            # detection = yolo_to_xyxy(detection, input_shape)
            # print(detection)
            
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

        if return_image:
            return detection, original_image
        else: 
            return detection, image_info
