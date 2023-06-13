import os
from asone.utils import get_names
import numpy as np
import warnings
import torch
import onnxruntime
from asone import utils
import super_gradients
import numpy as np
from super_gradients.training.processing import DetectionCenterPadding, StandardizeImage, NormalizeImage, ImagePermute, ComposeProcessing, DetectionLongestMaxSizeRescale
from super_gradients.training import models
from super_gradients.common.object_names import Models


class_names = [""]


class YOLOnasDetector:
    def __init__(self,
                 model_flag,
                 weights=None,
                 cfg=None,
                 use_onnx=True,
                 use_cuda=True,
                #  checkpoint_num_classes=80,
                 num_classes=80
                 ):
        
        
        self.model_flag = model_flag
        # self.checkpoint_num_classes = checkpoint_num_classes
        if not os.path.exists(weights):
            utils.download_weights(weights)
        
        self.num_classes = num_classes
        self.device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
        self.use_onnx = use_onnx

        # Load Model
        self.model = self.load_model(weights=weights)

    def load_model(self, weights):
        
            # model = super_gradients.training.models.get(name, 
            #             checkpoint_path=weights, 
            #             checkpoint_num_classes=self.checkpoint_num_classes,
            #             num_classes=self.num_classes).to(self.device)
    
        if self.model_flag == 160: 
            model = models.get(Models.YOLO_NAS_S,
                    checkpoint_path=weights,
                    num_classes=self.num_classes).to(self.device)
        elif self.model_flag == 161:
            model = models.get(Models.YOLO_NAS_M,
                    checkpoint_path=weights,
                    num_classes=self.num_classes).to(self.device)
        elif self.model_flag == 162:
            model = models.get(Models.YOLO_NAS_L,
                    checkpoint_path=weights,
                    num_classes=self.num_classes).to(self.device)
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

        if self.num_classes==80:
            self.model.set_dataset_processing_params(class_names=class_names,
            image_processor=ComposeProcessing(
                    [
                        DetectionLongestMaxSizeRescale(output_shape=(636, 636)),
                        DetectionCenterPadding(output_shape=(640, 640), pad_value=114),
                        StandardizeImage(max_value=255.0),
                        ImagePermute(permutation=(2, 0, 1)),
                    ]
                ),
            iou=iou_thres,conf=conf_thres,
            )
        original_image = image
        # Inference
        if self.use_onnx:
            pass
            
        else:

            detections = self.model.predict(image)
            image_info = {
                'width': original_image.shape[1],
                'height': original_image.shape[0],
            }
            detections = list(detections)    
            pred = detections[0].prediction
            bboxes_xyxy = pred.bboxes_xyxy
            confidence = pred.confidence
            labels = pred.labels

            confidence = confidence.reshape(-1,1)
            labels = labels.reshape(-1,1)
            arr = np.append(bboxes_xyxy, confidence, axis=1)
            predictions = np.append(arr, labels, axis=1)
       

        if return_image:
            return predictions, original_image
        else: 
            return predictions, image_info
        
