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


class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17",
               "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34",
               "35", "36", "37", "38", "39", "40", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51",
               "52", "53", "54", "55", "56", "57", "58", "59", "60", "61", "62", "63","64", "65", "66", "67", "68",
               "69", "70", "71", "72", "73", "74", "75", "76", "77", "78", "79"]


class YOLOnasDetector:
    def __init__(self,
                 weights=None,
                 cfg=None,
                 use_onnx=True,
                 use_cuda=True,
                 ):
        
        if not os.path.exists(weights):
            utils.download_weights(weights)
            
        self.device = 'cuda' if use_cuda and torch.cuda.is_available() else 'cpu'
        self.use_onnx = use_onnx

        # Load Model
        self.model = self.load_model(weights=weights)

    def load_model(self, weights):
        model_name = os.path.basename(weights)
        name, file_extension = os.path.splitext(model_name)
        
        model = super_gradients.training.models.get(name, checkpoint_path=weights, checkpoint_num_classes=80, num_classes=80).to(self.device)
        
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

        
        self.model.set_dataset_processing_params( class_names=class_names,
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
            # Input names of ONNX model on which it is exported
            # input_name = self.model.get_inputs()[0].name
            # # Run onnx model
            # pred = self.model.run([self.model.get_outputs()[0].name], {
            #                       input_name: processed_image})[0]
            # Run Pytorch model
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
        
