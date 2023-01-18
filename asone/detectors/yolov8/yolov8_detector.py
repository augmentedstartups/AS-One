import os
from asone import utils
from asone.utils import get_names
import onnxruntime
import torch
from .utils.yolov8_utils import prepare_input, process_output
import numpy as np
import warnings
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.nn.tasks import DetectionModel, attempt_load_one_weight


class YOLOv8Detector:
    def __init__(self,
                 weights=None,
                 use_onnx=False,
                 use_cuda=True):

        self.use_onnx = use_onnx
        self.device = 'cuda' if use_cuda else 'cpu'

        # If incase weighst is a list of paths then select path at first index
        weights = str(weights[0] if isinstance(weights, list) else weights)

        if not os.path.exists(weights):
            utils.download_weights(weights)

        # Load Model
        self.model = self.load_model(use_cuda, weights)

    def load_model(self, use_cuda, weights, fp16=False):

        # Device: CUDA and if fp16=True only then half precision floating point works
        self.fp16 = fp16 & (
            (not self.use_onnx or self.use_onnx) and self.device != 'cpu')

        # Load onnx
        if self.use_onnx:
            if use_cuda:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']

            model = onnxruntime.InferenceSession(weights, providers=providers)
        # Load Pytorch
        else:
            model, ckpt = attempt_load_one_weight(weights)
            model = AutoBackend(model, fp16=False, dnn=False).to(self.device)
            model.half() if self.fp16 else model.float()
        return model

    def detect(self, image: list,
               input_shape: tuple = (640, 640),
               conf_thres: float = 0.25,
               iou_thres: float = 0.45,
               max_det: int = 1000,
               filter_classes: bool = None,
               agnostic_nms: bool = True,
               with_p6: bool = False
               ) -> list:

        # Preprocess input image and also copying original image for later use
        original_image = image.copy()
        processed_image = prepare_input(
            image, input_shape, 32, False if self.use_onnx else True)

        # Perform Inference on the Image
        if self.use_onnx:
            # Run ONNX model
            input_name = self.model.get_inputs()[0].name
            prediction = self.model.run([self.model.get_outputs()[0].name], {
                input_name: processed_image})[0]
            prediction = torch.from_numpy(prediction)
        # Run Pytorch model
        else:
            processed_image = torch.from_numpy(processed_image).to(self.device)
            # Change image floating point precision if fp16 set to true
            processed_image = processed_image.half() if self.fp16 else processed_image.float()

            with torch.no_grad():
                prediction = self.model(processed_image, augment=False)

        detection = []
        # Postprocess prediction
        detection = process_output(prediction,
                                   original_image.shape[:2],
                                   processed_image.shape[2:],
                                   conf_thres,
                                   iou_thres,
                                   agnostic=agnostic_nms,
                                   max_det=max_det)

        image_info = {
            'width': original_image.shape[1],
            'height': original_image.shape[0],
        }

        if filter_classes:
            class_names = get_names()

            filter_class_idx = []
            if filter_classes:
                for _class in filter_classes:
                    if _class.lower() in class_names:
                        filter_class_idx.append(
                            class_names.index(_class.lower()))
                    else:
                        warnings.warn(
                            f"class {_class} not found in model classes list.")

            detection = detection[np.in1d(
                detection[:, 5].astype(int), filter_class_idx)]

        return detection, image_info
