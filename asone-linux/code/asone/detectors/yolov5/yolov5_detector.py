from .yolov5_utils import (
    non_max_suppression, scale_coords, letterbox)

import numpy as np
import torch
import onnxruntime
import os

class YOLOv5Detector:
    def __init__(self,
                 weights=os.path.join(os.path.dirname(
                     os.path.abspath(__file__)), './weights/yolov5s.onnx'),
                 use_cuda=True, use_onnx=False) -> None:

        if use_onnx:
            if use_cuda:
                providers = [
                    'CUDAExecutionProvider',
                    'CPUExecutionProvider'
                ]
            else:
                providers = ['CPUExecutionProvider']

        self.model = onnxruntime.InferenceSession(weights, providers=providers)

        # else:
        #     self.model = torch
        self.device = 'cuda' if use_cuda else 'cpu'

    def detect(self, image: list,
               conf_thres: float = 0.25,
               iou_thres: float = 0.45,
               classes: int = None,
               agnostic_nms: bool = False,
               input_shape=(640, 640),
               max_det: int = 1000) -> list:

        image0 = image.copy()
        image = letterbox(image, input_shape, stride=32, auto=False)[0]
        image = image.transpose((2, 0, 1))[::-1]
        image = np.ascontiguousarray(image, dtype=np.float32)
        image = image
        image /= 255
        if len(image.shape) == 3:
            image = image[None]  # expand for batch dim

        input_name = self.model.get_inputs()[0].name
        pred = self.model.run([self.model.get_outputs()[0].name],
                              {input_name: image})[0]
        if isinstance(pred, np.ndarray):
            pred = torch.tensor(pred, device=self.device)
        pred = non_max_suppression(
            pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        for i, det in enumerate(pred):  # per image
            if len(det):
                det[:, :4] = scale_coords(
                    image.shape[2:], det[:, :4], image0.shape).round()
                pred[i] = det
        dets = pred[0].cpu().numpy()
        image_info = {
            'width': image0.shape[1],
            'height': image0.shape[0],
        }
        return dets, image_info
