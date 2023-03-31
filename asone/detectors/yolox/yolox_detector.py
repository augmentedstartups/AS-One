
import os
from asone.utils import get_names
import numpy as np
import warnings

import torch
import onnxruntime

from asone import utils
from asone.detectors.yolox.yolox.utils import fuse_model, postprocess
from asone.detectors.yolox.yolox.exp import get_exp
from asone.detectors.yolox.yolox_utils import preprocess, multiclass_nms, demo_postprocess


class YOLOxDetector:
    def __init__(self,
                 model_name=None,
                 exp_file=None,
                 weights=None,
                 use_onnx=False,
                 use_cuda=False
                 ):

        self.use_onnx = use_onnx
        self.device = 'cuda' if use_cuda else 'cpu'

        if not os.path.exists(weights):
            utils.download_weights(weights)

        self.weights_name = os.path.basename(weights)

        if model_name is None:
            model_name = 'yolox-s'

        if exp_file is None:
            exp_file = os.path.join("exps", "default", "yolox_s.py")
        # Load Model
        if self.use_onnx:
            self.model = self.load_onnx_model(use_cuda, weights)
        else:
            self.model = self.load_torch_model(weights, exp_file, model_name)

    def load_onnx_model(self, use_cuda, weights):
        # Load onnx
        if use_cuda:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        model = onnxruntime.InferenceSession(weights, providers=providers)
        return model

    def load_torch_model(self, weights,
                         exp_file, model_name,
                         fp16=True, fuse=False):
        # Device: CUDA and if fp16=True only then half precision floating point works
        self.fp16 = bool(fp16) & (
            (not self.use_onnx or self.use_onnx) and self.device != 'cpu')
        exp = get_exp(exp_file, model_name)

        ckpt = torch.load(weights, map_location="cpu")

        # get number of classes from weights
        # head.cls_preds.0.weight weights contains number of classes so simply extract it and with in exp file.
        exp.num_classes = ckpt['model']['head.cls_preds.0.weight'].size()[0]
        self.classes = exp.num_classes
        model = exp.get_model()
        if self.device == "cuda":
            model.cuda()
            if self.fp16:  # to FP16
                model.half()
        model.eval()

        # load the model state dict
        model.load_state_dict(ckpt["model"])
        if fuse:
            model = fuse_model(model)
        return model

    def detect(self,
               image: list,
               input_shape: tuple = (640, 640),
               conf_thres: float = 0.25,
               iou_thres: float = 0.45,
               max_det: int = 1000,
               filter_classes: bool = None,
               agnostic_nms: bool = True,
               with_p6: bool = False,
               return_image=False
               ) -> list:

        if self.weights_name in ['yolox_tiny.onnx', 'yolox_nano.onnx']:
            input_shape = (416, 416)

        self.input_shape = input_shape

        # Image Preprocess for onnx models
        if self.use_onnx:
            processed_image, ratio = preprocess(image, self.input_shape)
        else:
            processed_image, ratio = preprocess(image, self.input_shape)
            processed_image = torch.from_numpy(processed_image).unsqueeze(0)
            processed_image = processed_image.float()
            if self.device == "cuda":
                processed_image = processed_image.cuda()
                if self.fp16:
                    processed_image = processed_image.half()

        detection = []
        # Inference
        if self.use_onnx:  # Run ONNX model
            # Model Input and Output
            model_inputs = {self.model.get_inputs(
            )[0].name: processed_image[None, :, :, :]}
            detection = self.model.run(None, model_inputs)[0]
            # Postprrocessing
            detection = demo_postprocess(
                detection, self.input_shape, p6=with_p6)[0]
            boxes = detection[:, :4]
            scores = detection[:, 4:5] * detection[:, 5:]
            boxes_xyxy = np.ones_like(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
            boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
            boxes_xyxy /= ratio
            detection = multiclass_nms(
                boxes_xyxy, scores, nms_thr=iou_thres, score_thr=conf_thres)

        # Run Pytorch model
        else:
            with torch.no_grad():
                prediction = self.model(processed_image)
                prediction = postprocess(prediction,
                                         self.classes,
                                         conf_thres,
                                         iou_thres,
                                         class_agnostic=agnostic_nms
                                         )[0]
                if prediction is not None:
                    prediction = prediction.detach().cpu().numpy()
                    bboxes = prediction[:, 0:4]
                    # Postprocessing
                    bboxes /= ratio
                    cls = prediction[:, 6]
                    scores = prediction[:, 4] * prediction[:, 5]
                    for box in range(len(bboxes)):
                        pred = np.append(bboxes[box], scores[box])
                        pred = np.append(pred, cls[box])
                        detection.append(pred)
                    detection = np.array(detection)
                else:
                    detection = prediction

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

        image_info = {
            'width': image.shape[1],
            'height': image.shape[0],
        }
       
        if return_image:
            return detection, image
        else: 
            return detection, image_info
        
