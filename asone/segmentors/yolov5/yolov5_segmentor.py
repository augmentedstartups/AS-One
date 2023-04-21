import os
from asone.utils import get_names
import numpy as np
import warnings
import torch
import onnxruntime

from asone.segmentors.yolov5.yolov5.utils.yolov5_utils import (non_max_suppression,
                                                              scale_coords,
                                                              letterbox,
                                                               process_mask,
                                                               Annotator, colors, scale_image)
from asone.segmentors.yolov5.yolov5.models.experimental import attempt_load
from asone import utils


class YOLOv5Segmentor:
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
        else: 
            model = attempt_load(weights, device=self.device, inplace=True, fuse=True)
            model.half() if self.fp16 else model.float()
        return model

    def image_preprocessing(self,
                            image: list,
                            input_shape=(640, 640))-> list:

        original_image = image.copy()
        image = letterbox(image, input_shape, stride=32, auto=True)[0]
        image = image.transpose((2, 0, 1))[::-1]
        image = np.ascontiguousarray(image, dtype=np.float32)
        image /= 255  # 0 - 255 to 0.0 - 1.0
        if len(image.shape) == 3:
            image = image[None]  # expand for batch dim  
        return original_image, image
    
    def segment(self, image: list,
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
            pass
            # Run Pytorch model  
        else:
            processed_image = torch.from_numpy(processed_image).to(self.device)
            # Change image floating point precision if fp16 set to true
            processed_image = processed_image.half() if self.fp16 else processed_image.float()

            pred, proto = self.model(processed_image, augment=False, visualize=False)[:2]

        # Post Processing
        if isinstance(pred, np.ndarray):
            pred = torch.tensor(pred, device=self.device)

        predictions = non_max_suppression(prediction=pred, conf_thres=conf_thres,
                                          iou_thres=iou_thres,
                                          agnostic=agnostic_nms, 
                                          max_det=max_det, nm=32)

        for i, prediction in enumerate(predictions):  # per image
            if len(prediction):

                predictions[i] = prediction
                proto[i] = proto
               # HWC
                prediction[:, :4] = scale_coords(
                    processed_image.shape[2:], prediction[:, :4], original_image.shape).round()
                masks = process_mask(proto[i], prediction[:, 6:], prediction[:, :4], original_image.shape[:2],
                                     upsample=True)

        detections = predictions[0].cpu().numpy()
        masks = masks.cpu().numpy()
        image_info = {
            'width': original_image.shape[1],
            'height': original_image.shape[0],
        }

        self.boxes = detections[:, :4]
        self.scores = detections[:, 4:5]
        self.class_ids = detections[:, 5:6]
        self.masks = masks

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
            return detections, masks, processed_image.cpu().numpy()
        else: 
            return detections, masks, image_info


if __name__ == '__main__':
    # Initialize YOLOv6 object detector
    from asone.utils.draw import draw_detections_and_masks
    import cv2
    model_type = 144
    weights = "AS-One/data/custom_weights/yolov5s-seg.pt"

    result = YOLOv5Segmentor(weights=weights, use_cuda=False, use_onnx=False)
    img = cv2.imread('AS-One/data/sample_imgs/test2.jpg')

    dets, masks, image_info = result.segment(image=img, return_image=False)

    img = draw_detections_and_masks(img, dets[:, :4], dets[:, 5:6], dets[:, 4:5], mask_alpha=0.8, mask_maps=masks)

    cv2.imshow("Result", img)
    cv2.waitKey(0)




 