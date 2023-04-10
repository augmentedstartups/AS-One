import cv2
import numpy as np
from ultralytics.yolo.utils import ops
import torch
from ultralytics.yolo.data.augment import LetterBox

def prepare_input(image, input_shape, stride, pt):
    input_tensor = LetterBox(input_shape, auto=pt, stride=stride)(image=image)
    input_tensor = input_tensor.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    input_tensor = np.ascontiguousarray(input_tensor).astype(np.float32)  # contiguous
    input_tensor /= 255.0  # 0 - 255 to 0.0 - 1.0
    input_tensor = input_tensor[None].astype(np.float32)
    return input_tensor


def process_output(detections, 
                   ori_shape, 
                   input_shape, 
                   conf_threshold, 
                   iou_threshold,
                   classes=None,
                   mlmodel=False,
                   agnostic=False,
                   max_det=300,
                   ):
    detections = ops.non_max_suppression(detections,
                                          conf_thres=conf_threshold,
                                          iou_thres=iou_threshold,
                                          classes=classes,
                                          agnostic=agnostic,
                                          max_det=max_det,
                                          )

    if mlmodel:
        detection = detections[0].cpu().numpy()
        return detection

    for i in range(len(detections)): 
        # Extract boxes from predictions
        detections[i][:, :4] = ops.scale_boxes(input_shape, detections[i][:, :4], ori_shape).round()

    
    return detections[0].cpu().numpy()


def rescale_boxes(boxes, ori_shape, input_shape):

    input_height, input_width = input_shape
    img_height, img_width = ori_shape
    # Rescale boxes to original image dimensions
    input_shape = np.array(
        [input_width, input_height, input_width, input_height])
    boxes = np.divide(boxes, input_shape, dtype=np.float32)
    boxes *= np.array([img_width, img_height, img_width, img_height])
    return boxes
