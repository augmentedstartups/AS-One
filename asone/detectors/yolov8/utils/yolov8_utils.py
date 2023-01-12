import cv2
import numpy as np
from ultralytics.yolo.utils import ops
import torch

def prepare_input(image, input_shape):
    input_height, input_width = input_shape
    input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize input image
    input_img = cv2.resize(input_img, (input_width, input_height))
    # Scale input pixel values to 0 to 1
    input_img = input_img / 255.0
    input_img = input_img.transpose(2, 0, 1)
    input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

    return input_tensor


def process_output(detections, 
                   ori_shape, 
                   input_shape, 
                   conf_threshold, 
                   iou_threshold,
                   classes=None,
                   agnostic=False,
                   multi_label=False,
                   labels=(),
                   max_det=300,
                   nm=0,  # number of masks
                   ):
    if not isinstance(detections, torch.Tensor):    
        detections = torch.from_numpy(detections)
    detections = ops.non_max_suppression(detections,
                                          conf_thres=conf_threshold,
                                          iou_thres=iou_threshold,
                                          classes=classes,
                                          agnostic=agnostic,
                                          multi_label=multi_label,
                                          labels=labels,
                                          max_det=max_det,
                                          nm=nm,  # number of masks
                                        )

    for i in range(len(detections)):

        # convert tensor to numpy array
        detections[i] = detections[i].cpu().numpy()
        
        # Extract boxes from predictions
        detections[i][:, :4] = rescale_boxes(
            detections[i][:, :4], ori_shape, input_shape)
    
    return detections


def rescale_boxes(boxes, ori_shape, input_shape):

    input_height, input_width = input_shape
    img_height, img_width = ori_shape
    # Rescale boxes to original image dimensions
    input_shape = np.array(
        [input_width, input_height, input_width, input_height])
    boxes = np.divide(boxes, input_shape, dtype=np.float32)
    boxes *= np.array([img_width, img_height, img_width, img_height])
    return boxes
