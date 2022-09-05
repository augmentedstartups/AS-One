
import numpy as np
import torch
import onnxruntime
import cv2
import os

from yolox_utils import preprocess, COCO_CLASSES, multiclass_nms, demo_postprocess, vis

class YOLOxDetector:
    def __init__(self,
                 weights=os.path.join(os.path.dirname(
                     os.path.abspath(__file__)), './weights/yolor_csp-640-640.onnx'),
                 use_cuda=True, use_onnx=True) -> None:

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
               with_p6 = False,
               classes: int = None,
               agnostic_nms: bool = False,
               input_shape=(640, 640),
               max_det: int = 1000) -> list:

        self.input_shape = input_shape
        # Image Preprocess
        img, ratio = preprocess(image, self.input_shape)
        # Model Input and Output
        model_inputs = {self.model.get_inputs()[0].name: img[None, :, :, :]}
        model_output = self.model.run(None, model_inputs)
        # Prediction Post Process
        predictions = demo_postprocess(model_output[0], self.input_shape, p6=with_p6)[0]

        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]

        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
        boxes_xyxy /= ratio
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=iou_thres, score_thr=conf_thres)
        if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
            origin_img = vis(image, final_boxes, final_scores, final_cls_inds,
                            conf=conf_thres, class_names=COCO_CLASSES)
        
        image_info = {
            'width': image.shape[1],
            'height': image.shape[0],
        }
    
        return dets, image_info

if __name__ == '__main__':
    model_path = "/home/ajmair/benchmarking/asone/asone-linux/code/asone/detectors/yolox/weights/yolox_m.onnx"
    # Initialize YOLOvx object detector
    yolox_detector = YOLOxDetector(model_path)
    img = cv2.imread('/home/ajmair/benchmarking/asone/asone-linux/test.jpeg')
    # Detect Objects
    result =  yolox_detector.detect(img)
    print(result)
    # cv2.imwrite("myoutput.jpg", result)
 

