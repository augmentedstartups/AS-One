from json import detect_encoding
import time
import cv2
import os
import numpy as np
import onnxruntime
from .yolov6_utils import prepare_input, process_output 


class YOLOv6Detector:
    def __init__(self, weights=
                       os.path.join
                       (os.path.dirname
                       (os.path.abspath(__file__)), './weights/yolov6t.onnx'),
                 use_cuda=True, use_onnx=True) -> None:

        if use_onnx:
            if use_cuda:
                providers = [
                            'CUDAExecutionProvider',
                            'CPUExecutionProvider'
                            ]
            else:
                providers = ['CPUExecutionProvider']
      
        self.model = onnxruntime.InferenceSession(weights,
                                                    providers = providers)
        # Get Model Input
        model_inputs = self.model.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        
        # Input shape
        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        
        # Get Model Output
        model_outputs = self.model.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    def detect(self, image: list,
               conf_thres: float = 0.3,
                input_shape=(640, 640),

               iou_thres: float = 0.5,) -> list:
        
        # Prepare Input
        img_height, img_width = image.shape[:2]
        input_tensor = prepare_input(image, self.input_width, self.input_height)
    
        # Perform Inference on the Image
        start = time.perf_counter()
        outputs = self.model.run(self.output_names, {self.input_names[0]: input_tensor})[0] 

        # Process Output
        boxes, scores, class_ids = process_output(outputs, img_height, img_width,
                                                 self.input_width, self.input_height,
                                                 conf_thres, iou_thres)
        det = []
        for box in range(len(boxes)):
            pred = np.append(boxes[box], scores[box])
            pred = np.append(pred, class_ids[box])
            det.append(pred)
  
        det = np.array(det)
        image_info = {
            'width': image.shape[1],
            'height': image.shape[0],
        }
        return det, image_info



if __name__ == '__main__':
    model_path = "/home/ajmair/benchmarking/yolov6_wrapper/yolov6n.onnx"
    # Initialize YOLOv6 object detector
    yolov6_detector = YOLOv6Detector(model_path)
    img = cv2.imread('/home/ajmair/benchmarking/yolov6_wrapper/persons.jpeg')
    # Detect Objects
    result =  yolov6_detector.detect(img)
    print(result)
 