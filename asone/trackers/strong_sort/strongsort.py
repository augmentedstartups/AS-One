import numpy as np
from .tracker.strong_sort import StrongSORT


class StrongSort:
    def __init__(self, detector) -> None:
        
        self.tracker = StrongSORT(model_weights='osnet_x0_25_msmt17.pt', device='cpu')
        self.detector = detector
        try:
            self.input_shape = tuple(detector.model.get_inputs()[0].shape[2:])
        except AttributeError as e:
            self.input_shape = (640, 640)
        
    def detect_and_track(self, image: np.ndarray, config: dict) -> tuple:
           
        _dets_xyxy, img = self.detector.detect(
            image, **config, return_image=True
            )
        
        bbox_xyxy = _dets_xyxy[:, :4]
        conf = _dets_xyxy[:, 4]
        classes = _dets_xyxy[:, 5]
        
        # if isinstance(_dets_xyxy, np.ndarray) and len(_dets_xyxy) > 0:
        dets = self.tracker.update(bbox_xyxy, conf, classes, img)
        dets = np.array(dets)
    
        if dets != []:
            bbox_xyxy = dets[:, :4]
            ids = dets[:, 4]
            class_ids = dets[:, 5]
            scores = dets[:, 6]
            return bbox_xyxy, ids, scores, class_ids
        else:
            return [], [], [], []
