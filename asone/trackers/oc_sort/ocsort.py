from motpy import Detection, MultiObjectTracker
import numpy as np
from .tracker.ocsort import OCSort


class OcSort:
    def __init__(self, detector) -> None:

        self.tracker = OCSort(det_thresh=0.2)
        self.detector = detector
        try:
            self.input_shape = tuple(detector.model.get_inputs()[0].shape[2:])
        except AttributeError as e:
            self.input_shape = (640, 640)
        
    def detect_and_track(self, image: np.ndarray, config: dict) -> tuple:
                       
        _dets_xyxy, image_info = self.detector.detect(
            image, **config
            )
        image_info = [image_info['height'], image_info['width']]
        if isinstance(_dets_xyxy, np.ndarray) and len(_dets_xyxy) > 0:
            dets = self.tracker.update(_dets_xyxy, image_info)
            bbox_xyxy = dets[:, :4]
            ids = dets[:, 4]
            class_ids = dets[:, 5]
            scores = dets[:, 6]

            return bbox_xyxy, ids, scores, class_ids
        return [],[],[],[]
