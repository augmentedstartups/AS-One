from motpy import Detection, MultiObjectTracker
import numpy as np


class Motpy:
    def __init__(self, detector, max_distance_between_points=30) -> None:

        self.tracker = MultiObjectTracker(dt=0.1)
        self.detector = detector
        try:
            self.input_shape = tuple(detector.model.get_inputs()[0].shape[2:])
        except AttributeError as e:
            self.input_shape = (640, 640)

    def detect_and_track(self, image: np.ndarray, config: dict) -> tuple:
                       
        _dets_xyxy, image_info = self.detector.detect(
            image, **config
            )
        class_ids = []
        ids = []
        bboxes_xyxy = []
        scores = []
        
        if isinstance(_dets_xyxy, np.ndarray) and len(_dets_xyxy) > 0:
            
            self.tracker.step(detections=[
                Detection(
                    box=box[:4],
                    score= box[4],
                    class_id=box[5]
                    )
                for box in _dets_xyxy
                ])
            
            
            bboxes_xyxy, ids,  scores, class_ids = self._tracker_update()
        return bboxes_xyxy, ids, scores, class_ids

    def _tracker_update(self):

        bboxes_xyxy = []
        class_ids = []
        scores = []
        ids = []

        tracked_objects = self.tracker.active_tracks()
        for obj in tracked_objects:
            bboxes_xyxy.append(obj[1:2][0].tolist())
            class_ids.append(obj[3])
            scores.append(obj[2])
            ids.append(1)
        return np.array(bboxes_xyxy), ids,  scores, class_ids
