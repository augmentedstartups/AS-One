from norfair import Detection, Tracker
import numpy as np


class NorFair:
    def __init__(self, detector, max_distance_between_points=30) -> None:

        self.tracker = Tracker(
            distance_function=self._euclidean_distance,
            distance_threshold=max_distance_between_points,
        )
        self.detector = detector
        try:
            self.input_shape = tuple(detector.model.get_inputs()[0].shape[2:])
        except AttributeError as e:
            self.input_shape = (640, 640)

    def _euclidean_distance(self, detection, tracked_object):
        return np.linalg.norm(detection.points - tracked_object.estimate)

    def detect_and_track(self, image: np.ndarray, config: dict) -> tuple:
                       
        _dets_xyxy, image_info = self.detector.detect(
            image, **config
            )

        class_ids = []
        ids = []
        bboxes_xyxy = []
        scores = []

        if isinstance(_dets_xyxy, np.ndarray) and len(_dets_xyxy) > 0:

            dets_xyxy = [
                Detection(
                    np.array([(box[2] + box[0])/2, (box[3] + box[1])/2]), data=box)
                for box in _dets_xyxy
                # if box[-1] == 2
            ]

            bboxes_xyxy, ids,  scores, class_ids = self._tracker_update(
                dets_xyxy, image_info)

        return bboxes_xyxy, ids,  scores, class_ids

    def _tracker_update(self, dets_xyxy: list, image_info: dict):

        bboxes_xyxy = []
        class_ids = []
        scores = []
        ids = []

        tracked_objects = self.tracker.update(detections=dets_xyxy)

        for obj in tracked_objects:
            det = obj.last_detection.data
            bboxes_xyxy.append(det[:4])
            class_ids.append(int(det[-1]))
            scores.append(int(det[-2]))
            ids.append(obj.id)
        return np.array(bboxes_xyxy), ids,  scores, class_ids
