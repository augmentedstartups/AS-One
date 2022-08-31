import norfair
from norfair import Detection, Tracker
import numpy as np


class NorFair:
    def __init__(self, detector, max_distance_between_points=30) -> None:
        
        self.tracker = Tracker(
            distance_function=self.euclidean_distance,
            distance_threshold=max_distance_between_points,
        )
        self.detector = detector


    def euclidean_distance(self, detection, tracked_object):
        return np.linalg.norm(detection.points - tracked_object.estimate)

    def inference(self, image):
        _detections, image_info = self.detector.detect(image)
        detections = [
            Detection(np.array([(box[2] + box[0])/2, (box[3] + box[1])/2]), data=box)
            for box in _detections
            # if box[-1] == 2
        ]

        tracked_objects = self.tracker.update(detections=detections)
        norfair.draw_points(image, detections)
        norfair.draw_tracked_objects(image, tracked_objects)
        return image
