from .tracker.byte_tracker import BYTETracker
import numpy as np
from asone import utils


class ByteTrack(object):
    def __init__(self, detector, min_box_area: int = 10) -> None:

        self.min_box_area = min_box_area
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        self.detector = detector
        try:
            self.input_shape = tuple(detector.model.get_inputs()[0].shape[2:])
        except AttributeError as e:
            self.input_shape = (640, 640)

        self.tracker = BYTETracker(frame_rate=30)

    def detect_and_track(self, image: np.ndarray, conf_thres: float = 0.25, filter_classes:list = None):

        dets_xyxy, image_info = self.detector.detect(
            image, input_shape=self.input_shape,
            conf_thres = conf_thres,
            filter_classes=filter_classes)

        class_ids = []
        ids = []
        bboxes_xyxy = []
        scores = []

        if isinstance(dets_xyxy, np.ndarray) and len(dets_xyxy) > 0:
            class_ids = [int(i) for i in dets_xyxy[:, -1].tolist()]
            bboxes_xyxy, ids, scores = self._tracker_update(
                dets_xyxy,
                image_info,
            )
        return bboxes_xyxy, ids, scores, class_ids

    def _tracker_update(self, dets: np.ndarray, image_info: dict):
        online_targets = []
        if dets is not None:
            online_targets = self.tracker.update(
                dets[:, :-1],
                [image_info['height'], image_info['width']],
                [image_info['height'], image_info['width']],
            )

        online_xyxys = []
        online_ids = []
        online_scores = []
        for online_target in online_targets:
            tlwh = online_target.tlwh
            track_id = online_target.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > self.min_box_area and not vertical:
                online_xyxys.append(utils.tlwh_to_xyxy(tlwh))
                online_ids.append(track_id)
                online_scores.append(online_target.score)

        return online_xyxys, online_ids, online_scores
