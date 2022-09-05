from .tracker import build_tracker
import numpy as np
import os
from asone import utils

class DeepSort:
    def __init__(self, detector, cfg=os.path.join(os.path.dirname(
            os.path.abspath(__file__)), 'tracker/configs/deep_sort.yaml'), use_cuda=True):

        self.tracker = build_tracker(cfg, use_cuda=use_cuda)
        self.detector = detector
        self.input_shape = tuple(detector.model.get_inputs()[0].shape[2:])

    def detect_and_track(self, image):

        dets_xyxy, image_info = self.detector.detect(
            image, input_shape=self.input_shape)
        image_info['im0'] = image

        class_ids = []
        ids = []
        bboxes_xyxy = []
        scores = []

        if isinstance(dets_xyxy, np.ndarray) and len(dets_xyxy) > 0:
            class_ids = dets_xyxy[:, -1].tolist()
            bboxes_xyxy, ids, class_ids = self._tracker_update(
                dets_xyxy,
                image_info,
            )

        return bboxes_xyxy, ids, [], class_ids

    def _tracker_update(self, dets_xyxy: np.ndarray, image_info: dict):

        bbox_xyxy = []
        ids = []
        object_id = []

        if dets_xyxy is not None:
            dets_xywh = np.array([np.array(utils.xyxy_to_xywh(det)) for det in dets_xyxy[:, :4]])

            outputs = self.tracker.update(
                dets_xywh, dets_xyxy[:, -2].tolist(), dets_xyxy[:, -1].tolist(), image_info['im0'])

            if len(outputs) > 0:
                bbox_xyxy = outputs[:, :4]
                ids = outputs[:, -2]
                object_id = outputs[:, -1]

        return bbox_xyxy, ids, object_id
