from .tracker import build_tracker
import numpy as np
import os
from asone import utils


class DeepSort:
    def __init__(self, detector, weights=None, use_cuda=True):

        if weights is None:
            weights = os.path.join(os.path.dirname(
                os.path.abspath(__file__)), "tracker/deep/checkpoint/ckpt.t7")

        if not os.path.exists(weights):
            utils.download_weights(weights)

        cfg = {
            'MAX_DIST': 0.2,
            'MIN_CONFIDENCE': 0.3,
            'NMS_MAX_OVERLAP': 0.5,
            'MAX_IOU_DISTANCE': 0.7,
            'MAX_AGE': 70,
            'N_INIT': 3,
            'NN_BUDGET': 100
        }

        self.tracker = build_tracker(weights, cfg, use_cuda=use_cuda)
        self.detector = detector
        try:
            self.input_shape = tuple(detector.model.get_inputs()[0].shape[2:])
        except AttributeError as e:
            self.input_shape = (640, 640)
            
    def detect_and_track(self, image: np.ndarray, config: dict) -> tuple:
                       
        dets_xyxy, image_info = self.detector.detect(
            image, **config
            )

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
            dets_xywh = np.array([np.array(utils.xyxy_to_xywh(det))
                                 for det in dets_xyxy[:, :4]])

            outputs = self.tracker.update(
                dets_xywh, dets_xyxy[:, -2].tolist(), dets_xyxy[:, -1].tolist(), image_info['im0'])

            if len(outputs) > 0:
                bbox_xyxy = outputs[:, :4]
                ids = outputs[:, -2]
                object_id = outputs[:, -1]

        return bbox_xyxy, ids, object_id
