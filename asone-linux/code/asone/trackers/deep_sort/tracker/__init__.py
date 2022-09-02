from .deep_sort import DeepSORT
from .utils.parser import get_config
import os

__all__ = ['DeepSORT', 'build_tracker']


def build_tracker(cfg_deep, use_cuda=True):
    cfg = get_config()
    cfg.merge_from_file(cfg_deep)

    return DeepSORT(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), cfg.DEEPSORT.REID_CKPT),
        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET, use_cuda=use_cuda)
