import os

from asone import utils
from asone.segmentors.utils.weights_path import get_weight_path
from asone.segmentors.segment_anything.sam import SamSegmentor


class Segmentor:
    def __init__(self, 
                 model_flag,
                 weights: str=None,
                 use_cuda: bool=True):
        
        if weights is None:
            weight = get_weight_path(model_flag)
        
        if not os.path.exists(weight):
            utils.download_weights(weight)
        
        self.model = self._select_segmentor(model_flag, weight, use_cuda)
    
    def _select_segmentor(self, model_flag, weights, use_cuda):
        if model_flag == 171:
            model = SamSegmentor(weights, use_cuda=use_cuda)
        return model
    
    def create_mask(self, bbox_xyxy, image):
        return self.model.create_mask(bbox_xyxy, image)
