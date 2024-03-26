import os
import numpy as np
import cv2
import torch

from asone import utils
from asone.segmentors.utils.weights_path import get_weight_path
from segment_anything import sam_model_registry, SamPredictor
from asone.utils.utils import PathResolver


class Segmentor:
    def __init__(self, 
                 model_flag,
                 weights: str=None):
        
        if weights is None:
            weight = get_weight_path(model_flag)
        
        if not os.path.exists(weight):
            utils.download_weights(weight)
            
        with PathResolver():
            if model_flag == 171:
                self.load_models(weight)

    def load_models(self, ckpt: str) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry["vit_h"](checkpoint=ckpt).to(device=device)
        self.model = SamPredictor(sam)
        
        return self.model
    
    
    def draw_masks_fromList(self, image, masks_generated, labels, colors=[0, 255, 0]):
        masked_image = image.copy()
        for i in range(len(masks_generated)):
            mask = masks_generated[i].squeeze()  # Squeeze to remove singleton dimension
            color = np.asarray(colors, dtype='uint8')
            mask_color = np.expand_dims(mask, axis=-1) * color  # Apply color to the mask

            # Apply the mask to the image
            masked_image = np.where(mask_color > 0, mask_color, masked_image)

        masked_image = masked_image.astype(np.uint8)
        return cv2.addWeighted(image, 0.5, masked_image, 0.5, 0)
    
    
    def create_mask(self, bbox_xyxy, image):
        self.model.set_image(image)
        
        input_boxes = torch.from_numpy(bbox_xyxy).to(self.model.device)
        transformed_boxes = self.model.transform.apply_boxes_torch(input_boxes, image.shape[:2])
        
        masks, _, _ = self.model.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        
        result_image = self.draw_masks_fromList(image, masks.cpu(), bbox_xyxy)
        
        return result_image
