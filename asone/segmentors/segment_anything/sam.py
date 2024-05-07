import numpy as np
import cv2
import torch

from segment_anything import sam_model_registry, SamPredictor
from asone.utils.utils import PathResolver


class SamSegmentor:
    def __init__(self,
                 weights: str=None,
                 use_cuda: bool = True):
        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        with PathResolver():
            self.model = self.load_models(weights)
        
    def load_models(self, ckpt: str) -> None:
        sam = sam_model_registry["vit_h"](checkpoint=ckpt).to(device=self.device)
        model = SamPredictor(sam)
        
        return model
    
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
        
        input_boxes = torch.from_numpy(bbox_xyxy).to(self.device)
        transformed_boxes = self.model.transform.apply_boxes_torch(input_boxes, image.shape[:2])
        
        masks, _, _ = self.model.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        
        # result_image = self.draw_masks_fromList(image, masks.cpu(), bbox_xyxy)
        
        return masks.cpu()
