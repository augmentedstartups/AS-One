import cv2

from asone.segmentors.yolov5 import YOLOv5Segmentor
from asone.detectors.utils.weights_path import get_weight_path


class Segmentor:
    def __init__(self,
                 model_flag: int,
                 weights: str = None,
                 use_cuda: bool = True,
                 recognizer: int = None):

        self.model = self._select_segmentor(model_flag, weights, use_cuda, recognizer)

    def _select_segmentor(self, model_flag, weights, cuda, recognizer):
        # Get required weight using model_flag
        # global _segmentor
        mlmodel = False
        if weights and weights.split('.')[-1] == 'onnx':
            onnx = True
            weight = weights
        elif weights and weights.split('.')[-1] == 'mlmodel':
            onnx = False
            weight = weights
            mlmodel = True
        elif weights:
            print("-1")
            onnx = False
            weight = weights
        else:
            mlmodel, onnx, weight = get_weight_path(model_flag)
        print(model_flag)
        if model_flag in range(143, 145):
            print('000')
            _segmentor = YOLOv5Segmentor(weights=weight,
                                       use_onnx=onnx,
                                       use_cuda=cuda)
        return _segmentor

    def get_segmentor(self):
        return self.model

    def segment(self,
               image: list,
               return_image=False,
               **kwargs: dict):
        return self.model.detect(image, return_image, **kwargs)


if __name__ == '__main__':
    # Initialize YOLOv5 instance segmentor
    from asone.utils.draw import draw_detections_and_masks
    model_type = 144
    weights = "AS-One/data/custom_weights/yolov5s-seg.pt"
    result = Segmentor(model_flag=model_type, weights=weights, use_cuda=False)
    model = result.get_segmentor()
    img = cv2.imread('AS-One/data/sample_imgs/test2.jpg')
    dets, masks, image_info = model.segment(image=img, return_image=False)

    img = draw_detections_and_masks(img, dets[:, :4], dets[:, 5:6], dets[:, 4:5], mask_alpha=0.8, mask_maps=masks)

    cv2.imshow("Result", img)
    cv2.waitKey(0)
