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
            onnx = False
            weight = weights
        else:
            mlmodel, onnx, weight = get_weight_path(model_flag)

        if model_flag in range(143, 144):
            print('000')
            _segmentor = YOLOv5Segmentor(weights=weight,
                                       use_onnx=onnx,
                                       use_cuda=cuda)
        return _segmentor

    def get_detector(self):
        return self.model

    def detect(self,
               image: list,
               return_image=False,
               **kwargs: dict):
        return self.model.detect(image, return_image, **kwargs)


if __name__ == '__main__':
    # Initialize YOLOv6 object detector
    model_type = 144
    result = Segmentor(model_flag=model_type, use_cuda=True)
    img = cv2.imread('/home/hd/PycharmProjects/AS-One/data/sample_imgs/test2.jpg')
    pred = result.get_detector(img)
    print(pred)
