import asone.asone
from .utils import draw_boxes
from .detectors import Detector
import cv2
import argparse

def main(args):
    img_path = args.image
    img = cv2.imread(img_path)
    detector = Detector(asone.YOLOV7_E6_ONNX, use_cuda=args.use_cuda).get_detector()
    dets, img_info = detector.detect(img)

    bbox_xyxy = dets[:, :4]
    scores = dets[:, 4]
    class_ids = dets[:, 5]

    img = draw_boxes(img, bbox_xyxy, class_ids=class_ids)
    cv2.imwrite('data/results/result.png', img)

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path of test image")
    parser.add_argument('--cpu', default=True, action='store_false', dest='use_cuda', help='If provided the model will run on cpu otherwise it will run on gpu')

    args = parser.parse_args()
    main(args)
