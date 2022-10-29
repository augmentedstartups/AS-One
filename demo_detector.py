import asone
from asone import utils
from asone.detectors import Detector
import cv2
import argparse

def main(args):
    img = cv2.imread('sample_imgs/test2.jpg')
    detector = Detector(asone.YOLOV7_E6_ONNX, use_cuda=args.use_cuda).get_detector()
    dets, img_info = detector.detect(img)

    bbox_xyxy = dets[:, :4]
    scores = dets[:, 4]
    class_ids = dets[:, 5]

    img = utils.draw_boxes(img, bbox_xyxy, class_ids=class_ids)
    cv2.imwrite('result.png', img)

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', default=True, action='store_false', dest='use_cuda', help='If provided the model will run on cpu otherwise it will run on gpu')

    args = parser.parse_args()
    main(args)
