import asone
from .utils import draw_boxes
from .detectors import Detector
import cv2
import argparse

def main(args):
    filter_classes = args.filter_classes

    if filter_classes:
        filter_classes = filter_classes.split(',')

    img_path = args.image
    img = cv2.imread(img_path)
    detector = Detector(asone.YOLOV7_PYTORCH, weights=args.weights, use_cuda=args.use_cuda)
    dets, img_info = detector.detect(img, filter_classes=filter_classes)

    bbox_xyxy = dets[:, :4]
    scores = dets[:, 4]
    class_ids = dets[:, 5]

    img = draw_boxes(img, bbox_xyxy, class_ids=class_ids, class_names=None) # class_names=['License Plate'] for custom_trained_model
    cv2.imwrite('data/results/result.png', img)

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path of test image")
    parser.add_argument('--cpu', default=True, action='store_false', dest='use_cuda', help='If provided the model will run on cpu otherwise it will run on gpu')
    parser.add_argument('--filter_classes', default=None, help='Class names seperated by comma (,). e.g. person,car ')
    parser.add_argument('-w', '--weights', default=None, help='Path of trained weights')

    args = parser.parse_args()
    main(args)
