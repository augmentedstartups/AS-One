import asone
from asone import ASOne
from .utils import draw_boxes, draw_text
import cv2
import argparse
import time
import os


def main(args):
    
    image_path = args.image
    detector = ASOne(asone.CRAFT, recognizer=asone.EASYOCR, use_cuda=args.use_cuda)
    img = cv2.imread(image_path)      
    results = detector.detect_text(img)
    img = draw_text(img, results)
    if args.display:
        cv2.imshow('Window', img)

    if args.save:
        cv2.imwrite("results.jpeg", img)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        return

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path of image")
    parser.add_argument('--cpu', default=True, action='store_false', dest='use_cuda', help='If provided the model will run on cpu otherwise it will run on gpu')
    parser.add_argument('--no_display', action='store_false', default=True, dest='display', help='if provided video will not be displayed')
    parser.add_argument('--no_save', action='store_false', default=True, dest='save', help='if provided video will not be saved')
    args = parser.parse_args()
    main(args)
