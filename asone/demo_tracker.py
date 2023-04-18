import asone
from asone import ASOne
from .utils import draw_boxes
import cv2
import argparse
import time
import os


def main(args):
    filter_classes = args.filter_classes
    video_path = args.video

    os.makedirs(args.output_path, exist_ok=True)

    if filter_classes:
        filter_classes = filter_classes.split(',')


    detect = ASOne(tracker=asone.BYTETRACK, detector=asone.YOLOV7_PYTORCH, 
                   use_cuda=True)

    track = detect.track_video(video_path, output_dir=args.output_path, 
                               save_result=args.save, display=args.display,
                               filter_classes=filter_classes)
 
    for bbox_details, frame_details in track:
        bbox_xyxy, ids, scores, class_ids = bbox_details
        frame, frame_num, fps = frame_details


if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Path of video")
    parser.add_argument('--cpu', default=True, action='store_false', dest='use_cuda', help='If provided the model will run on cpu otherwise it will run on gpu')
    parser.add_argument('--filter_classes', default=None, help='Class names seperated by comma (,). e.g. person,car ')
    parser.add_argument('-w', '--weights', default=None, help='Path of trained weights')
    parser.add_argument('-o', '--output_path', default='data/results', help='path of output file')
    parser.add_argument('--no_display', action='store_false', default=True, dest='display', help='if provided video will not be displayed')
    parser.add_argument('--no_save', action='store_false', default=True, dest='save', help='if provided video will not be saved')

    args = parser.parse_args()
    main(args)