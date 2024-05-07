import sys
import argparse
import asone
from asone import ASOne
import torch


def main(args):
    filter_classes = args.filter_classes

    if filter_classes:
        filter_classes = ['person']
    # Check if cuda available
    if args.use_cuda and torch.cuda.is_available():
        args.use_cuda = True
    else:
        args.use_cuda = False
     
    if sys.platform.startswith('darwin'):
        detector = asone.YOLOV7_MLMODEL 
    else:
        detector = asone.YOLOV7_PYTORCH
    
    detect = ASOne(
        detector=detector,
        weights=args.weights,
        use_cuda=args.use_cuda
        )
    # Get tracking function
    track = detect.detect_video(args.video_path,
                                output_dir=args.output_dir,
                                conf_thres=args.conf_thres,
                                iou_thres=args.iou_thres,
                                display=args.display,
                                filter_classes=filter_classes,
                                class_names=None) # class_names=['License Plate'] for custom weights
    
    # Loop over track_fn to retrieve outputs of each frame 
    for bbox_details, frame_details in track:
        bbox_xyxy, ids, scores, class_ids = bbox_details
        frame, frame_num, fps = frame_details
        print(frame_num)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('video_path', help='Path to input video')
    parser.add_argument('--cpu', default=True, action='store_false', dest='use_cuda',
                        help='run on cpu if not provided the program will run on gpu.')
    parser.add_argument('--no_save', default=True, action='store_false',
                        dest='save_result', help='whether or not save results')
    parser.add_argument('--no_display', default=True, action='store_false',
                        dest='display', help='whether or not display results on screen')
    parser.add_argument('--output_dir', default='data/results',  help='Path to output directory')
    parser.add_argument('--draw_trails', action='store_true', default=False,
                        help='if provided object motion trails will be drawn.')
    parser.add_argument('--filter_classes', default=None, help='Filter class name')
    parser.add_argument('-w', '--weights', default=None, help='Path of trained weights')
    parser.add_argument('-ct', '--conf_thres', default=0.25, type=float, help='confidence score threshold')
    parser.add_argument('-it', '--iou_thres', default=0.45, type=float, help='iou score threshold')

    args = parser.parse_args()

    main(args)