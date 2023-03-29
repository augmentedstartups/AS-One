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

    
    detector = ASOne(asone.YOLOV7_PYTORCH, weights=args.weights, use_cuda=args.use_cuda)

    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    FPS = cap.get(cv2.CAP_PROP_FPS)

    if args.save:
        video_writer = cv2.VideoWriter(
            os.path.basename(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            FPS,
            (int(width), int(height)),
        )
    
    frame_no = 1
    tic = time.time()

    prevTime = 0

    while True:
        start_time = time.time()

        ret, img = cap.read()
        if not ret:
            break
        frame = img.copy()
        
        dets, img_info = detector.detect(img, conf_thres=0.25, iou_thres=0.45)
        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime

        if dets is not None: 
            bbox_xyxy = dets[:, :4]
            scores = dets[:, 4]
            class_ids = dets[:, 5]
            img = draw_boxes(img, bbox_xyxy, class_ids=class_ids)

        cv2.line(img, (20, 25), (127, 25), [85, 45, 255], 30)
        cv2.putText(img, f'FPS: {int(fps)}', (11, 35), 0, 1, [
                    225, 255, 255], thickness=2, lineType=cv2.LINE_AA)


        frame_no+=1
        if args.display:
            cv2.imshow('Window', img)

        if args.save:
            video_writer.write(img)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

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