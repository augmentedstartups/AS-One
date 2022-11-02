import argparse
from trackers import Tracker
import argparse
import asone
import utils
from detectors import Detector
import cv2
import os
from loguru import logger
import time
import copy

def main(args):
    detector = Detector(asone.YOLOV7_E6_ONNX, use_cuda=args.use_cuda).get_detector()
    tracker = Tracker(asone.BYTETRACK, detector, use_cuda=args.use_cuda).get_tracker()

    cap = cv2.VideoCapture(args.video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    output_dir = 'results'
    if args.save_results:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, os.path.basename(args.video_path))
        logger.info(f"video save path is {save_path}")

        video_writer = cv2.VideoWriter(
            save_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (int(width), int(height)),
        )

    frame_id = 1
    tic = time.time()

    prevTime = 0

    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break
        im0 = copy.deepcopy(frame)

        bboxes_xyxy, ids, scores, class_ids = tracker.detect_and_track(
            frame)
        elapsed_time = time.time() - start_time

        logger.info(
            'frame {}/{} ({:.2f} ms)'.format(frame_id, int(frame_count),
                                             elapsed_time * 1000), )

        im0 = utils.draw_boxes(im0, bboxes_xyxy, identities=ids)

        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime
        cv2.line(im0, (20, 25), (127, 25), [85, 45, 255], 30)
        cv2.putText(im0, f'FPS: {int(fps)}', (11, 35), 0, 1, [
                    225, 255, 255], thickness=2, lineType=cv2.LINE_AA)

        if args.display:
            cv2.imshow(' Sample', im0)
        if args.save_results:
            video_writer.write(im0)

        frame_id += 1

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    tac = time.time()
    print(f'Total Time Taken: {tac - tic:.2f}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('video_path', help='Path to input video')
    parser.add_argument('--cpu', default=True,
                        action='store_false', dest='use_cuda', help='run on cpu')
    parser.add_argument('--display', default=False,
                        action='store_true', dest='display', help='Display Results')
    parser.add_argument('--save', default=False,
                        action='store_true', dest='save_results', help='Save Results')

    args = parser.parse_args()

    main(args)
