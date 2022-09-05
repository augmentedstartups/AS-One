import argparse
from asone import ASOne
import argparse


def main(args):
    asone = ASOne(tracker=args.tracker, detector=args.detector, use_cuda=args.use_cuda)
    asone.start_tracking(args.video_path, display=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('video_path', help='Path to input video')
    parser.add_argument('--tracker', default='byte_track', help='Path to input video')
    parser.add_argument('--detector',default='yolov5s', help='Path to input video')
    parser.add_argument('--cpu', default=True, action='store_false', dest='use_cuda', help='run on cpu')
    args = parser.parse_args()

    main(args)
