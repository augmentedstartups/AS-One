import argparse
import asone
from asone import ASOne
import argparse

def main(args):
    dt_obj = ASOne(tracker=asone.BYTETRACK, detector=asone.YOLOX_DARKNET_PYTORCH, use_cuda=args.use_cuda)
    dt_obj.start_tracking(args.video_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('video_path', help='Path to input video')
    parser.add_argument('--cpu', default=True, action='store_false', dest='use_cuda', help='run on cpu')
    args = parser.parse_args()

    main(args)
