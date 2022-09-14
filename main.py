import argparse
import asone
from asone import ASOne
import argparse

def main(args):
    dt_obj = ASOne(tracker=asone.NORFAIR, detector=asone.YOLOV7_PYTORCH, use_cuda=args.use_cuda)
    dt_obj.start_tracking(args.video_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('video_path', help='Path to input video')
    parser.add_argument('--tracker', type=int, default=0, help='tracker [0, 1, 2] for Bytetrack, deepsort, norfair')
    parser.add_argument('--detector',type= int, default=0, help='Path to input video')
    parser.add_argument('--cpu', default=True, action='store_false', dest='use_cuda', help='run on cpu')
    args = parser.parse_args()

    main(args)
