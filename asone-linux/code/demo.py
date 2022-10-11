import argparse
from asone import ASOne
import argparse
import asone

def main(args):
    dt_obj = ASOne(tracker=asone.BYTETRACK, detector=args.detector, use_cuda=args.use_cuda, use_onnx=True)
    dt_obj.start_tracking(args.video_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('video_path', help='Path to input video')
    parser.add_argument('--tracker', type=int, default=0, help='Path to input video')
    parser.add_argument('--detector',default='yolov5s', help='Path to input video')
    parser.add_argument('--cpu', default=True, action='store_false', dest='use_cuda', help='run on cpu')
    args = parser.parse_args()

    main(args)
