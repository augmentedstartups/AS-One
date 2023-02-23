import asone
from asone import utils
from asone import ASOne
import cv2


video_path = 'data/sample_imgs/sample_text.jpeg'
detector = ASOne(detector=asone.YOLOV7_PYTORCH, use_cuda=True) # Set use_cuda to False for cpu

filter_classes = [] # Set to None to detect all classes

cap = cv2.VideoCapture(video_path)

while True:
    _, frame = cap.read()
    if not _:
        break

    res = detector.detect(frame)
    print(res)