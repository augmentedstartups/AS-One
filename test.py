import asone
from asone import utils
from asone import ASOne
import cv2

video_path = 'data/sample_videos/test.mp4'
detector = ASOne(detector=asone.YOLOV5N_PYTORCH, weights='/Users/apple/Desktop/ramzan/asone/yolov5n.mlmodel' ,use_cuda=False) # Set use_cuda to False for cpu

filter_classes = [] # Set to None to detect all classes
img_path = '/Users/apple/Desktop/ramzan/asone/data/sample_imgs/test.jpg'
frame = cv2.imread(img_path)
dets, img_info = detector.detect(frame, filter_classes=filter_classes)

bbox_xyxy = dets[:4]
# scores = dets[:, 4]
# class_ids = dets[:, 5]

frame = utils.draw_boxes(frame, bbox_xyxy, class_ids=[1])

cv2.imshow('result', frame)
cv2.imwrite('res.jpg', frame)