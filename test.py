import asone
from asone import utils
from asone import ASOne
import cv2
import numpy as np


video_path = 'data/sample_videos/test.mp4'
detector = ASOne(detector=asone.YOLOV5N_MLMODEL, weights='yolov5n.mlmodel' ,use_cuda=False) # Set use_cuda to False for cpu

filter_classes = [] # Set to None to detect all classes
img_path = 'data/sample_imgs/test2.jpg'
frame = cv2.imread(img_path)
dets, img_info= detector.detect(frame, filter_classes=filter_classes, conf_thres=0.39)

bbox_xyxy = dets[:, :4]
scores = dets[:, 4]
class_ids = dets[:, 5]

frame = utils.draw_boxes(frame, bbox_xyxy, class_ids=class_ids)
cv2.imwrite('results.jpg', frame)