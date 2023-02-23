import asone
from asone import utils
from asone import ASOne
import cv2
from asone import utils


img_path = 'data/sample_imgs/sample_text.jpeg'
detector = ASOne(detector=asone.CRAFT ,use_cuda=True) # Set use_cuda to False for cpu

img = cv2.imread(img_path)
dets, img_info = detector.detect(img) 
bbox_xyxy = dets[:, :4]
scores = dets[:, 4]
class_ids = dets[:, 5]

img = utils.draw_boxes(img, bbox_xyxy, class_ids=class_ids)

cv2.imwrite("results.jpg", img)