import asone
from asone import utils
from asone import ASOne
import cv2
from asone import utils


img_path = 'data/sample_imgs/sample_text.jpeg'
detector = ASOne(detector=asone.CRAFT, recognizer=asone.STANDARD ,use_cuda=True) # Set use_cuda to False for cpu

img = cv2.imread(img_path)

results = detector.detect_text(img) 
img = utils.draw_text(img, results)
cv2.imwrite("results.jpg", img)