import asone
from asone import utils
from asone import ASOne
import cv2
import matplotlib.pyplot as plt
from asone import utils


video_path = 'data/sample_imgs/sample_text.jpeg'
detector = ASOne(detector=asone.CRAFT, recognizer=asone.STANDARD, use_cuda=True) # Set use_cuda to False for cpu
cap = cv2.VideoCapture(video_path)

while True:
    _, frame = cap.read()
    if not _:
        break

    results = detector.detect_text(frame, languages=['en'])
    frame = utils.draw_text(frame ,results)
    cv2.imwrite("results.jpg", frame)