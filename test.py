import asone
from asone import utils
from asone import ASOne
import cv2


video_path = 'data/sample_imgs/sample_text.jpeg'
detector = ASOne(detector=asone.Text_Detector, use_cuda=True) # Set use_cuda to False for cpu

filter_classes = [] # Set to None to detect all classes

cap = cv2.VideoCapture(video_path)

while True:
    _, frame = cap.read()
    if not _:
        break

    dets, img_info = detector.detect(frame, languages=['en'])
    bbox_xyxy = dets[:, :4]
    scores = dets[:, 4]
    class_ids = dets[:, 5]
    frame = utils.draw_boxes(frame, bbox_xyxy, class_ids=class_ids)

    cv2.imwrite("khkljsdj.jpg", frame)
    # cv2.imshow('result', frame)

    # if cv2.waitKey(25) & 0xFF == ord('q'):
    #     break
    cv2.waitKey(0)