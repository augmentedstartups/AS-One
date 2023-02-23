from easyocr_rec.easyocr_recognizer import EasyocrRecognizer
import cv2


ocr = EasyocrRecognizer(detect_network='dbnet18',)
input_path = "sample_text.jpeg"
img = cv2.imread(input_path)
print(ocr.text_recognizer(img))