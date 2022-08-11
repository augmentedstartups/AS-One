import cv2

img = cv2.imread('test.jpg')
cv2.imshow("TEST DISPLAY", cv2.resize(img, (600, 600))
)
cv2.waitKey(0)