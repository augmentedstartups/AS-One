import cv2
import numpy as np

img = np.zeros((400, 600, 3))

cap = cv2.VideoCapture(0)

while True:
    if not cap.isOpened():
        print('Unable to load camera. Use the command "xhost +"')
        pass

    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('Video', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()