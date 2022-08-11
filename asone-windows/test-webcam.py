import cv2
import numpy as np

img = np.zeros((400, 600, 3))

##Fetch IP Address
# importing socket module
import socket
# getting the hostname by socket.gethostname() method
hostname = socket.gethostname()
# getting the IP address using socket.gethostbyname() method
ip_address = socket.gethostbyname(hostname)
# capturing the video from ip stream 
cap = cv2.VideoCapture(f'http://{ip_address}:56000/mjpeg')
# cap.open("")

while True:
    if not cap.isOpened():
        print('Unable to load camera. Use the command "xhost +"')
        pass

    # Capture frame-by-frame
    ret, frame = cap.read()
    print(frame)
    # Display the resulting frame
    cv2.imshow('Video', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
