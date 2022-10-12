# ASOne

![croped](https://user-images.githubusercontent.com/107035454/195083948-4873d60a-3ac7-4279-8770-535488f4a097.png)


# Introduction

Asone is a python wrapper for multiple detection and tracking algorithms all at one place. Different trackers such as `ByteTrack`, `DeepSort` or `NorFair` can be integrated with different versions of `YOLO` with minimum lines of code.
This python wrapper provides yolo models in both `ONNX` and `PyTorch` versions.


### Prerequisite

- If using windows, Make sure you have [MS Build tools](https://devblogs.microsoft.com/cppblog/announcing-visual-c-build-tools-2015-standalone-c-tools-for-build-environments) installed in system. 


Usage:

```
pip install numpy Cython

pip install asone

# for windows
pip install -e git+https://github.com/samson-wang/cython_bbox.git#egg=cython-bbox
# for linux
pip install cython-bbox

# for cpu
pip install torch torchvision


# for gpu
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113


```

Sample code:

```
import asone
from asone import ASOne

dt_obj = ASOne(tracker=asone.BYTETRACK, detector=asone.YOLOX_DARKNET_PYTORCH, use_cuda=True)
dt_obj.track_video('sample_videos/test.mp4')

# To track using webcam
dt_obj.track_webcam()

```
Results on provided sample video

https://user-images.githubusercontent.com/107035454/195079926-aee47eac-0430-4ada-8cc7-cc9d1d13c889.mp4

Sample code to use detector:

```
import asone
from asone import utils
from asone.detectors import Detector
import cv2

img = cv2.imread('sample_imgs/test2.jpg')
detector = Detector(asone.YOLOV7_E6_ONNX, use_cuda=True).get_detector()
dets, img_info = detector.detect(img)

bbox_xyxy = dets[:, :4]
scores = dets[:, 4]
class_ids = dets[:, 5]

img = utils.draw_boxes(img, bbox_xyxy, class_ids=class_ids)
cv2.imwrite('result.png', img)

```