# ASOne

![croped](https://user-images.githubusercontent.com/107035454/195083948-4873d60a-3ac7-4279-8770-535488f4a097.png)

#### Table of Contents
- [Introduction](#introduction)
- [Prerequisite](#prerequisite)
- [Installation](#installation)
- [Usage](#usage)
- [Benchmarks](asone-linux/Instructions/Benchmarking.md)

### Introduction

Asone is a python wrapper for multiple detection and tracking algorithms all at one place. Different trackers such as `ByteTrack`, `DeepSort` or `NorFair` can be integrated with different versions of `YOLO` with minimum lines of code.
This python wrapper provides yolo models in both `ONNX` and `PyTorch` versions.

### Prerequisite

- If you want to use `GPU` make sure you have `GPU` drivers installed in your system. Follow instructions given in [driver installation](asone-linux/Instructions/Driver-Installations.md).
- If using windows, Make sure you have [MS Build tools](https://aka.ms/vs/17/release/vs_BuildTools.exe) installed in system. 
- [Download git for windows](https://git-scm.com/download/win) if not installed.

### Installation

For linux

```
python3 -m venv .env
source .env/bin/activate

pip install numpy Cython
pip install cython-bbox

pip install asone


# for cpu
pip install torch torchvision

# for gpu
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113

```

For windows

```
python -m venv .env
.env\Scripts\activate
pip install numpy Cython
pip install -e git+https://github.com/samson-wang/cython_bbox.git#egg=cython-bbox

pip install asone

# for cpu
pip install torch torchvision

# for gpu
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
```



Use detector on a img using gpu. (Set `use_cuda` flag in sample code to Flase to test it on cpu.)

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

# Change detector
detector = Detector(asone.YOLOX_S_PYTORCH, use_cuda=True).get_detector()
```

You can also run the `demo_detector.py` to test the above code.

```
# run on gpu
python demo_detector.py

# run on cpu
python demo_detector.py --cpu
```


Use tracker on sample video using gpu. (Set `use_cuda` flag in sample code to Flase to test it on cpu.)


```
import asone
from asone import ASOne

dt_obj = ASOne(tracker=asone.BYTETRACK, detector=asone.YOLOX_DARKNET_PYTORCH, use_cuda=True)
dt_obj.track_video('sample_videos/test.mp4')

# To track using webcam
# dt_obj.track_webcam()
```


dt_obj = ASOne(tracker=asone.BYTETRACK, detector=asone.YOLOX_DARKNET_PYTORCH, use_cuda=True)
[Note]: To change Detector or Tracker, you only have to change the flag. (flags are provided in [benchmark](asone-linux/Instructions/Benchmarking.md) tables.)

```
dt_obj = ASOne(tracker=asone.DEEPSORT, detector=asone.YOLOX_S_PYTORCH, use_cuda=True)
```
