# AS-One : A Modular Libary for YOLO Object Detection and Object Tracking `BETA`

![croped](https://user-images.githubusercontent.com/107035454/195083948-4873d60a-3ac7-4279-8770-535488f4a097.png)

#### Table of Contents
1. Introduction
2. Prerequisites
3. Clone the Repo
4. Installation
5. Running AS-One
6. [Usage](#usage)
7. [Benchmarks](asone/linux/Instructions/Benchmarking.md)

## 1. Introduction

AS-One is a python wrapper for multiple detection and tracking algorithms all at one place. Different trackers such as `ByteTrack`, `DeepSort` or `NorFair` can be integrated with different versions of `YOLO` with minimum lines of code.
This python wrapper provides YOLO models in both `ONNX` and `PyTorch` versions. We plan to offer support for future versions of YOLO when they get released.

This is One Library for most of your computer vision needs.

If you would like to dive deeper into YOLO Object Detection and Tracking, then check out our [courses](https://www.augmentedstartups.com/store) and [projects](https://store.augmentedstartups.com)

[<img src="https://s3.amazonaws.com/kajabi-storefronts-production/blogs/22606/images/0FDx83VXSYOY0NAO2kMc_ASOne_Windows_Play.jpg" width="50%">](https://www.youtube.com/watch?v=K-VcpPwcM8k)

Watch the step-by-step tutorial

## 2. Prerequisites

- Make sure to install `GPU` drivers in your system if you want to use `GPU` . Follow [driver installation](asone/linux/Instructions/Driver-Installations.md) for further instructions.
- Make sure you have [MS Build tools](https://aka.ms/vs/17/release/vs_BuildTools.exe) installed in system if using windows. 
- [Download git for windows](https://git-scm.com/download/win) if not installed.

## 3. Clone the Repo

Navigate to an empty folder of your choice.

```git clone https://github.com/augmentedstartups/AS-One.git```

Change Directory to AS-One

```cd AS-One```

## 4. Installation

### For `Linux`

```shell
python3 -m venv .env
source .env/bin/activate

pip install numpy Cython
pip install cython-bbox

pip install asone


# for CPU
pip install torch torchvision

# for GPU
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113

```

### For `Windows 10/11`

```shell
python -m venv .env
.env\Scripts\activate
pip install numpy Cython
pip install -e git+https://github.com/samson-wang/cython_bbox.git#egg=cython-bbox

pip install asone

# for CPU
pip install torch torchvision

# for GPU
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
or
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio===0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

### Run AS-One
```
python main.py data/sample_videos/test.mp4
```

### Run in `Google Colab`

 <a href="https://drive.google.com/file/d/1xy5P9WGI19-PzRH3ceOmoCgp63K6J_Ls/view?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>


## 5. Running AS-One

Run `main.py` to test tracker on `data/sample_videos/test.mp4` video

```python
import asone
from asone import utils
from asone.detectors import Detector
import cv2

img = cv2.imread('data/sample_imgs/test2.jpg')
detector = Detector(asone.YOLOV7_E6_ONNX, use_cuda=True) # Set use_cuda to False for cpu

filter_classes = ['person'] # Set to None to detect all classes

dets, img_info = detector.detect(img, , filter_classes=filter_classes)

bbox_xyxy = dets[:, :4]
scores = dets[:, 4]
class_ids = dets[:, 5]

img = utils.draw_boxes(img, bbox_xyxy, class_ids=class_ids)
cv2.imwrite('result.png', img)
```

### Use Custom Trained Weights
Use your custom weights of a detector model trained on custom data by simply providing path of the weights file.

```python
import asone
from asone import utils
from asone.detectors import Detector
import cv2

img = cv2.imread('data/sample_imgs/test2.jpg')
detector = Detector(asone.YOLOV7_PYTORCH, weights="data/custom_weights/yolov7_custom.pt", use_cuda=True) # Set use_cuda to False for cpu

filter_classes = ['person'] # Set to None to detect all classes

dets, img_info = detector.detect(img, , filter_classes=filter_classes)

bbox_xyxy = dets[:, :4]
scores = dets[:, 4]
class_ids = dets[:, 5]

img = utils.draw_boxes(img, bbox_xyxy, class_ids=class_ids, class_names=['License Plate']) # class_names are names of classes in your dataset
cv2.imwrite('result.png', img)
```
### Changing Detector Models
Change detector by simply changing detector flag. The flags are provided in [benchmark](asone/linux/Instructions/Benchmarking.md) tables.

```python
# Change detector
detector = Detector(asone.YOLOX_S_PYTORCH, use_cuda=True)
```

Run the `asone/demo_detector.py` to test detector.

```shell
# run on gpu
python -m asone.demo_detector data/sample_imgs/test2.jpg

# run on cpu
python -m asone.demo_detector data/sample_imgs/test2.jpg --cpu
```

## Object Tracking

### Video
Use tracker on sample video using gpu. 

```python
import asone
from asone import ASOne

# Instantiate Asone object
dt_obj = ASOne(tracker=asone.BYTETRACK, detector=asone.YOLOX_DARKNET_PYTORCH, use_cuda=True)

filter_classes = ['person'] # set to None to track all classes

# Get tracking function
track_fn = dt_obj.track_video('data/sample_videos/test.mp4', output_dir='data/results', save_result=True, display=True, filter_classes=filter_classes)

# Loop over track_fn to retrieve outputs of each frame 
for bbox_details, frame_details in track_fn:
    bbox_xyxy, ids, scores, class_ids = bbox_details
    frame, frame_num, fps = frame_details
    # Do anything with bboxes here

# To track using webcam
# Get tracking function
track_fn = dt_obj.track_webcam(cam_id=0, output_dir='data/results', save_result=True, display=True, filter_classes=filter_classes)

# Loop over track_fn to retrieve outputs of each frame 
for bbox_details, frame_details in track_fn:
    bbox_xyxy, ids, scores, class_ids = bbox_details
    frame, frame_num, fps = frame_details
    # Do anything with bboxes here
```
### Use Custom Trained Weights for Detector
Use your custom weights of a detector model trained on custom data by simply providing path of the weights file.

```python
import asone
from asone import ASOne

# Instantiate Asone object
dt_obj = ASOne(tracker=asone.BYTETRACK, detector=asone.YOLOX_DARKNET_PYTORCH, weights='data/custom_weights/yolov7_custom.pt', use_cuda=True)

filter_classes = ['person'] # set to None to track all classes

# Get tracking function
track_fn = dt_obj.track_video('data/sample_videos/test.mp4', output_dir='data/results', save_result=True, display=True, filter_classes=filter_classes, class_names=['License Plate']) #class_names are class names in your custom data

# Loop over track_fn to retrieve outputs of each frame 
for bbox_details, frame_details in track_fn:
    bbox_xyxy, ids, scores, class_ids = bbox_details
    frame, frame_num, fps = frame_details
    # Do anything with bboxes here
```

### Changing Detector and Tracking Models

Change Tracker by simply changing the tracker flag.

The flags are provided in [benchmark](asone/linux/Instructions/Benchmarking.md) tables.

```python
dt_obj = ASOne(tracker=asone.BYTETRACK, detector=asone.YOLOX_DARKNET_PYTORCH, use_cuda=True)
// Change tracker
dt_obj = ASOne(tracker=asone.DEEPSORT, detector=asone.YOLOX_DARKNET_PYTORCH, use_cuda=True)
```

```python
dt_obj = ASOne(tracker=asone.DEEPSORT, detector=asone.YOLOX_S_PYTORCH, use_cuda=True)
```

To setup ASOne using Docker follow instructions given in [docker setup](asone/linux/Instructions/Docker-Setup.md) 

# ToDo
- [x] First Release
- [x] Import trained models
- [ ] Simplify code even further
- [ ] Add support for other Trackers and Detectors
- [ ] M1/2 Apple Silicon Compatibility

|Offered By: |Maintained By:|
|-------------|-------------|
|[![AugmentedStarups](https://user-images.githubusercontent.com/107035454/195115263-d3271ef3-973b-40a4-83c8-0ade8727dd40.png)](https://augmentedstartups.com)|[![AxcelerateAI](https://user-images.githubusercontent.com/107035454/195114870-691c8a52-fcf0-462e-9e02-a720fc83b93f.png)](https://axcelerate.ai/)|
