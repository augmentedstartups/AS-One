# AS-One : A Modular Libary for YOLO Object Detection and Object Tracking `BETA`

![croped](https://user-images.githubusercontent.com/107035454/195083948-4873d60a-3ac7-4279-8770-535488f4a097.png)

#### Table of Contents
- [Introduction](#introduction)
- [Prerequisite](#prerequisite)
- [Installation](#installation)
- [Usage](#usage)
- [Benchmarks](asone/linux/Instructions/Benchmarking.md)

## 1. Introduction

Asone is a python wrapper for multiple detection and tracking algorithms all at one place. Different trackers such as `ByteTrack`, `DeepSort` or `NorFair` can be integrated with different versions of `YOLO` with minimum lines of code.
This python wrapper provides yolo models in both `ONNX` and `PyTorch` versions.

## 2. Prerequisite

- Make sure to install `GPU` drivers in your system if you want to use `GPU` . Follow [driver installation](asone/linux/Instructions/Driver-Installations.md) for further instructions.
- Make sure you have [MS Build tools](https://aka.ms/vs/17/release/vs_BuildTools.exe) installed in system if using windows. 
- [Download git for windows](https://git-scm.com/download/win) if not installed.

## 3. Installation

For `Linux`

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

For `Windows 10/11`

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

## 4. Clone the Repo

Navigate to an empty folder of your choice.

```git clone https://github.com/augmentedstartups/AS-One.git```

Change Directory to AS-One

```cd AS-One```

## 5. Running AS-One

Run `main.py` to test tracker on `data/sample_videos/test.mp4` video

```
# run on gpu
python main.py data/sample_videos/test.mp4

# run on cpu
python main.py data/sample_videos/test.mp4 --cpu
```

### Usage
#### Detector
Use detector on an image using GPU

```
import asone
from asone import utils
from asone.detectors import Detector
import cv2

img = cv2.imread('data/sample_imgs/test2.jpg')
detector = Detector(asone.YOLOV7_E6_ONNX, use_cuda=True).get_detector() # Set use_cuda to False for cpu

dets, img_info = detector.detect(img)

bbox_xyxy = dets[:, :4]
scores = dets[:, 4]
class_ids = dets[:, 5]

img = utils.draw_boxes(img, bbox_xyxy, class_ids=class_ids)
cv2.imwrite('result.png', img)
```

Change detector by simply changing detector flag. flags are provided in [benchmark](asone/linux/Instructions/Benchmarking.md) tables.

```
# Change detector
detector = Detector(asone.YOLOX_S_PYTORCH, use_cuda=True).get_detector()
```

Run the `asone/demo_detector.py` to test detector.

```
# run on gpu
python -m asone.demo_detector data/sample_imgs/test2.jpg

# run on cpu
python -m asone.demo_detector data/sample_imgs/test2.jpg --cpu
```


#### Tracker
Use tracker on sample video using gpu. 


```
import asone
from asone import ASOne

# Instantiate Asone object
dt_obj = ASOne(tracker=asone.BYTETRACK, detector=asone.YOLOX_DARKNET_PYTORCH, use_cuda=args.use_cuda)

# Get tracking function
track_fn = dt_obj.track_video('data/sample_videos/test.mp4', output_dir='data/results', save_result=True, display=True)

# Loop over track_fn to retrieve outputs of each frame 
for bbox_details, frame_details in track_fn:
    bbox_xyxy, ids, scores, class_ids = bbox_details
    frame, frame_num, fps = frame_details
    # Do anything with bboxes here

# To track using webcam
# Get tracking function
track_fn = dt_obj.track_webcam(cam_id=0, output_dir='data/results', save_result=True, display=True)

# Loop over track_fn to retrieve outputs of each frame 
for bbox_details, frame_details in track_fn:
    bbox_xyxy, ids, scores, class_ids = bbox_details
    frame, frame_num, fps = frame_details
    # Do anything with bboxes here
```

Change Tracker by simply changing the tracker flag.

flags are provided in [benchmark](asone/linux/Instructions/Benchmarking.md) tables.

```
dt_obj = ASOne(tracker=asone.BYTETRACK, detector=asone.YOLOX_DARKNET_PYTORCH, use_cuda=True)
// Change tracker
dt_obj = ASOne(tracker=asone.DEEPSORT, detector=asone.YOLOX_DARKNET_PYTORCH, use_cuda=True)
```



```
dt_obj = ASOne(tracker=asone.DEEPSORT, detector=asone.YOLOX_S_PYTORCH, use_cuda=True)
```

Results on provided sample video

https://user-images.githubusercontent.com/107035454/195079926-aee47eac-0430-4ada-8cc7-cc9d1d13c889.mp4

To setup ASOne using Docker follow instructions given in [docker setup](asone/linux/Instructions/Docker-Setup.md) 

# ToDo
- [x] First Release
- [ ] Simplify code even further
- [ ] Add support for other Trackers and Detectors

|Offered By: |Maintained By:|
|-------------|-------------|
|[![AugmentedStarups](https://user-images.githubusercontent.com/107035454/195115263-d3271ef3-973b-40a4-83c8-0ade8727dd40.png)](https://augmentedstartups.com)|[![AxcelerateAI](https://user-images.githubusercontent.com/107035454/195114870-691c8a52-fcf0-462e-9e02-a720fc83b93f.png)](https://axcelerate.ai/)|

