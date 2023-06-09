# AS-One : A Modular Library for YOLO Object Detection and Object Tracking

[<img src="https://kajabi-storefronts-production.kajabi-cdn.com/kajabi-storefronts-production/file-uploads/themes/2151476941/settings_images/65d82-0d84-6171-a7e0-5aa180b657d5_Black_with_Logo.jpg" width="100%">](https://www.youtube.com/watch?v=K-VcpPwcM8k)





#### Table of Contents
1. Introduction
2. Prerequisites
3. Clone the Repo
4. Installation
    - [Linux](#4-installation)
    - [Windows 10/11](#4-installation) 
    - [MacOS](#4-installation) 
5. Running AS-One
6. [Sample Code Snippets](#6-sample-code-snippets)
7. [Model Zoo](asone/linux/Instructions/Benchmarking.md)

## 1. Introduction
==UPDATE: YOLO-NAS is OUT==

AS-One is a python wrapper for multiple detection and tracking algorithms all at one place. Different trackers such as `ByteTrack`, `DeepSORT` or `NorFair` can be integrated with different versions of `YOLO` with minimum lines of code.
This python wrapper provides YOLO models in `ONNX`, `PyTorch` & `CoreML`  flavors. We plan to offer support for future versions of YOLO when they get released.

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
<details open>
<summary>For Linux</summary>

```shell
python3 -m venv .env
source .env/bin/activate

pip install numpy Cython
pip install cython-bbox asone onnxruntime-gpu==1.12.1
pip install super-gradients==3.1.1
# for CPU
pip install torch torchvision
# for GPU
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
```
</details>

<details>
<summary> For Windows 10/11</summary>

```shell
python -m venv .env
.env\Scripts\activate
pip install numpy Cython 
pip install lap
pip install -e git+https://github.com/samson-wang/cython_bbox.git#egg=cython-bbox

pip install asone onnxruntime-gpu==1.12.1
pip install super-gradients==3.1.1
# for CPU
pip install torch torchvision

# for GPU
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
or
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio===0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```
</details>
<details>
<summary>For MacOS</summary>

```shell
python3 -m venv .env
source .env/bin/activate

pip install numpy Cython
pip install cython-bbox asone
pip install super-gradients==3.1.1
# for CPU
pip install torch torchvision
```
</details>

## 5. Running AS-One

Run `main.py` to test tracker on `data/sample_videos/test.mp4` video

```
python main.py data/sample_videos/test.mp4
```

### Run in `Google Colab`

 <a href="https://drive.google.com/file/d/1xy5P9WGI19-PzRH3ceOmoCgp63K6J_Ls/view?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>


## 6. Sample Code Snippets
<details>
<summary>6.1. Object Detection</summary>

```python
import asone
from asone import utils
from asone import ASOne
import cv2

video_path = 'data/sample_videos/test.mp4'
detector = ASOne(detector=asone.YOLOV7_PYTORCH, use_cuda=True) # Set use_cuda to False for cpu

filter_classes = ['car'] # Set to None to detect all classes

cap = cv2.VideoCapture(video_path)

while True:
    _, frame = cap.read()
    if not _:
        break

    dets, img_info = detector.detect(frame, filter_classes=filter_classes)

    bbox_xyxy = dets[:, :4]
    scores = dets[:, 4]
    class_ids = dets[:, 5]

    frame = utils.draw_boxes(frame, bbox_xyxy, class_ids=class_ids)

    cv2.imshow('result', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
```

Run the `asone/demo_detector.py` to test detector.

```shell
# run on gpu
python -m asone.demo_detector data/sample_videos/test.mp4

# run on cpu
python -m asone.demo_detector data/sample_videos/test.mp4 --cpu
```

<details>
<summary>6.1.1 Use Custom Trained Weights for Detector</summary>

<!-- ### 6.1.2 Use Custom Trained Weights -->

Use your custom weights of a detector model trained on custom data by simply providing path of the weights file.

```python
import asone
from asone import utils
from asone import ASOne
import cv2

video_path = 'data/sample_videos/license_video.webm'
detector = ASOne(detector=asone.YOLOV7_PYTORCH, weights='data/custom_weights/yolov7_custom.pt', use_cuda=True) # Set use_cuda to False for cpu

class_names = ['license_plate'] # your custom classes list

cap = cv2.VideoCapture(video_path)

while True:
    _, frame = cap.read()
    if not _:
        break

    dets, img_info = detector.detect(frame)

    bbox_xyxy = dets[:, :4]
    scores = dets[:, 4]
    class_ids = dets[:, 5]

    frame = utils.draw_boxes(frame, bbox_xyxy, class_ids=class_ids, class_names=class_names) # simply pass custom classes list to write your classes on result video

    cv2.imshow('result', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
```
</details>

<details>
<summary>6.1.2. Changing Detector Models </summary>

Change detector by simply changing detector flag. The flags are provided in [benchmark](asone/linux/Instructions/Benchmarking.md) tables.
* Our library now supports YOLOv5, YOLOv7, and YOLOv8 on macOS.
```python
# Change detector
detector = ASOne(detector=asone.YOLOX_S_PYTORCH, use_cuda=True)

# For macOs
# YOLO5
detector = ASOne(detector=asone.YOLOV5X_MLMODEL)
# YOLO7
detector = ASOne(detector=asone.YOLOV7_MLMODEL)
# YOLO8
detector = ASOne(detector=asone.YOLOV8L_MLMODEL)
```

</details>

</details>

<details>
<summary>6.2. Object Tracking </summary>

Use tracker on sample video. 

```python
import asone
from asone import ASOne

# Instantiate Asone object
detect = ASOne(tracker=asone.BYTETRACK, detector=asone.YOLOV7_PYTORCH, use_cuda=True) #set use_cuda=False to use cpu

filter_classes = ['person'] # set to None to track all classes

# ##############################################
#           To track using video file
# ##############################################
# Get tracking function
track = detect.track_video('data/sample_videos/test.mp4', output_dir='data/results', save_result=True, display=True, filter_classes=filter_classes)

# Loop over track to retrieve outputs of each frame 
for bbox_details, frame_details in track:
    bbox_xyxy, ids, scores, class_ids = bbox_details
    frame, frame_num, fps = frame_details
    # Do anything with bboxes here

# ##############################################
#           To track using webcam
# ##############################################
# Get tracking function
track = detect.track_webcam(cam_id=0, output_dir='data/results', save_result=True, display=True, filter_classes=filter_classes)

# Loop over track to retrieve outputs of each frame 
for bbox_details, frame_details in track:
    bbox_xyxy, ids, scores, class_ids = bbox_details
    frame, frame_num, fps = frame_details
    # Do anything with bboxes here

# ##############################################
#           To track using web stream
# ##############################################
# Get tracking function
stream_url = 'rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4'
track = detect.track_stream(stream_url, output_dir='data/results', save_result=True, display=True, filter_classes=filter_classes)

# Loop over track to retrieve outputs of each frame 
for bbox_details, frame_details in track:
    bbox_xyxy, ids, scores, class_ids = bbox_details
    frame, frame_num, fps = frame_details
    # Do anything with bboxes here
```

[Note] Use can use custom weights for a detector model by simply providing path of the weights file. in `ASOne` class.

<details>
<summary>6.2.1 Changing Detector and Tracking Models</summary>

<!-- ### Changing Detector and Tracking Models -->

Change Tracker by simply changing the tracker flag.

The flags are provided in [benchmark](asone/linux/Instructions/Benchmarking.md) tables.

```python
detect = ASOne(tracker=asone.BYTETRACK, detector=asone.YOLOV7_PYTORCH, use_cuda=True)
# Change tracker
detect = ASOne(tracker=asone.DEEPSORT, detector=asone.YOLOV7_PYTORCH, use_cuda=True)
```

```python
# Change Detector
detect = ASOne(tracker=asone.DEEPSORT, detector=asone.YOLOX_S_PYTORCH, use_cuda=True)
```
</details>


Run the `asone/demo_detector.py` to test detector.

```shell
# run on gpu
python -m asone.demo_detector data/sample_videos/test.mp4

# run on cpu
python -m asone.demo_detector data/sample_videos/test.mp4 --cpu
```
</details>
<details>
<summary>6.3. Text Detection</summary>

Sample code to detect text on an image

```python
# Detect and recognize text
import asone
from asone import utils
from asone import ASOne
import cv2


img_path = 'data/sample_imgs/sample_text.jpeg'
ocr = ASOne(detector=asone.CRAFT, recognizer=asone.EASYOCR, use_cuda=True) # Set use_cuda to False for cpu
img = cv2.imread(img_path)
results = ocr.detect_text(img) 
img = utils.draw_text(img, results)
cv2.imwrite("data/results/results.jpg", img)
```

Use Tracker on Text
```python
import asone
from asone import ASOne

# Instantiate Asone object
detect = ASOne(tracker=asone.DEEPSORT, detector=asone.CRAFT, recognizer=asone.EASYOCR, use_cuda=True) #set use_cuda=False to use cpu

# ##############################################
#           To track using video file
# ##############################################
# Get tracking function
track = detect.track_video('data/sample_videos/GTA_5-Unique_License_Plate.mp4', output_dir='data/results', save_result=True, display=True)

# Loop over track to retrieve outputs of each frame 
for bbox_details, frame_details in track:
    bbox_xyxy, ids, scores, class_ids = bbox_details
    frame, frame_num, fps = frame_details
    # Do anything with bboxes here
```

Run the `asone/demo_ocr.py` to test ocr.

```shell
# run on gpu
 python -m asone.demo_ocr data/sample_videos/GTA_5-Unique_License_Plate.mp4

# run on cpu
 python -m asone.demo_ocr data/sample_videos/GTA_5-Unique_License_Plate.mp4 --cpu
```

</details>

<details>
<summary>6.4. Pose Estimation</summary>

Sample code to estimate pose on an image

```python
# Pose Estimation
import asone
from asone import utils
from asone import PoseEstimator
import cv2

img_path = 'data/sample_imgs/test2.jpg'
pose_estimator = PoseEstimator(estimator_flag=asone.YOLOV8M_POSE, use_cuda=True) #set use_cuda=False to use cpu
img = cv2.imread(img_path)
kpts = pose_estimator.estimate_image(img) 
img = utils.draw_kpts(img, kpts)
cv2.imwrite("data/results/results.jpg", img)
```
* Now you can use Yolov8 and Yolov7-w6 for pose estimation. The flags are provided in [benchmark](asone/linux/Instructions/Benchmarking.md) tables.

```python
# Pose Estimation on video
import asone
from asone import PoseEstimator

video_path = 'data/sample_videos/football1.mp4'
pose_estimator = PoseEstimator(estimator_flag=asone.YOLOV7_W6_POSE, use_cuda=True) #set use_cuda=False to use cpu
estimator = pose_estimator.estimate_video(video_path, save=True, display=True)
for kpts, frame_details in estimator:
    frame, frame_num, fps = frame_details
    print(frame_num)
    # Do anything with kpts here
```

Run the `asone/demo_pose_estimator.py` to test Pose estimation.

```shell
# run on gpu
 python -m asone.demo_pose_estimator data/sample_videos/football1.mp4

# run on cpu
 python -m asone.demo_pose_estimator data/sample_videos/football1.mp4 --cpu
```

</details>

To setup ASOne using Docker follow instructions given in [docker setup](asone/linux/Instructions/Docker-Setup.md) 

# ToDo
- [x] First Release
- [x] Import trained models
- [x] Simplify code even further
- [x] Updated for YOLOv8
- [x] OCR and Counting
- [x] OCSORT, StrongSORT, MoTPy
- [x] M1/2 Apple Silicon Compatibility
- [x] Pose Estimation YOLOv7/v8
- [x] YOLO-NAS
- [ ] SAM Integration

|Offered By: |Maintained By:|
|-------------|-------------|
|[![AugmentedStarups](https://user-images.githubusercontent.com/107035454/195115263-d3271ef3-973b-40a4-83c8-0ade8727dd40.png)](https://augmentedstartups.com)|[![AxcelerateAI](https://user-images.githubusercontent.com/107035454/195114870-691c8a52-fcf0-462e-9e02-a720fc83b93f.png)](https://axcelerate.ai/)|
