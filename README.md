# AS-One : A Modular Library for YOLO Object Detection and Object Tracking



<div align="center">
  <p>
    <a align="center" href="" target="https://badge.fury.io/py/asone">
      <img
        width="100%"
        src="https://kajabi-storefronts-production.kajabi-cdn.com/kajabi-storefronts-production/file-uploads/themes/2151476941/settings_images/65d82-0d84-6171-a7e0-5aa180b657d5_Black_with_Logo.jpg" width="100%">
      <a href="https://www.youtube.com/watch?v=K-VcpPwcM8k" style="display:inline-block;padding:10px 20px;background-color:red;color:white;text-decoration:none;font-size:16px;font-weight:bold;border-radius:5px;transition:background-color 0.3s;" target="_blank">Watch Video</a>

    
  </p>

  <br>

  <br>

[![PyPI version](https://badge.fury.io/py/asone.svg)](https://badge.fury.io/py/asone)
[![python-version](https://img.shields.io/pypi/pyversions/supervision)](https://badge.fury.io/py/asone)
[![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1xy5P9WGI19-PzRH3ceOmoCgp63K6J_Ls/view?usp=sharing)
[![start with why](https://img.shields.io/badge/version-1.1.0-green)](https://github.com/augmentedstartups/AS-One)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)

</div>

## 👋 Hello

==UPDATE: YOLO-NAS is OUT==

AS-One is a python wrapper for multiple detection and tracking algorithms all at one place. Different trackers such as `ByteTrack`, `DeepSORT` or `NorFair` can be integrated with different versions of `YOLO` with minimum lines of code.
This python wrapper provides YOLO models in `ONNX`, `PyTorch` & `CoreML` flavors. We plan to offer support for future versions of YOLO when they get released.

This is One Library for most of your computer vision needs.

If you would like to dive deeper into YOLO Object Detection and Tracking, then check out our [courses](https://www.augmentedstartups.com/store) and [projects](https://store.augmentedstartups.com)

[<img src="https://s3.amazonaws.com/kajabi-storefronts-production/blogs/22606/images/0FDx83VXSYOY0NAO2kMc_ASOne_Windows_Play.jpg" width="50%">](https://www.youtube.com/watch?v=K-VcpPwcM8k)

Watch the step-by-step tutorial 🤝



## 💻 Install
<details><summary> 🔥 Prerequisites</summary>

- Make sure to install `GPU` drivers in your system if you want to use `GPU` . Follow [driver installation](asone/linux/Instructions/Driver-Installations.md) for further instructions.
- Make sure you have [MS Build tools](https://aka.ms/vs/17/release/vs_BuildTools.exe) installed in system if using windows.
- [Download git for windows](https://git-scm.com/download/win) if not installed.
</details>

```bash
pip install asone
```

<details>
<summary> 👉 Install from Source</summary>

### 💾 Clone the Repository

Navigate to an empty folder of your choice.

`git clone https://github.com/augmentedstartups/AS-One.git`

Change Directory to AS-One

`cd AS-One`

<details open>
<summary> 👉 For Linux</summary>


```shell
python3 -m venv .env
source .env/bin/activate

pip install -r requirements.txt

# for CPU
pip install torch torchvision
# for GPU
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
```


</details>

<details>
<summary> 👉 For Windows 10/11</summary>

```shell
python -m venv .env
.env\Scripts\activate
pip install numpy Cython
pip install lap
pip install -e git+https://github.com/samson-wang/cython_bbox.git#egg=cython-bbox

pip install asone onnxruntime-gpu==1.12.1
pip install typing_extensions==4.7.1
pip install super-gradients==3.1.3
# for CPU
pip install torch torchvision

# for GPU
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
or
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio===0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

</details>
<details>
<summary> 👉 For MacOS</summary>

```shell
python3 -m venv .env
source .env/bin/activate


pip install -r requirements.txt

# for CPU
pip install torch torchvision
```

</details>
</details>

##  Quick Start 🏃‍♂️

Use tracker on sample video.

```python
import asone
from asone import ASOne

detect = ASOne(tracker=asone.BYTETRACK, detector=asone.YOLOV9_C, use_cuda=True)
track = detect.track_video('data/sample_videos/test.mp4', filter_classes=['car'])

for bbox_details, frame_details in track:
    frame, _ , _ = frame_details
    frame = ASOne.draw(frame, bbox_details)
```


### Run in `Google Colab` 💻


<a href="https://drive.google.com/file/d/1xy5P9WGI19-PzRH3ceOmoCgp63K6J_Ls/view?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

##  Sample Code Snippets 📃

<details>
<summary>6.1 👉 Object Detection</summary>

```python
import asone
from asone import ASOne

detector = ASOne(detector=asone.YOLOV9_C, use_cuda=True) # Set use_cuda to False for cpu
vid = detector.video_reader('data/sample_videos/test.mp4')

for img in vid:
    dets, img_info = detector.detect(img)
    annotations = ASOne.draw(img, dets)
```

Run the `asone/demo_detector.py` to test detector.

```shell
# run on gpu
python -m asone.demo_detector data/sample_videos/test.mp4

# run on cpu
python -m asone.demo_detector data/sample_videos/test.mp4 --cpu
```


<details>
<summary>6.1.1 👉 Use Custom Trained Weights for Detector</summary>
<!-- ### 6.1.2 Use Custom Trained Weights -->

Use your custom weights of a detector model trained on custom data by simply providing path of the weights file.

```python
import asone
from asone import ASOne

detector = ASOne(detector=asone.YOLOV9_C, weights='data/custom_weights/yolov7_custom.pt', use_cuda=True) # Set use_cuda to False for cpu
vid = detector.video_reader('data/sample_videos/license_video.mp4')

for img in vid:
    dets, img_info = detector.detect(img)
    frame = ASOne.draw(img, dets, class_names=['license_plate'])
```

</details>

<details>
<summary>6.1.2 👉 Changing Detector Models </summary>

Change detector by simply changing detector flag. The flags are provided in [benchmark](asone/linux/Instructions/Benchmarking.md) tables.

- Our library now supports YOLOv5, YOLOv7, and YOLOv8 on macOS.

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
<summary>6.2 👉 Object Tracking </summary>

Use tracker on sample video.

```python
import asone
from asone import ASOne

# Instantiate Asone object
detect = ASOne(tracker=asone.BYTETRACK, detector=asone.YOLOV9_C, use_cuda=True) #set use_cuda=False to use cpu
track = detect.track_video('data/sample_videos/test.mp4', filter_classes=['car'])

# Loop over track to retrieve outputs of each frame
for bbox_details, frame_details in track:
    frame, _ , _ = frame_details
    frame = ASOne.draw(frame, bbox_details)
    # Do anything with bboxes here
```

[Note] Use can use custom weights for a detector model by simply providing path of the weights file. in `ASOne` class.

<details>
<summary>6.2.1 👉 Changing Detector and Tracking Models</summary>

<!-- ### Changing Detector and Tracking Models -->

Change Tracker by simply changing the tracker flag.

The flags are provided in [benchmark](asone/linux/Instructions/Benchmarking.md) tables.

```python
detect = ASOne(tracker=asone.BYTETRACK, detector=asone.YOLOV9_C, use_cuda=True)
# Change tracker
detect = ASOne(tracker=asone.DEEPSORT, detector=asone.YOLOV9_C, use_cuda=True)
```

```python
# Change Detector
detect = ASOne(tracker=asone.DEEPSORT, detector=asone.YOLOX_S_PYTORCH, use_cuda=True)
```

</details>

Run the `asone/demo_tracker.py` to test detector.

```shell
# run on gpu
python -m asone.demo_tracker data/sample_videos/test.mp4

# run on cpu
python -m asone.demo_tracker data/sample_videos/test.mp4 --cpu
```

</details>

<details>
<summary>6.3 👉 Segmentation</summary>


```python
import asone
from asone import ASOne

detect = ASOne(detector=asone.YOLOV9_C, segmentor=asone.SAM, use_cuda=True) #set use_cuda=False to use cpu
track = detect.detect_video('data/sample_videos/test.mp4', filter_classes=['car'])

for bbox_details, frame_details in track:
    frame, frame_num, fps = frame_details
    frame = ASOne.draw_masks(frame, bbox_details) # Draw masks
```
</details>

<details>
<summary>6.4 👉 Text Detection</summary>
  
Sample code to detect text on an image

```python
# Detect and recognize text
import asone
from asone import ASOne, utils
import cv2

ocr = ASOne(detector=asone.CRAFT, recognizer=asone.EASYOCR, use_cuda=True) # Set use_cuda to False for cpu
img = cv2.imread('data/sample_imgs/sample_text.jpeg')
results = ocr.detect_text(img)
img = utils.draw_text(img, results)
```

Use Tracker on Text

```python
import asone
from asone import ASOne

# Instantiate Asone object
detect = ASOne(tracker=asone.DEEPSORT, detector=asone.CRAFT, recognizer=asone.EASYOCR, use_cuda=True) #set use_cuda=False to use cpu
track = detect.track_video('data/sample_videos/GTA_5-Unique_License_Plate.mp4')

# Loop over track to retrieve outputs of each frame
for bbox_details, frame_details in track:
    frame, _, _ = frame_details
    frame = ASOne.draw(frame, bbox_details)

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
<summary>6.5 👉 Pose Estimation</summary>


Sample code to estimate pose on an image

```python
# Pose Estimation
import asone
from asone import PoseEstimator, utils
import cv2

pose_estimator = PoseEstimator(estimator_flag=asone.YOLOV8M_POSE, use_cuda=True) #set use_cuda=False to use cpu
img = cv2.imread('data/sample_imgs/test2.jpg')
kpts = pose_estimator.estimate_image(img)
img = utils.draw_kpts(img, kpts)
```

- Now you can use Yolov8 and Yolov7-w6 for pose estimation. The flags are provided in [benchmark](asone/linux/Instructions/Benchmarking.md) tables.

```python
# Pose Estimation on video
import asone
from asone import PoseEstimator, utils

pose_estimator = PoseEstimator(estimator_flag=asone.YOLOV7_W6_POSE, use_cuda=True) #set use_cuda=False to use cpu
estimator = pose_estimator.estimate_video('data/sample_videos/football1.mp4')
for kpts, frame_details in estimator:
    frame, _, __ = frame_details
    img = utils.draw_kpts(frame, kpts)
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

To setup ASOne using Docker follow instructions given in [docker setup](asone/linux/Instructions/Docker-Setup.md)🐳

### ToDo 📝

- [x] First Release
- [x] Import trained models
- [x] Simplify code even further
- [x] Updated for YOLOv8
- [x] OCR and Counting
- [x] OCSORT, StrongSORT, MoTPy
- [x] M1/2 Apple Silicon Compatibility
- [x] Pose Estimation YOLOv7/v8
- [x] YOLO-NAS
- [x] SAM Integration


| Offered By 💼 :                                                                                                                                                  | Maintained By 👨‍💻 :                                                                                                                                    |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| [![AugmentedStarups](https://user-images.githubusercontent.com/107035454/195115263-d3271ef3-973b-40a4-83c8-0ade8727dd40.png)](https://augmentedstartups.com) | [![AxcelerateAI](https://user-images.githubusercontent.com/107035454/195114870-691c8a52-fcf0-462e-9e02-a720fc83b93f.png)](https://axcelerate.ai/) |
