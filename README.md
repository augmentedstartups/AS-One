# ASOne

![croped](https://user-images.githubusercontent.com/107035454/195083948-4873d60a-3ac7-4279-8770-535488f4a097.png)

#### Table of Contents
- [Introduction](#introduction)
- [Benchmarking](#benchmarking)
    - [Trackers](#trackers)
        - [DeepSort](#deepsort)
        - [ByteTrack](#bytetrack)
        - [NorFair](#norfair)
    - [Detectors](#detectors)
        - [YOLOv5](#yolov5)
        - [YOLOv6](#yolov6)
        - [YOLOv7](#yolov7)
        - [YOLOr](#yolor)
        - [YOLOx](#yolox)

- Asone Library Installation
    - [Install In Docker Container](#install-in-docker-container)
        - [Prerequisite](#prerequisite)
        - [Installation](#installation)
    - [Install Locally](#install-locally)



# Introduction

Asone is a python wrapper for multiple detection and tracking algorithms all at one place. Different trackers such as `ByteTrack`, `DeepSort` or `NorFair` can be integrated with different versions of `YOLO` with minimum lines of code.
This python wrapper provides yolo models in both `ONNX` and `PyTorch` versions.

### Prerequisite

- If using windows, Make sure you have [MS Build tools](https://aka.ms/vs/17/release/vs_BuildTools.exe) installed in system. 
- [Download git for windows](https://git-scm.com/download/win) if not installed.

Usage:

```
# linux
python3 -m venv .env
source .env/bin/activate

# windows
python -m venv .env
.env\Scripts\activate

pip install numpy Cython

pip install asone


# for windows
pip install -e git+https://github.com/samson-wang/cython_bbox.git#egg=cython-bbox

# for linux
pip install cython-bbox

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

### To change Detector or Tracker, you only have to chnage the flag.

change Tracker:

```
dt_obj = ASOne(tracker=asone.BYTETRACK, detector=asone.YOLOX_DARKNET_PYTORCH, use_cuda=True)
// Chnage tracker
dt_obj = ASOne(tracker=asone.DEEPSORT, detector=asone.YOLOX_DARKNET_PYTORCH, use_cuda=True)
```
change Detector:

```
dt_obj = ASOne(tracker=asone.BYTETRACK, detector=asone.YOLOX_DARKNET_PYTORCH, use_cuda=True)
// Chnage detector
dt_obj = ASOne(tracker=asone.BYTERACK, detector=asone.YOLOX_S_PYTORCH, use_cuda=True)
```
# Benchmarking
## Hardware Used:
- CPU: Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz
- GPU: 8GB (RTX2080)  

## Trackers

#### DeepSort

| Model           |  Model Flag  |  FPS-GPU   | FPS-CPU
|---------------- |-----------| -----------| --------
|DeepSort-ONNX-Yolov5s|DEEPSORT|13|3.2|
|DeepSort-Pytorch-Yolov5s|DEEPSORT|13|3.2|

### ByteTrack

| Model           |  Model Flag  |  FPS-GPU   | FPS-CPU
|---------------- |-----------| -----------| --------
|ByteTrack-ONNX-YOLOv5s|BYTETRACK|33.7|17.4|
|ByteTrack-Pytorch-Sample-YOLOv5s|BYTETRACK|33.7|17.4|

### NorFair

| Model           |  Model Flag  |  FPS-GPU   | FPS-CPU
|---------------- |-----------| -----------| --------
|tryolab-ONNX-YOLOv5s|NORFAIR|25.8|12|
|tryolab-Pytorch-YOLOv5s|NORFAIR|25.8|12|

## Detectors
### YOLOv5
|    Pytorch                      |ONNX                         |
|:-------------------------------:|:-----------------------------:|
|<table>  <thead>  <tr><th>Model Name / Model Flag</th>  <th>FPS-GPU</th>  <th>FPS-CPU</th>    </tr>  </thead>  <tbody>  <tr><td>YOLOV5X6_PYTORCH</td>  <td>20.8</td>  <td>3.69</td> </tr>  <tr> <td>YOLOV5S_PYTORCH</td> <td>57.25</td>  <td>25.4</td>    </tr>  <tr> <td>YOLOV5N_PYTORCH</td> <td>68</td>  <td>45</td>    </tr> <tr> <td>YOLOV5M_PYTORCH</td> <td>54</td>  <td>14</td>    </tr><tr> <td>YOLOV5L_PYTORCH</td> <td>40.06</td>  <td>8.28</td> </tr><tr> <td>YOLOV5X_PYTORCH</td> <td>28.8</td>  <td>4.32</td>    </tr><tr> <td>YOLOV5N6_PYTORCH</td> <td>63.5</td>  <td>39</td>    </tr><tr> <td>YOLOV5S6_PYTORCH</td> <td>58</td>  <td>23</td>    </tr><tr> <td>YOLOV5M6_PYTORCH</td> <td>49</td>  <td>10</td>    </tr><tr> <td>YOLOV5L6_PYTORCH </td> <td>33</td>  <td>6.5</td>    </tr> </tbody>  </table>| <table>  <thead>  <tr><th>Model Name / Model Flag</th>  <th>FPS-GPU</th>  <th>FPS-CPU</th>    </tr>  </thead>  <tbody>  <tr><td>YOLOV5X6_ONNX</td>  <td>2.58</td>  <td>2.46</td> </tr>  <tr> <td>YOLOV5S_ONNX</td> <td>17</td>  <td>16.35</td>    </tr>  <tr> <td>YOLOV5N_ONNX</td> <td>57.25</td>  <td>35.23</td>    </tr> <tr> <td>YOLOV5M_ONNX</td> <td>45.8</td>  <td>11.17</td>    </tr><tr> <td>YOLOV5L_ONNX</td> <td>4.07</td>  <td>4.36</td> </tr><tr> <td>YOLOV5X_ONNX</td> <td>2.32</td>  <td>2.6</td>    </tr><tr> <td>YOLOV5N6_ONNX</td> <td>28.6</td>  <td>32.7</td>    </tr><tr> <td>YOLOV5S6_ONNX</td> <td>17</td>  <td>16.35</td>    </tr><tr> <td>YOLOV5M6_ONNX</td> <td>7.5</td>  <td>7.6</td>    </tr><tr> <td>YOLOV5L6_ONNX   </td> <td>3.7</td>  <td>3.98</td>    </tr> </tbody>  </table>|

### YOLOv6
|    Pytorch                      |ONNX                         |
|:-------------------------------:|:-----------------------------:|
|<table>  <thead>  <tr><th>Model Name / Model Flag</th>  <th>FPS-GPU</th>  <th>FPS-CPU</th>    </tr>  </thead>  <tbody>  <tr><td>YOLOV6N_PYTORCH</td>  <td>65.4</td>  <td>35.32</td> </tr>  <tr> <td>YOLOV6T_PYTORCH</td> <td>63</td>  <td>15.21</td>    </tr>  <tr> <td>YOLOV6S_PYTORCH</td> <td>49.24</td>  <td>20</td>    </tr> <tr> <td>YOLOV6M_PYTORCH</td> <td>35</td>  <td>9.96</td>    </tr><tr> <td>YOLOV6L_PYTORCH</td> <td>31</td>  <td>6.2</td> </tr><tr> <td>YOLOV6L_RELU_PYTORCH</td> <td>27</td>  <td>6.3</td>    </tr><tr> <td>YOLOV6S_REPOPT_PYTORCH</td> <td>63.5</td>  <td>39</td>    </tr> </tbody>  </table>| <table>  <thead>  <tr><th>Model Name / Model Flag</th>  <th>FPS-GPU</th>  <th>FPS-CPU</th>    </tr>  </thead>  <tbody>  <tr><td>YOLOV6N_ONNX</td>  <td>50</td>  <td>30</td> </tr>  <tr> <td>YOLOV6T_ONNX</td> <td>45.8</td>  <td>16</td>    </tr>  <tr> <td>YOLOV6S_ONNX</td> <td>41</td>  <td>13.8</td>    </tr> <tr> <td>YOLOV6M_ONNX</td> <td>25</td>  <td>6.07</td>    </tr><tr> <td>YOLOV6L_ONNNX</td> <td>17.7</td>  <td>3.32</td> </tr><tr> <td>YOLOV6L_RELU_ONNX</td> <td>19.15</td>  <td>4.36</td>    </tr><tr> <td>YOLOV6S_REPOPT_ONNX</td> <td>63.5</td>  <td>39</td>    </tr> </tbody>  </table>|

### YOLOv7
|    Pytorch                      |ONNX                         |
|:-------------------------------:|:-----------------------------:|
|<table>  <thead>  <tr><th>Model Name / Model Flag</th>  <th>FPS-GPU</th>  <th>FPS-CPU</th>    </tr>  </thead>  <tbody>  <tr><td>YOLOV7_TINY_PYTORCH</td>  <td>53</td>  <td>19</td> </tr>  <tr> <td>YOLOV7_PYTORCH</td> <td>38</td>  <td>6.83</td>    </tr>  <tr> <td>YOLOV7_X_PYTORCH</td> <td>28</td>  <td>4.36</td>    </tr> <tr> <td>YOLOV7_W6_PYTORCH</td> <td>32.7</td>  <td>7.26</td>    </tr><tr> <td>YOLOV7_E6_PYTORCH</td> <td>15.26</td>  <td>3.07</td> </tr><tr> <td>YOLOV7_D6_PYTORCH</td> <td>21</td>  <td>3.78</td>    </tr><tr> <td>YOLOV7_E6E_PYTORCH</td> <td>24</td>  <td>3.36</td>    </tr> </tbody>  </table>| <table>  <thead>  <tr><th>Model Name / Model Flag</th>  <th>FPS-GPU</th>  <th>FPS-CPU</th>    </tr>  </thead>  <tbody>  <tr><td>YOLOV7_TINY_ONNX</td>  <td>41.6</td>  <td>22</td> </tr>  <tr> <td>YOLOV7_ONNX</td> <td>26</td>  <td>3.78</td>    </tr>  <tr> <td>YOLOV7_X_ONNX</td> <td>19.08</td>  <td>2.35</td>    </tr> <tr> <td>YOLOV7_W6_ONNX</td> <td>28.6</td>  <td>5.2</td>    </tr><tr> <td>YOLOV7_E6_ONNX</td> <td>14.3</td>  <td>2.97</td> </tr><tr> <td>YOLOV7_D6_ONNX</td> <td>18.32</td>  <td>2.58</td>    </tr><tr> <td>YOLOV7_E6E_ONNX</td> <td>15.26</td>  <td>2.09</td>    </tr> </tbody>  </table>|

### YOLOr
|    Pytorch                      |ONNX                         |
|:-------------------------------:|:-----------------------------:|
|<table>  <thead>  <tr><th>Model Name / Model Flag</th>  <th>FPS-GPU</th>  <th>FPS-CPU</th>    </tr>  </thead>  <tbody>  <tr><td>YOLOR_CSP_X_PYTORCH</td>  <td>28.6</td>  <td>1.83</td> </tr>  <tr> <td>YOLOR_CSP_X_STAR_PYTORCH</td> <td>30</td>  <td>1.76</td>    </tr>  <tr> <td>YOLOR_CSP_STAR_PYTORCH</td> <td>38.1</td>  <td>2.86</td>    </tr> <tr> <td>YOLOR_CSP_PYTORCH</td> <td>38</td>  <td>2.77</td>    </tr><tr> <td>YOLOR_P6_PYTORCH</td> <td>20</td>  <td>1.57</td> </tr></tbody>  </table>| <table>  <thead>  <tr><th>Model Name / Model Flag</th>  <th>FPS-GPU</th>  <th>FPS-CPU</th>    </tr>  </thead>  <tbody>  <tr><td>YOLOR_CSP_X_ONNX</td>  <td>15.7</td>  <td>2.53</td> </tr>  <tr> <td>YOLOR_CSP_X_STAR_ONNX</td> <td>15.79</td>  <td>2.05</td>    </tr>  <tr> <td>YOLOR_CSP_STAR_ONNX</td> <td>18.32</td>  <td>3.34</td>    </tr> <tr> <td>YOLOR_CSP_ONNX</td> <td>15.7</td>  <td>2.53</td>    </tr><tr> <td>YOLOR_P6_ONNX</td> <td>25.4</td>  <td>5.58</td> </tr></tbody>  </table>|

### YOLOx
|    Pytorch                      |ONNX                         |
|:-------------------------------:|:-----------------------------:|
|<table>  <thead>  <tr><th>Model Name / Model Flag</th>  <th>FPS-GPU</th>  <th>FPS-CPU</th>    </tr>  </thead>  <tbody>  <tr><td>YOLOX_L_PYTORCH</td>  <td>2.58</td>  <td>2.31</td> </tr>  <tr> <td>YOLOX_NANO_PYTORCH</td> <td>35</td>  <td>32</td>    </tr>  <tr> <td>YOLOX_TINY_PYTORCH</td> <td>25.4</td>  <td>25.4</td>    </tr> <tr> <td>YOLOX_DARKNET_PYTORCH</td> <td>2</td>  <td>1.94</td>    </tr><tr> <td>YOLOX_S_PYTORCH</td> <td>9.54</td>  <td>9.7</td> </tr><tr> <td>YOLOX_M_PYTORCH</td> <td>4.4</td>  <td>4.36</td>    </tr><tr> <td>YOLOX_X_PYTORCH</td> <td>15.64</td>  <td>1.39</td>    </tr> </tbody>  </table>| <table>  <thead>  <tr><th>Model Name / Model Flag</th>  <th>FPS-GPU</th>  <th>FPS-CPU</th>    </tr>  </thead>  <tbody>  <tr><td>YOLOX_L_ONNX</td>  <td>22.9</td>  <td>3.07</td> </tr>  <tr> <td>YOLOX_NANO_ONNX</td> <td>59</td>  <td>54</td>    </tr>  <tr> <td>YOLOX_TINY_ONNX</td> <td>60</td>  <td>35</td>    </tr> <tr> <td>YOLOX_DARKNET_ONNX</td> <td>24</td>  <td>3.36</td>    </tr><tr> <td>YOLOX_S_ONNX</td> <td>45</td>  <td>13.8</td> </tr><tr> <td>YOLOX_M_ONNX</td> <td>32</td>  <td>6.54</td>    </tr><tr> <td>YOLOX_X_ONNX</td> <td>15.79</td>  <td>2.03</td>    </tr> </tbody>  </table>|





# Asone Library Installation

## Install In Docker Container

### Prerequisite

- Make sure you have docker installed in your system. if not, reffer to docker installation for [Linux](asone-linux/README.md), [Windows](asone-windows/README.md)
- If using windows, Make sure you have [MS Build tools](https://aka.ms/vs/17/release/vs_BuildTools.exe) installed in system. 


### Installation

1. Clone the repo

```
git clone https://github.com/axcelerateai/asone.git
cd asone
```
2. If using windows, Run this command in command prompt.
```
set PWD=%cd%
```
2. Run docker coompose command.

```
# To test on Linux with GPU 
docker compose run linux-gpu

# To test on Windows with GPU 
docker compose run windows-gpu
```

```
# To test on Linux with CPU 
docker compose run linux

# To test on Windows with CPU 
docker compose run windows
```

3. In docker terminal.

```
# if using gpu
python main.py [VIDEO_PATH] --save

# if using cpu
python main.py [VIDEO_PATH] --cpu --save
```




|Offered By: |Maintained By:|
|-------------|-------------|
|[![AugmentedStarups](https://user-images.githubusercontent.com/107035454/195115263-d3271ef3-973b-40a4-83c8-0ade8727dd40.png)](https://augmentedstartups.com)|[![AxcelerateAI](https://user-images.githubusercontent.com/107035454/195114870-691c8a52-fcf0-462e-9e02-a720fc83b93f.png)](https://axcelerate.ai/)|

