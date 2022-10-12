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
- Asone Library Installation
    - [Install In Docker Container](#install-in-docker-container)
        - [Prerequisite](#prerequisite)
        - [Installation](#installation)
    - [Install Locally](#install-locally)



# Introduction

Asone is a python wrapper for multiple detection and tracking algorithms all at one place. Different trackers such as `ByteTrack`, `DeepSort` or `NorFair` can be integrated with different versions of `YOLO` with minimum lines of code.
This python wrapper provides yolo models in both `ONNX` and `PyTorch` versions.

Usage:

```
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
dt_obj.start_tracking('sample_videos/test.mp4')
```
Results on provided sample video

https://user-images.githubusercontent.com/107035454/195079926-aee47eac-0430-4ada-8cc7-cc9d1d13c889.mp4


# Benchmarking

## Trackers

#### DeepSort

| Model           |   Hardware | FPS-GPU | FPS-CPU
|----------------|----------------|-----------| -----------|
|[DeepSort-Pytorch-YOLOv3](https://github.com/ZQPei/deep_sort_pytorch)|GPU: 8GB (RTX2080)<br>CPU:  Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz|12.2|2.5|
|[DeepSort-ONNX-YOLOv3](https://github.com/ZQPei/deep_sort_pytorch)|GPU: 8GB (RTX2080)<br>CPU:  Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz|10.3|2.7|
|[PP-DeepSort](https://github.com/PaddlePaddle/PaddleDetection)|GPU: 8GB (RTX2080)<br>CPU:  Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz|7.9|-|
|[mmtracking-DeepSort](https://github.com/open-mmlab/mmtracking/tree/master/configs/mot/deepsort)|GPU: 8GB (RTX2080)<br>CPU:  Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz|8.2|-|
|[StrongSort-Pytorch-Yolov5](https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet.git)|GPU: 8GB (RTX2080)<br>CPU:  Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz|8.5|1|
|[DeepSort-ONNX-Yolov5](https://github.com/ZQPei/deep_sort_pytorch)|GPU: 8GB (RTX2080)<br>CPU:  Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz|13|3.2|
|[StrongSort-ONNX-Yolov5](https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet.git)|GPU: 8GB (RTX2080)<br>CPU:  Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz|11|5.5|

### ByteTrack

| Model           |   Hardware | FPS-GPU | FPS-CPU
|----------------|----------------|-----------| -----------|
|[ByteTrack-Pytorch-YOLOX](https://github.com/ifzhang/ByteTrack)|GPU: 8GB (RTX2080)<br>CPU:  Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz|9|0.5|
|[ByteTrack-ONNX-YOLOX](https://github.com/ifzhang/ByteTrack)|GPU: 8GB (RTX2080)<br>CPU:  Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz|4|0.5|
|[yolox-bytetrack-mcmot-sample](https://github.com/Kazuhito00/yolox-bytetrack-mcmot-sample)|GPU: 8GB (RTX2080)<br>CPU:  Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz|48|-|
|[ByteTrack-Sample-YOLOX](https://github.com/Kazuhito00/yolox-bytetrack-mcmot-sample)|GPU: 8GB (RTX2080)<br>CPU:  Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz|20.9|7|
|[ByteTrack-ONNX-YOLOv5s](https://github.com/ifzhang/ByteTrack)|GPU: 8GB (RTX2080)<br>CPU:  Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz|20.1|10.7|
|[ByteTrack-Sample-YOLOv5s](https://github.com/Kazuhito00/yolox-bytetrack-mcmot-sample)|GPU: 8GB (RTX2080)<br>CPU:  Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz|33.7|17.4|

### NorFair

| Model           |   Hardware | FPS-GPU | FPS-CPU
|----------------|----------------|-----------| -----------|
|[tryolab-YOLOv4](https://github.com/tryolabs/norfair)|GPU: 8GB (RTX2080)<br>CPU:  Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz|27.3|3.7|
|[tryolab-ONNX-YOLOv5s](https://github.com/tryolabs/norfair)|GPU: 8GB (RTX2080)<br>CPU:  Intel(R) Core(TM) i9-9900K CPU @ 3.60GHz|25.8|12|





# Asone Library Installation

## Install In Docker Container

### Prerequisite

- Make sure you have docker installed in your system. if not, reffer to docker installation for [Linux](asone-linux/README.md), [Windows](asone-windows/README.md)
- If using windows, Make sure you have [MS Build tools](https://devblogs.microsoft.com/cppblog/announcing-visual-c-build-tools-2015-standalone-c-tools-for-build-environments) installed in system, 

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
docker compose run asone

# To test on Windows with CPU 
docker compose run windows
```

3. In docker terminal.

```
# if using gpu
python main.py [VIDEO_PATH]

# if using cpu
python main.py [VIDEO_PATH] --cpu
```



## Install Locally

1. Clone the repo

```
git clone https://github.com/axcelerateai/asone.git
cd asone
```

2. Create virtual env.

```
python3 -m venv .env
```
3. Activate venv

```
# linux
source .env/bin/activate

# windows
.env\Scripts\activate
```

4. Install asone

```
pip install numpy Cython

pip install .
```

5. Intall pre-requisite

```
# for windows
pip install -e git+https://github.com/samson-wang/cython_bbox.git#egg=cython-bbox

# for linux
pip install cython-bbox
```

6. Install torch

```
# for gpu
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
```

7. Test it by runiing main.py

```
# if using gpu
python main.py [VIDEO_PATH]

# if using cpu
python main.py [VIDEO_PATH] --cpu
```
