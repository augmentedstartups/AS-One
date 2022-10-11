# ASOne

#### Table of Contents
- [Introduction](introduction)
- System Setup
    - [Linux](asone-linux)
    - [Windows](asone-windows)
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
pip install asone
```

```
import asone
from asone import ASOne

dt_obj = ASOne(tracker=asone.BYTETRACK, detector=asone.YOLOX_DARKNET_PYTORCH, use_cuda=True)
dt_obj.start_tracking([VIDEO_PATH])

```

- `VIDEO_PATH` = Path of input video


# Asone Library Installation

## Install In Docker Container

### Prerequisite

- Make sure you have docker installed in your system. if not reffer to [Docker Setup](asone-linux/README.md)


### Installation

1. Clone the repo

```
git clone https://github.com/axcelerateai/asone.git
cd asone
```

2. Run docker coompose command.

```
# if you wanna test on gpu system
docker compose run linux-gpu
```

```
# if you wanna test on cpu system
docker compose run linux
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

# Windows

.env\Scripts\activate
```

4. Intall pre-requisite

```
pip install numpy Cython
```
```
# for windows
pip install -e git+https://github.com/samson-wang/cython_bbox.git#egg=cython-bbox

# for linux
pip install cython-bbox
```

5. Install torch

```
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```
6. Install asone

```
pip install .
```

5. Test it by runiing main.py

```
# if using gpu
python main.py [VIDEO_PATH]

# if using cpu
python main.py [VIDEO_PATH] --cpu
```
