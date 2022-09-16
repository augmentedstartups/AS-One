# ASOne

#### Table of Contents
- System Setup
    - [Linux](asone-linux)
    - [Windows](asone-windows)
- [Asone Library Installation](#asone-library-installation)


# Asone Library Installation

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
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```
6. Install asone

```
pip install .
```

5. Test it by runiing demo.py

```
python main.py [VIDEO_PATH]
```
