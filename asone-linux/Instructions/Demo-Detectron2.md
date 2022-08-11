# ASOne

#### Table of Contents

- [Docker Demo](#docker-demo)
  - [Docker Demo](#docker-demo-1)
    - [Setup Detectron](#setup-detectron)
    - [Demo using docker compose](#demo-using-docker-compose-file)
    - [Test Detectron2](#test-detectron2)

# Docker Installation

- If you haven't installed docker first install it by following provided instructions [here](../)

## Docker Demo

### Setting Up detectron2

1. Clone the Repo

```
git clone https://github.com/facebookresearch/detectron2.git
```

2. Goto the detectron2 directory

```
cd detectron2
```

3. Download some sample images in this folder

### Demo Using Docker Compose File

1. Run container without gpu

```
docker compose run linux
```

2. Run container with gpu

```
docker compose run linux-gpu
```

- To test DISPLAY is shared with docker properly:

```
python main.py
```

- if an image show up then everything is working properly.

- if you see an error saying `qt.qpa.xcb: could not connect to display` that means your display is not accessible to docker.

Try this:
```
sudo xhost +local:docker
```

- To build and run docker container manually follow instructions for [Manual Build](Manual-Build.md)


### Test Detectron2

1. After docker container starts properly, in docker terminal change directory using.

```
cd detectron2
```

2. In Docker terminal run demo.py file

```
python demo/demo.py --input [PATH_TO_TEST_IMAGE]  --output [PATH_TO_OUTPUT_IMAGE] \
  --opts MODEL.DEVICE [DEVICE] \
  MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
```

- `PATH_TO_TEST_IMAGE` = Path of test image
- `PATH_TO_OUTPUT_IMAGE` = Path of Results
- `DEVICE` = device to use i.e. `cpu` or `gpu`

e.g.
```
python demo/demo.py --input ../test.jpeg  --output ../result.jpg \
  --opts MODEL.DEVICE gpu \
  MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
```
