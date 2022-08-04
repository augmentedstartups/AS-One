# ASOne

#### Table of Contents

- [Docker Intallation](#docker-installation)
  - [Ubuntu](#ubuntu)
    - [Install Using Shell Script](#install-using-shell-script)
    - [Manual Installation](#manuall-install)
- [Docker Demo](#docker-demo)
  - [Docker Build](#docker-build)
  - [Docker Demo](#docker-demo-1)
    - [Setup Detectron](#setup-detectron)
    - [Demo using docker compose](#demo-using-docker-compose-file)
    - [Demo using docker](#demo-using-docker)

# Docker Installation

## Ubuntu

#### Install using Shell Script

```
chmod a+x docker-installation.sh
./docker-installation.sh
```

- [NOTE] If there is an error while installing docker, try removing apt-lists and resinstalling.

```
sudo rm -rf /var/lib/apt/lists/*
```

#### Manuall Install

1. Run following command to remove all old versions on docker

```
sudo apt-get remove docker docker-engine docker.io containerd runc
```

2. Set up Repository

- Update the apt package index and install packages to allow apt to use a repository over HTTPS:

```
sudo apt-get update
sudo apt-get install \
    ca-certificates \
    curl \
    gnupg \
    lsb-release
```

- Add Docker’s official GPG key:

```
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
```

- Use the following command to set up the repository:

```
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

3. Install Docker Engine

- Update the apt package index, and install the latest version of Docker Engine, containerd, and Docker Compose:

```
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin
```

# Docker Demo

## Docker Build

#### Build docker contanier

```
docker build -t [IMAGE_NAME]:[TAG] .
```

e.g. `docker build -t asone:latest` .

  - `IMAGE_NAME` = Asign a name to image
  - `TAG` = Asign a tag to image

## Docker Demo

#### Setting Up detectron2

1. Clone the Repo

```
git clone https://github.com/facebookresearch/detectron2.git
```

2. Goto the detectron2 directory

```
cd detectron2
```

3. Download some sample images in this folder

#### Demo Using Docker Compose File

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

#### Demo Using Docker

1. Run Docker Contaner

```
docker run --gpus all --env="DISPLAY" --net=host -v [PATH_TO_LOCAL_DIR]:/workspace/  -it [IMAGE_NAME]:[TAG]
```

  - `PATH_TO_LOCAL_DIR` = Path to detectron2 directory or use `${PWD}` if already in that directory

e.g. 
```
docker run --gpus all --env="DISPLAY" --net=host -v $PWD:/workspace/  -it asone:latest
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
python demo/demo.py --input test.jpeg  --output result.jpg \
  --opts MODEL.DEVICE gpu \
  MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
```
