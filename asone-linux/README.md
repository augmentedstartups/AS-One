# ASOne

#### Table of Contents
- [Docker Intallation](#docker-installation)
  - [Ubuntu](#ubuntu)
    - [Prerequisite](#prerequisite)
    - [Install Using Shell Script](#install-using-shell-script)
- [Test Docker](#test-docker)
  - [Test using Compose File](#test-using-docker-compose-file)
  - [Test Detectron2](#test-detectron2)

# Docker Installation

## Ubuntu
#### Prerequisite
1. For systems with `GPU` please verify you have nvidia drivers installed. run

```
nvidia-smi
```
if you see something like the following. Then you can continue running [shell script](#install-using-shell-script)

![](imgs/nvidia-drivers.png)

 or you can install nvidia drivers using following command:

```
sudo apt-get install nvidia-driver-YYY nvidia-dkms-YYY
```
- `YYY`= Nvidia driver version

e.g `sudo apt-get install nvidia-driver-510 nvidia-dkms-510`

#### Install using Shell Script

```
chmod a+x docker-installation.sh
./docker-installation.sh
```
  - For systems with `GPU` run following commands after installing docker.
  ```
  sudo apt-get install -y nvidia-docker2
  sudo systemctl restart docker

  sudo xhost +local:docker
  ``` 

- [NOTE] If there is an error while installing docker, try removing apt-lists and resinstalling.

```
sudo rm -rf /var/lib/apt/lists/*
```

- In case shell script keeps failing or you want to install manually follow steps in [Manual Installation](Instructions/Manual-Installation.md)


# Test Docker

## Test Using Docker Compose File

1. Run container without GPU

```
docker compose run linux
```

2. Run container with GPU

```
docker compose run linux-GPU
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

- To build and run docker container manually follow instructions for [Manual Build](Instructions/Manual-Build.md)


## Test Detectron2

To test detectron2 follow instuctions in [Demo Detectron2](Instructions/Demo-Detectron2.md)