# ASOne

#### Table of Contents

- [Docker Intallation](#docker-installation)
  - [Ubuntu](#ubuntu)
    - [Install Using Shell Script](#install-using-shell-script)
- [Test Docker](#test-docker)
  - [Test using Compose File](#test-using-docker-compose-file)
  - [Test Detectron2](#test-detectron2)

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

- In case shell script keeps failing or you want to install manually follow steps in [Manual Installation](Manual-Installation.md)


# Test Docker

## Test Using Docker Compose File

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


## Test Detectron2

To test detectron2 follow instuctions in [Demo Detectron2](Demo-Detectron2.md)