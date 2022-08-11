# ASOne



# Docker Manual Build

## Docker Installation

- If you haven't installed docker first install it by following provided instructions [here](../)

## Build Image Manually

1. Run the follwoing command to build docker image

```
docker build -t [IMAGE_NAME]:[TAG] .
```
  - `IMAGE_NAME` = Asign a name to image
  - `TAG` = Asign a tag to image

e.g. `docker build -t asone:latest .` 

## Run Build Image

1. To run the build image in docker container with `cpu`.

```
docker run --env="DISPLAY" --net=host -v [PATH_TO_LOCAL_DIR]:/workspace/  -it [IMAGE_NAME]:[TAG]
```
  - `IMAGE_NAME` = Asign a name to image
  - `TAG` = Asign a tag to image
  - `PATH_TO_LOCAL_DIR` = Path to detectron2 directory or use `$PWD` if already in that directory


e.g `docker run --env="DISPLAY" --net=host -v $PWD:/workspace/ -it asone:latest`

2. To run th ebuild image in docker container with `gpu`

```
docker run --gpus all --env="DISPLAY" --net=host -v [PATH_TO_LOCAL_DIR]:/workspace/  -it [IMAGE_NAME]:[TAG]
```
  - `IMAGE_NAME` = Asign a name to image
  - `TAG` = Asign a tag to image
  - `PATH_TO_LOCAL_DIR` = Path to detectron2 directory or use `$PWD` if already in that directory

e.g `docker run --gpus all --env="DISPLAY" --net=host -v $PWD:/workspace/ -it asone:latest`