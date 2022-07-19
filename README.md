# ASOne

## Setup detectron2
1. Clone the Repo
```
git clone https://github.com/facebookresearch/detectron2.git
```
2. Goto the detectron2 directory
```
cd detectron2
```
3. Download some sample images in this folder

## Docker Setup

1. Build docker contanier
```
docker build -t [IMAGE_NAME]:[TAG]
```

- IMAGE_NAME = Asign a name to image
- TAG = Asign a tag to image

2. Run Docker Contaner

```
docker run --env="DISPLAY" --net=host -v ${PWD}:/workbase/ --name gui -it [IMAGE_NAME]:[TAG]
```

3. In Docker terminal run demo.py file

```
python demo/demo.py --input [PATH_TO_TEST_IMAGE]  --output [PATH_TO_OUTPUT_IMAGE] \
  --opts MODEL.DEVICE [DEVICE] \ 
  MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
```

- PATH_TO_TEST_IMAGE = Path of test image
- PATH_TO_OUTPUT_IMAGE = Path of Results
- DEVICE = device to use i.e. cpu or gpu
