# ASOne

## Docker Installation for Windows Systems

### Steps to Install Docker on Windows:
- Kindly read the requirements section carefully, Given below!
- Go to [Docker Official Website](https://docs.docker.com/desktop/install/windows-install/)
- Download the installer file from the website
- Open the installer and make sure to select both the checkboxes

### System Requirements
Windows machine must meet the following requirements to successfully install the docker:

**With WSL 2 backend**
Type **winver** in RUN to check the version of the installed windows.

- Windows 11 64-bit: Home or Pro version 21H2 or higher,\
  or Enterprise or Education version 21H2 or     higher
- Windows 10 64-bit: Home or Pro 21H1 (build 19043) or higher,\
  or Enterprise or Education 20H2 (build 19042) or higher
- Enable the WSL 2 feature on Windows. For detailed instructions,\
  refer to the [wsl installation](https://docs.microsoft.com/en-us/windows/wsl/install)

- The following hardware prerequisites are required to successfully\
  run WSL 2 on Windows 10 or Windows 11:  

- 64-bit processor with Second Level Address Translation (SLAT)

- 4GB system RAM

- BIOS-level hardware virtualization support must be enabled in the \
  BIOS settings. For more information, see [Virtualization](https://docs.docker.com/desktop/troubleshoot/topics/)

- Download and install the [Linux kernel update package](https://docs.microsoft.com/en-us/windows/wsl/install-manual#step-4---download-the-linux-kernel-update-package)








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
docker run --gpus all --env="DISPLAY" --net=host -v [PATH_TO_LOCAL_DIR]:/workbase/  -it [IMAGE_NAME]:[TAG]
```
- PATH_TO_LOCAL_DIR = Path to detectron2 directory or use `${PWD}` if already in that directory

3. In Docker terminal run demo.py file

```
python demo/demo.py --input [PATH_TO_TEST_IMAGE]  --output [PATH_TO_OUTPUT_IMAGE] \
  --opts MODEL.DEVICE [DEVICE] \ 
  MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
```

- PATH_TO_TEST_IMAGE = Path of test image
- PATH_TO_OUTPUT_IMAGE = Path of Results
- DEVICE = device to use i.e. cpu or gpu
